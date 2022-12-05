import random
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
import os
# torch.multiprocessing.set_start_method('spawn')
from multiprocessing import set_start_method
from config_pr import opt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# from pytorchtools import EarlyStopping
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import math
import re
from torch.nn.functional import normalize
from pyvi.ViTokenizer import tokenize
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from scipy import sparse
import csv
import pickle
from underthesea import text_normalize

def process_actor_director(x):
    if type(x) != str:
        return ["other"]
    else:
        if len(x)==0:
            return ["other"]
        return x.split(",")

def check_list(list_x, all_x):
    new_list = []
    for x in list_x[0]:
        if x in all_x:
            new_list.append(x)
        else:
            new_list.append("other")    
    return [new_list]

def normalizeString(s):
    # Tách dấu câu nếu kí tự liền nhau
    s = re.sub(r"([.!?,\-\&\(\)\[\]])", r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
    
    s = text_normalize(s.lower())
    
    return s

def get_unique_category(pd_film):
    list_category_unique = []
    categorys = pd_film[pd_film['raw_category_name'].notnull()]['raw_category_name'].apply(lambda x: x.split(","))
    for list_category in categorys:
        # print(list_category)
        for category in list_category:
            if category not in list_category_unique:
                list_category_unique.append(category)
    # print(list_category_unique)
    return list_category_unique

def get_all_category(pd_film_series):
    unique_category_film_series = get_unique_category(pd_film_series)
    return unique_category_film_series

pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_episode = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)

#Onehot Actor-2237, 1067
pd_film_series['actor_id'] = pd_film_series['actor_id'].apply(lambda x: process_actor_director(x))
list_actor = pd_film_series['actor_id'].tolist()
dict_film_actor = pd_film_series[['series_id', 'actor_id']].set_index('series_id').T.to_dict('list')
unique_list_actor = []
for x in list_actor:
    unique_list_actor.extend(x)
unique_list_actor = list(set(unique_list_actor))
list_numbers_of_actor = {}

for actor_id in unique_list_actor:
    for list_actor_id in list_actor:
        if actor_id in list_actor_id:
            if list_numbers_of_actor.get(actor_id) is None:
                list_numbers_of_actor[actor_id] = 1
            else:
                list_numbers_of_actor[actor_id] +=1

filter_actor = []
for actor_id in list_numbers_of_actor.keys():
    if list_numbers_of_actor[actor_id] > 1:
        print(actor_id)
        filter_actor.append(actor_id)
    else:
        filter_actor.append("other")
filter_actor = list(set(filter_actor))
filter_actor.sort()            
onehot_actor=MultiLabelBinarizer()
onehot_actor.fit([filter_actor])

#Onehot director-444
pd_film_series['director_id'] = pd_film_series['director_id'].apply(lambda x: process_actor_director(x))
list_director = pd_film_series['director_id'].tolist()
dict_film_director = pd_film_series[['series_id', 'director_id']].set_index('series_id').T.to_dict('list')
unique_list_director = []
for x in list_director:
    unique_list_director.extend(x)
unique_list_director = list(set(unique_list_director))
list_numbers_of_director = {}

for director_id in unique_list_director:
    for list_director_id in list_director:
        if director_id in list_director_id:
            if list_numbers_of_director.get(director_id) is None:
                list_numbers_of_director[director_id] = 1
            else:
                list_numbers_of_director[director_id] +=1

filter_director = []
for director_id in list_numbers_of_director.keys():
    if list_numbers_of_director[director_id] > 1:
        print(director_id)
        filter_director.append(director_id)
    else:
        filter_director.append("other")
filter_director = list(set(filter_director))
filter_director.sort()            
onehot_director=MultiLabelBinarizer()
onehot_director.fit([filter_director])

dict_films_duration = pd_film_episode.groupby('series_id')['duration'].agg([("duration", "sum")]).reset_index().set_index('series_id').T.to_dict('list')
ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan.dropna(inplace=True)
pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_series_drop_nan.dropna(inplace=True)
        
onehot_is_series=OneHotEncoder(handle_unknown='ignore',sparse=False)
onehot_is_series.fit(pd_film_series_drop_nan[['is_series']])
all_films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_series)['series_id'].apply(lambda x: str(x)).tolist()
onehot_categorical=MultiLabelBinarizer()
all_category = get_all_category(pd_film_series)
onehot_categorical.fit([all_category])
    
onehot_country=OneHotEncoder(handle_unknown='ignore',sparse=False)
pd_film_series_country = pd.DataFrame(pd_film_series, columns = ['country'])
pd_film_series_country = pd_film_series_country[pd_film_series_country['country'].notnull()]
onehot_country.fit(pd_film_series_country[['country']])
model_tokenizer= AutoTokenizer.from_pretrained("vinai/phobert-base")
bertsentence= SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device="cuda:0")

def get_record_by_item_id(item_id, pd_film_series):
    record = pd_film_series[pd_film_series['series_id'] == int(item_id)].to_dict('records')[0]
    return record

def get_ft_item(film_id):
    record_trg_item = get_record_by_item_id(film_id, pd_film_series)
    duration = dict_films_duration[int(film_id)][0]
    fe_record_trg_item = embedding_item(film_id, duration, record_trg_item, bertsentence,onehot_is_series,onehot_country,onehot_categorical)
    return fe_record_trg_item

def embedding_item(film_id, duration, record, bertsentence,onehot_is_series,onehot_country,onehot_categorical):
    data={}
    
    if type(record['description']) != str:
        if type(record['series_name']) != str:
            data['description'] = "other"
        else:
            data['description'] = normalizeString(str(record['series_name']))
    else:           
        data['description']= normalizeString(str(record['series_name']) + " " + str(record['description']))
        
    data['country']=[record['country']]
    try:
        data['raw_category_name'] = record['raw_category_name'].split(",")
    except: 
        data['raw_category_name'] = ""
               
    data['director_name']=record['director_name']
    
    data['actor_name']=record['actor_name']
    
    data['is_series']=[record['is_series']]
    
    data['duration']= duration
    
    if np.isnan(record['release_year']):
        # return None
        data['release_year'] = 2022
    else:        
        data['release_year']=record['release_year']
    
    sentences=[tokenize(data['description'])]
    
    description_emb= torch.FloatTensor(bertsentence.encode(sentences))
    
    is_series_emb=onehot_is_series.transform([data['is_series']])
    is_series_emb=torch.FloatTensor(is_series_emb)
    
    duration=torch.FloatTensor([[data['duration']]])
    
    release_year=torch.FloatTensor([[data['release_year']]])

    if type(data['country']) == str:
        country=onehot_country.transform([data['country']]) 
    else:
        country=onehot_country.transform([[""]])               
    country=torch.FloatTensor(country)
    
    category=onehot_categorical.transform([data['raw_category_name']])
    category=torch.FloatTensor(category)
    
    #Actor
    actor = onehot_actor.transform(check_list(dict_film_actor[int(film_id)], filter_actor))
    actor = torch.FloatTensor(actor)
    
    #Director
    director = onehot_director.transform(check_list(dict_film_director[int(film_id)], filter_director))
    director = torch.FloatTensor(director)
    
    # print(dict_film_actor[int(film_id)])
    # print(dict_film_director[int(film_id)])
    # print(actor.size())
    # print(director.size())
    # emb_film = torch.concat([description_emb, country, category,actor, director, is_series_emb, duration,release_year], 1)
    # print(emb_film.size())
    # return emb_film.detach().cpu().numpy()[0]
    # return (normalize(description_emb).to(opt.device), normalize(country).to(opt.device), normalize(category).to(opt.device),normalize(actor).to(opt.device), 
    #         normalize(director).to(opt.device), normalize(is_series_emb).to(opt.device),normalize(duration).to(opt.device), normalize(release_year).to(opt.device))
    
    return (description_emb.to(opt.device), country.to(opt.device), category.to(opt.device),actor.to(opt.device), 
            director.to(opt.device), is_series_emb.to(opt.device),duration.to(opt.device), release_year.to(opt.device))
    
pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
dict_fe_item = {}
list_film_remove = []
list_ft_film = []

for item_id in list_film_id:
    dict_fe_item[item_id] = get_ft_item(item_id)
#     if dict_fe_item[item_id] is None:          
#         list_film_remove.append(item_id)

# for item_id in list_film_remove:
#     list_film_id.remove(item_id)

with open('raw_features_unnormal.pickle', 'wb') as handle:
    pickle.dump(dict_fe_item, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                
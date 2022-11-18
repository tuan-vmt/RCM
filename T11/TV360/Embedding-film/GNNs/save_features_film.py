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

def normalizeString(s):
    # Tách dấu câu nếu kí tự liền nhau
    s = re.sub(r"([.!?,\-\&\(\)\[\]])", r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
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
bertsentence= SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device="cpu")

def get_record_by_item_id(item_id, pd_film_series):
    record = pd_film_series[pd_film_series['series_id'] == int(item_id)].to_dict('records')[0]
    return record

def get_ft_item(film_id):
    record_trg_item = get_record_by_item_id(film_id, pd_film_series)
    duration = dict_films_duration[int(film_id)][0]
    fe_record_trg_item = embedding_item(duration, record_trg_item, bertsentence,onehot_is_series,onehot_country,onehot_categorical,model_tokenizer)
    return fe_record_trg_item

def embedding_item(duration, record, bertsentence,onehot_is_series,onehot_country,onehot_categorical,bert_words):
    data={}
    
    if type(record['description']) != str:
        return None
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
    
    if type(data['actor_name']) == str:
        actor=[bert_words.encode(data['actor_name'])]
    else:
        actor=[bert_words.encode("")]                
    actor=torch.FloatTensor(actor)
    if actor.shape[1]<16:
        actor=F.pad(actor,(1,15-actor.shape[1]),mode='constant', value=0)
    elif actor.shape[1]>16:
        actor=actor[:,:16]
        
    if type(data['director_name']) == str:      
        director=[bert_words.encode(data['director_name'])]
    else:
        director=[bert_words.encode('')]
                       
    director=torch.FloatTensor(director)
    if director.shape[1]<16:
        director=F.pad(director,(1,15-director.shape[1]),mode='constant', value=0)
    elif director.shape[1]>16:
        director=director[:,:16]
    emb_film = torch.concat([normalize(description_emb), normalize(country), normalize(category),normalize(actor), normalize(director), is_series_emb, duration,release_year], 1)
    # print(emb_film.size())
    return emb_film.detach().cpu().numpy()[0]

pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
dict_fe_item = {}
list_film_remove = []
list_ft_film = []

for item_id in list_film_id:
    dict_fe_item[item_id] = get_ft_item(item_id)
    if dict_fe_item[item_id] is None:          
        list_film_remove.append(item_id)

for item_id in list_film_remove:
    list_film_id.remove(item_id)

with open('raw_features.pickle', 'wb') as handle:
    pickle.dump(dict_fe_item, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                
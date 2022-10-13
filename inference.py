import random
import numpy as np
import json
import torch
from tqdm import tqdm
import model
from model import TV360Recommend
from dataloader import Tv360Dataset, normalizeString, clean, unique, get_all_category, get_record_by_item_id
import os
# torch.multiprocessing.set_start_method('spawn')
from multiprocessing import set_start_method
from config_path import opt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from sklearn.metrics import classification_report
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

pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_episode = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)
ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan.dropna(inplace=True)
pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_series_drop_nan.dropna(inplace=True)
pd_film_episode_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)
pd_film_episode_drop_nan.dropna(inplace=True)
        
onehot_is_series=OneHotEncoder(handle_unknown='ignore',sparse=False)
onehot_is_series.fit(pd_film_series_drop_nan[['is_series']])

onehot_categorical=MultiLabelBinarizer()
all_category = get_all_category(pd_film_series, pd_film_episode)
onehot_categorical.fit([all_category])
    
onehot_country=OneHotEncoder(handle_unknown='ignore',sparse=False)
pd_film_series_country = pd.DataFrame(pd_film_series, columns = ['country'])
pd_film_series_country = pd_film_series_country[pd_film_series_country['country'].notnull()]
pd_film_episode_country = pd.DataFrame(pd_film_episode, columns = ['country'])
pd_film_episode_country = pd_film_episode_country[pd_film_episode_country['country'].notnull()]
pd_country = pd.concat([pd_film_series_country, pd_film_episode_country])
onehot_country.fit(pd_country[['country']])
model_tokenizer= AutoTokenizer.from_pretrained("vinai/phobert-base")
bertsentence= SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device=opt.device)
            
def get_ft_user(user_id):
    user_info = ccai[ccai['user_id'] == int(user_id)].to_dict('records')[0]
    onehot_province_user=OneHotEncoder(handle_unknown='ignore',sparse=False)
    onehot_province_user.fit(ccai_drop_nan[['province_name']])
    if type(user_info['province_name']) == str:
        onehot_province_user = onehot_province_user.transform([[user_info['province_name']]])[0]
    else:
        onehot_province_user = onehot_province_user.transform([[""]])[0]        
        
    gender = user_info['gender']
    age = user_info['age']    
    if math.isnan(gender):
        return None
    if math.isnan(age):
        return None              
    ccai_embedding = np.hstack((np.array([gender, age]), onehot_province_user))
    ccai_embedding = torch.FloatTensor(ccai_embedding)
    return ccai_embedding

def get_history(user_id):
    path_list_file_log_film = []
    for file_path in os.listdir(opt.folder + "data/"):
        if file_path.find("log_film") >= 0:
            path_list_file_log_film.append(file_path)
    l_hst_items = []
    total_watch_duration = {}       
    for log_film_path in path_list_file_log_film:
        # print(log_film_path)
        hst_users_info = pd.read_csv(opt.folder + "data/"  + log_film_path)
        hst_users_info = hst_users_info.dropna() 
        hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: clean(x))
        hst_users_info['user_id'] = hst_users_info['user_id'].apply(lambda x: clean(x))
        
        hst_items_id = unique(list(hst_users_info[hst_users_info['user_id'] == user_id]['content_id']))
        # l_hst_items = [(u,int(log_film_path[9:13])) for u in hst_items]
        for item in hst_items_id:
            watch_duration = list(hst_users_info[(hst_users_info['user_id'] == user_id) & (hst_users_info['content_id'] == str(item))]['watch_duration'])[0]
            l_hst_items.append((item,int(log_film_path[9:13])))
            if total_watch_duration.get(item) is None:
                total_watch_duration[item] = watch_duration
            else:
                total_watch_duration[item] += watch_duration
    
    def sort_date(date):
        return date[1]               
    l_hst_items.sort(reverse=False, key = sort_date)
    
    hst_items = []
    if len(l_hst_items) == 0:
        return None
    elif len(l_hst_items) < opt.numbers_of_hst_films:
        count = opt.numbers_of_hst_films // len(l_hst_items) + 1        
        for i in range(count):
            hst_items.extend(l_hst_items)
        hst_items = hst_items[:opt.numbers_of_hst_films]
    elif len(l_hst_items) >= opt.numbers_of_hst_films:
        hst_items = l_hst_items[-opt.numbers_of_hst_films:]
    hst_items = [y[0] for y in hst_items]
    
    return hst_items, total_watch_duration     

def embedding_item(record, type_record, bertsentence,onehot_is_series,onehot_country,onehot_categorical,bert_words):
    data={}
    
    if type_record == "series":
        if type(record['description']) != str:
            return None
        data['description']= normalizeString(str(record['series_name']) + " " + str(record['description']))
    else:  
        if type(record['episode_description']) != str:
            return None      
        data['description']= normalizeString(str(record['episode_name'])+ " " + str(record['episode_description']))
        
    data['country']=[record['country']]
    try:
        data['raw_category_name'] = record['raw_category_name'].split(",")
    except: 
        data['raw_category_name'] = ""
               
    data['director_name']=record['director_name']
    
    data['actor_name']=record['actor_name']
    
    data['is_series']=[record['is_series']]
    
    data['duration']=record['duration']
    
    if np.isnan(record['release_year']):
        return None
        # data['release_year'] = 2022
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

    # print(description_emb.size())
    # print(country.size())
    # print(category.size())
    # print(actor.size())
    # print(director.size())
    # print(is_series_emb.size())
    # print(duration.size())
    # print(release_year.size())
    return (normalize(description_emb).unsqueeze(0).to(opt.device), normalize(country).unsqueeze(0).to(opt.device), normalize(category).unsqueeze(0).to(opt.device),normalize(actor).unsqueeze(0).to(opt.device), normalize(director).unsqueeze(0).to(opt.device), is_series_emb.unsqueeze(0).to(opt.device), duration.unsqueeze(0).to(opt.device),release_year.unsqueeze(0).to(opt.device))


def get_ft_item(film_id):
    record_trg_item, type_record_trg_item = get_record_by_item_id(film_id, pd_film_series, pd_film_episode)
    fe_record_trg_item = embedding_item(record_trg_item, type_record_trg_item, bertsentence,onehot_is_series,onehot_country,onehot_categorical,model_tokenizer)
    return fe_record_trg_item,record_trg_item


if __name__ == "__main__":
    ccai_embedding = get_ft_user("69959644").unsqueeze(0)
    fe_trg_item,_ = get_ft_item("732613")
    
    hst_film_id, total_watch_duration = get_history("69959644")
    fe_hst_items = []
    list_rating = []
    for film_id in hst_film_id:
        fe_item, record_item = get_ft_item(film_id)
        rating = min(float(total_watch_duration[film_id])/float(record_item['duration']), 1)
        if rating > 0.5:
            print(film_id)
        list_rating.append(rating)
        fe_hst_items.append(fe_item)
    print(list_rating)    
    model = TV360Recommend().to(opt.device)
    model.load_state_dict(torch.load("save_models1/30.pth"))
    model.eval()
    with torch.no_grad():
        inputs = (fe_hst_items, fe_trg_item, ccai_embedding, [list_rating])
        outputs = model(inputs)
        print(outputs)   
    
    
        
# def inference(user_id, film_id):
#     device = opt.device
    
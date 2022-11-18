import random
import numpy as np
import json
import torch
from tqdm import tqdm
import argparse
import model
from model import TV360Recommend
from dataloader import find_seri_id, normalizeString, clean, unique, get_all_category, get_record_by_item_id
import os
# torch.multiprocessing.set_start_method('spawn')
from multiprocessing import set_start_method
from config_path import opt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action = 'ignore', category = UserWarning)
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

pd_film_series = pd.read_csv(opt.folder + "data/" + opt.path_film_series)
pd_film_episode = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
dict_films_duration = pd_film_episode.groupby('series_id')['duration'].agg(
    [("duration", "sum")]).reset_index().set_index('series_id').T.to_dict('list')
ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan.dropna(inplace=True)
pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/" + opt.path_film_series)
pd_film_series_drop_nan.dropna(inplace=True)

onehot_is_series = OneHotEncoder(handle_unknown='ignore', sparse=False)
onehot_is_series.fit(pd_film_series_drop_nan[['is_series']])
all_films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_series)['series_id'].apply(lambda x: str(x)).tolist()
onehot_categorical = MultiLabelBinarizer()
all_category = get_all_category(pd_film_series)
onehot_categorical.fit([all_category])

onehot_country = OneHotEncoder(handle_unknown='ignore', sparse=False)
pd_film_series_country = pd.DataFrame(pd_film_series, columns=['country'])
pd_film_series_country = pd_film_series_country[pd_film_series_country['country'].notnull()]
onehot_country.fit(pd_film_series_country[['country']])
model_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
bertsentence = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device="cpu")


def get_ft_user(user_id):
    user_info = ccai[ccai['user_id'] == int(user_id)].to_dict('records')[0]
    onehot_province_user = OneHotEncoder(handle_unknown='ignore', sparse=False)
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


def get_history(user_id, day, list_film_id):
    path_file_user_info = opt.path_file_user_info
    folder = opt.folder
    path_list_file_log_film = []

    for file_path in os.listdir(folder + "data/"):
        if file_path.find("log_film") >= 0:
            path_list_file_log_film.append(file_path)
    users_info = pd.read_csv(folder + path_file_user_info)

    list_hst_users_info = []
    for log_film_path in path_list_file_log_film:
        # print(hst_users_info['content_id'])
        if log_film_path != "log_film_0625_0630.csv":
            hst_users_info_5_days = pd.read_csv(opt.folder + "data/" + log_film_path)
            list_hst_users_info.append(hst_users_info_5_days)

    hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)
    # hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
    hst_users_info['user_id'] = hst_users_info['user_id'].apply(lambda x: clean(x))
    hst_users_info = hst_users_info[hst_users_info['user_id'] == user_id]
    if len(hst_users_info) == 0:
        print("The user hasn't watched any movies yet")
    hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
    hst_users_info = hst_users_info[hst_users_info['content_id'].isin(list_film_id)]
    hst_users_info = hst_users_info[hst_users_info['partition'] < day]
    hst_film_id_group_by_user = hst_users_info.groupby(["user_id", "content_id"])['watch_duration'].agg(
        [("watch_duration", "sum")]).reset_index()
    hst_film_id_group_by_user['partition'] = \
    hst_users_info.groupby(["user_id", "content_id"])['partition'].apply(lambda x: min(x)).reset_index()['partition']

    users_items = hst_film_id_group_by_user.groupby(['user_id'])['content_id'].apply(list).reset_index()
    users_items['partition'] = hst_film_id_group_by_user.groupby('user_id')['partition'].apply(list).reset_index()[
        'partition']
    users_items['watch_duration'] = \
    hst_film_id_group_by_user.groupby('user_id')['watch_duration'].apply(list).reset_index()['watch_duration']
    users_items = users_items.set_index('user_id').T.to_dict('list')

    def sort_date(value):
        v = list(zip(value[0], value[1], value[2]))
        v.sort(key=lambda x: x[1])
        if len(v) < 40:
            v = v + (40 - len(v)) * [v[-1]]
        # print(list(zip(value[0], value[1], value[2])).sort(key=lambda x: x[1]))
        return v[-opt.numbers_of_hst_films:]

    hst_users_items = {k: sort_date(v) for k, v in users_items.items()}
    print(hst_users_items)
    return hst_users_items


def embedding_item(duration, record, bertsentence, onehot_is_series, onehot_country, onehot_categorical, bert_words):
    data = {}
    
    if type(record['description']) != str:
        return None
    data['description'] = normalizeString(str(record['series_name']) + " " + str(record['description']))
        
    data['country'] = [record['country']]
    try:
        data['raw_category_name'] = record['raw_category_name'].split(", ")
    except: 
        data['raw_category_name'] = ""
               
    data['director_name'] = record['director_name']
    
    data['actor_name'] = record['actor_name']
    
    data['is_series'] = [record['is_series']]
    
    data['duration'] = duration
    
    if np.isnan(record['release_year']):
        data['release_year'] = 2022
    else:        
        data['release_year'] = record['release_year']
    
    sentences = [tokenize(data['description'])]
    
    description_emb = torch.FloatTensor(bertsentence.encode(sentences))
    
    is_series_emb = onehot_is_series.transform([data['is_series']])
    is_series_emb = torch.FloatTensor(is_series_emb)
    
    duration = torch.FloatTensor([[data['duration']]])
    
    release_year = torch.FloatTensor([[data['release_year']]])

    if type(data['country']) == str:
        country = onehot_country.transform([data['country']]) 
    else:
        country = onehot_country.transform([[""]])               
    country = torch.FloatTensor(country)
    
    category = onehot_categorical.transform([data['raw_category_name']])
    category = torch.FloatTensor(category)
    
    if type(data['actor_name']) == str:
        actor = [bert_words.encode(data['actor_name'])]
    else:
        actor = [bert_words.encode("")]                
    actor = torch.FloatTensor(actor)
    if actor.shape[1] < 16:
        actor = F.pad(actor, (1, 15-actor.shape[1]), mode = 'constant', value = 0)
    elif actor.shape[1] > 16:
        actor = actor[:, :16]
        
    if type(data['director_name']) == str:      
        director = [bert_words.encode(data['director_name'])]
    else:
        director = [bert_words.encode('')]
                       
    director = torch.FloatTensor(director)
    if director.shape[1] < 16:
        director = F.pad(director, (1, 15-director.shape[1]), mode = 'constant', value = 0)
    elif director.shape[1] > 16:
        director = director[:, :16]
    return (normalize(description_emb).unsqueeze(0).to(opt.device), normalize(country).unsqueeze(0).to(opt.device), normalize(category).unsqueeze(0).to(opt.device), normalize(actor).unsqueeze(0).to(opt.device), normalize(director).unsqueeze(0).to(opt.device), is_series_emb.unsqueeze(0).to(opt.device), duration.unsqueeze(0).to(opt.device), release_year.unsqueeze(0).to(opt.device))


def get_ft_user(user_id):
    try:
        user_info = ccai[ccai['user_id'] == int(user_id)].to_dict('records')[0]
        onehot_province_user = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        onehot_province_user.fit(ccai_drop_nan[['province_name']])
        if type(user_info['province_name']) == str:
            onehot_province_user = onehot_province_user.transform([[user_info['province_name']]])[0]
        else:
            onehot_province_user = onehot_province_user.transform([[""]])[0]
    except:
        print(user_id)
        return None              
        
    gender = user_info['gender']
    age = user_info['age']    
    if math.isnan(gender):
        return None
    if math.isnan(age):
        return None              
    ccai_embedding = np.hstack((np.array([gender, age]), onehot_province_user))
    ccai_embedding = torch.FloatTensor(ccai_embedding)
    return ccai_embedding


def get_ft_item(film_id):
    record_trg_item = get_record_by_item_id(film_id, pd_film_series)
    duration = dict_films_duration[int(film_id)][0]
    fe_record_trg_item = embedding_item(duration, record_trg_item, bertsentence, onehot_is_series, onehot_country, onehot_categorical, model_tokenizer)
    return fe_record_trg_item


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default = 'TV360/RCM/save_models/6_best_accuracy.pth', help = 'initial weights path')
    parser.add_argument('--device', default='cpu', help = 'cuda device, i.e. cuda:0 or 0, 1, 2, 3 or cpu')
    parser.add_argument('--user-id',type=str, default = "100082365")
    parser.add_argument('--day', type=int, default=20220625, help='Day from 20220625 to 20220630')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_opt(True)
    ccai_embedding = get_ft_user(args.user_id).unsqueeze(0)
    # fe_trg_item = get_ft_item(args.series_id)

    list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
       
    model = TV360Recommend().to(args.device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    dict_fe_item = {}
    list_film_remove = []
    list_ft_film = []
    for item_id in list_film_id:
        dict_fe_item[item_id] = get_ft_item(item_id)
        if dict_fe_item[item_id] is None:
            list_film_remove.append(item_id)

    for item_id in list_film_remove:
        list_film_id.remove(item_id)

    for i in range(8):
        list_ft_i = [dict_fe_item[item_id][i] for item_id in list_film_id]
        ft_i = torch.cat(list_ft_i, 0)
        # print(ft_i.size())
        list_ft_film.append(ft_i)
        # print(dict_fe_item['14495'])
    length_list_ft_film = len(list_film_id)
    list_film_id.sort()
    onehot_film = MultiLabelBinarizer()
    onehot_film.fit([list_film_id])
    ccai_embedding = torch.cat(length_list_ft_film * [ccai_embedding.unsqueeze(0)], 0)
    hst_users_items = get_history(args.user_id, args.day, list_film_id)
    fe_hst_items = []
    list_rating = []
    for (film_id, partition, watch_duration) in hst_users_items[args.user_id]:
        batch_fe_item = []
        fe_item = dict_fe_item[film_id]
        for i in range(8):
            ft_i = torch.cat(length_list_ft_film * [fe_item[i]], 0)
            batch_fe_item.append(ft_i)
        rating = min(float(watch_duration) / float(dict_films_duration[int(film_id)][0]), 1)
        rating = torch.cat(length_list_ft_film * [torch.tensor([rating])], 0)
        list_rating.append(rating)
        fe_hst_items.append(batch_fe_item)

    with torch.no_grad():
        inputs = (fe_hst_items, list_ft_film, ccai_embedding, list_rating)
        outputs = model(inputs)
        print(outputs)   
    
    
        
# def inference(user_id, film_id):
#     device = args.device
    
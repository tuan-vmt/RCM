import random
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
import model
from model import TV360Recommend
from dataloader import Tv360Dataset, normalizeString, clean, unique, get_all_category, get_record_by_item_id, find_seri_id
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
from scipy import sparse
import bottleneck as bn
import pickle5 as pickle
from copy import deepcopy
from pytorch_model_summary import summary

pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_episode = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)
dict_films_duration = pd_film_episode.groupby('series_id')['duration'].agg([("duration", "sum")]).reset_index().set_index('series_id').T.to_dict('list')
ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
all_profile_users_id = ccai['profile_id'].tolist()
all_profile_users_id = [str(id) for id in all_profile_users_id]
ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
ccai_drop_nan.dropna(inplace=True)
pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_series_drop_nan.dropna(inplace=True)
    
def get_ft_item(film_id, data):
    
    if data[str(film_id)] is None:
        return None
    else:
        description_emb, country, category, actor, director, is_series_emb, duration ,release_year = data[str(film_id)]
        return (description_emb.unsqueeze(0).to(args.device), country.unsqueeze(0).to(args.device), category.unsqueeze(0).to(args.device),
                actor.unsqueeze(0).to(args.device), director.unsqueeze(0).to(args.device), 
                is_series_emb.unsqueeze(0).to(args.device), duration.unsqueeze(0).to(args.device),
                release_year.unsqueeze(0).to(args.device)
        )
    
def get_ft_user(user_id):
    try:
        user_info = ccai[ccai["profile_id"] == int(user_id)].to_dict('records')[0]
        onehot_province_user=OneHotEncoder(handle_unknown='ignore',sparse=False)
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

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='TV360/RCM/save_models/85_best_accuracy.pth', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. cuda:0 or 0,1,2,3 or cpu')
    parser.add_argument('--top-k', type=int, default=100, help='top-k film co do yeu thich cao nhat')

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = parse_opt(True)
    list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
    dict_film_durations = pd_film_series[["series_id", "duration"]]
    dict_film_durations = dict_film_durations.set_index('series_id').T.to_dict('list')
    path_file_user_info = opt.path_file_user_info
    folder = opt.folder
    path_list_file_log_film = []

    for file_path in os.listdir(folder + "data/"):
        if file_path.find("log_film") >= 0:
            path_list_file_log_film.append(file_path)
    users_info = pd.read_csv(folder + path_file_user_info)
    # users_info.dropna()
    # users_items = {}
    list_hst_users_info = []
    for log_film_path in path_list_file_log_film:
        # print(hst_users_info['content_id'])
        if log_film_path != "log_film_0625_0630.csv":
            hst_users_info_5_days = pd.read_csv(opt.folder + "data/" + log_film_path)
            list_hst_users_info.append(hst_users_info_5_days)
         # hst_users_info = hst_users_info.dropna()
         
    model = TV360Recommend().to(args.device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    dict_fe_item = {}
    list_film_remove = []
    list_ft_film = []
    raw_features_file = '/home/admin1/mnt_raid/source/tuanvm/AirFlow/airflow-tutorial/TV360/RCM/raw_features_unnormal.pickle'
    with open(raw_features_file, 'rb') as f:
        data = pickle.load(f)
    for item_id in list_film_id:
        dict_fe_item[item_id] = get_ft_item(item_id, data)
        if dict_fe_item[item_id] is None:          
            list_film_remove.append(item_id)    
    
    for item_id in list_film_remove:
        list_film_id.remove(item_id) 
    
    list_film_id.sort()
    print("Length Film Eval:", len(list_film_id))
    for i in range(8):
        list_ft_i = [dict_fe_item[item_id][i] for item_id in list_film_id]
        ft_i = torch.cat(list_ft_i, 0)
        # print(ft_i.size())
        list_ft_film.append(ft_i)              
    # print(dict_fe_item['14495'])
    length_list_ft_film = len(list_film_id)
    
    #Hst Film      
    hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)    
    # hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
    hst_users_info["profile_id"] = hst_users_info["profile_id"].apply(lambda x: clean(x))
    hst_users_info = hst_users_info[hst_users_info["profile_id"].isin(all_profile_users_id)]
    hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
    hst_users_info = hst_users_info[hst_users_info['content_id'].isin(list_film_id)]
    hst_users_info = hst_users_info[(hst_users_info['content_id'] != "-1") & (hst_users_info["profile_id"] != "-1")]
    hst_film_id_group_by_user = hst_users_info.groupby(["profile_id", "content_id"])['watch_duration'].agg([("watch_duration", "sum")]).reset_index()
    hst_film_id_group_by_user['partition'] = hst_users_info.groupby(["profile_id", "content_id"])['partition'].apply(lambda x: max(x)).reset_index()['partition']
    # hst_film_id_group_by_user['partition'] = hst_users_info.groupby(["profile_id", "content_id"])['partition'].apply(list).reset_index()['partition']
    users_items = hst_film_id_group_by_user.groupby(["profile_id"])['content_id'].apply(list).apply(lambda x: x if len(x) > 0 else -1).reset_index()
    # users_items = hst_film_id_group_by_user.groupby(["profile_id"])['content_id'].apply(list).apply(lambda x: x if len(x) >= opt.numbers_of_hst_films else -1).reset_index()
    #apply(lambda x: x if len(x) >= opt.numbers_of_hst_films else -1)
    users_items['partition'] = hst_film_id_group_by_user.groupby("profile_id")['partition'].apply(list).reset_index()['partition']
    users_items['watch_duration'] = hst_film_id_group_by_user.groupby("profile_id")['watch_duration'].apply(list).reset_index()['watch_duration']
    users_items = users_items[users_items['content_id'] != -1]
    # print(users_items)
    users_items = users_items.set_index("profile_id").T.to_dict('list')
    
    dict_mask_index_film_user = {}
          
    def sort_date(key, value):
        v = list(zip(value[0], value[1], value[2]))
        v.sort(key=lambda x: x[1])
        # print(list(zip(value[0], value[1], value[2])).sort(key=lambda x: x[1]))
        if len(v) < opt.numbers_of_hst_films:
            v.extend((opt.numbers_of_hst_films - len(v)) * [v[-1]])               
        return v[-opt.numbers_of_hst_films:]
    
    hst_users_items = {k: sort_date(k, v) for k, v in users_items.items()} 
           
    onehot_film = MultiLabelBinarizer()
    onehot_film.fit([list_film_id])
    pred = []
    # print(hst_users_items)
    print("Length User: ", len(list(hst_users_items.keys())))       
    with torch.no_grad():   
        for idx, key in enumerate(hst_users_items.keys()):
            # print(i)
            if idx > 200:
                break
            pred_key = []
            hst_films_key = hst_users_items[key]
            ccai_embedding = get_ft_user(key)
            if ccai_embedding is None:
                continue
            ccai_embedding = torch.cat(length_list_ft_film*[ccai_embedding.unsqueeze(0)], 0)
            fe_hst_items = []
            list_rating = []
            for (film_id,partition, watch_duration) in hst_films_key:
                batch_fe_item = []
                fe_item = dict_fe_item[film_id]
                for i in range(8):
                    ft_i = torch.cat(length_list_ft_film*[fe_item[i]], 0)
                    batch_fe_item.append(ft_i)
                    
                if watch_duration > opt.duration_threshold:
                    rating = 1
                else:
                    rating = watch_duration/opt.duration_threshold
                # if watch_duration > dict_film_durations[int(film_id)][0]:
                #     rating = 1
                # else:
                #     rating = watch_duration/dict_film_durations[int(film_id)][0]
                # print(rating)      
                rating = torch.cat(length_list_ft_film*[torch.tensor([rating])], 0)
                
                list_rating.append(rating)
                fe_hst_items.append(batch_fe_item)
                
            list_ft_film1 = deepcopy(list_ft_film)   
            inputs = (fe_hst_items, list_ft_film1, ccai_embedding, list_rating)
            # print(summary(model, inputs))
            # print(model)
            outputs = model(inputs)
            outputs = outputs[:, 0].detach().cpu().numpy()
            indexs = bn.argpartition(-np.array([outputs]), args.top_k, axis=1)
            sort_idx = indexs[:, :args.top_k]
            sort_idx = sorted(sort_idx[0], key=lambda x: outputs[x], reverse=True)
            outputs_film = [list_film_id[idx] for idx in sort_idx]
            print("---------------------")
            
            print(outputs_film)
            pred.append(outputs_film)
            
    pred = np.array(pred)
    print(pred.shape)
    
    
    
                
                
                
            
        
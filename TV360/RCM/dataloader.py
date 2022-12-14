import re
import sys
import inspect
import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
import os
from config_path import opt
import math
import time
from pyvi.ViTokenizer import tokenize
from torch.nn import functional as F
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
import pickle5 as pickle
import random

all_films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_series)['series_id'].apply(lambda x: str(x)).tolist()
all_episode_id = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
dict_films_duration = all_episode_id.groupby('series_id')['duration'].agg([("duration", "sum")]).reset_index().set_index('series_id').T.to_dict('list')
list_eps_series = all_episode_id[['episode_id', 'series_id']].set_index('episode_id').T.to_dict('dict')

def normalizeString(s):
    # Tách dấu câu nếu kí tự liền nhau
    s = re.sub(r"([.!?,\-\&\(\)\[\]])", r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def clean(x):
    if str(x).find("\'") >= 0 or str(x).find("\"") >= 0 or str(x).find("-") >= 0 or str(x).find("=") >= 0 \
            or str(x).find("<") >= 0 or re.search('[a-zA-Z]', str(x)) is not None:
        return "-1"
    else:
        return str(int(x))

def find_seri_id(x):
    if str(x).find("\'") >= 0 or str(x).find("\"") >= 0 or str(x).find("-") >= 0 or str(x).find("=") >= 0 \
            or str(x).find("<") >= 0 or re.search('[a-zA-Z]', str(x)) is not None:
        return "-1"
    else:
        film_id = int(x)
        # print(film_id)
        if film_id in list_eps_series.keys():
            # print(film_id)
            return str(list_eps_series[film_id]['series_id'])
        else:
            return "-1"
        
def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list


def split_data():
    print("Start")
    st = time.time()
    path_file_user_info = opt.path_file_user_info
    folder = opt.folder
    path_list_file_log_film = []

    for file_path in os.listdir(folder + "data/"):
        if file_path.find("log_film") >= 0:
            path_list_file_log_film.append(file_path)
    users_info = pd.read_csv(folder + path_file_user_info)
    # users_info.dropna()
    all_users_id = [str(user_id) for user_id in users_info['profile_id'].tolist()]
    # users_items = {}
    list_hst_users_info = []
    for log_film_path in path_list_file_log_film:
        # print(hst_users_info['content_id'])
        if log_film_path != "log_film_0625_0630.csv":
            hst_users_info_5_days = pd.read_csv(opt.folder + "data/" + log_film_path)
            list_hst_users_info.append(hst_users_info_5_days)
         # hst_users_info = hst_users_info.dropna()
    hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)    
    # hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
    hst_users_info['profile_id'] = hst_users_info['profile_id'].apply(lambda x: clean(x))
    hst_users_info = hst_users_info[hst_users_info['profile_id'].isin(all_users_id)]
    hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
    hst_users_info = hst_users_info[(hst_users_info['content_id'] != "-1") & (hst_users_info['profile_id'] != "-1")]
    hst_film_id_group_by_user = hst_users_info.groupby(["profile_id", "content_id"])['watch_duration'].agg([("watch_duration", "sum")]).reset_index()
    hst_film_id_group_by_user['partition'] = hst_users_info.groupby(["profile_id", "content_id"])['partition'].apply(list).reset_index()['partition']

    users_items = hst_film_id_group_by_user.groupby(['profile_id'])['content_id'].apply(list).apply(lambda x: x if len(x) >= 1.5*opt.numbers_of_hst_films else -1).reset_index()
    users_items['partition'] = hst_film_id_group_by_user.groupby('profile_id')['partition'].apply(list).reset_index()['partition']
    users_items['watch_duration'] = hst_film_id_group_by_user.groupby('profile_id')['watch_duration'].apply(list).reset_index()['watch_duration']
    users_items = users_items[users_items['content_id'] != -1]
    users_items = users_items.set_index('profile_id').T.to_dict('list')
    print("Time Split 1: ", time.time()- st)        
    def sort_date(value):
        # v = list(zip(value[0], value[1], value[2]))
        # v.sort(key=lambda x: x[1])
        # # print(list(zip(value[0], value[1], value[2])).sort(key=lambda x: x[1]))
        # return v
        v = []
        for film_id, list_date, watch_duration in zip(value[0], value[1], value[2]):
            list_date = list(set(list_date))
            len_date = len(list_date)            
            v.extend(list(zip(len_date*[film_id], list_date, len_date*[watch_duration])))
        v.sort(key=lambda x: x[1])
        return v
    
    users_items = {k: sort_date(v) for k, v in users_items.items()}
    # Sort by Date
    # filter_list_user = list(users_items.keys())

    # Split History, Train & Val Users-Item
    train_target_users_items = []
    val_target_users_items = []
    list_target_users_item = []
    
    for key in users_items.keys():
        list_hst_key = users_items[key][:opt.numbers_of_hst_films]
        for idx in range(opt.numbers_of_hst_films, len(users_items[key])):
            label = 1 if (users_items[key][idx][2] > opt.duration_threshold) else 0
            list_target_users_item.append((key, list_hst_key, users_items[key][idx], label))
                   
    for key in users_items.keys():
        # hst_users_items[key] = users_items[key][:opt.numbers_of_hst_films]
        for idx in range(len(users_items[key])):
            # print(item)
            # print(type(item))
            if idx == len(users_items[key]) - 1:
                continue
            else:
                if idx + opt.numbers_of_hst_films <= len(users_items[key]) - 1:
                    list_hst_key = users_items[key][idx:(idx + opt.numbers_of_hst_films)]
                    target_item = users_items[key][idx + opt.numbers_of_hst_films]
                    label = 1 if (target_item[2] > opt.duration_threshold) else 0
                    list_target_users_item.append((key, list_hst_key, target_item, label))
                else:
                    length_hst = len(users_items[key]) - 1 - idx
                    list_hst_key =  users_items[key][idx:(idx + length_hst)]
                    list_hst_key.extend((opt.numbers_of_hst_films - length_hst) * [users_items[key][-1]])
                    target_item = users_items[key][-1]
                    label = 1 if (target_item[2] > opt.duration_threshold) else 0
                    list_target_users_item.append((key, list_hst_key, target_item, label))
                    
    random.shuffle(list_target_users_item)
                    
    length_train_trg_users_items = int(0.7 * (len(list_target_users_item)))
    list_train_trg_users_items = list_target_users_item[:length_train_trg_users_items]
    list_val_trg_users_items = list_target_users_item[length_train_trg_users_items:]

    train_target_users_items.extend(list_train_trg_users_items)
    val_target_users_items.extend(list_val_trg_users_items)

    target_users_items = {}
    target_users_items['train'] = train_target_users_items
    target_users_items['val'] = val_target_users_items
    # print(target_users_items)
    # print(hst_users_items)
    print("Done")
    return target_users_items

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


def get_record_by_item_id(item_id, pd_film_series):
    record = pd_film_series[pd_film_series['series_id'] == int(item_id)].to_dict('records')[0]
    return record

def get_ft_user(ccai, onehot_province_user, user_id):
    try:
        user_info = ccai[ccai["profile_id"] == int(user_id)].to_dict('records')[0]
        if type(user_info['province_name']) == str:
            onehot_province_user = onehot_province_user.transform([[user_info['province_name']]])[0]
        else:
            onehot_province_user = onehot_province_user.transform([[""]])[0]
    except:
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

class Tv360Dataset(data.Dataset):
    def __init__(self, target_users_items, phase="train"):
        self.ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan.dropna(inplace=True)
        self.target_users_items = target_users_items[phase]
        self.labels = [it[3] for it in self.target_users_items]
        # self.rating_imdb_rating = {}
        # print(self.pd_film_series['series_id'])
        self.list_ft_user = {}
        self.onehot_province_user = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.onehot_province_user.fit(self.ccai_drop_nan[['province_name']])

        raw_features_file = 'raw_features_unnormal.pickle'
        with open(raw_features_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print("Len data: --------------------------------", len(self.labels))
        print(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # key, list_hst_key, target_item, label
        user_id, list_hst_item_id, target_item, label = self.target_users_items[idx]
        item_id = target_item[0]
        list_rating = []
        # CCAI Embedding
        if self.list_ft_user.get(int(user_id)) is not None:
            ccai_embedding = self.list_ft_user[int(user_id)]
        else:    
            user_info = self.ccai[self.ccai['profile_id'] == int(user_id)].to_dict('records')[0]
            if type(user_info['province_name']) == str:
                onehot_province_user = self.onehot_province_user.transform([[user_info['province_name']]])[0]
            else:
                onehot_province_user = self.onehot_province_user.transform([[""]])[0]
            gender = user_info['gender']
            age = user_info['age']
            if math.isnan(gender):
                return None
            if math.isnan(age):
                return None
            
            ccai_embedding = np.hstack((np.array([gender, age]), onehot_province_user))
            ccai_embedding = torch.FloatTensor(ccai_embedding)
            self.list_ft_user[int(user_id)] = ccai_embedding
            
        # Feature Target Item
        fe_record_trg_item = self.data[str(item_id)]   
        if fe_record_trg_item is None:
            return None
        # Feature History Items
        fe_hst_items = []
        for (item_hst_id, _, total_watch_duration) in list_hst_item_id:
            # duration = dict_films_duration[int(item_id)][0]
            # prefer = float(total_watch_duration) / float(duration)
            if total_watch_duration >= opt.duration_threshold:
                rating = 1
            else:
                rating = float(total_watch_duration) / opt.duration_threshold
            # if total_watch_duration >= self.dict_film_durations[int(item_hst_id)][0]:
            #     rating = 1
            # else:
            #     rating = float(total_watch_duration) / self.dict_film_durations[int(item_hst_id)][0]
            # rating = float(total_watch_duration) / opt.duration_threshold    
            # rating = min(float(total_watch_duration) / float(duration), 1)
            list_rating.append(rating)
            ebd = self.data[item_hst_id]
            if ebd is None:
                return None
            fe_hst_items.append(ebd)
        return (fe_hst_items, fe_record_trg_item, ccai_embedding, list_rating), self.labels[idx]
    
    
class EvalTv360Dataset(data.Dataset):
    def __init__(self, target_users_items):
        self.ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan.dropna(inplace=True)
        self.target_users_items = target_users_items
        # self.rating_imdb_rating = {}
        # print(self.pd_film_series['series_id'])
        self.list_ft_user = {}
        self.onehot_province_user = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.onehot_province_user.fit(self.ccai_drop_nan[['province_name']])

        raw_features_file = 'raw_features_unnormal.pickle'
        with open(raw_features_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.target_users_items)

    def __getitem__(self, idx):
        # key, list_hst_key, target_item, label
        user_id, list_hst_item_id, target_item = self.target_users_items[idx]
        list_rating = []
        # CCAI Embedding
        if self.list_ft_user.get(int(user_id)) is not None:
            ccai_embedding = self.list_ft_user[int(user_id)]
        else:
            ccai_embedding = get_ft_user(self.ccai, self.onehot_province_user, user_id)
            self.list_ft_user[int(user_id)] = ccai_embedding
        if ccai_embedding is None:
            return None   
        # Feature Target Item
        fe_record_trg_item = self.data[str(target_item)]
        if fe_record_trg_item is None:
            return None
        # Feature History Items
        fe_hst_items = []
        for (item_hst_id, _, total_watch_duration) in list_hst_item_id:
            # duration = dict_films_duration[int(item_id)][0]
            # prefer = float(total_watch_duration) / float(duration)
            if total_watch_duration >= opt.duration_threshold:
                rating = 1
            else:
                rating = float(total_watch_duration) / opt.duration_threshold
            # if total_watch_duration >= self.dict_film_durations[int(item_hst_id)][0]:
            #     rating = 1
            # else:
            #     rating = float(total_watch_duration) / self.dict_film_durations[int(item_hst_id)][0]
            # rating = float(total_watch_duration) / opt.duration_threshold    
            # rating = min(float(total_watch_duration) / float(duration), 1)
            list_rating.append(rating)
            ebd = self.data[item_hst_id]
            if ebd is None:
                return None
            fe_hst_items.append(ebd)
        return (fe_hst_items, fe_record_trg_item, ccai_embedding, list_rating), int(user_id)

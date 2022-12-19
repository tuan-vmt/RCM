import random
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
import os
# torch.multiprocessing.set_start_method('spawn')
from multiprocessing import set_start_method
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
from config_path import opt
from torch.nn.functional import normalize
from pyvi.ViTokenizer import tokenize
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from scipy import sparse
import bottleneck as bn
from config_path import opt
import pickle
from multiprocessing import Pool

all_films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_series)['series_id'].apply(lambda x: str(x)).tolist()
all_episode_id = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
dict_films_duration = all_episode_id.groupby('series_id')['duration'].agg([("duration", "sum")]).reset_index().set_index('series_id').T.to_dict('list')
list_eps_series = all_episode_id[['episode_id', 'series_id']].set_index('episode_id').T.to_dict('dict')

def intersection(lst1, lst2):
    # print(list(set(lst1[0]) & set(lst2[0])))
    return len(list(set(lst1[0]) & set(lst2[0])))

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
        
path_list_file_log_film = []

for file_path in os.listdir(opt.folder + "data/"):
    if file_path.find("log_film") >= 0:
        path_list_file_log_film.append(file_path)
users_info = pd.read_csv(opt.folder + opt.path_file_user_info)
# users_info.dropna()
all_users_id = [str(user_id) for user_id in users_info['user_id'].tolist()]
# users_items = {}
list_hst_users_info = []
for log_film_path in path_list_file_log_film:
    print(opt.folder + "data/" + log_film_path)
    hst_users_info_5_days = pd.read_csv(opt.folder + "data/" + log_film_path)
    list_hst_users_info.append(hst_users_info_5_days)
hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)
hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
hst_users_info['profile_id'] = hst_users_info['profile_id'].apply(lambda x: clean(x))

dict_film_profile = hst_users_info.groupby(['content_id'])['profile_id'].apply(list).reset_index()
dict_film_profile = dict_film_profile[dict_film_profile['content_id'] != "-1"]
dict_film_profile = dict_film_profile.set_index('content_id').T.to_dict('list')
# print(dict_film_profile)
dict_ground_truth_film_profile = {}

sort_key = list(dict_film_profile.keys())
sort_key.sort()
print(sort_key)
print(len(sort_key))

for key1 in sort_key:
    for key2 in sort_key:
        if key1 == key2:
            continue  
        if (key1, key2) not in dict_ground_truth_film_profile.keys() and (key2, key1) not in dict_ground_truth_film_profile.keys():
            # print(dict_film_profile[key1])
            # print(dict_film_profile[key2])
            intersect = intersection(dict_film_profile[key1], dict_film_profile[key2])
            if intersect > opt.threshold_nb_of_user:
                score = 1
            else:
                score = intersect / opt.threshold_nb_of_user
            dict_ground_truth_film_profile[(key1, key2)] = score
            # dict_ground_truth_film_profile[(key2, key1)] = score
        else:
            continue
    
with open('groundtruth_feature_weighting_1.pickle', 'wb') as handle:
    pickle.dump(dict_ground_truth_film_profile, handle, protocol=pickle.HIGHEST_PROTOCOL)       
        
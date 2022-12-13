from config_path import opt
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import math
from torch import nn
from config_path import opt
import time
import re
from sklearn.preprocessing import MultiLabelBinarizer
import os
import sys
from underthesea import text_normalize

c = (1 > 0) ? 1 : 0
a = text_normalize('Ðảm baỏ chất lựơng phòng thí nghịêm hoá học')
print(a)
exit()
films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_series)
all_episode_id = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
# print(type(all_episode_id[['episode_id', 'series_id']]))
list_eps_series = all_episode_id[['episode_id', 'series_id']].set_index('episode_id').T.to_dict('dict')
pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
pd_film_episode = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)
list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
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
            # print("HIHI")
            return "-1"
        
def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list

# print(list_film_id)
# list_film_id.sort()
path_list_file_log_film = []

for file_path in os.listdir(opt.folder + "data/"):
    if file_path.find("log_film") >= 0:
        path_list_file_log_film.append(file_path)
list_hst_users_info = []
for log_film_path in path_list_file_log_film:
    # print(hst_users_info['content_id'])
    hst_users_info_5_days = pd.read_csv(opt.folder + "data/" + log_film_path)
    list_hst_users_info.append(hst_users_info_5_days)

users_info = pd.read_csv(opt.folder + opt.path_file_user_info)
# users_info.dropna()
all_users_id = [str(user_id) for user_id in users_info['user_id'].tolist()]
hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)    
# hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
hst_users_info['user_id'] = hst_users_info['user_id'].apply(lambda x: clean(x))
hst_users_info = hst_users_info[hst_users_info['user_id'].isin(all_users_id)]
hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
hst_users_info = hst_users_info[hst_users_info['content_id'].isin(list_film_id)]
hst_users_info = hst_users_info[(hst_users_info['content_id'] != "-1") & (hst_users_info['user_id'] != "-1")]
hst_film_id_group_by_user = hst_users_info.groupby(["content_id"])['user_id'].apply(list).reset_index()
hst_film_id_group_by_user = hst_film_id_group_by_user[['content_id', 'user_id']].set_index('content_id').T.to_dict('list')
a = hst_film_id_group_by_user['8257'][0]
print("-----------------------------")
b = hst_film_id_group_by_user['1835'][0]
d = []
c = 0
for x in a:
    if x in b:
        d.append(x)
        c+=1
print(list(set(d)))        
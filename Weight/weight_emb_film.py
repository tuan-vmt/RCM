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
# warnings.filterwarnings(action='ignore', category=UserWarning)
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
import pickle

def intersection(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))

def score_release_year(year1, year2):
    score = (100 - abs(int(year1) - int(year2)))/100
    print(year1)
    print(year2)
    print(score)
    return score

def score_imdb_rating(film1, film2):
    score = (10 - abs(film1 - film2))/10
    return score

def score_category(list_category_film1, list_category_film2):
    intersect= intersection(list_category_film1, list_category_film2)
    if intersect > opt.threshold_category:
        return 1.0
    else:
        return intersect/opt.threshold_category

def score_country(country1, country2):
    if country1 == country2:
        return 1.0
    else:
        return 0.0
    
def score_actor(list_actor_film1, list_actor_film2):
    intersect = intersection(list_actor_film1, list_actor_film2)
    if intersect > opt.threshold_actor:
        return 1.0
    else:
        return intersect/opt.threshold_actor
        
def score_director(list_director_film1, list_director_film2):
    intersect = intersection(list_director_film1, list_director_film2)
    if intersect > opt.threshold_director:
        return 1.0
    else:
        return intersect/opt.threshold_director
       
def get_unique_category(pd_film):
    list_category_unique = []
    categorys = pd_film[pd_film['raw_category_id'].notnull()]['raw_category_id'].apply(lambda x: x.split(","))
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

def process_actor_director(x):
    if type(x) != str:
        return ["other"]
    else:
        if len(x)==0:
            return ["other"]
        return x.split(",")

def process_release_year(x):
    if np.isnan(x):
        return 2022
    else:
        return x
    
def process_rating(x):
    if np.isnan(x):
        return 6.5
    else:
        return x    
         
pd_film_series = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)

pd_film_series['series_id'] = pd_film_series['series_id'].apply(lambda x: str(int(x)))

#Rating
pd_film_series['imdb_rating'] = pd_film_series['imdb_rating'].apply(lambda x: process_rating(x))

#Release Year
pd_film_series['release_year'] = pd_film_series['release_year'].apply(lambda x: process_release_year(x))

#Director
pd_film_series['director_id'] = pd_film_series['director_id'].apply(lambda x: process_actor_director(x))
list_director = pd_film_series['director_id'].tolist()
unique_list_director = []
for x in list_director:
    unique_list_director.extend(x)
unique_list_director = list(set(unique_list_director))

#Actor
pd_film_series['actor_id'] = pd_film_series['actor_id'].apply(lambda x: process_actor_director(x))
list_actor = pd_film_series['actor_id'].tolist()
unique_list_actor = []
for x in list_actor:
    unique_list_actor.extend(x)
unique_list_actor = list(set(unique_list_actor))

#Raw Category
list_unique_raw_category = get_all_category(pd_film_series)

pd_film_series['raw_category_id'] = pd_film_series['raw_category_id'].apply(lambda x: process_actor_director(x))
dict_pd_film_series = pd_film_series[["series_id", "release_year", "imdb_rating", "country", "raw_category_id", "director_id", "actor_id"]].set_index('series_id').T.to_dict('list')

    
# list_director = pd_film_series['director_id'].tolist()

len_list_unique_raw_category = len(list_unique_raw_category)
len_unique_list_director = len(unique_list_director)
len_unique_list_actor = len(unique_list_actor)

print(dict_pd_film_series['1850'])
dict_weights_film = {}
print(len(dict_pd_film_series))

sort_key = list(dict_pd_film_series.keys())
sort_key.sort()
print(sort_key)
print(len(sort_key))
#Score Film
for key1 in sort_key:
    for key2 in sort_key:
        if key2 == key1:
            continue
        if (key1, key2) not in dict_weights_film.keys() and (key2, key1) not in dict_weights_film.keys():
            sc_category = score_category(dict_pd_film_series[key1][3], dict_pd_film_series[key2][3])
            sc_actor = score_actor(dict_pd_film_series[key1][5], dict_pd_film_series[key2][5])
            sc_year = score_release_year(dict_pd_film_series[key1][0], dict_pd_film_series[key2][0])
            sc_country = score_country(dict_pd_film_series[key1][2], dict_pd_film_series[key2][2])
            sc_director = score_director(dict_pd_film_series[key1][4], dict_pd_film_series[key2][4])
            sc_rating = score_imdb_rating(dict_pd_film_series[key1][1], dict_pd_film_series[key2][1])
            dict_weights_film[(key1, key2)] = [sc_year, sc_rating, sc_country, sc_category, sc_director, sc_actor]
            # dict_weights_film[(key2, key1)] = dict_weights_film[(key1, key2)]
        else:
            continue

exit()
with open('feature_weighting.pickle', 'wb') as handle:
    pickle.dump(dict_weights_film, handle, protocol=pickle.HIGHEST_PROTOCOL)          






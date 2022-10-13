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

phobert = AutoModel.from_pretrained("vinai/phobert-base")
model_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")


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
    all_users_id = [str(user_id) for user_id in users_info['user_id'][:4000].tolist()]
    users_items = {}
    for log_film_path in path_list_file_log_film:
        # print(log_film_path)
        hst_users_info = pd.read_csv(folder + "data/" + log_film_path)
        # print(hst_users_info)
        # print(self.all_users_id)
        hst_users_info = hst_users_info.dropna()
        # print(hst_users_info['content_id'])
        st1 = time.time()
        hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: clean(x))
        hst_users_info['user_id'] = hst_users_info['user_id'].apply(lambda x: clean(x))
        print("Clean Time: ", time.time() - st1)
        # print(hst_users_info['content_id'][:100])
        hst_user_info_group_by_user = hst_users_info.groupby(['user_id'])
        hst_watch_duration_group_by_user = hst_user_info_group_by_user['watch_duration'].apply(list).reset_index(name='watch_duration').set_index('user_id')['watch_duration'].to_dict()
        hst_film_id_group_by_user = hst_user_info_group_by_user['content_id'].apply(list).reset_index(name='film_id').set_index('user_id')['film_id'].to_dict()
        unique_users = list(set(hst_users_info['user_id']))
        for user_id in unique_users:
            if user_id not in all_users_id:
                continue
            hst_items = hst_film_id_group_by_user[str(user_id)]
            list_watch_duration = hst_watch_duration_group_by_user[str(user_id)]
            l_hst_items = list(zip(hst_items, len(hst_items)*[int(log_film_path[9:13])], list_watch_duration))
            if users_items.get(str(int(user_id))) is None:
                users_items[str(int(user_id))] = []
                users_items[str(int(user_id))].extend(l_hst_items)
            else:
                users_items[str(int(user_id))].extend(l_hst_items)

            def sort_date(date):
                return date[1]

            users_items[str(int(user_id))].sort(reverse=False, key=sort_date)
    print("Time Split 1: ", time.time()- st)        
    # print(users_items)                        
    filter_users_items = {}
    for key in users_items.keys():
        if len(users_items[key]) >= 1.5 * opt.numbers_of_hst_films:
            filter_users_items[key] = users_items[key]

    # Sort by Date
    # print(filter_users_items)
    # filter_list_user = list(filter_users_items.keys())

    # Split History, Train & Val Users-Item
    hst_users_items = {}
    train_target_users_items = []
    val_target_users_items = []

    for key in filter_users_items.keys():
        # hst_users_items[key] = filter_users_items[key][:opt.numbers_of_hst_films]
        length_users_item_key = len(filter_users_items[key])
        list_target_items_unique = unique([filter_users_items[key][i][0] for i in range(length_users_item_key)])
        list_item_date = {}
        list_watch_duration_item_date = {}
        list_target_users_item = []

        for item in list_target_items_unique:
            for i in range(length_users_item_key):
                if filter_users_items[key][i][0] == item:
                    if list_item_date.get(item) is None:
                        list_item_date[item] = []
                        list_item_date[item].append(filter_users_items[key][i][1])
                        list_watch_duration_item_date[item] = 0
                        list_watch_duration_item_date[item] += filter_users_items[key][i][2]
                    else:
                        list_item_date[item].append(filter_users_items[key][i][1])
                        list_watch_duration_item_date[item] += filter_users_items[key][i][2]
        hst_users_items[key] = []
        for idx, item in enumerate(list_item_date.keys()):
            if idx < opt.numbers_of_hst_films:
                hst_users_items[key].append((item, list_item_date[item], list_watch_duration_item_date[item]))
            else:
                list_target_users_item.append((key, item, list_item_date[item], list_watch_duration_item_date[item]))

        length_train_trg_users_items = int(0.7 * (len(list_target_users_item) - opt.numbers_of_hst_films))
        list_train_trg_users_items = list_target_users_item[
                                     opt.numbers_of_hst_films:(opt.numbers_of_hst_films + length_train_trg_users_items)]
        list_val_trg_users_items = list_target_users_item[(opt.numbers_of_hst_films + length_train_trg_users_items):]

        train_target_users_items.extend(list_train_trg_users_items)
        val_target_users_items.extend(list_val_trg_users_items)

    target_users_items = {}
    target_users_items['train'] = train_target_users_items
    target_users_items['val'] = val_target_users_items
    print("Time Split: ", time.time()- st)  
    return hst_users_items, target_users_items
    # film_episode = pd.read_csv(folder + 'data/tv360_film_episode.csv')
    # film_series = pd.read_csv(folder + 'data/tv360_film_series.csv')


def embedding_dataloader(record, type_record, bertsentence, onehot_is_series, onehot_country, onehot_categorical,
                         bert_words):
    data = {}

    if type_record == "series":
        if type(record['description']) != str:
            return None
        data['description'] = normalizeString(str(record['series_name']) + " " + str(record['description']))
    else:
        if type(record['episode_description']) != str:
            return None
        data['description'] = normalizeString(str(record['episode_name']) + " " + str(record['episode_description']))

    data['country'] = [record['country']]
    try:
        data['raw_category_name'] = record['raw_category_name'].split(",")
    except:
        data['raw_category_name'] = ""

    data['director_name'] = record['director_name']

    data['actor_name'] = record['actor_name']

    data['is_series'] = [record['is_series']]

    data['duration'] = record['duration']

    if np.isnan(record['release_year']):
        return None
        # data['release_year'] = 2022
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
        actor = F.pad(actor, (1, 15 - actor.shape[1]), mode='constant', value=0)
    elif actor.shape[1] > 16:
        actor = actor[:, :16]

    if type(data['director_name']) == str:
        director = [bert_words.encode(data['director_name'])]
    else:
        director = [bert_words.encode('')]

    director = torch.FloatTensor(director)
    if director.shape[1] < 16:
        director = F.pad(director, (1, 15 - director.shape[1]), mode='constant', value=0)
    elif director.shape[1] > 16:
        director = director[:, :16]

    return (
    normalize(description_emb).to(opt.device), normalize(country).to(opt.device), normalize(category).to(opt.device),
    normalize(actor).to(opt.device), normalize(director).to(opt.device), is_series_emb.to(opt.device),
    duration.to(opt.device), release_year.to(opt.device))


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


def get_all_category(pd_film_series, pd_film_episode):
    unique_category_film_series = get_unique_category(pd_film_series)
    unique_category_film_episode = get_unique_category(pd_film_episode)
    return unique(unique_category_film_series + unique_category_film_episode)


def get_record_by_item_id(item_id, pd_film_series, pd_film_episode):
    if int(item_id) in list(pd_film_series['series_id']):
        record = pd_film_series[pd_film_series['series_id'] == int(item_id)].to_dict('records')[0]
        type = "series"
        return record, type

    elif int(item_id) in list(pd_film_episode['episode_id']):
        record = pd_film_episode[pd_film_episode['episode_id'] == int(item_id)].to_dict('records')[0]
        type = "episode"
        return record, type

    else:
        return None



class Tv360Dataset(data.Dataset):
    def __init__(self, bertsentence, hst_users_items, target_users_items, phase="train"):
        self.bertsentence = bertsentence
        self.hst_users_items = hst_users_items
        self.ccai = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
        self.ccai_drop_nan.dropna(inplace=True)
        self.org_target_users_items = target_users_items[phase]
        self.target_users_items = []
        self.labels = []
        # self.rating_imdb_rating = {}
        self.pd_film_series = pd.read_csv(opt.folder + "data/" + opt.path_film_series)
        self.pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/" + opt.path_film_series)
        self.pd_film_series_drop_nan.dropna(inplace=True)
        # print(self.pd_film_series['series_id'])
        self.pd_film_episode = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
        self.pd_film_episode_drop_nan = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)
        self.pd_film_episode_drop_nan.dropna(inplace=True)
        self.list_ft_user = {}
        self.list_ft_item = {}
        self.list_duration = {}
        for i, (user_id, item_id, dates, watch_duration) in enumerate(self.org_target_users_items):
            if int(item_id) in list(self.pd_film_series['series_id']):
                if self.list_duration.get(int(item_id)) is None:
                    duration = list(self.pd_film_series[self.pd_film_series['series_id'] == int(item_id)]['duration'])[0]
                    self.list_duration[int(item_id)] = duration
                else:
                    duration = self.list_duration[int(item_id)]
                if duration == 0:
                    continue           
                self.labels.append(min(float(watch_duration) / float(duration), 1))
                # self.labels.append(float(watch_duration)/float(duration))
                self.target_users_items.append((user_id, item_id, dates, watch_duration))
            elif int(item_id) in list(self.pd_film_episode['episode_id']):
                if self.list_duration.get(int(item_id)) is None:
                    duration = list(self.pd_film_episode[self.pd_film_episode['episode_id'] == int(item_id)]['duration'])[0]
                    self.list_duration[int(item_id)] = duration
                else:
                    duration = self.list_duration[int(item_id)]
                if duration == 0:
                    continue        
                self.labels.append(min(float(watch_duration) / float(duration), 1))
                # self.labels.append(float(watch_duration)/float(duration))
                self.target_users_items.append((user_id, item_id, dates, watch_duration))

        self.onehot_is_series = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.onehot_is_series.fit(self.pd_film_series_drop_nan[['is_series']])

        self.onehot_categorical = MultiLabelBinarizer()
        all_category = get_all_category(self.pd_film_series, self.pd_film_episode)
        self.onehot_categorical.fit([all_category])

        self.onehot_country = OneHotEncoder(handle_unknown='ignore', sparse=False)
        pd_film_series_country = pd.DataFrame(self.pd_film_series, columns=['country'])
        pd_film_series_country = pd_film_series_country[pd_film_series_country['country'].notnull()]
        pd_film_episode_country = pd.DataFrame(self.pd_film_episode, columns=['country'])
        pd_film_episode_country = pd_film_episode_country[pd_film_episode_country['country'].notnull()]
        pd_country = pd.concat([pd_film_series_country, pd_film_episode_country])
        self.onehot_country.fit(pd_country[['country']])

        self.onehot_province_user = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.onehot_province_user.fit(self.ccai_drop_nan[['province_name']])

        print("Len data: --------------------------------", len(self.labels))
        test = np.array(self.labels)
        print(test)
        # print(self.hst_users_items)
        # print(len(self.hst_users_items))
        # print(len(self.target_users_items))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_id, item_id, _, _ = self.target_users_items[idx]
        list_hst_item_id = self.hst_users_items[user_id]
        list_rating = []
        # CCAI Embedding
        if self.list_ft_user.get(int(user_id)) != None:
            ccai_embedding = self.list_ft_user[int(user_id)]
        else:    
            user_info = self.ccai[self.ccai['user_id'] == int(user_id)].to_dict('records')[0]
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
            
        try:
            record_trg_item, type_record_trg_item = get_record_by_item_id(item_id, self.pd_film_series,
                                                                          self.pd_film_episode)
        except:
            return None
        # Feature Target Item
        if self.list_ft_item.get(int(item_id)) != None:
            fe_record_trg_item = self.list_ft_item[int(item_id)]
        else:        
            fe_record_trg_item = embedding_dataloader(record_trg_item, type_record_trg_item, self.bertsentence,
                                                  self.onehot_is_series, self.onehot_country, self.onehot_categorical,
                                                  model_tokenizer)
            self.list_ft_item[int(item_id)] = fe_record_trg_item
            
        if fe_record_trg_item is None:
            return None
        # Feature History Items
        fe_hst_items = []
        for (item_hst_id, _, total_watch_duration) in list_hst_item_id:
            duration = self.list_duration[int(item_id)]
            if duration == 0:
                return None
            rating = min(float(total_watch_duration) / float(duration), 1)
            list_rating.append(rating)
            try:
                hst_record_item, type_record_item = get_record_by_item_id(item_hst_id, self.pd_film_series,
                                                                          self.pd_film_episode)
            except:
                return None
            if self.list_ft_item.get(int(item_hst_id)) != None:
                ebd = self.list_ft_item[int(item_hst_id)]
            else:    
                ebd = embedding_dataloader(hst_record_item, type_record_item, self.bertsentence, self.onehot_is_series,
                                       self.onehot_country, self.onehot_categorical, model_tokenizer)
                self.list_ft_item[int(item_hst_id)] = ebd
            if ebd is None:
                return None
            fe_hst_items.append(ebd)
        return (fe_hst_items, fe_record_trg_item, ccai_embedding, list_rating), self.labels[idx]

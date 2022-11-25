import random
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
import model
from new_model import TV360Recommend
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

def ndcg(X_pred, heldout_batch, k=50):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    X_pred: predicted ranking scores
    heldout_batch: binary relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    
    return DCG / IDCG


def recall(X_pred, heldout_batch, k=50):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

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
    return (normalize(description_emb).unsqueeze(0).to(args.device), normalize(country).unsqueeze(0).to(args.device), normalize(category).unsqueeze(0).to(args.device),normalize(actor).unsqueeze(0).to(args.device), normalize(director).unsqueeze(0).to(args.device), is_series_emb.unsqueeze(0).to(args.device), duration.unsqueeze(0).to(args.device),release_year.unsqueeze(0).to(args.device))
    
def get_ft_user(user_id):
    try:
        user_info = ccai[ccai['user_id'] == int(user_id)].to_dict('records')[0]
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
    parser.add_argument('--weights', type=str, default='TV360/RCM/save_models/123_best_accuracy.pth', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=1024, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. cuda:0 or 0,1,2,3 or cpu')
    parser.add_argument('--start-day', type=int, default=20220626, help='Date from 20220625 to 20220630')
    parser.add_argument('--end-day', type=int, default=20220626, help='Date from 20220625 to 20220630')

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = parse_opt(True)
    if args.start_day > args.end_day:
        print("Invalid Date!")
        exit()
        
    list_film_id = pd_film_series['series_id'].apply(lambda x: str(x)).tolist()
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
    for item_id in list_film_id:
        dict_fe_item[item_id] = get_ft_item(item_id)
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
    
    #List Item Eval
    eval_users_info = pd.read_csv(opt.folder + "data/log_film_0625_0630.csv")
    # eval_users_info.dropna()
    eval_users_info['user_id'] = eval_users_info['user_id'].apply(lambda x: clean(x))
    eval_users_info['content_id'] = eval_users_info['content_id'].apply(lambda x: find_seri_id(x))
    eval_users_info = eval_users_info[eval_users_info['content_id'].isin(list_film_id)]
    eval_users_info = eval_users_info[(eval_users_info['content_id'] != "-1") & (eval_users_info['user_id'] != "-1")]
    list_time = range(args.start_day, args.end_day + 1)
    eval_users_info = eval_users_info[eval_users_info['partition'].isin(list_time)]
    # print(eval_users_info)
    all_users_id_eval = [str(user_id) for user_id in eval_users_info['user_id'].tolist()]
    eval_users_info = eval_users_info.groupby(["user_id"])['content_id'].apply(list).apply(lambda x: unique(x)).reset_index()
    eval_users_info = eval_users_info[['user_id', 'content_id']].set_index('user_id').T.to_dict('list')
    
    #Hst Film      
    hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)    
    # hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
    hst_users_info['user_id'] = hst_users_info['user_id'].apply(lambda x: clean(x))
    hst_users_info = hst_users_info[hst_users_info['user_id'].isin(all_users_id_eval)]
    hst_users_info['content_id'] = hst_users_info['content_id'].apply(lambda x: find_seri_id(x))
    hst_users_info = hst_users_info[hst_users_info['content_id'].isin(list_film_id)]
    hst_users_info = hst_users_info[(hst_users_info['content_id'] != "-1") & (hst_users_info['user_id'] != "-1")]
    hst_film_id_group_by_user = hst_users_info.groupby(["user_id", "content_id"])['watch_duration'].agg([("watch_duration", "sum")]).reset_index()
    # hst_film_id_group_by_user['partition'] = hst_users_info.groupby(["user_id", "content_id"])['partition'].apply(lambda x: min(x)).reset_index()['partition']
    hst_film_id_group_by_user['partition'] = hst_users_info.groupby(["user_id", "content_id"])['partition'].apply(list).reset_index()['partition']
    users_items = hst_film_id_group_by_user.groupby(['user_id'])['content_id'].apply(list).apply(lambda x: x if len(x) > 0 else -1).reset_index()
    # users_items = hst_film_id_group_by_user.groupby(['user_id'])['content_id'].apply(list).apply(lambda x: x if len(x) >= opt.numbers_of_hst_films else -1).reset_index()
    #apply(lambda x: x if len(x) >= opt.numbers_of_hst_films else -1)
    users_items['partition'] = hst_film_id_group_by_user.groupby('user_id')['partition'].apply(list).reset_index()['partition']
    users_items['watch_duration'] = hst_film_id_group_by_user.groupby('user_id')['watch_duration'].apply(list).reset_index()['watch_duration']
    users_items = users_items[users_items['content_id'] != -1]
    # print(users_items)
    users_items = users_items.set_index('user_id').T.to_dict('list')
            
    # def sort_date(value):
    #     v = list(zip(value[0], value[1], value[2]))
    #     v.sort(key=lambda x: x[1])
    #     # print(list(zip(value[0], value[1], value[2])).sort(key=lambda x: x[1]))
    #     if len(v) < opt.numbers_of_hst_films:
    #         v.extend((opt.numbers_of_hst_films - len(v)) * [v[-1]])
    #     return v[-opt.numbers_of_hst_films:]
    # hst_users_items = {k: sort_date(v) for k, v in users_items.items()}
    
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
        if len(v) < opt.numbers_of_hst_films:
            v.extend((opt.numbers_of_hst_films - len(v)) * [v[-1]])
        return v[-opt.numbers_of_hst_films:]       
    
    hst_users_items = {k: sort_date(v) for k, v in users_items.items()} 
           
    onehot_film = MultiLabelBinarizer()
    onehot_film.fit([list_film_id])        
    heldout_batch = []
    pred = []
    # print(hst_users_items)
    print("Length User: ", len(list(hst_users_items.keys())))       
    with torch.no_grad():   
        for i, key in enumerate(hst_users_items.keys()):
            # print(i)
            if i > 200:
                break
            # print(eval_users_info[key])
            if len(eval_users_info[key]) == 0:
                print("Empty Film Item")
                continue
            heldout_batch_key = onehot_film.transform(eval_users_info[key])[0]
            if 1 not in heldout_batch_key:
                print(key)
                print(eval_users_info[key])
                exit()
            pred_key = []
            hst_films_key = hst_users_items[key]
            ccai_embedding = get_ft_user(key)
            if ccai_embedding is None:
                continue
            heldout_batch.append(heldout_batch_key)
            ccai_embedding = torch.cat(args.batch_size*[ccai_embedding.unsqueeze(0)], 0)
            fe_hst_items = []
            list_rating = []
            for (film_id,partition, watch_duration) in hst_films_key:
                batch_fe_item = []
                fe_item = dict_fe_item[film_id]
                for i in range(8):
                    ft_i = torch.cat(args.batch_size*[fe_item[i]], 0)
                    batch_fe_item.append(ft_i)
                # rating = min(float(watch_duration)/float(dict_films_duration[int(film_id)][0]), 1)
                
                # prefer = float(watch_duration)/float(dict_films_duration[int(film_id)][0])
                # if prefer > opt.prefer_threshold:
                #     rating = 1
                # else:
                #     rating = prefer/opt.prefer_threshold
                if watch_duration > opt.duration_threshold:
                    rating = 1
                else:
                    rating = watch_duration/opt.duration_threshold  
                rating = torch.cat(args.batch_size*[torch.tensor([rating])], 0)
                
                list_rating.append(rating)
                fe_hst_items.append(batch_fe_item)

            nb_batch = (length_list_ft_film // args.batch_size) + 1
            pred_user = []
            for i in range(nb_batch):
                z = min(args.batch_size*(i+1), length_list_ft_film)
                len_batch = z - args.batch_size*i
                batch_ft_film = []
                for i1 in range(8):
                    batch_ft_film.append(list_ft_film[i1][args.batch_size*i: z])
                if len_batch < args.batch_size:
                    for i2 in range(len(hst_films_key)): 
                        list_rating[i2] = list_rating[i2][:len_batch] 
                        for i3 in range(len(fe_hst_items[i2])):
                            fe_hst_items[i2][i3] = fe_hst_items[i2][i3][:len_batch]   
                
                inputs = (fe_hst_items, batch_ft_film, ccai_embedding[:len_batch], list_rating)
                outputs = model(inputs)
                outputs = outputs[:, 0].detach().cpu().numpy()
                pred_user.extend(outputs)
            # pred_key.append(outputs[:, 0].detach().cpu().numpy())   
            pred.append(pred_user)
            
    pred = np.array(pred)
    heldout_batch = sparse.csr_matrix(np.array(heldout_batch))
    recall_ = recall(pred, heldout_batch)
    ndcg_ = ndcg(pred, heldout_batch)
    print(recall_)
    print("---------")
    print(ndcg_)
    print("RECALL MEAN: ")
    print(sum(recall_)/len(recall_))
    
    print("NDCG MEAN: ")
    print(sum(ndcg_)/len(ndcg_))
    
    
    
                
                
                
            
        
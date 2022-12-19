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
# import pickle5 as pickle
import pickle
from copy import deepcopy
from pytorch_model_summary import summary

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
    # print(ccai_embedding.size())
    return ccai_embedding

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='TV360/RCM/save_models/85_best_accuracy.pth', help='initial weights path')
    parser.add_argument('--mask', action='store_true', help='Che nhung film trong history film cua user')
    parser.add_argument('--batch-size', type=int, default=3, help='total batch size for all GPUs, -1 for autobatch')
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
    raw_features_file = 'raw_features_unnormal.pickle'
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
    
    #List Item Eval
    eval_users_info = pd.read_csv(opt.folder + "data/log_film_0625_0630.csv")
    # eval_users_info.dropna()
    eval_users_info["profile_id"] = eval_users_info["profile_id"].apply(lambda x: clean(x))
    eval_users_info['content_id'] = eval_users_info['content_id'].apply(lambda x: find_seri_id(x))
    eval_users_info = eval_users_info[eval_users_info['content_id'].isin(list_film_id)]
    eval_users_info = eval_users_info[(eval_users_info['content_id'] != "-1") & (eval_users_info["profile_id"] != "-1")]
    list_time = range(args.start_day, args.end_day + 1)
    eval_users_info = eval_users_info[eval_users_info['partition'].isin(list_time)]
    # print(eval_users_info)
    all_users_id_eval = [str(user_id) for user_id in eval_users_info["profile_id"].tolist()]
    eval_users_info = eval_users_info.groupby(["profile_id"])['content_id'].apply(list).apply(lambda x: unique(x)).reset_index()
    eval_users_info = eval_users_info[["profile_id", 'content_id']].set_index("profile_id").T.to_dict('list')
    
    #Hst Film      
    hst_users_info = pd.concat(list_hst_users_info, ignore_index=True, sort=False)    
    # hst_users_info = pd.read_csv(opt.folder + "data/log_film_0601_0605.csv")
    hst_users_info["profile_id"] = hst_users_info["profile_id"].apply(lambda x: clean(x))
    hst_users_info = hst_users_info[hst_users_info["profile_id"].isin(all_users_id_eval)]
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
        
        dict_mask_index_film_user[key] = []
        list_film_remove = []
        if args.mask:
            ground_truth = eval_users_info[key][0]
            for item in v:
                if item[0] in ground_truth:
                    if item[0] in list_film_remove:
                        continue
                    else:
                        list_film_remove.append(item[0])
                        index_list_film = list_film_id.index(item[0])
                        dict_mask_index_film_user[key].append(index_list_film)
            for item_id in list_film_remove:
                eval_users_info[key][0].remove(item_id)
            # print(dict_mask_index_film_user[key])            
            # print(list_film_remove)
            # print(eval_users_info[key][0])
            # print("------------------")    
            # if '8167' in list_film_remove:
            #     exit()
                        
        return v[-opt.numbers_of_hst_films:]
    hst_users_items = {k: sort_date(k, v) for k, v in users_items.items()}
    # def sort_date(value):
    #     # v = list(zip(value[0], value[1], value[2]))
    #     # v.sort(key=lambda x: x[1])
    #     # # print(list(zip(value[0], value[1], value[2])).sort(key=lambda x: x[1]))
    #     # return v
    #     v = []
    #     for film_id, list_date, watch_duration in zip(value[0], value[1], value[2]):
    #         list_date = list(set(list_date))
    #         len_date = len(list_date)            
    #         v.extend(list(zip(len_date*[film_id], list_date, len_date*[watch_duration])))
    #     v.sort(key=lambda x: x[1])
    #     if len(v) < opt.numbers_of_hst_films:
    #         v.extend((opt.numbers_of_hst_films - len(v)) * [v[-1]])
    #     return v[-opt.numbers_of_hst_films:]       
    
    # hst_users_items = {k: sort_date(v) for k, v in users_items.items()} 
           
    onehot_film = MultiLabelBinarizer()
    onehot_film.fit([list_film_id])        
    heldout_batch = []
    pred = []
    # print(hst_users_items)
    print("Length User: ", len(list(hst_users_items.keys())))       
    with torch.no_grad():   
        for idx, key in enumerate(hst_users_items.keys()):
            # print(i)
            if idx > 200:
                break
            # print(eval_users_info[key])
            if len(eval_users_info[key][0]) == 0:
                # print("Empty Film Item")
                continue
            heldout_batch_key = onehot_film.transform(eval_users_info[key])[0]
            hst_films_key = hst_users_items[key]
            ccai_embedding = get_ft_user(key)
            if ccai_embedding is None:
                continue
            heldout_batch.append(heldout_batch_key)
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
            fe_hst_items1 = deepcopy(fe_hst_items)  
            list_ft_film1 = deepcopy(list_ft_film)
            print(ccai_embedding.size())
            inputs = (fe_hst_items1, list_ft_film1, ccai_embedding, list_rating)
            # print(summary(model, inputs))
            # print(model)
            outputs = model(inputs)
            outputs = outputs[:, 0].detach().cpu().numpy()
            if args.mask:
                for index in dict_mask_index_film_user[key]:
                    outputs[index] = 0     
            pred.append(outputs)
            
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

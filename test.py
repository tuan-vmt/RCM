from config_path import opt
import pandas as pd


from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import math
from sklearn.metrics import accuracy_score
from torch.nn.functional import normalize
from model import TV360Recommend
from torch import nn
from config_path import opt
import time

all_films_id = pd.read_csv(opt.folder + "data/" + opt.path_film_episode)

x = all_films_id[['episode_id', 'series_id']].apply(list).to_dict()
print(x)

# # b = torch.tensor([1,1,1,1,0,0,1,1]).to("cuda:0")
# # print(get_evaluation(a, b))
# # b = torch.rand(8, 1, 1)
# # c = torch.rand(8, 1, 1)

# # d = torch.concat((a,b,c), 1)
# # # print(d.size())
# def check_tensor_nan(x):
#     x1 = torch.ones(x.size())
#     x2 = (x==x)
#     print(x1)
#     print(x2[0][0][1])
#     return x1==x2
    
# a = torch.tensor([[[0.6921, np.nan]],

#         [[0.1664, 0.0766]],

#         [[0.2484, 0.1851]],

#         [[0.3963, 0.2112]],

#         [[0.2945, 0.1751]],

#         [[0.1241, 0.3501]],

#         [[0.0118, 0.8963]],

#         [[0.7782, 0.8237]]]).to("cuda:0")
# if not torch.all(torch.eq(a, a)).cpu().numpy():
#     print("HAHA")
# pd_film_series_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_series)
# # pd_film_series_drop_nan.dropna(inplace=True)
# pd_film_series_drop_nan = pd.DataFrame(pd_film_series_drop_nan, columns = ['country'])

# pd_film_episode_drop_nan = pd.read_csv(opt.folder + "data/"  + opt.path_film_episode)
# # pd_film_series_drop_nan.dropna(inplace=True)
# pd_film_episode_drop_nan = pd.DataFrame(pd_film_episode_drop_nan, columns = ['country'])
# pd_film_episode_drop_nan = pd_film_episode_drop_nan[pd_film_episode_drop_nan['country'].notnull()]
# print(pd_film_episode_drop_nan)
# # print(np.isnan(pd_film_series_drop_nan['release_year'][67]))
# print(pd_film_series_drop_nan)
# x = pd.concat([pd_film_series_drop_nan, pd_film_episode_drop_nan])
# print(x[['country']])
# # print(pd.concat([pd_film_series_drop_nan[['country']] + pd_film_episode_drop_nan[['country']]], names=["country"]))
# exit()
# # print(str(math.nan))
# print(0.3*[5, 10, 15, 20])
# ccai_drop_nan = pd.read_csv(opt.folder + opt.path_file_user_info)
# ccai_drop_nan.dropna(inplace=True)
# onehot_province_user=OneHotEncoder(handle_unknown='ignore',sparse=False)
# onehot_province_user.fit(ccai_drop_nan[['province_name']])
# onehot_province_user = onehot_province_user.transform([["Hà Nội"]])[0]
# print(len(onehot_province_user))
# # print(pd_film_series_drop_nan[pd_film_series_drop_nan['description'].isnull()])
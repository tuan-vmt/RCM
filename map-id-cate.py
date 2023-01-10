import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("../datasets/data_log/tv360_film_series.csv")
data = data[["series_id","raw_category_id","raw_category_name"]]
data = data.replace(np.nan,"other")
# print(data)
# df1 = data[data['raw_category_name'].str.split(',').map(len)==1]
with open('../pickle-101/list_id_film.pickle', 'rb') as pickle_file:
    series_id = pickle.load(pickle_file)
with open('../pickle-101/label.pickle', 'rb') as pickle_file:
    label = pickle.load(pickle_file)
print(len(series_id))
df = pd.DataFrame({'series_id':series_id,'label':label})
# print(data1.head)
# data[(data["raw_category_name"].str.split(','))==1]
# print(data["raw_category_name"].str.split(','))

from collections import Counter
df_last = pd.merge(data,df,on="series_id")

df_filter = df_last[df_last['raw_category_name'].str.split(',').map(len)==1]
# cate = df_filter["raw_category_name"].unique()
fin_max=[None]*15
for i in range(0,15):
    tmp = Counter(df_filter[df_filter['label'] == i]["raw_category_name"])
    print(tmp)
    fin_max[i] =  max(tmp, key=tmp.get)
# print(df_filter.head)
# print(Counter(df_filter[df_filter['raw_category_name']=="Tâm lý"]["label"]))
print(fin_max)
print(df_filter)

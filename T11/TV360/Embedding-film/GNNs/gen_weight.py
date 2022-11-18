import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

dt_seri = pd.read_csv("/content/drive/MyDrive/vtt11/tv360_film_series.csv")
dt_seri_filter = dt_seri[["series_id","release_year","country","raw_category_id","director_id","actor_id","imdb_rating"]]
dt_seri_filter = dt_seri_filter.dropna()
dt_in = pd.DataFrame(columns=['item1', 'item2', 'release_year','country','raw_category_id','director_id','actor_id','imdb_rating'])

dt_in = pd.DataFrame(columns=['item1', 'item2', 'release_year','country','raw_category_id','director_id','actor_id','imdb_rating'])
dt_in
for index, row in dt_seri_filter.iterrows():
  for index1, row1 in dt_seri_filter.iterrows():
    if index >index1:
      continue
    y = (300-abs(row["release_year"]-row1["release_year"]))/300
    # print(y)

    if row1["country"] == row["country"]:
      c = 1/35
    else:
      c =0
    # print(c)

    id0 = row['raw_category_id'].split(',')
    id1 = row1['raw_category_id'].split(',')
    temp = set(id0)
    lst3 = [value for value in id1 if value in temp]
    g = len(lst3)/28
    # print(g)

    if row['director_id'] == row1['director_id']:
      d=1
    else:
      d=0
    # print(d)

    id0 = row['actor_id'].split(',')
    id1 = row1['actor_id'].split(',')
    temp = set(id0)
    lst3 = [value for value in id1 if value in temp]
    a = len(lst3)/4416
    #print(a)

    r = (10-abs(row['imdb_rating']-row1['imdb_rating']))/10
    # print(r)

    new_row = {'item1':row['series_id'],'item2':row1['series_id'],'release_year':y,'country':c,'raw_category_id':g,'director_id': d,'actor_id':a,'imdb_rating':r}
    dt_in = dt_in.append(new_row,ignore_index=True)
    
dt_in.to_csv("weight.csv", sep='\t', encoding='utf-8')

list_log = ["log_film_0601_0605.csv","log_film_0605_0610.csv","log_film_0610_0615.csv", "log_film_0615_0620.csv","log_film_0620_0625.csv","log_film_0625_0630.csv"]
df = pd.DataFrame()
for i in list_log:
    df1 = pd.read_csv("datasets/data_log/"+i)
    df = pd.concat([df,df1], axis=0)
    
dict_seri ={}
for index, row in dt_seri_filter.iterrows():
    dict_seri[row['series_id']]=[]
df_epidode = pd.read_csv("/home/gemai/mnt_raid1/datpt1/Viettel/T11-2022/datasets/data_log/tv360_film_espisode.csv")
l_e = df_epidode["episode_id"].values.tolist()
l_s = dt_seri_filter["series_id"].values.tolist()

dict_id ={}
for index, row in df_epidode.iterrows():
    dict_id[row['episode_id']] = row['series_id']
from tqdm import tqdm
count =0
dict_seri ={}
for index, row in dt_seri_filter.iterrows():
    dict_seri[row['series_id']]=[]
    
dict_id ={}
for index, row in df_epidode.iterrows():
    dict_id[row['episode_id']] = row['series_id']  

for index, row in tqdm(df.iterrows()):
    id_film, id_user = row['content_id'],row['user_id']
    if id_film in l_s:
        dict_seri[id_film].append(id_user)
    else:
        
        if id_film in l_e:
            id_seri = dict_id[id_film]  
            if id_seri in l_s:
                dict_seri[id_seri].append(id_user)            
df_w = pd.read_csv('/home/gemai/mnt_raid1/datpt1/Viettel/T11-2022/weight.csv',sep="\t",index_col=0)
def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))
l_score=[]
for index, row in df_w.iterrows():
    i1 =row['item1']
    i2 = row['item2']
    if dict_seri.get(i1) is not None and dict_seri.get(i2) is not None:
        score = min(len(intersect(dict_seri[i1],dict_seri[i2]))/10,1)
    else:
        score =0
    l_score.append(score)
    
m = np.asarray(l_score)
df_w["score"] = m
df_w.to_csv("weight.csv", sep='\t', encoding='utf-8')


feature_cols =['release_year','country','raw_category_id','director_id','actor_id','imdb_rating']
X = df_w[feature_cols] # Features
y = df_w.score # Target variable

model = LinearRegression()
model.fit(X,y)

print(model.coef_)

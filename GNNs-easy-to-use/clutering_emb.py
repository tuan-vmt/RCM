import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score

# cleaning, plotting and dataframes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open('/home/gemai/mnt_raid1/datpt1/Viettel/T11-2022/EmbeddingFilm/GNNs-easy-to-use/result/graphsage_default_emb.pickle', 'rb') as pickle_file:
    content = pickle.load(pickle_file)
data = pd.DataFrame.from_dict(content)

scaler = MinMaxScaler()

# scale down our data
df_scaled = scaler.fit_transform(data)

# inertia = []

# possible_K_values = [i for i in range(2,40)]

# for each_value in possible_K_values:
    
#     # iterate through, taking each value from 
#     model = KMeans(n_clusters=each_value, init='k-means++',random_state=32)
    
#     # fit it
#     model.fit(df_scaled)
    
#     # append the inertia to our array
#     inertia.append(model.inertia_)

# plt.plot(possible_K_values, inertia)
# plt.title('The Elbow Method')

# plt.xlabel('Number of Clusters')

# plt.ylabel('Inertia')

# plt.savefig('foo.png')
# plt.show()
model = KMeans(n_clusters=15, init='k-means++',random_state=32)

# re-fit our model
model.fit(df_scaled)

from sklearn.decomposition import PCA
pca = PCA(2)
data = pca.fit_transform(df_scaled)

model = KMeans(n_clusters = 15, init = "k-means++")
label = model.fit_predict(data)
plt.figure(figsize=(10,10))
uniq = np.unique(label)
for i in uniq:
   plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)

plt.legend()
plt.savefig('foo1.png')

silhouette_score_average = silhouette_score(df_scaled, model.predict(df_scaled))

print("Score sil arg: ",silhouette_score_average)


# silhouette_score_individual = silhouette_samples(df_scaled, model.predict(df_scaled))

# bad_k_values = {}

# # iterate through to find any negative values
# for each_silhouette in silhouette_score_individual:
        
#     # if we find a negative, lets start counting them
#     if each_silhouette < 0:
        
#         if each_value not in bad_k_values:
#             bad_k_values[each_value] = 1
        
#         else:
#             bad_k_values[each_value] += 1
# for key, val in bad_k_values.items():
#     print(f' This Many Clusters: {key} | Number of Negative Values: {val}')
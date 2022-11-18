import csv
import pandas as pd
import numpy as np
import pickle

# with open('raw_features.pickle', 'rb') as f:
#     data = pickle.load(f)

print(np.dot([2, 3], [4, 5]))
# feature_weights = 
df = pd.read_csv("TV360/EmbeddingFilm/GNNs-easy-to-use/weight.csv", sep="\t")
print(list(set(df['item1'].apply(lambda x: str(int(x))).tolist())))
df = df.groupby(["item1", "item2"]).sum().reset_index().set_index(["item1", "item2"]).T.to_dict('list')
print(df[(8257, 1835)])
import pickle
import json
import numpy as np
from config_pr import opt


def unique(list1):
    list_set = set(list1)
    unique_list = (list(list_set))
    return unique_list

with open("/home/gemai/mnt_raid1/datpt1/Viettel/T11-2022/folder_pickle/feature_weighting.pickle", 'rb') as f:
        dict_film_weights = pickle.load(f)

feature_weights = [0, 0.1304048, 0, 0.7227898, 0, 0.3818457]
dict_keys = {}
dict_scores = {}  
dict_key_scores = {}     
for key in dict_film_weights.keys():
    score = np.dot(feature_weights, dict_film_weights[key])
    dict_scores[key] = np.dot(feature_weights, dict_film_weights[key])
        
    if dict_keys.get(key[0]) is None:
        dict_keys[key[0]] = []
        dict_key_scores[key[0]] = []
        dict_keys[key[0]].append(key[1])
        dict_key_scores[key[0]].append(score)
    else:
        dict_keys[key[0]].append(key[1])
        dict_key_scores[key[0]].append(score)

count_node = 0
for key in dict_keys.keys():
    if max(dict_key_scores[key]) < opt.score_threshold:
        count_node+=1
        max_index = dict_key_scores[key].index(max(dict_key_scores[key]))
        dict_scores[key, dict_keys[key][max_index]] = opt.score_threshold
print(count_node)
with open('score.pickle', 'wb') as handle:
    pickle.dump(dict_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)        
                  


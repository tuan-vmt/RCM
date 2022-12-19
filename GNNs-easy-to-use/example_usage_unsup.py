import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from GNNs_unsupervised import GNN
import pandas as pd
import pickle
from config_pr import opt

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def example_with_cora():
    raw_features_file = '/home/gemai/mnt_raid1/tuanvm/Viettel/TV360/EmbeddingFilm/GNNs-easy-to-use/raw_features_gnn.pickle'
    with open(raw_features_file, 'rb') as f:
        data = pickle.load(f)
    del data["0"]     
    key_none = []
    for key in data.keys():
        if data[key] is None:
            key_none.append(key)
    for key in key_none:
        del data[key]        
             
    list_film_id = list(data.keys())
    list_film_id = [int(id) for id in list_film_id]
    list_film_id.sort()
    # print(list_film_id)
    # Load features
    feat_data = []
    node_map = {} # map node to Node_ID
    
    for i, film_id in enumerate(list_film_id):
        feat_data.append(data[str(film_id)].tolist()[:-4])
        # print(feat_data)
        # exit()
        node_map[film_id] = i

    with open("score.pickle", 'rb') as f:
        dict_score = pickle.load(f)
        
    raw_features = np.asarray(feat_data)
    print(raw_features[0])
    # load adjacency matrix
    row = []
    col = []
    count = 0
    for key in dict_score.keys():
        node_1, node_2 = key
        if (int(node_1) not in list_film_id) or (int(node_2) not in list_film_id):
            continue
        weights = dict_score[key]
        # print(weights)
        if weights >= opt.score_threshold:
            row.extend([node_map[int(node_1)], node_map[int(node_2)]])
            col.extend([node_map[int(node_2)], node_map[int(node_1)]])
            count+=1
    print("Count: ", count)      
    row = np.asarray(row)
    col = np.asarray(col)
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))

    """
    Example of using GraphSAGE for unsupervised learning.
    using CUDA and print training progress
    """
    gnn = GNN(adj_matrix, features=raw_features, supervised=False, model=opt.model, device='cpu',epochs=opt.epochs, lr=opt.lr, unsup_loss_type=opt.loss)
    # train the model
    gnn.fit()
    # get the node embeddings with the trained GAT
    embs = gnn.generate_embeddings()
    # evaluate the embeddings with logistic regression
    print(len(embs))
    print(type(embs))
    name = opt.model + '_default_emb'
    with open(f'result/{name}.pickle', 'wb') as handle:
        pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(len(embs))
    print("EDone")

if __name__ == "__main__":
    example_with_cora()

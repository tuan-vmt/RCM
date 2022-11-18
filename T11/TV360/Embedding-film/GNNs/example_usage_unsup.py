import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from GNNs_unsupervised import GNN
import pandas as pd
import pickle
import config_pr
import argparse

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def generate_embedding(args):
    
    #######Preprocessing#########
    weights_threshold = config_pr.weights_threshold
    feature_weights = config_pr.feature_weights
    raw_features_file = config_pr.raw_features_file_dir
    #load data
    with open(raw_features_file, 'rb') as f:
        data = pickle.load(f)
    key_none = []
    for key in data.keys():
        if data[key] is None:
            key_none.append(key)
    for key in key_none:
        del data[key]        
             
    list_key_film_id = list(data.keys())
    list_key_film_id = [int(i) for i in list_key_film_id]
    # Load features
    feat_data = []
    node_map = {} # map node to Node_ID
    
    df = pd.read_csv("weight.csv", sep="\t")
    list_film_id = list(set(df['item1'].apply(lambda x: int(x)).tolist()))
    list_film_id = intersection(list_key_film_id, list_film_id)
    dict_film_weights = df.groupby(["item1", "item2"]).sum().reset_index().set_index(["item1", "item2"]).T.to_dict('list')
    
    for i, film_id in enumerate(list_film_id):
        feat_data.append(data[str(film_id)].tolist())
        node_map[film_id] = i

    raw_features = np.asarray(feat_data)
    # load adjacency matrix
    row = []
    col = []
    for key in dict_film_weights.keys():
        node_1, node_2 = key
        if (int(node_1) not in list_film_id) or (int(node_2) not in list_film_id):
            continue
        weights = np.dot(feature_weights, dict_film_weights[key][1:7])
        if weights > weights_threshold:
            row.extend([node_map[int(node_1)], node_map[int(node_2)]])
            col.extend([node_map[int(node_2)], node_map[int(node_1)]])
    row = np.asarray(row)
    col = np.asarray(col)
    adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))

    """
    Path of using GraphSAGE for unsupervised learning.
    using CUDA and print training progress
    """
    gnn = GNN(adj_matrix, config_pr, features=raw_features)
    # train the model
    gnn.fit()
    # get the node embeddings with the trained GAT
    embs = gnn.generate_embeddings()
    # evaluate the embeddings with logistic regression
    #save embs
    print(type(embs))
    name = config_pr.model + '_default_emb'
    with open(f'result/{name}.pickle', 'wb') as handle:
        pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(len(embs))
    print("EDone")

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Graph Embedding model')
    # arguments for optimization

    parser.add_argument('--name', type=str, default='default_emb',
                        help='name of the finetune')
    args = parser.parse_args()
    generate_embedding(args)

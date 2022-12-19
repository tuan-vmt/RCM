import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import pickle

with open("groundtruth_feature_weighting.pickle", 'rb') as f:
    ground_truth = pickle.load(f)

with open("feature_weighting.pickle", 'rb') as f:
    score_weight = pickle.load(f)

# matrix = [0.11547538, 0, 0.99209856, 0, 0.55746724]
# matrix = [0.1304048, 0, 0.7227898, 0, 0.3818457]
# for key in score_weight.keys():
#     if score_weight[key][0] < 0:
#         print(score_weight[key])
#     # a = np.dot(matrix, score_weight[key][1:])
#     # if a>0.8:
#     #     print(a)
# exit()    
X = []
y = []

for key in sorted(ground_truth.keys()):
    if "-1" in key:
        continue
    if 0 not in score_weight[key] and ground_truth[key] > 0 and score_weight[key][0] > 0:
        X.append(score_weight[key][1:])
        y.append(ground_truth[key])
# print(X)
model = LinearRegression()
model.fit(X,y)

print(model.coef_)  

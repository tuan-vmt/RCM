import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import PairwiseDistance
import numpy as np
import os
from torch.nn import functional as F
import math
from config_path import opt
from torch.nn.functional import normalize
import time

class TabTransformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768,64)
        self.linear2 = nn.Linear(132,32)
        self.linear3 = nn.Linear(36,16)
        self.linear4 = nn.Linear(39,16)
        self.n = 6  #Number of Attention Layer 
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=16, nhead=8, batch_first=True, activation=nn.LeakyReLU(0.1), dropout=0.3), self.n)

    def forward(self, x):
        #Descriptions Embedding
        x[0] = x[0].to(opt.device)
        x[1] = x[1].to(opt.device)
        x[2] = x[2].to(opt.device)
        x[3] = x[3].to(opt.device)
        x[4] = x[4].to(opt.device)
        x[5] = x[5].to(opt.device)
        x[6] = x[6].to(opt.device)
        x[7] = x[7].to(opt.device)
        
        fe_descriptions = self.linear1(x[0]).reshape(-1, 1, 64)
        
        #Country Embedding     
        fe_country = self.linear3(x[1])
        
        #Category Embedding 
        fe_categories = self.linear4(x[2])
        
        fe_info_film = torch.cat((fe_country, fe_categories, x[3], x[4]), 1)
        
        #Self Attention
        fe_info_film_1 = self.transformer(fe_info_film)
                     
        fe_info_film_1 = fe_info_film_1.reshape(-1, 1, 64)
        
        fe_info_film_2 = torch.cat((x[5], x[6],x[7]), 2)
        # print(ebd_info2.size())
        
        fe_item_ebd = torch.cat((fe_descriptions, fe_info_film_1, fe_info_film_2), 2)
               
        fe_item_ebd = self.linear2(fe_item_ebd)
        
        return fe_item_ebd
    
    
    
class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()     
        self.n = 1 #Number of Attention Layer
        self.d_model = 32
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True, activation=nn.LeakyReLU(0.1), dropout=0.3), self.n).to("cuda:0")
        self.inplanes = opt.numbers_of_hst_films
        self.bn1 = nn.BatchNorm1d(self.inplanes + 1, eps=1e-05)
        
    def forward(self, x, list_rating, fe_target_item):      
        items = torch.stack(x).float().to("cuda:0").permute(1,0,2,3).reshape(-1,opt.numbers_of_hst_films, self.d_model)
        ratings = torch.stack(list_rating).float().to("cuda:0").permute(1,0).unsqueeze(-1)
        fe_user_prefer = torch.mul(items,ratings)
        fe_user_prefer = torch.cat((fe_user_prefer, fe_target_item), 1)
        fe_user_prefer = self.bn1(fe_user_prefer)
        fe_user_prefer = self.transformer(fe_user_prefer)
        return fe_user_prefer



class WideModel(nn.Module):
    
    def __init__(self,input_dim = 32*opt.numbers_of_hst_films + 32):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
  
    def forward(self, x):
        # x = torch.cat((x1, x2), 1)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
    
    
    
class DeepModel(nn.Module):
    
    def __init__(self,input_dim = 32*opt.numbers_of_hst_films + 64):
        super().__init__()
        self.linear1 = nn.Linear(65,32)
        self.fc =nn.Sequential(
            nn.Linear(input_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(32, 1)
        )
  
    def forward(self, x1, x4):
        x4 = self.linear1(x4).unsqueeze(1)
        x =  torch.cat((x1, x4), 1)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # print(x)
        return x


           
class TV360Recommend(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.TabTransformerTargetItem =  nn.DataParallel(TabTransformer(), device_ids=[0]).to(opt.device)
        self.TransformerLayer = TransformerLayer().to(opt.device)
        self.listTabTransformerHistoryItem = []
        self.linear_pairwise = nn.Linear(opt.numbers_of_hst_films, 1)
        self.wide_model = WideModel().to(opt.device)
        self.deep_model = DeepModel().to(opt.device)
    
    def forward(self, x):
        fe_hst_items, fe_target_item, ccai_embedding, list_rating = x
        
        #CCAI Embedding
        ccai_embedding = normalize(ccai_embedding.to(opt.device))    
                
        #Target Item Embedding
        fe_target_item = self.TabTransformerTargetItem(fe_target_item)
        
        #List History Item Embedding
        list_fe_hst_items = []
        for _, hst_item in enumerate(fe_hst_items):
            list_fe_hst_items.append(self.TabTransformerTargetItem(hst_item))
        fe_user_prefer = self.TransformerLayer(list_fe_hst_items, list_rating, fe_target_item)
        
        #Wide & Deep Model
        
        #Wide Model
        fe_pairwise = self.wide_model(fe_user_prefer)
        
        #Deep Model
        fe_deep = self.deep_model(fe_user_prefer, ccai_embedding)
        
        #Add Wide&Deep
        output = torch.add(fe_pairwise, fe_deep)
        output = torch.sigmoid(output)
        return output                
        

        
        
                   
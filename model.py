import torch
import torch.nn as nn
from config_path import opt
import math
from torch.autograd import Variable

class TabTransformer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 64)
        self.linear2 = nn.Linear(132, 32)
        self.linear3 = nn.Linear(36, 16)
        self.linear4 = nn.Linear(39, 16)
        self.n = 6  # Number of Attention Layer
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=16, nhead=8, batch_first=True,
                                                                            activation=nn.LeakyReLU(0.1),
                                                                            dropout=0.1), self.n)

    def forward(self, x):
        # Descriptions Embedding
        fe_descriptions = self.linear1(x[0]).reshape(-1, 1, 64)
        
        # Country Embedding
        fe_country = self.linear3(x[1])
        
        # Category Embedding
        fe_categories = self.linear4(x[2])
        
        fe_info_film = torch.cat((fe_country, fe_categories, x[3], x[4]), 1)
        
        # Self Attention
        fe_info_film_1 = self.transformer(fe_info_film)
                     
        fe_info_film_1 = fe_info_film_1.reshape(-1, 1, 64)
        
        fe_info_film_2 = torch.cat((x[5], x[6], x[7]), 2)
        # print(ebd_info2.size())
        
        fe_item_ebd = torch.cat((fe_descriptions, fe_info_film_1, fe_info_film_2), 2)
               
        fe_item_ebd = self.linear2(fe_item_ebd)
        
        return fe_item_ebd
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0)        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        x = x*math.sqrt(self.d_model)
        seq_length = x.size(1)
        
        pe = Variable(self.pe[:, :seq_length], requires_grad=False)
        
        if x.is_cuda:
            pe.cuda()
        # cộng embedding vector với pe 
        x = x + pe
        x = self.dropout(x)
        
        return x
    
    
class TransformerLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Number of Attention Layer
        self.n = 1
        self.d_model = 32
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8,
                                                                            batch_first=True,
                                                                            activation=nn.LeakyReLU(0.1),
                                                                            dropout=0.1), self.n).to(opt.device)
        self.inplanes = opt.numbers_of_hst_films
        self.pe = PositionalEncoder(d_model=self.d_model, max_seq_length=opt.numbers_of_hst_films+1, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 32))
        
    def forward(self, x, list_rating):
        cls_tokens = self.cls_token.expand(x[0].shape[0], -1, -1)
        items = torch.stack(x).float().to(opt.device).permute(1, 0, 2, 3)\
            .reshape(-1, opt.numbers_of_hst_films, self.d_model)
        if len(list_rating) > 1:
            ratings = torch.stack(list_rating).float().to(opt.device).permute(1, 0).unsqueeze(-1)
        else:
            ratings = torch.FloatTensor(list_rating).to(opt.device).unsqueeze(-1)
        fe_user_prefer = torch.mul(items, ratings)
        fe_user_prefer = torch.cat((fe_user_prefer, cls_tokens), dim=1)
        fe_user_prefer = self.pe(fe_user_prefer)
        fe_user_prefer = self.transformer(fe_user_prefer)
        # print(fe_user_prefer.size())
        return fe_user_prefer[:, -1]


class DeepModel(nn.Module):
    
    def __init__(self, input_dim=129):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x1, x2, x4):
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x4 = torch.flatten(x4, start_dim=1)
        x = torch.cat((x1, x2, x4), 1)
        x = self.fc(x)
        return x


class TV360Recommend(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.TabTransformerTargetItem = TabTransformer().to(opt.device)
        self.TransformerLayer = TransformerLayer().to(opt.device)
        self.linear_pairwise = nn.Linear(opt.numbers_of_hst_films, 1)
        self.deep_model = DeepModel().to(opt.device)
    
    def forward(self, x):
        fe_hst_items, fe_target_item, ccai_embedding, list_rating = x

        # CCAI Embedding
        ccai_embedding = ccai_embedding.to(opt.device)

        # Target Item Embedding
        fe_target_item = self.TabTransformerTargetItem(fe_target_item)

        # List History Item Embedding
        list_fe_hst_items = []
        for _, hst_item in enumerate(fe_hst_items):
            list_fe_hst_items.append(self.TabTransformerTargetItem(hst_item))
        fe_user_prefer = self.TransformerLayer(list_fe_hst_items, list_rating)
        
        # Wide & Deep Model
        
        # Wide Model
        fe_pairwise = torch.nn.CosineSimilarity(dim=2)(fe_user_prefer, fe_target_item)
        # print(fe_pairwise)
        # Deep Model
        fe_deep = self.deep_model(fe_user_prefer, fe_target_item, ccai_embedding)
        
        # Add Wide&Deep
        output = torch.add(fe_pairwise, fe_deep)
        output = torch.sigmoid(output)
        return output


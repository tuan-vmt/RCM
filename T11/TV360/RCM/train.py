import random
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F
from tqdm import tqdm
import model
from model import TV360Recommend
from dataloader import split_data, Tv360Dataset 
import os
# torch.multiprocessing.set_start_method('spawn')
from multiprocessing import set_start_method
from config_path import opt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import time


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    else:
        return torch.utils.data.dataloader.default_collate(batch)
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', help='Load Pretrained Model')
    parser.add_argument('--weights', type=str, default='best.pt', help='initial weights path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_opt(True)
    bertsentence= SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device=args.device)
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    max = 10
    print(args.device)
    #Split Data Train, Val and Test
    hst_users_items, target_users_items = split_data() 
    #Load Dataset  
    
    train_dataset = Tv360Dataset(bertsentence, hst_users_items, target_users_items, phase="train")
    val_dataset = Tv360Dataset(bertsentence, hst_users_items, target_users_items, phase="val")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    #Load Model
    model = TV360Recommend()
    if args.pretrained:
        model.load_state_dict(torch.load(args.weights))
    model = model.to(args.device)
    #Learning Rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    
    #Loss Function
    criterior = nn.MSELoss()
    
    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch, args.epochs))
        epoch_loss = 0.0
        # epoch_corrects = 0
        # Training
        count_loss = 0
        count_acc = 0
        model.train()
        
        for batch_idx, items in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if items is None:
                continue
            (inputs, labels) = items
            labels = labels.to(args.device).float()
            # print("\n\nstart batch---------------------",epoch)
            outputs = model(inputs)
            outputs = outputs[:, 0].float()
            # epoch_corrects += get_evaluation(outputs, labels)
            
            loss = criterior(outputs, labels)
            # print(labels.size()[0])
            count_loss += labels.size()[0]
            count_acc += len(outputs)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / count_loss
        # epoch_accuracy = epoch_corrects / count_acc
        # print("Epoch Accuracy: ", epoch_accuracy)
        if (epoch+1) % 5 == 0:
            PATH = "TV360/RCM/save_models/" + "{}.pth".format(epoch+1)
            torch.save(model.state_dict(), PATH)         
        print("Train loss: {:.10f}".format(epoch_loss))
        # print(epoch_accuracy)
        #Eval
        model.eval()
        epoch_loss = 0.0
        epoch_corrects = 0
        count_loss = 0
        count_acc = 0
        with torch.no_grad():  
            for batch_idx, items in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                if items is None:
                    continue
                (inputs, labels) = items
                labels = labels.to(args.device).float()
                # print("\n\nstart batch---------------------",epoch)
                outputs = model(inputs)
                outputs = outputs[:, 0].float()
                loss = criterior(outputs, labels)
                # epoch_corrects += get_evaluation(outputs, labels)
                epoch_loss += loss.item()
                count_loss += labels.size()[0]
                count_acc += len(outputs)
            epoch_loss = epoch_loss / count_loss
            # epoch_accuracy = epoch_corrects / count_acc
            if (epoch_loss < max):
                max = epoch_loss
                PATH = "TV360/RCM/save_models/" + str(epoch+1)+"_best_accuracy.pth"
                torch.save(model.state_dict(), PATH)
            print("Valid loss: {:.10f}".format(epoch_loss))
            # print(epoch_accuracy)

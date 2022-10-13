import random
import numpy as np
import json
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
from sklearn.metrics import classification_report
# from pytorchtools import EarlyStopping
import time

def collate_fn(batch):
    batch = list(filter(lambda x:x is not None, batch)) # filter out all the Nones
    if len(batch) == 0:
        return None
    else:
        return torch.utils.data.dataloader.default_collate(batch)

def get_evaluation(outputs, labels):
    y_prob = np.array(outputs.cpu().data.numpy())
    label = np.array(labels.cpu().data.numpy())
    y_prob = np.where(y_prob >= 0.5, 1, y_prob)
    y_prob = np.where(y_prob < 0.5, 0, y_prob)
    # print(classification_report(y_prob, label))
    return sum(y_prob == label)

if __name__ == "__main__":
    
    bertsentence= SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base', device=opt.device)
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    num_epochs = 301
    device = opt.device
    batch_size = 16
    max = 10
    
    #Split Data Train, Val and Test
    hst_users_items, target_users_items = split_data() 
    #Load Dataset  
    st = time.time()
    train_dataset = Tv360Dataset(bertsentence, hst_users_items, target_users_items, phase="train")
    print("TIME LOAD TRAIN DATASET: ", time.time() - st)
    val_dataset = Tv360Dataset(bertsentence, hst_users_items, target_users_items, phase="val")
    print("TIME LOAD VAL DATASET: ", time.time() - st)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn, shuffle=False)
    model = TV360Recommend().to(opt.device)
    model.load_state_dict(torch.load("TV360/RCM/save_models/300.pth"))
    #Learning Rate
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
    
    #Loss Function
    criterior = nn.BCELoss()
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
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
            labels = labels.to(device).float()
            # print("\n\nstart batch---------------------",epoch)
            outputs = model(inputs)
            outputs = outputs[:, 0].float()
            # print(outputs)
            # print(labels)
            # epoch_corrects += get_evaluation(outputs, labels)
            loss = criterior(outputs, labels)
            count_loss += 1
            count_acc += len(outputs)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / count_loss
        # epoch_accuracy = epoch_corrects / count_acc
        # print("Epoch Accuracy: ", epoch_accuracy)
        if (epoch % 5 == 0):
            PATH = "TV360/RCM/save_models/" + "{}.pth".format(epoch)
            torch.save(model.state_dict(), PATH)         
        print("Train loss: {:.10f}".format(epoch_loss))
        #Evaling-------------------------------------------------
        model.eval()
        epoch_loss = 0.0
        # epoch_corrects = 0
        count_loss = 0
        count_acc = 0
        with torch.no_grad():  
            for batch_idx, items in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                if items is None:
                    continue
                (inputs, labels) = items
                labels = labels.to(device).float()
                # print("\n\nstart batch---------------------",epoch)
                outputs = model(inputs)
                outputs = outputs[:, 0].float()
                loss = criterior(outputs, labels)
                # epoch_corrects += get_evaluation(outputs, labels)
                epoch_loss += loss.item()
                count_loss += 1
                count_acc += len(outputs)
            epoch_loss = epoch_loss / count_loss
            # epoch_accuracy = epoch_corrects / count_acc
            if (epoch_loss < max):
                max = epoch_loss
                PATH = "TV360/RCM/save_models/" + str(epoch)+"_best_accuracy.pth"
                torch.save(model.state_dict(), PATH)
            print("Valid loss: {:.10f}".format(epoch_loss))

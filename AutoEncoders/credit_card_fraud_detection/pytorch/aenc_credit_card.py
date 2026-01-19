import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
import kagglehub
import matplotlib.pyplot as plt

from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


# device setup

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# hyperparameters

batch_size = 100
epochs = 10


# data preprocessing

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

df = pd.read_csv(path+'/creditcard.csv') #, header=None)
#print(f"df -> {df}")

df_time = df.loc[:, 'Time']
max_t = df_time.max()
min_t = df_time.min()
df_time = df_time.map(lambda x: (x - min_t)/(max_t - min_t))

df_amt = df.loc[:, 'Amount']
max_a = df_amt.max()
min_a = df_amt.min()
df_amt = df_amt.map(lambda x: (x - min_a)/(max_a - min_a))

df_fea = df.drop(columns=["Time", "Amount", "Class"])
max_f = df_fea.max().max()
min_f = df_fea.min().min()
df_fea = df_fea.map(lambda x: (x - min_f)/(max_f - min_f))

df = pd.concat([df_time, df_amt, df_fea,  df.loc[:, 'Class']], axis=1)
print(f"df -> {df}")

normal_data = df[df[df.columns[-1]] == 0]
abnormal_data = df[df[df.columns[-1]] == 1]

#print (f"len normal_data: {len(normal_data)}")
#print (f"len abnormal_data: {len(abnormal_data)}")

data_x, data_y = normal_data.iloc[:,:-1], normal_data.iloc[:,-1]
anm_data_x, anm_data_y = abnormal_data.iloc[:,:-1], abnormal_data.iloc[:,-1]
print(f"data_x -> {data_x}")
print(f"data_y -> {data_y}")

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, shuffle=True)
anm_train_x, anm_test_x, anm_train_y, anm_test_y = train_test_split(anm_data_x, anm_data_y, test_size=0.1, shuffle=True)

class credit_card_dataset(Dataset):

    def __init__(self, data, label):
        self.data = torch.tensor(np.array(data), dtype=torch.float)
        self.label = torch.tensor(np.array(label), dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    


train_data = credit_card_dataset(train_x, train_y)
train_dload = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = credit_card_dataset(anm_test_x, anm_test_y)
test_dload = DataLoader(test_data, batch_size=64)


# model construction

class Credit_Card_Fraud_Detection_Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.h1  = nn.LazyLinear(800)
        self.h2  = nn.LazyLinear(500)
        self.out = nn.LazyLinear(100)

        self.rlu = nn.ReLU()

    def forward(self, x):

        x = self.rlu(self.h1(x))
        x = self.rlu(self.h2(x))
        x = self.rlu(self.out(x))

        return x

class Credit_Card_Fraud_Detection_Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.h1  = nn.LazyLinear(500)
        self.h2  = nn.LazyLinear(800)
        self.out = nn.LazyLinear(30)

        self.rlu = nn.ReLU()
        self.sig = nn.Sigmoid()        

    def forward(self, x):

        x = self.rlu(self.h1(x))
        x = self.rlu(self.h2(x))
        x = self.sig(self.out(x))

        return x

class Credit_Card_Fraud_Detection(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc  = Credit_Card_Fraud_Detection_Encoder()
        self.dec  = Credit_Card_Fraud_Detection_Decoder()
        

    def forward(self, x):

        x = self.enc(x)
        x = self.dec(x)

        return x

    
#model = build_Credit_Card_Fraud_Detection()
model = Credit_Card_Fraud_Detection().to(device)

# model settings

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.L1Loss()

# model training

def train(dataloader, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0
    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, x) # sending original input as label for reconstruction 

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)


print ("\nTraining: ")
Epochs = epochs
for i in range(0, Epochs):
    loss = train(train_dload, model, optimizer, loss_fn)
    print (f"Epoch {i}/{Epochs}: Training loss -> {loss}")


# model testing

def test(dataloader, model, loss_fn):

    model.eval()

    loss_v = 0
    cnt = 0
    for x, y in dataloader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, x)

        loss_v = loss_v + loss.item()
        cnt = cnt + 1

    print (f"Testing Loss: {loss_v/cnt}")
    

print ("\nTesting: ")
test(test_dload, model, loss_fn)
    

# model prediction

def predict(model):
    model.eval()
    with torch.no_grad():

        loss_v = []

        for x, y in train_data:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, x)

            loss_v.append(loss.item())

        loss = np.array(loss_v)
        #print (f"Train preds: {pred}")
        print (f"Train loss: {loss[:50]}")

        threshold = np.mean(loss) + np.std(loss)
        print("\nThreshold: ", threshold)

        loss_v = []

        for x, y in test_data:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, x)

            loss_v.append(loss.item())

        loss = np.array(loss_v)
        print (f"Test loss: {loss[:50]}")

        test_loss = torch.tensor(loss) #.numpy()
        preds = torch.lt(test_loss, threshold)
        print (f"preds: {preds[:50]}")

        correct = 0
        count = 0
        for p in preds:
            if not p:
                correct = correct + 1
            count = count + 1

        print (f"\nAccuracy: {correct/count}")


print ("\nSamples Prediction: ")
predict(model)

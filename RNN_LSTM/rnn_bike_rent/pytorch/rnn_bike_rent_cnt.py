import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers
import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


# device setup

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# hyper paramters

batch_size = 100

    
# data preprocessing

path = kagglehub.dataset_download("marklvl/bike-sharing-dataset")

print("Path to dataset files:", path)

data = tf.keras.utils.get_file("hour.csv", origin=f"file://{path}/hour.csv")

df = pd.read_csv(data)
print (f"df -> {df[df['dteday'] == '2011-01-01']}")


def get_features(df, timesteps):

    x_lst = []
    y_lst = []
    i = 0

    while i+timesteps < len(df):

        x, y = df.iloc[i:i+timesteps, :-1], df.iloc[i+timesteps-1, -1]
        
        x = np.array(x.loc[:, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']])
        x_lst.append(x)
        y = np.array(y)
        y_lst.append(y)

        i = i + 1

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
#test_x, test_y = get_features(test_x, test_y)
                                 
print (f"x -> {x[0]},\ny -> {y[0]}")

train_cnt = int(.90 * len(df))

train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]


class bike_dataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


train_data = bike_dataset(train_x, train_y)
train_dload = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = bike_dataset(test_x, test_y)
test_dload = DataLoader(test_data, batch_size=1)


# model construction

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(13, 100, batch_first=True)
        #self.rnn = nn.GRU(13, 100, batch_first=True)
        #self.rnn = nn.LSTM(13, 100, batch_first=True)
        self.h1  = nn.LazyLinear(500)
        #self.bn  = nn.LazyBatchNorm1d()
        self.out = nn.LazyLinear(1)

        self.rlu = nn.ReLU()

        
    def forward(self, x):

        # for RNN, GRU
        h0 = torch.zeros(1,x.size(0),100).to(device)
        x,_  = self.rnn(x, h0)
        x  = self.rlu(self.h1(x[:,-1,:]))
        x  = self.out(x) #self.bn(x))

        # for LSTM
        #h0 = torch.zeros(1,x.size(0),100).to(device)
        #c0 = torch.zeros(1,x.size(0),100).to(device)
        #x, (hn , cn) = self.rnn(x, (h0, c0))
        #x  = self.rlu(self.h1(x[:,-1,:]))
        #x  = self.out(x) #self.bn(x))

        return x

model = NeuralNet().to(device)
#summary(model, input_size=(10, 13))

# model settings

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


# model training

def train(dataLoader, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0    
    for batch, (x, y) in enumerate(dataLoader):

        x, y = x.to(device), y.to(device)

        y = y.unsqueeze(1)
        # forward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)
        

print ("\nTraining: ")
Epochs = 20
for i in range(0, Epochs):
    loss = train(train_dload, model, optimizer, loss_fn)
    print (f"Epoch {i}/{Epochs}: Training loss -> {loss}")


# model testing

def test(dataLoader, model, loss_fn):

    model.eval()

    loss_v = 0
    cnt = 0
    for x,y in dataLoader:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss_v = loss_v + loss_fn(pred, y).item()

        cnt = cnt + 1

    print (f"Testing Loss: {loss_v/cnt}")


print ("\nTesting: ")
test(test_dload, model, loss_fn)
    

# model prediction

def predict(model, count=10):
    model.eval()
    with torch.no_grad():

        i = 0
        for x,y in test_dload:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            print (f"pred: {pred}, y: {y}")

            i = i + 1
            if i == count:
                break

print ("\nSamples Prediction: ")
predict(model, 10)

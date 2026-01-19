import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.datasets
from keras import layers
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

iris_dataset = sklearn.datasets.load_iris(as_frame=True)

print (f"dataset -> {iris_dataset['frame']}")


class Iris_Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data), dtype=torch.float)
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    
train_cnt = int(.90 * len(iris_dataset['frame']))

train_x = iris_dataset['frame'].iloc[:train_cnt, :-1]
train_y = iris_dataset['frame'].iloc[:train_cnt, -1]
train_data = Iris_Dataset(train_x, train_y)
train_dataloader = DataLoader(train_data, batch_size=64)

#print (f"train_x -> {train_x}, train_y -> {train_y}")

test_x = iris_dataset['frame'].iloc[train_cnt:, :-1]
test_y = iris_dataset['frame'].iloc[train_cnt:, -1]
test_data = Iris_Dataset(test_x, test_y)
test_dataloader = DataLoader(test_data, batch_size=1)

#print (f"test_x -> {test_x}, test_y -> {test_y}")

#print (f"num features -> {len(test_x.columns)}")

if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")
print(f"Using {device} device")



class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(4, 200)
        self.h2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 3)

        self.r  = nn.ReLU()
        self.s  = nn.Sigmoid()
        self.b1 = nn.BatchNorm1d(200)
        self.b2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.h1(x)
        x = self.r(x)
        x = self.b1(x)
        x = self.h2(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.output(x)
        x = self.s(x)

        return x #torch.sigmoid(x)

    
model = NeuralNet().to(device) #NNModel() #.to(device) #NeuralNet().to(device)
#model.float() #double()
print (model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()


def train(model, train_dataloader, optimizer, loss_fn):

    model.train()

    for batch, (x,y) in enumerate(train_dataloader):

        x , y = x.to(device), y.to(device)
        pred = model(x)
        #y = y.unsqueeze(1)
        #print(f"y shape -> {y.shape}")
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 == 0:
            print (f"Training Loss -> {loss.item()}")

def test(model, test_dataloader, loss_fn):

    size = len(test_dataloader.dataset)
    model.eval()

    with torch.no_grad():

        loss_v = 0
        correct = 0
        for x, y in test_dataloader:

            x , y = x.to(device), y.to(device)
            pred = model(x)
            #y = y.unsqueeze(1)
            loss = loss_fn(pred, y)

            loss_v = loss_v + loss.item()
            if float(pred.argmax()) == y:
                correct = correct + 1 #pred.round() == y #(pred.argmax(1) == y).type(torch.float).sum().item()

        correct = correct/size
        print (f"Testing Loss -> {loss_v/size}, Accuracy -> {correct}")


Epochs = 10
for i in range(0, Epochs):
    print (f"Epoch : {i}")
    train(model, train_dataloader, optimizer, loss_fn)
    test(model, test_dataloader, loss_fn)
    print ("----------------")

# Prediction samples
model.eval()
with torch.no_grad():

    correct = 0
    i = 0
    for x, y in test_dataloader: #i in range(0, 40):
        x , y = x.to(device), y.to(device)
        pred = model(x)
        if float(pred.argmax()) == y:
            correct = correct + 1
        print (f"{i} -> pred: {pred}, pred_id: {int(pred.argmax())}, y: {y}")

        if i == 39:
            break
        i = i + 1

    print (f"Prediction Accuracy: {correct/15}")

'''
model.eval()
correct = 0
for i in range(0, 40):
    
    with torch.no_grad():
        x, y = test_data[i][0], test_data[i][1]
        x , y = x.to(device), y.to(device)
        pred = model(torch.tensor(x))
        if pred.round() == y:
            correct = correct + 1
        print (f"{i} -> pred: {pred}, y: {y}")

print (f"Prediction Accuracy: {correct/40}")
'''

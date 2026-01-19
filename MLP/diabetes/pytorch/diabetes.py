import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sklearn.datasets

class diabetes_data(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(np.array(data))
        self.labels = torch.tensor(np.array(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = sklearn.datasets.load_diabetes(as_frame=True)

print (f"data -> {dataset['frame']}")
#print (f"x -> {dataset['frame'].iloc[:, :-1]}")
#print (f"y -> {dataset['frame'].iloc[:, -1]}")

train_cnt = int(len(dataset['frame']) * .91)

train_x = dataset['frame'].iloc[:train_cnt, :-1]
train_y = dataset['frame'].iloc[:train_cnt, -1]

#print (f"train x -> {train_x}")
#print (f"train y -> {train_y}")

train_data = DataLoader(diabetes_data(train_x, train_y), batch_size=1)

test_x = dataset['frame'].iloc[train_cnt:, :-1]
test_y = dataset['frame'].iloc[train_cnt:, -1]

#print (f"test x -> {test_x}")
#print (f"test y -> {test_y}")

test_data_set = diabetes_data(test_x, test_y)
test_data = DataLoader(test_data_set, batch_size=64)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(10, 750)
        self.r1 = nn.ReLU()
        self.h2 = nn.Linear(750, 500)
        self.r2 = nn.ReLU()
        self.output = nn.Linear(500, 1)

    def forward(self, x):
        x = self.h1(x)
        x = self.r1(x)
        x = self.h2(x)
        x = self.r2(x)
        x = self.output(x)

        return x

model = NeuralNet()
model.double()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.MSELoss()

def train(model, train_data, optimizer, loss_fn):

    model.train()

    for batch, (x, y) in enumerate(train_data):

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print (f"Train Loss -> {loss.item()}")


def test(model, test_data, loss_fn):

    model.eval()

    with torch.no_grad():
        loss_v = 0
        for x, y in test_data:

            pred = model(x)
            loss = loss_fn(pred, y)

            loss_v = loss_v + loss.item()

        print (f"Test Loss -> {loss_v}")

epochs = 100
for i in range(0, epochs):
    print (f"Epoch {i+1}:")
    train(model, train_data, optimizer, loss_fn)
    test(model, test_data, loss_fn)
    print ("---------------")


model.eval()
for i in range(0, 10):
    x, y = test_data_set[i][0], test_data_set[i][1]
    pred = model(torch.tensor(x))
    print (f"{i} -> pred: {pred}, y: {y}")

    

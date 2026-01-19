import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing

housing_datasets = fetch_california_housing()

print(f"data len -> {len(housing_datasets['data'])}, target len -> {len(housing_datasets['target'])}")

train_cnt = int(.80 * len(housing_datasets['data']))
test_cnt  = len(housing_datasets['data']) - train_cnt

print (f"train cnt -> {train_cnt}, test cnt -> {test_cnt}")

class Housing_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_x = housing_datasets['data'][train_cnt:]
train_y = housing_datasets['target'][train_cnt:]
train_data = Housing_Dataset(train_x, train_y)
train_data_loader = DataLoader(train_data, batch_size=64)

test_x  = housing_datasets['data'][:train_cnt]
test_y  = housing_datasets['target'][:train_cnt]
test_data = Housing_Dataset(test_x, test_y)
test_data_loader = DataLoader(test_data, batch_size=64)


#for x, y in train_data_loader:
#    print (f"x -> {x}")
#    print (f"y -> {y}")
#    break

#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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
        self.h1   = nn.Linear(8, 500)
        self.relu = nn.ReLU()
        self.output = nn.Linear(500, 1)

    def forward(self, x):
        #x = x.to(torch.float32)
        x = self.h1(x)
        x = self.relu(x)
        x = self.output(x)

        return x

model = NeuralNet() #.to(device)
model.double()
print (model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters()) #, lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):

    model.train()
    for batch, (x,y) in enumerate(dataloader):
        #x , y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%10 == 0:
            print (f"loss -> {loss.item()}")


def test(dataloader, model, loss_fn):

    model.eval()
    test_loss = 0

    with torch.no_grad():

        for x, y in dataloader:
            #x , y = x.to(device), y.to(device)
            pred = model(x)

            test_loss = test_loss + loss_fn(pred, y).item()

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_data_loader, model, loss_fn, optimizer)
    test(test_data_loader, model, loss_fn)
print("Done!")


model.eval()

for i in range(0, 10):
    x, y = torch.tensor(test_data[i][0]), torch.tensor(test_data[i][1])
    with torch.no_grad():
        #x = x.to(device)
        pred = model(x)
        predicted, actual = pred, y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from keras import layers


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# hyper paramters

batch_size = 100

# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_cnt = int(.10 * len(x_train))
test_cnt  = int(.10 * len(x_test))

x_train = x_train[:train_cnt]
y_train = y_train[:train_cnt]

x_test = x_test[:test_cnt]
y_test = y_test[:test_cnt]

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


class mnist_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_data = mnist_dataset(x_train, y_train)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = mnist_dataset(x_test, y_test)
test_dataloader = DataLoader(test_data, batch_size=1)
    

# model construction

class Cnn_Mnist_Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1  = nn.LazyConv2d(1, 3)
        self.conv2  = nn.LazyConv2d(1, 3)
        self.mlp    = nn.LazyLinear(800)
        self.bn     = nn.LazyBatchNorm1d()
        self.output = nn.Linear(800, 10)

        self.maxpl  = nn.MaxPool2d(2)
        self.rlu    = nn.ReLU()
        self.flt    = nn.Flatten()
        self.sfm    = nn.Softmax(dim=1)

    def forward(self, x):

        #x = self.bn(x)
        x = self.rlu(self.conv1(x))
        x = self.rlu(self.conv2(x))
        x = self.flt(self.maxpl(x))
        x = self.bn(self.mlp(x))
        x = self.rlu(x)
        x = self.sfm(self.output(x))

        return x


model = Cnn_Mnist_Classifier().to(device)

summary(model, input_size=(1, 28, 28))
        
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()

# model training

def train(train_dataloader, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0
    for batch, (x, y) in enumerate(train_dataloader):

        x, y = x.to(device), y.to(device)

        # fordward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)


print ("\nTraining: ")
Epochs = 10
for i in range(0, Epochs):
    loss = train(train_dataloader, model, optimizer, loss_fn)
    print (f"Epoch {i}/{Epochs}: Training loss -> {loss}")

           
# model testing

def test(test_dataloader, model, loss_fn):

    model.eval()
    with torch.no_grad():
        
        count = 0
        total_loss = 0
        acc = 0
        for x, y in test_dataloader:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss = total_loss + loss.item()
            count = count + 1

            if pred.argmax() == y:
                acc = acc + 1

        print (f"Testing loss: {total_loss/count}, accuracy: {acc/count}")


print ("\nTesting: ")
test(test_dataloader, model, loss_fn)

    
# model prediction

def predict(model, count=10):

    model.eval()
    with torch.no_grad():

        i = 0
        for x, y in test_dataloader:

            if i == count:
                break

            i = i + 1
            x, y = x.to(device), y.to(device)
            pred = model(x)

            print (f"pred: {pred.argmax()}, y: {y}")

print ("\nSamples Prediction: ")
predict(model, 10)




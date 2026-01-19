import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# hyper paramters

batch_size = 100

# data preprocessing

#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Example normalization
    ])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_dload = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_dload  = DataLoader(testset, batch_size=1)

train_x, train_y = trainset[0]
print (f"train_x shape: {train_x.shape}, train_y: {train_y}")

    
# model construction

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1  = nn.LazyConv2d(32, 3)
        self.conv2  = nn.LazyConv2d(64, 3)
        self.conv3  = nn.LazyConv2d(64, 3)
        self.maxpl1 = nn.MaxPool2d(3)
        self.maxpl2 = nn.MaxPool2d(2)
        self.flt    = nn.Flatten()
        self.h1     = nn.LazyLinear(500)
        self.bn     = nn.LazyBatchNorm1d()
        self.output = nn.LazyLinear(100)
        
        self.rlu  = nn.ReLU()
        #self.sfm = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.rlu(self.conv1(x))
        x = self.rlu(self.conv2(self.maxpl1(x)))
        x = self.rlu(self.conv3(self.maxpl2(x)))
        x = self.bn(self.rlu(self.h1(self.flt(x))))
        x = self.output(x)
        
        return x


model = NeuralNet().to(device)

summary(model, input_size=(3, 32, 32))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()


# model training

def train(train_dload, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0
    for batch, (x,y) in enumerate(train_dload):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)


print ("\nTraining: ")
Epochs = 10
for i in range(0, Epochs):
    loss = train(train_dload, model, optimizer, loss_fn)
    print (f"Epoch {i}/{Epochs}: Training loss -> {loss}")


# model testing

def test(test_dload, model, loss_fn):

    model.eval()

    count  = 0
    acc    = 0
    loss_v = 0
    for x, y in test_dload:

        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss_v = loss_v + loss.item()
        count = count + 1

        if pred.argmax() == y:
            acc = acc + 1

    print (f"Testing loss: {loss_v/count}, Accuracy: {acc/count}")


print ("\nTesting: ")
test(test_dload, model, loss_fn)


# model prediction

def predict(model, count=10):

    model.eval()
    with torch.no_grad():

        i = 0
        for x, y in test_dload:

            if i == 10:
                break

            i = i + 1
            x, y = x.to(device), y.to(device)
            pred = model(x)

            print (f"pred: {pred.argmax()}, y: {y}")


print ("\nSamples Prediction: ")
predict(model, 10)


import os, sys
import kagglehub

import numpy as np
#import tensorflow as tf
from keras import layers
import keras, keras_hub

#from tensorflow.keras.preprocessing.image import load_img

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.utils.data import random_split

import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms


class_names  = ['glioma','meningioma','notumor','pituitary']
num_classes = 4
batch_size  = 32
image_size  = 256
patch_size  = 16
num_patches = (image_size//patch_size) ** 2
nheads      = 4
num_layers  = 4
hidden_size = 64
mlp_units   = 1024

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

train_data = path+'/Training'
test_data  = path+'/Testing'

transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        #transforms.ToTensor(),
        transforms.Grayscale(),
        #transforms.RandomRotation(degrees=(-0.20,0.20)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.Resize((image_size, image_size)),
    ])

train_dataset = datasets.ImageFolder(root=train_data, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data, transform=transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, validation_dataset = random_split(train_dataset, [train_size, val_size])

train_dload = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dload  = DataLoader(test_dataset, batch_size=1, shuffle=True)


print (f"len train_data: {len(train_dataset)}")


class PatchEncoder(nn.Module):
    def __init__(self, patch_size, num_patches, hidden_size):
        super().__init__()
        self.psize    = patch_size
        self.npatches = num_patches
        self.proj     = nn.LazyLinear(hidden_size)
        self.pos_emb  = nn.Embedding(self.npatches, hidden_size)
        self.unfold   = nn.Unfold(kernel_size=self.psize, stride=self.psize)

    def forward(self, img):

        #batch_size = ?
        patches = self.unfold(img).permute(0,2,1).to(device) # permute -> batch_size, num_patches, patch_size
        emb_inp = torch.arange(0, self.npatches).to(device)
        out     = self.proj(patches) + self.pos_emb(emb_inp)
        
        return out


# model construction

class ViT(nn.Module):

    def __init__(self, patch_size, num_patches, hidden_size, num_layers):
        super().__init__()

        self.nlayers = num_layers
        self.hdim    = hidden_size

        self.patch_enc = PatchEncoder(patch_size, num_patches, hidden_size).to(device)

        self.trans_enc_lyr = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=nheads, dim_feedforward=mlp_units, batch_first=True) 
        self.trans_enc = nn.TransformerEncoder(self.trans_enc_lyr, num_layers=self.nlayers)
        
        #self.trans_enc1 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=nheads, dim_feedforward=mlp_units, batch_first=True)
        #self.trans_enc2 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=nheads, dim_feedforward=mlp_units, batch_first=True)
        #self.trans_enc3 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=nheads, dim_feedforward=mlp_units, batch_first=True)
        #self.trans_enc4 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=nheads, dim_feedforward=mlp_units, batch_first=True)
        
        self.flt  = nn.Flatten()
        self.h1   = nn.LazyLinear(mlp_units)
        self.h2   = nn.LazyLinear(mlp_units)
        self.out  = nn.LazyLinear(num_classes)
        self.bn  = nn.LazyBatchNorm1d()
        #self.ln  = nn.LayerNorm(self.hdim)
        
        self.rlu = nn.ReLU()
        self.smx = nn.Softmax(dim=1)

    def forward(self, x):

        #x = self.ln(self.patch_enc(x))
        x = self.patch_enc(x)

        #x = self.trans_enc1(x)
        #x = self.trans_enc2(x)
        #x = self.trans_enc3(x)
        #x = self.trans_enc4(x)

        x = self.trans_enc(x)
        
        x = self.flt(x)

        #x = self.rlu(self.h1(x))
        #x = self.rlu(self.h2(x))

        # adding batchNorm to hidden layers (apply only to one hidden layer) seems to improve accuracy
        x = self.bn(self.rlu(self.h1(x)))
        #x = self.rlu(self.h2(x))
        
        x = self.smx(self.out(x))
        
        return x


model = ViT(patch_size, num_patches, hidden_size, num_layers).to(device)
#model = PatchEncoder(patch_size, num_patches, hidden_size).to(device)

#dummy_input = torch.randn(10, 1, 256, 256)
#model(dummy_input)

#sys.exit(-1)

#summary(model, input_size=(1, 256, 256))
print (model)

# model settings

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


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
    print (f"Epoch {i+1}/{Epochs}: Training loss -> {loss}")


# model testing

def test(test_dload, model, loss_fn):

    model.eval()

    with torch.no_grad():
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


# prediction

def predict(model, count=10):
    model.eval()
    with torch.no_grad():

        i = 0
        for x, y in test_dload:

            if i == 10:
                break

            x, y = x.to(device), y.to(device)
            pred = model(x)

            print (f"pred: {pred.argmax()}, prd : {pred}, y: {y}")
            i = i + 1

            
print ("\nSamples Prediction: ")
predict(model, count=10)

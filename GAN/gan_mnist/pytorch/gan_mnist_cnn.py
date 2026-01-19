import os, sys
import keras
#import keras_hub
import numpy as np
#import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


class mnist_data(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(np.array(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_images = x_train.reshape(-1, 1, 28, 28).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
noise_dim = 100

train_dload = DataLoader(mnist_data(train_images), batch_size=BATCH_SIZE, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense = nn.LazyLinear(256*7*7, bias=False)
        self.convTrp1 = nn.LazyConvTranspose2d(128, (5, 5), stride=(1, 1), padding=2, bias=False)
        self.convTrp2 = nn.LazyConvTranspose2d(64, (5, 5), stride=(2, 2), padding=2, bias=False)
        self.convTrp3 = nn.LazyConvTranspose2d(1, (5, 5), stride=(2, 2), padding=1, output_padding=1, bias=False)
        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        
        self.lrlu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.dense(x)
        x = self.lrlu(self.bn1(x))
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, 256, 7, 7))

        x = self.lrlu(self.bn2(self.convTrp1(x)))
        x = self.lrlu(self.bn3(self.convTrp2(x)))
        x = self.tanh(self.convTrp3(x))

        return x

    
generator = Generator().to(device)

#noise = torch.randn(16, noise_dim, device=device) #torch.normal(mean=0.0,std=1.0,size=(16, 100))
#generator.train(False)
#generated_image = generator(noise)
#generated_image = generated_image.to(device)
#print (f"generated_image shape: {generated_image.shape}")
#sys.exit(-1)

print (generator)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.LazyConv2d(64, (5, 5), stride=(2, 2), padding=2)
        self.conv2 = nn.LazyConv2d(128, (5, 5), stride=(2, 2), padding=2)
        self.dense = nn.LazyLinear(1)

        self.lrlu = nn.LeakyReLU()
        self.flt = nn.Flatten()
        self.drp = nn.Dropout(p=0.3)

    def forward(self, x):

        x = self.drp(self.lrlu(self.conv1(x)))
        x = self.drp(self.lrlu(self.conv2(x)))
        x = self.flt(x)
        x = self.dense(x)

        return x


discriminator = Discriminator().to(device)

#dummy = train_dataset.take(1).get_single_element()[0]
#print (f"real input shape: {dummy.shape}")
#print (f"fake input shape: {generated_image.shape}")
#decision = discriminator(generated_image)
#print (decision)
#sys.exit(-1)

print(discriminator)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

cross_entropy = nn.BCEWithLogitsLoss()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(real_output, torch.ones_like(real_output))
    fake_loss = cross_entropy(fake_output, torch.zeros_like(fake_output))
    total_loss = (real_loss + fake_loss) #/2
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(fake_output, torch.ones_like(fake_output))


def train(generator, discriminator, train_data, generator_optimizer, discriminator_optimizer):

    generator.train()
    discriminator.train()

    gen_loss_cml  = 0
    disc_loss_cml = 0

    batch_cnt = 0
    for batch, images in enumerate(train_data):

        #noise = torch.normal(mean=0.0,std=1.0,size=(BATCH_SIZE, 100))
        noise = torch.randn(BATCH_SIZE, noise_dim, device=device)

        generated_images = generator(noise)
        fake_output = discriminator(generated_images)
        gen_loss  = generator_loss(fake_output)

        generator_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        generator_optimizer.step()

        
        generated_images = generator(noise)
        fake_output = discriminator(generated_images) #.detach())
        images = images.to(device)
        real_output = discriminator(images)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        discriminator_optimizer.zero_grad()
        disc_loss.backward()
        discriminator_optimizer.step()

        gen_loss_cml  = gen_loss_cml + gen_loss.item()
        disc_loss_cml = disc_loss_cml + disc_loss.item()
        batch_cnt = batch_cnt + 1
        
    return (gen_loss_cml/batch_cnt), (disc_loss_cml/batch_cnt)


print ("\nTraining: ")
Epochs = 20
for i in range(0, Epochs):
    gen_loss, disc_loss = train(generator, discriminator, train_dload, generator_optimizer, discriminator_optimizer)
    print (f"Epoch {i+1}/{Epochs}: Training loss -> Generator: {gen_loss}, Discriminator: {disc_loss}")


def display_images(tensor_img, num_img=16, size=(1, 28, 28)):
    tensor_img = (tensor_img + 1) / 2
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_img], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()

num_examples_to_generate = 16
seed =  torch.randn(num_examples_to_generate, noise_dim, device=device)

generator.train(False)
seed = seed.to(device)
predictions = generator(seed)

display_images(predictions, num_img=16)


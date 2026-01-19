import os, sys
import kagglehub

import numpy as np
import tensorflow as tf
from keras import layers

from tensorflow.keras.preprocessing.image import load_img

class_names  = ['NORMAL', 'PNEUMONIA']
num_classes = 2
batch_size  = 32
image_size  = 256
patch_size  = 16
num_patches = (image_size // patch_size) ** 2
nheads      = 4
num_layers  = 15
hidden_size = 64
mlp_units   = 1024

path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
path = path+"/chest_xray/chest_xray"

print("Path to dataset files:", path)

train_data = path+'/train'
test_data  = path+'/test'
vald_data  = path+'/val'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size,image_size),
    shuffle=True,
    #seed=50,
    #validation_split=0.1,  
    #subset='training',
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    vald_data,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size,image_size),
    shuffle=True,
    #seed=50,
    #validation_split=0.1,
    #subset='validation',
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size,image_size),
    #shuffle=True,
    #seed=50,
)

print(f"len train_dataset: {len(train_dataset)}")
print(f"len validation_dataset: {len(validation_dataset)}")
print(f"len test_dataset: {len(test_dataset)}")
#print(f"train_dataset: {next(iter(train_dataset.take(1)))}")

augment_layers = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=(-0.20, 0.20)),
        layers.RandomFlip(seed=15),
    ])


def process_data(image, label):

    augmented_image = augment_layers(image)
    resized_image   = tf.image.resize(augmented_image, size=(image_size, image_size))
    return resized_image, label

training_dataset = (
    train_dataset.
    shuffle(1000)
    .map(process_data,num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

#print(f"training_dataset: {len(training_dataset)}")
#print(f"training_dataset: {next(iter(training_dataset.take(1)))}")

vald_dataset=(
   validation_dataset.shuffle(1000)
    .prefetch(tf.data.AUTOTUNE)
)


class PatchEncoder(layers.Layer):
    def __init__(self, patch_size, hidden_size):
        super().__init__()

        self.proj    = layers.Dense(hidden_size)
        self.pos_emb = layers.Embedding(input_dim=num_patches, output_dim=hidden_size)
        self.psize   = patch_size

    def call(self, img):
        batch_size = tf.shape(img)[0]
        patches = tf.image.extract_patches(images=img, sizes=[1, self.psize, self.psize, 1], strides=[1, self.psize, self.psize, 1], rates=[1,1,1,1], padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        emb_inp = tf.range(0, num_patches)
        out = self.proj(patches) + self.pos_emb(emb_inp)

        return out
        

class TransformerEncoder(layers.Layer):
    def __init__(self, nheads, hidden_size):
        super().__init__()

        self.n1 = layers.LayerNormalization()
        self.n2 = layers.LayerNormalization()
        self.mhead_att = layers.MultiHeadAttention(nheads, hidden_size)
        self.h1 = layers.Dense(hidden_size, activation="gelu")
        self.h2 = layers.Dense(hidden_size, activation="gelu")

    def call(self, inp):
        
        
        x  = self.mhead_att(inp, inp)
        #print (f"mhead shape: {x.shape}")
        x0 = layers.Add()([x, inp])
        x1 = self.n1(x0)
        x2 = self.h1(x1)
        x3 = layers.Add()([x1, x2])
        out = self.n2(x3)
        
        return out


class ViTrf(tf.keras.Model):
    def __init__(self, nheads, hidden_size, nlayers, mlp_units):
        super().__init__()

        #self.inp = layers.Input(shape=(256,256,1))
        self.nlayers = nlayers
        self.patch_enc = PatchEncoder(patch_size, hidden_size)
        self.trs_enc   = [TransformerEncoder(nheads, hidden_size) for i in range(nlayers)]
        self.flt       = layers.Flatten()
        self.h1        = layers.Dense(mlp_units, activation="relu")
        self.h2        = layers.Dense(mlp_units, activation="relu")
        self.out       = layers.Dense(num_classes, activation="softmax")

    def call(self, inp):

        x = self.patch_enc(inp)

        for i in range(self.nlayers):
            x = self.trs_enc[i](x)

        x = self.flt(x)
        x = self.h1(x)
        x = self.h2(x)
        out = self.out(x)

        return out


ViT = ViTrf(nheads, hidden_size, num_layers, mlp_units)

#ViT.summary()

ViT.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #"Adam", #
            loss="categorical_crossentropy",
            metrics=["accuracy"])

ViT.fit(train_dataset,
        epochs=10,
        validation_data=validation_dataset)

loss, acc = ViT.evaluate(test_dataset)

print (f"Testing loss: {loss}, Accuracy: {acc}")

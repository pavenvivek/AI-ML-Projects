import os, sys
import kagglehub

import numpy as np
import tensorflow as tf
from keras import layers
import keras, keras_hub

from tensorflow.keras.preprocessing.image import load_img

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

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

train_data = path+'/Training'
test_data  = path+'/Testing'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size,image_size),
    shuffle=True,
    seed=50,
    validation_split=0.1,  
    subset='training',
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_data,
    labels='inferred',
    label_mode='categorical',
    class_names=class_names,
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(image_size,image_size),
    shuffle=True,
    seed=50,
    validation_split=0.1,
    subset='validation',
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

#print(f"train_dataset: {len(train_dataset)}")
#print(f"train_dataset: {next(iter(train_dataset.take(1)))}")

augment_layers = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=(-0.20, 0.20)),
        layers.RandomFlip(seed=15),
    ])


def process_data(image, label):

    augmented_image = augment_layers(image)
    resized_image   = tf.image.resize(augmented_image, size=(image_size, image_size))

    #return resized_image, label
    return image, label


training_dataset = (
    train_dataset.
    shuffle(1000)
    #.map(process_data,num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

#print(f"training_dataset: {len(training_dataset)}")
#print(f"training_dataset: {next(iter(training_dataset.take(1)))}")

vald_dataset=(
   validation_dataset.shuffle(1000)
    .prefetch(tf.data.AUTOTUNE)
)


class PatchEncoder(layers.Layer):
    def __init__(self, patch_size, num_patches, hidden_size):
        super().__init__()

        self.proj    = layers.Dense(hidden_size)
        self.pos_emb = layers.Embedding(input_dim=num_patches, output_dim=hidden_size)
        #self.pos_emb = keras_hub.layers.PositionEmbedding(sequence_length=256)
        self.psize   = patch_size
        self.npatches = num_patches

    def call(self, img):
        batch_size = tf.shape(img)[0]
        patches = tf.image.extract_patches(images=img, sizes=[1, self.psize, self.psize, 1], strides=[1, self.psize, self.psize, 1], rates=[1,1,1,1], padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
        emb_inp = tf.range(0, self.npatches)
        out = self.proj(patches) + self.pos_emb(emb_inp)
        #out = self.proj(patches) + self.pos_emb(emb_inp)
        
        return out
        

# Functional

def build_ViT():
    inp   = layers.Input(shape=(256, 256, 1))
    x     = PatchEncoder(patch_size, num_patches, hidden_size)(inp)

    for _ in range(num_layers):
        x = keras_hub.layers.TransformerEncoder(intermediate_dim=hidden_size, num_heads=nheads)(x)

    flt   = layers.Flatten()(x)
    h1    = layers.Dense(mlp_units, activation="relu")(flt)
    h2    = layers.Dense(mlp_units, activation="relu")(h1)
    out   = layers.Dense(num_classes, activation="softmax")(h2)


    model = keras.Model(inputs=inp, outputs=out, name="vision_transformer")

    return model


# Subclass

class ViT(keras.Model):

    def __init__(self):
        super().__init__()

        self.patch_enc = PatchEncoder(patch_size, num_patches, hidden_size)
        self.trs_enc   = [keras_hub.layers.TransformerEncoder(intermediate_dim=hidden_size, num_heads=nheads) for _ in range(num_layers)]
        self.h1        = layers.Dense(mlp_units, activation="relu")
        self.h2        = layers.Dense(mlp_units, activation="relu")
        self.out       = layers.Dense(num_classes, activation="softmax")

        self.flt       = layers.Flatten()
        
    def call(self, x):

        x = self.patch_enc(x)

        for i in range(num_layers):
            x = self.trs_enc[i](x)

        x = self.flt(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x


#ViT = build_ViT()
ViT = ViT()

ViT.summary()

ViT.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #"Adam", #
            loss="categorical_crossentropy",
            metrics=["accuracy"])

ViT.fit(training_dataset,
        epochs=10,
        validation_data=vald_dataset)

loss, acc = ViT.evaluate(test_dataset)

print (f"Testing loss: {loss}, Accuracy: {acc}")

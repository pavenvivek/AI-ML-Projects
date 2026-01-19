import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt


# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_images = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
noise_dim = 100

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# model construction

class Generator(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(256, input_shape=(100,))
        self.dense2 = layers.Dense(512)
        self.dense3 = layers.Dense(784, activation='tanh')

        self.lrlu = layers.LeakyReLU()
        self.drp  = layers.Dropout(0.2)
        
        
    def call(self, x):

        x = self.drp(self.lrlu(self.dense1(x)))
        x = self.drp(self.lrlu(self.dense2(x)))
        x = self.dense3(x)

        return x


generator = Generator()

noise = tf.random.normal([5, 100])
generated_image = generator(noise, training=False)
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#plt.show()

generator.summary()


class Discriminator(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = layers.Dense(512, input_shape=(784,))
        self.dense2 = layers.Dense(256)
        self.dense3 = layers.Dense(1)

        self.lrlu = layers.LeakyReLU()
        self.drp  = layers.Dropout(0.2)
        self.flt  = layers.Flatten()
        
        
    def call(self, x):

        x = self.flt(x)
        x = self.drp(self.lrlu(self.dense1(x)))
        x = self.drp(self.lrlu(self.dense2(x)))
        x = self.dense3(x)

        return x


discriminator = Discriminator()

#dummy = train_dataset.take(1).get_single_element()[0]
#print (f"real input shape: {dummy.shape}")
#print (f"fake input shape: {generated_image.shape}")
decision = discriminator(generated_image)
print (decision)

discriminator.summary()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# training

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss
    

def train(dataset, epochs):
    for epoch in range(epochs):

        gen_loss  = 0
        disc_loss = 0
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print (f"Epoch: {epoch}")
        print (f"Trainin Loss -> Generator: {gen_loss}, Discriminator: {disc_loss}")

EPOCHS = 50
train(train_dataset, EPOCHS)


# evaluation

def display_image(image):

    fig = plt.figure(figsize=(4, 4))
    for i in range(image.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.show()


noise = tf.random.normal([16, noise_dim])
generated_image = generator(noise, training=False)
generated_image = tf.reshape(generated_image, [16, 28, 28, 1])
display_image(generated_image)


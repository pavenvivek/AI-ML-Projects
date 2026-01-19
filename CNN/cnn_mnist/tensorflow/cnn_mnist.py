import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers

# hyperparameters

batch_size = 100

# data preprocessing

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

train_cnt = int(.10 * len(x_train))
test_cnt  = int(.10 * len(x_test))

x_train = x_train[:train_cnt]
y_train = y_train[:train_cnt]

x_test = x_test[:test_cnt]
y_test = y_test[:test_cnt]

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def preprocess(data, label):

    data = tf.cast(data, dtype=tf.float32)

    return (data, label)

train_data = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE) # 
test_data  = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# model construction

# Functional
def build_Cnn_Mnist():
    inputs = layers.Input(shape=(28, 28, 1))
    conv1  = layers.Conv2D(1, 3, activation="relu")(inputs)
    conv2  = layers.Conv2D(1, 3, activation="relu")(conv1)
    maxpl  = layers.MaxPooling2D(2,2)(conv2)
    #globpl = layers.GlobalAveragePooling2D()(maxpl)
    flt    = layers.Flatten()(maxpl)
    mlp    = layers.Dense(800, activation="relu")(flt)
    outputs = layers.Dense(10, activation="softmax")(mlp)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn")

    return model


# Subclass
class Cnn_Mnist_Classifier(keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1  = layers.Conv2D(1, 3, activation="relu")
        self.conv2  = layers.Conv2D(1, 3, activation="relu")
        self.mlp    = layers.Dense(800) #, activation="relu")
        self.bn     = layers.BatchNormalization()
        self.out    = layers.Dense(10, activation="softmax")

        self.maxpl  = layers.MaxPooling2D(2,2)
        self.flt    = layers.Flatten()
        self.rlu    = layers.ReLU()

    def call(self, x):

        #x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flt(self.maxpl(x))
        x = self.mlp(x)
        x = self.rlu(self.bn(x))
        x = self.out(x)

        return x


#model = build_Cnn_Mnist()
model = Cnn_Mnist_Classifier()

input_shape = (batch_size, 28, 28, 1)
dummy_input = tf.random.uniform(input_shape, dtype=tf.float32) 
model(dummy_input)
model.summary()


# model settings

model.compile(optimizer="Adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])


# model training

print ("\nTraining: ")
model.fit(train_data, epochs=10)


# model testing

print ("\nTesting: ")
loss, acc = model.evaluate(test_data)

print (f"Loss: {loss}, Accuracy: {acc}")


# model prediction

print ("\nSamples Prediction: ")
for i in range(0, 10):
    x = tf.cast(x_test[i], dtype=tf.float32)
    pred = model.predict(np.array([x]))
    print (f"pred: {pred.argmax()}, y: {y_test[i]}")

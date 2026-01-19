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

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

train_cnt = int(1 * len(x_train))
test_cnt  = int(1 * len(x_test))

x_train = tf.constant(x_train[:train_cnt], dtype=np.float32)
y_train = tf.constant(y_train[:train_cnt])

x_test = tf.constant(x_test[:test_cnt], dtype=np.float32)
y_test = tf.constant(y_test[:test_cnt])

print (f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def preprocess(data, label):

    #data = tf.cast(data, dtype=tf.float32)

    return (data, label)

train_data = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE) # 
test_data  = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)



# model construction

# Functional
def build_Cnn_Cifar():

    inputs = layers.Input(shape=(32, 32, 3))
    conv1  = layers.Conv2D(32, 3, activation="relu")(inputs)
    #b1     = layers.BatchNormalization()(conv1)
    maxpl  = layers.MaxPooling2D(pool_size=(3, 3))(conv1)
    conv2  = layers.Conv2D(64, 3, activation="relu")(maxpl)
    #b2     = layers.BatchNormalization()(conv2)
    maxpl  = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3  = layers.Conv2D(64, 3, activation="relu")(maxpl)
    #b3     = layers.BatchNormalization()(conv3)
    flt    = layers.Flatten()(conv3)
    h1     = layers.Dense(500, activation="relu")(flt)
    b4     = layers.BatchNormalization()(h1)
    outputs = layers.Dense(100, activation="softmax")(b4)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn")
    
    return model

# Subclass
class Cnn_Cifar_Classifier(keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1  = layers.Conv2D(32, 3, activation="relu")
        self.conv2  = layers.Conv2D(64, 3, activation="relu")
        self.conv3  = layers.Conv2D(64, 3, activation="relu")
        self.h1     = layers.Dense(500, activation="relu")
        self.bn     = layers.BatchNormalization()
        self.out    = layers.Dense(100, activation="softmax")
        
        self.maxpl1 = layers.MaxPooling2D(3)
        self.maxpl2 = layers.MaxPooling2D(2)
        self.flt    = layers.Flatten()
        #self.rlu    = layers.ReLU()

        
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(self.maxpl1(x))
        x = self.conv3(self.maxpl2(x))
        x = self.bn(self.h1(self.flt(x)))
        #x = self.rlu(x)
        x = self.out(x)
        
        return x


#model = build_Cnn_Cifar() 
model = Cnn_Cifar_Classifier() 

input_shape = (batch_size, 32, 32, 3)
dummy_input = tf.random.uniform(input_shape, dtype=tf.float32) 
model(dummy_input, training=False)
model.summary()

# model settings

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

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
    x, y = x_test[i], y_test[i]
    pred = model.predict(np.array([x]))

    print (f"{i} -> pred: {pred.argmax()}, y: {y}")



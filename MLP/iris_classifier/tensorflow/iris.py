import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.datasets
from keras import layers

iris_dataset = sklearn.datasets.load_iris(as_frame=True)

print (f"dataset -> {iris_dataset['frame']}")

train_cnt = int(.90 * len(iris_dataset['frame']))

train_x = tf.constant(iris_dataset['frame'].iloc[:train_cnt, :-1])
train_y = tf.constant(iris_dataset['frame'].iloc[:train_cnt, -1])

print (f"train_x -> {train_x}, train_y -> {train_y}")

test_x = tf.constant(iris_dataset['frame'].iloc[train_cnt:, :-1])
test_y = tf.constant(iris_dataset['frame'].iloc[train_cnt:, -1])

print (f"test_x -> {test_x}, test_y -> {test_y}")

#print (f"num features -> {len(test_x.columns)}")

inputs = layers.Input(shape=(4,))
h1     = layers.Dense(200, activation="relu")(inputs)
#b1     = layers.BatchNormalization()(h1)
h2     = layers.Dense(100, activation="relu")(h1)
#b2     = layers.BatchNormalization()(h2)
outputs = layers.Dense(3, activation="softmax")(h2)

model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_cls")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_x, train_y, epochs=50, batch_size=50)

loss, acc = model.evaluate(test_x, test_y)

print (f"Loss -> {loss}, Accuracy -> {acc}")

for i in range(0, 10):
    x, y = test_x[i], test_y[i]
    pred = model.predict(np.array([x]))
    print (f"pred -> {pred}, y -> {y}")


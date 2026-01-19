import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.datasets
from keras import layers

dataset = sklearn.datasets.load_diabetes(as_frame=True)

print (f"data -> {dataset['frame']}")
#print (f"x -> {dataset['frame'].iloc[:, :-1]}")
#print (f"y -> {dataset['frame'].iloc[:, -1]}")

train_cnt = int(len(dataset['frame']) * .91)

train_x = tf.constant(dataset['frame'].iloc[:train_cnt, :-1])
train_y = tf.constant(dataset['frame'].iloc[:train_cnt, -1])

#print (f"train x -> {train_x}")
#print (f"train y -> {train_y}")

test_x = tf.constant(dataset['frame'].iloc[train_cnt:, :-1])
test_y = tf.constant(dataset['frame'].iloc[train_cnt:, -1])

#print (f"test x -> {test_x}")
#print (f"test y -> {test_y}")


inputs = layers.Input(shape=(10,))
h1    = layers.Dense(750, activation="relu")(inputs)
h2    = layers.Dense(500, activation="relu")(h1)
outputs = layers.Dense(1)(h2)

model = keras.Model(inputs=inputs, outputs=outputs, name="mlp")

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(train_x, train_y, epochs=100)

loss = model.evaluate(test_x, test_y)

print (f"Loss -> {loss}")

for i in range(0, 10):
    x = np.array([test_x[i]])
    y = test_y[i]

    #print (f"x -> {x}, y -> {y}")
    pred = model.predict(x)

    print (f"{i} -> pred: {pred}, y: {y}")


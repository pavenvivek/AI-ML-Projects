import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.datasets
from keras import layers

cancer_dataset = sklearn.datasets.load_breast_cancer(as_frame=True)

print (f"dataset -> {cancer_dataset['frame']}")

train_cnt = int(.90 * len(cancer_dataset['frame']))

train_x = tf.constant(cancer_dataset['frame'].iloc[:train_cnt, :-1])
train_y = tf.constant(cancer_dataset['frame'].iloc[:train_cnt, -1])

print (f"train_x -> {train_x}, train_y -> {train_y}")

test_x = tf.constant(cancer_dataset['frame'].iloc[train_cnt:, :-1])
test_y = tf.constant(cancer_dataset['frame'].iloc[train_cnt:, -1])

print (f"test_x -> {test_x}, test_y -> {test_y}")

#print (f"num features -> {len(test_x.columns)}")

inputs = layers.Input(shape=(30,))
h1     = layers.Dense(800, activation="relu")(inputs)
#b1     = layers.BatchNormalization()(h1)
h2     = layers.Dense(500, activation="relu")(h1)
#b2     = layers.BatchNormalization()(h2)
outputs = layers.Dense(1, activation="sigmoid")(h2)

model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_cls")

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_x, train_y, epochs=50)

loss, acc = model.evaluate(test_x, test_y)

print (f"Loss -> {loss}, Accuracy -> {acc}")

for i in range(0, 10):
    x, y = test_x[i], test_y[i]
    pred = model.predict(np.array([x]))
    print (f"pred -> {pred}, y -> {y}")



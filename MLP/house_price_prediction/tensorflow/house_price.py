import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from keras import layers

housing_datasets = fetch_california_housing()

print(f"data len -> {len(housing_datasets['data'])}, target len -> {len(housing_datasets['target'])}")

print (f"shape: {housing_datasets['data'].shape}")

train_cnt = int(.80 * len(housing_datasets['data']))
test_cnt  = len(housing_datasets['data']) - train_cnt

print (f"train cnt -> {train_cnt}, test cnt -> {test_cnt}")

train_x = tf.constant(housing_datasets['data'][train_cnt:])
train_y = tf.constant(housing_datasets['target'][train_cnt:])
test_x  = tf.constant(housing_datasets['data'][:train_cnt])
test_y  = tf.constant(housing_datasets['target'][:train_cnt])

inputs  = layers.Input(shape=(8,))
h1      = layers.Dense(500, activation="relu")(inputs)
outputs = layers.Dense(1)(h1)

model = keras.Model(inputs=inputs, outputs=outputs, name="mlp")

model.compile(optimizer="Adam", loss="mean_squared_error") #, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=1, batch_size=10)

loss = model.evaluate(test_x, test_y)

print (f"Loss -> {loss}")

for i in range(0, 10):
    #print (f"input: {test_x[i]}")
    prd = model.predict(np.array([test_x[i]]))
    print (f"{i} : prediction -> {prd}, label -> {test_y[i]}")

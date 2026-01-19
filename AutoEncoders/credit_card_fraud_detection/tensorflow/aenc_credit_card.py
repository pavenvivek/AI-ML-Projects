import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
import kagglehub
import matplotlib.pyplot as plt

from keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# hyperparameters

batch_size = 100
epochs = 10

# data preprocessing

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

df = pd.read_csv(path+'/creditcard.csv') #, header=None)
#print(f"df -> {df}")

df_time = df.loc[:, 'Time']
max_t = df_time.max()
min_t = df_time.min()
df_time = df_time.map(lambda x: (x - min_t)/(max_t - min_t))

df_amt = df.loc[:, 'Amount']
max_a = df_amt.max()
min_a = df_amt.min()
df_amt = df_amt.map(lambda x: (x - min_a)/(max_a - min_a))

df_fea = df.drop(columns=["Time", "Amount", "Class"])
max_f = df_fea.max().max()
min_f = df_fea.min().min()
df_fea = df_fea.map(lambda x: (x - min_f)/(max_f - min_f))

df = pd.concat([df_time, df_amt, df_fea,  df.loc[:, 'Class']], axis=1)
print(f"df -> {df}")

normal_data = df[df[df.columns[-1]] == 0]
abnormal_data = df[df[df.columns[-1]] == 1]

#print (f"len normal_data: {len(normal_data)}")
#print (f"len abnormal_data: {len(abnormal_data)}")

data_x, data_y = normal_data.iloc[:,:-1], normal_data.iloc[:,-1]
anm_data_x, anm_data_y = abnormal_data.iloc[:,:-1], abnormal_data.iloc[:,-1]
print(f"data_x -> {data_x}")
print(f"data_y -> {data_y}")

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, shuffle=True)
#anm_train_x, anm_test_x, anm_train_y, anm_test_y = train_test_split(anm_data_x, anm_data_y, test_size=0.1, shuffle=True)


# model construction

# Functional
def build_Credit_Card_Fraud_Detection():

    # Encoder
    inputs = layers.Input(shape=(30,))
    h1     = layers.Dense(800, activation="relu")(inputs)
    h2     = layers.Dense(500, activation="relu")(h1)
    output = layers.Dense(100, activation="relu")(h1)
    encoder = keras.Model(inputs, output, name="aenc_encoder")

    # Decoder
    inputs = layers.Input(shape=(100,))
    h1     = layers.Dense(500, activation="relu")(inputs)
    h2     = layers.Dense(800, activation="relu")(h1)
    output = layers.Dense(30, activation="sigmoid")(h2)
    decoder = keras.Model(inputs, output, name="aenc_decoder")

    # Model
    inputs = layers.Input(shape=(30,))
    l1     = encoder(inputs)
    output = decoder(l1)
    model = keras.Model(inputs, output, name="aenc")

    return model


# Subclass
class Credit_Card_Fraud_Detection_Encoder(keras.layers.Layer):

    def __init__(self):
        super().__init__()

        self.h1  = layers.Dense(800, activation="relu")
        self.h2  = layers.Dense(500, activation="relu")
        self.out = layers.Dense(100, activation="relu")
        

    def call(self, x):

        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x

class Credit_Card_Fraud_Detection_Decoder(keras.layers.Layer):

    def __init__(self):
        super().__init__()

        self.h1  = layers.Dense(500, activation="relu")
        self.h2  = layers.Dense(800, activation="relu")
        self.out = layers.Dense(30, activation="sigmoid")
        

    def call(self, x):

        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x

class Credit_Card_Fraud_Detection(keras.Model):

    def __init__(self):
        super().__init__()

        self.enc  = Credit_Card_Fraud_Detection_Encoder()
        self.dec  = Credit_Card_Fraud_Detection_Decoder()
        

    def call(self, x):

        x = self.enc(x)
        x = self.dec(x)

        return x

    
#model = build_Credit_Card_Fraud_Detection()
model = Credit_Card_Fraud_Detection()
model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=optimizer,
              loss="mae")


# model training

model.fit(train_x, train_x,
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True)

# model testing

loss = model.evaluate(test_x, test_x)

print (f"Testing Loss: {loss}")

# model prediction

pred = model.predict(train_x)
train_loss = tf.keras.losses.mae(pred, train_x)

print (f"\nTraining Loss values: {train_loss[:50]}")

threshold = np.mean(train_loss) + np.std(train_loss)
print("\nThreshold: ", threshold)

pred = model.predict(anm_data_x)
test_loss = tf.keras.losses.mae(pred, anm_data_x)
print (f"\nTesting Loss values: {test_loss[:50]}")

test_loss = test_loss.numpy()
preds = tf.math.less(test_loss, threshold)
print (f"preds: {preds[:50]}")

correct = 0
count = 0

for p in preds:
    if not p:
        correct = correct + 1
    count = count + 1

print (f"\nAccuracy: {correct/count}")


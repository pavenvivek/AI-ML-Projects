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


df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
#print(f"bf df -> {df}")

max_v = df.iloc[:,:-1].max().max()
min_v = df.iloc[:,:-1].min().min()

print (f"min_val -> {min_v}, max_val -> {max_v}")

# normalize values to be between 0 and 1 (provided using sigmoid for reconstruction)
df = pd.concat([df.iloc[:,:-1].map(lambda x: (x - min_v)/(max_v - min_v)) , df.iloc[:,-1]], axis=1)
normal_data = df[df[df.columns[-1]] == 1]
abnormal_data = df[df[df.columns[-1]] == 0]

data_x, data_y = normal_data.iloc[:,:-1], normal_data.iloc[:,-1]
anm_data_x, anm_data_y = abnormal_data.iloc[:,:-1], abnormal_data.iloc[:,-1]

print(f"data_x -> {data_x}")
print(f"data_y -> {data_y}")

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1, shuffle=True)
anm_train_x, anm_test_x, anm_train_y, anm_test_y = train_test_split(anm_data_x, anm_data_y, test_size=0.1, shuffle=True)


'''
plt.grid()
plt.plot(np.arange(140), train_x[0])
plt.title("A Normal ECG")
plt.show()

plt.grid()
plt.plot(np.arange(140), anm_train_x[0])
plt.title("An Anomalous ECG")
plt.show()

sys.exit(-1)
'''

#sys.exit(-1)


# model construction

# Functional
def build_AutoEncoder_ECG():
    inputs = layers.Input(shape=(140,))
    h1     = layers.Dense(800, activation="relu")(inputs)
    h2     = layers.Dense(500, activation="relu")(h1)
    output = layers.Dense(100, activation="relu")(h2)

    encoder = keras.Model(inputs, output, name="aenc_encoder")

    inputs = layers.Input(shape=(100,))
    h1     = layers.Dense(500, activation="relu")(inputs)
    h2     = layers.Dense(800, activation="relu")(h1)
    output = layers.Dense(140, activation="sigmoid")(h2)

    decoder = keras.Model(inputs, output, name="aenc_decoder")


    inputs = layers.Input(shape=(140,))
    l1     = encoder(inputs)
    output = decoder(l1)

    model = keras.Model(inputs, output, name="aenc")

    return model


# Subclass
class ECG_Encoder(keras.Model):

    def __init__(self):
        super().__init__()

        self.h1  = layers.Dense(800, activation="relu")
        self.h2  = layers.Dense(500, activation="relu")
        self.out = layers.Dense(100, activation="relu")

        #self.rlu = layers.ReLU()

    def call(self, x):

        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x


class ECG_Decoder(keras.Model):

    def __init__(self):
        super().__init__()

        self.h1 = layers.Dense(500, activation="relu")
        self.h2 = layers.Dense(800, activation="relu")
        self.out = layers.Dense(140, activation="sigmoid")

        #self.rlu = nn.ReLU()

    def call(self, x):

        x = self.h1(x)
        x = self.h2(x)
        x = self.out(x)

        return x


class ECG_AutoEncoder(keras.Model):

    def __init__(self):
        super().__init__()

        self.enc = ECG_Encoder()
        self.dec = ECG_Decoder()

    def call(self, x):

        x = self.enc(x)
        x = self.dec(x)

        return x

model = build_AutoEncoder_ECG()
#model = ECG_AutoEncoder()

model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mae")


# model training

model.fit(train_x, train_x,
          epochs=20,
          batch_size=64)

# model testing

loss = model.evaluate(test_x, test_x)

print (f"Testing Loss: {loss}")

# model prediction

pred = model.predict(train_x)
train_loss = tf.keras.losses.mae(pred, train_x)

print (f"\nTraining Loss values: {train_loss[:50]}")

threshold = np.mean(train_loss) + np.std(train_loss)
print("\nThreshold: ", threshold)

pred = model.predict(anm_test_x)
test_loss = tf.keras.losses.mae(pred, anm_test_x)

print (f"\nTesting Loss values: {test_loss[:50]}")

test_loss = test_loss.numpy()
#test_loss[0] = 0.00152
preds = tf.math.less(test_loss, threshold)
#print (f"preds: {preds[:50]}")
#print (f"anm_test_y: {anm_test_y[:50]}")

print (f"\nAccuracy: {accuracy_score(anm_test_y, preds)}")


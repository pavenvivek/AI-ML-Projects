import os, sys
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

#from sklearn.datasets import fetch_california_housing
from keras import layers
import kagglehub


# hyperparameters

batch_size = 100


# data preprocessing

path = kagglehub.dataset_download("olegshpagin/mastercard-stock-price-prediction-dataset")
print("Path to dataset files:", path)

data = tf.keras.utils.get_file("MA.US_D1.csv", origin=f"file://{path}/D1/MA.US_D1.csv")
df = pd.read_csv(data, sep=",")
df = df.loc[:, ['low']] #['open', 'high', 'low', 'close']]
#print (f"df -> {df}")

def get_features(df, timesteps):

    x_lst = []
    y_lst = []
    i = 0

    while i+timesteps < len(df):

        #x, y = df.iloc[i:i+timesteps, :-1], df.iloc[i+timesteps-1, -1]
        x, y = df.iloc[i:i+timesteps], df.iloc[i+timesteps]
        
        x = np.array(x)
        x_lst.append(x)

        y = np.array(y)
        y_lst.append(y)

        i = i + 1

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
#print (f"x -> {x[0]},\ny -> {y[0]}")


train_cnt = int(.90 * len(df))
train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]
print (f"train: x -> {test_x[0]},\ny -> {test_y[0]}")

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))


def preprocess(data, label):

    #data = tf.cast(data, dtype=tf.float32)

    return (data, label)

train_data = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=len(train_x)).batch(batch_size).prefetch(tf.data.AUTOTUNE) # 
test_data  = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)



# model construction

# Functional
def build_Rnn_Stock_Price():
    inputs = layers.Input(shape=(timesteps, 1))
    rnn    = layers.SimpleRNN(100)(inputs)
    #rnn    = layers.GRU(100)(inputs)
    h1     = layers.Dense(500, activation="relu")(rnn)
    outputs = layers.Dense(1)(h1)
    model = keras.Model(inputs=inputs, outputs=outputs, name="rnn")

    return model

# Subclass
class Rnn_Stock_Price_Prediction(keras.Model):

    def __init__(self):
        super().__init__()

        #self.rnn  = layers.SimpleRNN(100)
        self.rnn = layers.GRU(100)
        self.h1   = layers.Dense(500, activation="relu")
        self.out  = layers.Dense(1)
        

    def call(self, x):

        x = self.rnn(x)
        x = self.h1(x)
        x = self.out(x)

        return x


#model = build_Rnn_Stock_Price()
model = Rnn_Stock_Price_Prediction()
model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mean_squared_error")

# model training

print ("\nTraining: ")
model.fit(train_data, epochs=40)


# model testing

print ("\nTesting: ")
loss = model.evaluate(test_data)

print (f"Loss: {loss}")


# model prediction

print ("\nSamples Prediction: ")
for i in range(0, 50):

    #print (f"test_x[i] -> {test_x[i]}")
    
    pred = model.predict(np.array([test_x[i]]))

    print (f"pred: {pred}, y: {test_y[i]}")

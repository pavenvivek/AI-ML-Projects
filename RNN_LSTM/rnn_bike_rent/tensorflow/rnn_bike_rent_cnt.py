import os
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers
import kagglehub


# hyperparameters

batch_size = 100


# data preprocessing

path = kagglehub.dataset_download("marklvl/bike-sharing-dataset")
print("Path to dataset files:", path)

data = tf.keras.utils.get_file("hour.csv", origin=f"file://{path}/hour.csv")
df = pd.read_csv(data)
print (f"df -> {df[df['dteday'] == '2011-01-01']}")


def get_features(df, timesteps):

    x_lst = []
    y_lst = []
    i = 0

    while i+timesteps < len(df):

        #print (f"val -> {train_x.loc[i:i+23, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]}")
        #print (f"val -> {np.array(train_x.loc[i:i+23, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']])}")

        x, y = df.iloc[i:i+timesteps, :-1], df.iloc[i+timesteps-1, -1]
        
        x = np.array(x.loc[:, ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']])
        #x = np.pad(x, pad_width=((0,timesteps-len(x)), (0,0)))
        x_lst.append(x)

        y = np.array(y)
        #y = np.pad(y, pad_width=(0,timesteps-len(y)))
        y_lst.append(y)

        i = i + 1

    return np.array(x_lst), np.array(y_lst)


timesteps = 10
x, y = get_features(df,timesteps)
#test_x, test_y = get_features(test_x, test_y)
print (f"x -> {x[0]},\ny -> {y[0]}")

train_cnt = int(.90 * len(df))
train_x, train_y = x[:train_cnt], y[:train_cnt]
test_x, test_y = x[train_cnt:], y[train_cnt:]

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))


def preprocess(data, label):

    #data = tf.cast(data, dtype=tf.float32)

    return (data, label)

train_data = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=len(train_x)).batch(batch_size).prefetch(tf.data.AUTOTUNE) # 
test_data  = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

                                 
# model construction


# Functional
def build_Rnn_Bike_Rent_Prediction():
    inputs  = layers.Input(shape=(timesteps, 13))
    rnn1    = layers.SimpleRNN(10)(inputs)
    #rnn1    = layers.LSTM(10)(inputs)
    #rnn1    = layers.GRU(10, return_sequences=True)(inputs)
    #rnn2    = layers.GRU(10)(rnn1)
    h1      = layers.Dense(200, activation="relu")(rnn1)
    outputs = layers.Dense(1)(h1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RNN")

    return model


# Subclass
class Rnn_Bike_Rent_Prediction(keras.Model):

    def __init__(self):
        super().__init__()

        self.rnn = layers.SimpleRNN(10)
        #self.rnn = layers.GRU(10)
        #self.rnn = layers.LSTM(10)
        self.h1  = layers.Dense(200)
        self.out = layers.Dense(1)

        self.rlu = layers.ReLU()

        
    def call(self, x):

        x  = self.rnn(x)
        x  = self.h1(x)
        x  = self.out(x)

        return x

    
model = build_Rnn_Bike_Rent_Prediction()
#model = Rnn_Bike_Rent_Prediction()
model.summary()

# model settings

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss="mean_squared_error")

# model training

print ("\nTraining: ")
model.fit(train_data, epochs=20, shuffle=True)


# model testing

print ("\nTesting: ")
loss = model.evaluate(test_data)

print (f"Loss: {loss}")


# model prediction

print ("\nSamples Prediction: ")
for i in range(0, 10):

    #print (f"test_x[i] -> {test_x[i]}")    
    pred = model.predict(np.array([test_x[i]]))
    print (f"pred: {pred}, y: {test_y[i]}")

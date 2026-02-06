import os, sys
import keras
import keras_hub
import numpy as np

import shutil
from keras import ops
import pathlib
import tensorflow as tf

import keras, keras_hub
from keras import layers


# Parameters

# Preprocessing params.
BATCH_SIZE = 32
SEQ_LENGTH = 128
MASK_RATE = 0.10 #75
PREDICTIONS_PER_SEQ = 32

# Model params.
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5

# Training params.
LEARNING_RATE = 5e-5
EPOCHS = 3


# Data download

path = keras.utils.get_file(
    origin="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    extract=True,
)

print (f"download path for sst: {path}")

sst_dir = os.path.expanduser("~/.keras/datasets/SST-2/")

# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

print (f"download path for vocab: {vocab_file}")

# Load SST-2.
sst_train_ds = tf.data.experimental.CsvDataset(
    sst_dir + "train.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
) #.batch(BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
) #.batch(BATCH_SIZE)


# Data Preprocessing

tokenizer = keras_hub.models.BertTokenizer.from_preset("bert_base_en_uncased",)
preprocessor = keras_hub.models.BertPreprocessor(tokenizer, sequence_length=SEQ_LENGTH) #.from_preset("bert_base_en_uncased")


def preprocess(sentences, labels):

    inputs = preprocessor(sentences)
    inputs = {"token_ids" : inputs["token_ids"],
              "segment_ids" : inputs["segment_ids"],
              "padding_mask" : inputs["padding_mask"]}

    return inputs["token_ids"], labels


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
train_data = (
    sst_train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
vald_data = (
    sst_val_ds.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


class LLM_Text_Classifier(keras.Model):

    def __init__(self, model_llm_text):
        super().__init__()

        self.llm_text = model_llm_text
        self.out      = layers.Dense(1, activation="sigmoid")

        
    def call(self, x):

        x = self.llm_text(x)
        x = self.out(x[:,0,:])

        return x
    

# Load from local pre-trained model - Run masked_llm.py before this to build encoder_model.keras

model_llm_text = keras.models.load_model("./encoder_model.keras") #, compile=True)
model = LLM_Text_Classifier(model_llm_text)


# model training and evalution

model.compile(
    optimizer=keras.optimizers.AdamW(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=vald_data,
    epochs=EPOCHS)

loss, acc = model.evaluate(vald_data)

print (f"Testing loss: {loss}, accuracy: {acc}")


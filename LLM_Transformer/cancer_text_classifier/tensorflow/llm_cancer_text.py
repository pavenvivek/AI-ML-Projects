import os,sys
import keras
import keras_hub
import numpy as np
#import matplotlib.pyplot as plt

import shutil
from keras import ops
import pathlib
import tensorflow as tf

import keras, keras_hub
from keras import layers
import kagglehub
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Data download

# Download pretraining data.
'''
path = keras.utils.get_file(
    origin="https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz",  #"https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    extract=True,
)

print (f"download path for wiki: {path}")
'''

# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)


path = kagglehub.dataset_download("falgunipatel19/biomedical-text-publication-classification")
print("Path to dataset files:", path)


df = pd.read_csv(path+"/alldata_1_for_kaggle.csv", encoding='MacRoman', index_col=0)
le = LabelEncoder()
df['0'] = le.fit_transform(df['0'])

print (f"df -> {df}")
print (f"df -> {df.columns}")

train_x, train_y = df.iloc[:,-1], df.iloc[:,0]

print (f"train_x -> {train_x}")
print (f"train_y -> {train_y}")

train_data_ft, test_data_ft = keras.utils.split_dataset((np.array(train_x), np.array(train_y)), shuffle=True, left_size=0.9)
print (f"train_data cnt -> {int(train_data_ft.cardinality())}")
print (f"test_data cnt -> {int(test_data_ft.cardinality())}")

#print (f"dataset -> {next(iter(train_data.take(1)))}")


#sys.exit(-1)


# Parameters

# Preprocessing params.
NUM_CLASSES = 3
PRETRAINING_BATCH_SIZE = 64 #128
FINETUNING_BATCH_SIZE = 32
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
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 2 #8
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3


# Data Preprocessing

#train_data = tf.data.TextLineDataset(wiki_dir+"wiki.train.tokens").filter(lambda x: tf.strings.length(x) > 100).batch(PRETRAINING_BATCH_SIZE)
#validation_data = tf.data.TextLineDataset(wiki_dir+"wiki.valid.tokens").filter(lambda x: tf.strings.length(x) > 100).batch(PRETRAINING_BATCH_SIZE)

#print(f"training_dataset: {len(train_data)}")
#num_elements = train_data.reduce(0, lambda x, _: x + 1).numpy()
#print(f"Number of elements: {num_elements}")

#print(f"training_dataset: {next(iter(train_data.take(1)))}")
#print(f"training_dataset: {train_data.take(1).get_single_element()}")

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True
)

masker = keras_hub.layers.MaskedLMMaskGenerator(
    vocabulary_size=tokenizer.vocabulary_size(),
    mask_selection_rate=0.10,
    mask_token_id=tokenizer.token_to_id("[MASK]"),
    mask_selection_length=PREDICTIONS_PER_SEQ
)

'''
def preprocess(inputs):
    out = masker(tokenizer(inputs))

    x = {"token_ids" : out["token_ids"], "mask_positions" : out["mask_positions"]}
    y = out["mask_ids"]
    w = out["mask_weights"]
    
    return x, y, w

train_data = (
    train_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) #.batch(PRETRAINING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

vald_data = (
    validation_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) #.batch(PRETRAINING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
'''
#print(f"training_dataset: {train_data.take(1).get_single_element()}")



# model construction


inputs = layers.Input(shape=(SEQ_LENGTH,), dtype="int32")
x = keras_hub.layers.TokenAndPositionEmbedding(
    vocabulary_size=tokenizer.vocabulary_size(),
    sequence_length=SEQ_LENGTH,
    embedding_dim=MODEL_DIM
)(inputs)

x = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(x)
x = keras.layers.Dropout(rate=DROPOUT)(x)


for _ in range(NUM_LAYERS):
    x = keras_hub.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            layer_norm_epsilon=NORM_EPSILON
        )(x)

encoder_model = keras.Model(inputs, x, name="encoder")
encoder_model.summary()

inputs = {
    "token_ids" : layers.Input(shape=(SEQ_LENGTH,), dtype="int32"),
    "mask_positions"  : layers.Input(shape=(PREDICTIONS_PER_SEQ,), dtype="int32")
}

enc_out = encoder_model(inputs["token_ids"])

outputs = keras_hub.layers.MaskedLMHead(
    vocabulary_size=tokenizer.vocabulary_size(),
    activation="softmax"
)(enc_out, mask_positions=inputs["mask_positions"])

model = keras.Model(inputs, outputs, name="pretrain-model")

model.summary()

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=PRETRAINING_LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    weighted_metrics=["accuracy"], #sparse_categorical_accuracy"]
    #jit_compile=True
)

#model.fit(
#    train_data,
#    validation_data=vald_data,
#    epochs=4
#)

#encoder_model.save("encoder_model.keras")



# Fine Tuning


def preprocess(sentences, labels):
    return tokenizer(sentences), labels


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.

train_data = (
    train_data_ft
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_data = (
    test_data_ft
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Preview a single input example.
#print(f"data -> {vald_data.take(1).get_single_element()}")


#encoder_model = keras.models.load_model("encoder_model.keras", compile=False)

#print (f"test -> {encoder_model(vald_data.take(1).get_single_element()[0]).shape}, {encoder_model(vald_data.take(1).get_single_element()[0])[:,0,:].shape}")


inputs  = layers.Input(shape=(SEQ_LENGTH,), dtype="int32")
enc_out = encoder_model(inputs)
#glob    = layers.GlobalAveragePooling1D()(enc_out) #[0])
output  = layers.Dense(NUM_CLASSES, activation="softmax")(enc_out[:,0,:])  #(glob)

model = keras.Model(inputs, output, name="finetune")
model.summary()

model.compile(
    optimizer=keras.optimizers.AdamW(FINETUNING_LEARNING_RATE),
    loss="sparse_categorical_crossentropy", #"binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    #validation_data=vald_data,
    epochs=10)

loss, acc = model.evaluate(test_data)

print (f"Testing loss: {loss}, accuracy: {acc}")


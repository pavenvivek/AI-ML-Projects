import os, sys
import keras
import keras_hub
import numpy as np

from keras import ops
import pathlib
import tensorflow as tf

import keras, keras_hub
from keras import layers


# Parameters

# Preprocessing params.
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
FINETUNING_LEARNING_RATE = 5e-5
FINETUNING_EPOCHS = 3


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
) #.batch(FINETUNING_BATCH_SIZE)
sst_val_ds = tf.data.experimental.CsvDataset(
    sst_dir + "dev.tsv", [tf.string, tf.int32], header=True, field_delim="\t"
) #.batch(FINETUNING_BATCH_SIZE)


# Preview a single input example.
#sample = sst_train_ds.take(1).get_single_element()
#print(f"sample data -> {sst_train_ds[:4]}")
#sys.exit(-1)


# Data Preprocessing

#tokenizer =  keras_hub.models.BertTokenizer(vocabulary=vocab_file, lowercase=True)
tokenizer = keras_hub.models.BertTokenizer.from_preset("bert_base_en_uncased",)
preprocessor = keras_hub.models.BertPreprocessor(tokenizer, sequence_length=SEQ_LENGTH) #.from_preset("bert_base_en_uncased")

def preprocess(sentences, labels):

    inputs = preprocessor(sentences)
    inputs = {"token_ids" : inputs["token_ids"],
              "segment_ids" : inputs["segment_ids"],
              "padding_mask" : inputs["padding_mask"]}

    return inputs, labels


#i = 0
#for s in sst_train_ds:
#    print (f"s -> {s}")
#    print (f"preprocess -> {preprocess(s[0], s[1])}\n")
#    if i == 10:
#        break
#    i = i + 1

#sys.exit(-1)


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
train_data = (
    sst_train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
vald_data = (
    sst_val_ds.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# model construction

# from keras backbone pre-built and pre-trained

inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "segment_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="segment_ids"),
    "padding_mask": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="padding_mask"),
}


# load backbone using one of the following APIs

## Step 1

#--

# Pretrained BERT encoder. This uses pre-trained weights. (Set load_weights=False to load random weights)
# below setting num_layers will work but setting hidden_dim will throw error because the model is pretrained with fixed dimension
# which cannot be changed. However, the some pretrained layers can be dropped.
#backbone = keras_hub.models.Backbone.from_preset("bert_base_en_uncased", num_layers=5) #, hidden_dim=256) 

#--

# or

#--

# Randomly initialized BERT encoder with a custom config. This uses only the bert model construction and not the pre-trained weights.

backbone = keras_hub.models.BertBackbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_dim=MODEL_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    max_sequence_length=SEQ_LENGTH,
)

#--

## Step 2

#-- 

# attach a classification head (dense layer)

x = backbone(inputs)

# either use x['pooled_output'] or x['sequence_output'][:,0,:] - both are same
output  = layers.Dense(1, activation="sigmoid")(x['pooled_output'])
#output  = layers.Dense(1, activation="sigmoid")(x['sequence_output'][:,0,:])

model = keras.Model(inputs, output, name="finetune")

#--

# or

#--
'''
# the following attaches a classification head (a dense layer with output num_classes and given activation)
model = keras_hub.models.BertClassifier(
    backbone=backbone,
    activation="sigmoid",
    num_classes=1
)


# the following uses a pretrained model and attaches classification head to it. num_layers has no effect here.
model = keras_hub.models.BertTextClassifier.from_preset(
    "bert_tiny_en_uncased", #"bert_base_en_uncased",
    activation="sigmoid",
    num_classes=1,
    preprocessor=None, # need this explicit here
)
'''
#--

model.summary()


'''
dummy_input = train_data.take(1).get_single_element()[0]
print(f"dummy input data -> {dummy_input}")
sample_output = model(dummy_input)

model_llm_text.summary()
model.summary()
'''

'''
print(f"sample output data -> {sample_output}")
print(f"sequence_output data shape -> {sample_output['sequence_output'].shape}")
print(f"sequence_output data -> {sample_output['sequence_output']}")
print(f"sequence_output data (last token) shape -> {sample_output['sequence_output'][:,0,:].shape}")
print(f"sequence_output data (last token) -> {sample_output['sequence_output'][:,0,:]}")
print(f"pooled_output data -> {sample_output['pooled_output']}")
print ("=============")
'''
#sys.exit(-1)

# model training and evalution

model.compile(
    optimizer=keras.optimizers.AdamW(FINETUNING_LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    #validation_data=vald_data,
    epochs=FINETUNING_EPOCHS)

loss, acc = model.evaluate(vald_data)

print (f"Testing loss: {loss}, accuracy: {acc}")


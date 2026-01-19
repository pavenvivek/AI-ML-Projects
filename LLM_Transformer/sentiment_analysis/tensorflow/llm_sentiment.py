import os, sys
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


# Parameters

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 1 #64 #128
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



# Data Preprocessing
'''
tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
    #dtype='string'
)
'''
#tokens = tokenizer.tokenize(sample_data)
#print (f"tokens -> {tokens}")


#print ("sample_data: ")
#sample_data = next(iter(sst_train_ds.take(1)))
#print(sample_data)
#print ("=============")

#tokenizer =  keras_hub.models.BertTokenizer(vocabulary=vocab_file, lowercase=True)
tokenizer = keras_hub.models.BertTokenizer.from_preset("bert_base_en_uncased",)
preprocessor = keras_hub.models.BertPreprocessor(tokenizer, sequence_length=SEQ_LENGTH) #.from_preset("bert_base_en_uncased")


def preprocess(sentences, labels):

    inputs = preprocessor(sentences)
    inputs = {"token_ids" : inputs["token_ids"],
              "segment_ids" : inputs["segment_ids"],
              "padding_mask" : inputs["padding_mask"]}

    return inputs["token_ids"], labels
    #return inputs, labels


#def preprocess(sentences, labels):
    #return tokenizer(sentences), labels
    

# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
train_data = (
    sst_train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
vald_data = (
    sst_val_ds.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(FINETUNING_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Preview a single input example.
#sample = train_data.take(1).get_single_element()
#print(f"sample data -> {sample}")
#print ("=============")
#print (f"test -> {encoder_model(vald_data.take(1).get_single_element()[0]).shape}, {encoder_model(vald_data.take(1).get_single_element()[0])[:,0,:].shape}")
#sys.exit(-1)


# model construction

# from scratch

# Functional
def build_LLM_Sentiment():

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

    # load from pretrained model
    #encoder_model = keras.models.load_model("./encoder_model.keras") #, compile=True)
    #encoder_model.summary()

    enc_out = encoder_model(inputs)
    output  = layers.Dense(1, activation="sigmoid")(enc_out[:,0,:])
    model = keras.Model(inputs, output, name="finetune")

    return model


# Subclass
class TokenAndPositionEmbedding(keras.layers.Layer):

    def __init__(self):
        super().__init__()

        self.tok_and_pos = keras_hub.layers.TokenAndPositionEmbedding(
                               vocabulary_size=tokenizer.vocabulary_size(),
                               sequence_length=SEQ_LENGTH,
                               embedding_dim=MODEL_DIM
                           )

    def call(self, x):

        x = self.tok_and_pos(x)
        return x
        

class LLM_Text(keras.Model):

    def __init__(self):
        super().__init__()

        self.tok_and_pos = TokenAndPositionEmbedding()

        self.ln  = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.drp = keras.layers.Dropout(rate=DROPOUT)

        self.trs_enc = [keras_hub.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=DROPOUT, layer_norm_epsilon=NORM_EPSILON) for _ in range(NUM_LAYERS)]

        
    def call(self, x):

        x = self.tok_and_pos(x)
        x = self.drp(self.ln(x))

        for i in range(NUM_LAYERS):
            x = self.trs_enc[i](x)

        return x


class LLM_Text_Classifier(keras.Model):

    def __init__(self, model_llm_text):
        super().__init__()

        self.llm_text = model_llm_text
        self.out      = layers.Dense(1, activation="sigmoid")

        
    def call(self, x):

        x = self.llm_text(x)
        x = self.out(x[:,0,:])

        return x

    
    
# load from pretrained model
#encoder_model = keras.models.load_model("./encoder_model.keras") #, compile=True)

#encoder_model.summary()

#enc_out = encoder_model(inputs)
#output  = layers.Dense(1, activation="sigmoid")(enc_out[:,0,:])

#output  = layers.Dense(1, activation="sigmoid")(x[:,0,:])


'''
# from keras backbone pre-built and pre-trained

inputs = {
    "token_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="token_ids"),
    "segment_ids": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="segment_ids"),
    "padding_mask": keras.Input(shape=(SEQ_LENGTH,), dtype="int32", name="padding_mask"),
}

#backbone = keras_hub.models.Backbone.from_preset("bert_base_en_uncased", num_layers=3) #, hidden_dim=256)

backbone = keras_hub.models.BertBackbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_dim=MODEL_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    max_sequence_length=SEQ_LENGTH,
)

x = backbone(inputs)
#output  = layers.Dense(1, activation="sigmoid")(x['pooled_output'])
output  = layers.Dense(1, activation="sigmoid")(x['sequence_output'][:,0,:])
'''
#inputs = layers.Input(shape=(SEQ_LENGTH,), dtype="int32")

# Load from local pre-trained model
model_llm_text = LLM_Text()
#model_llm_text = keras.models.load_model("./encoder_model.keras") #, compile=True)

#enc_out = model_llm_text(inputs)
#output  = layers.Dense(1, activation="sigmoid")(enc_out[:,0,:])
#model = keras.Model(inputs, output, name="finetune")

model = LLM_Text_Classifier(model_llm_text)  #keras.Model(inputs, output, name="finetune")
#model = build_LLM_Sentiment()

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
    epochs=3)

loss, acc = model.evaluate(vald_data)

print (f"Testing loss: {loss}, accuracy: {acc}")


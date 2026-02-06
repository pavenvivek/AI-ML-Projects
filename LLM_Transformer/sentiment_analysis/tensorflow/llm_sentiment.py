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
BATCH_SIZE = 32
SEQ_LENGTH = 128

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
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
vald_data = (
    sst_val_ds.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Preview a single input example.
#sample = train_data.take(1).get_single_element()
#print(f"sample data -> {sample}")
#print ("=============")
#print (f"test -> {encoder_model(vald_data.take(1).get_single_element()[0]).shape}, {encoder_model(vald_data.take(1).get_single_element()[0])[:,0,:].shape}")
#sys.exit(-1)


# model construction

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

# from scratch
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, seq_len, embed_dim):

        super().__init__()

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb   = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

    def call(self, x):

        seq_len = x.shape[-1] #ops.shape(x)[-1]
        pos     = ops.arange(start=0, stop=seq_len, step=1)
        pos_emb = self.pos_emb(pos)
        tok_emb = self.token_emb(x)
        out     = tok_emb + pos_emb

        return out


# building multiheadattention from scratch 
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):

        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):

        matmul_qk     = tf.matmul(q, k, transpose_b=True)
        dk            = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        output            = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q, mask=None):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output


class TransformerBlock_Sc(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):

        super().__init__()

        self.att  = MultiHeadAttention(d_model, num_heads)
        self.ffn  = keras.Sequential([layers.Dense(dff, activation="relu"), layers.Dense(d_model),])
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)

        self.drp1 = layers.Dropout(dropout_rate)
        self.drp2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):

        attn_out = self.att(x, x, x, mask=mask)
        attn_out = self.drp1(attn_out, training=training)
        out1     = self.ln1(x + attn_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.drp2(ffn_out, training=training)
        out2    = self.ln2(out1 + ffn_out)

        return out2


# using keras implenmentaion of multiheadattention
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super().__init__()

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.drp1 = layers.Dropout(rate)
        self.drp2 = layers.Dropout(rate)

    def call(self, x):

        att_out = self.att(x, x)
        att_out = self.drp1(att_out)
        out1    = self.ln1(x + att_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.drp2(ffn_out)
        out2    = self.ln2(out1 + ffn_out)

        return out2

    
class LLM_Text(keras.Model):

    def __init__(self):
        super().__init__()

        # from library
        #self.tok_and_pos = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=tokenizer.vocabulary_size(), sequence_length=SEQ_LENGTH, embedding_dim=MODEL_DIM)

        # from local
        self.tok_and_pos = TokenAndPositionEmbedding(tokenizer.vocabulary_size(), SEQ_LENGTH, MODEL_DIM)
        
        self.ln  = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.drp = keras.layers.Dropout(rate=DROPOUT)

        # from library
        #self.trs_enc = [keras_hub.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=DROPOUT, layer_norm_epsilon=NORM_EPSILON) for _ in range(NUM_LAYERS)]

        # from local
        #self.trs_enc = [TransformerBlock(MODEL_DIM, NUM_HEADS, INTERMEDIATE_DIM) for _ in range(NUM_LAYERS)]

        # from local scratch
        self.trs_enc = [TransformerBlock_Sc(MODEL_DIM, NUM_HEADS, INTERMEDIATE_DIM) for _ in range(NUM_LAYERS)]

        
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
    

model_llm_text = LLM_Text()
model = LLM_Text_Classifier(model_llm_text)
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


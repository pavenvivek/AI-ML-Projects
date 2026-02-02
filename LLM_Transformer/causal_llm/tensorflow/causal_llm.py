import os
import keras
import keras_hub
import numpy as np

from keras import ops, layers
import pathlib
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings


# hyper-parameters

BATCH_SIZE = 64
MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens

EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
VOCAB_SIZE = 5000  # Limits parameters in model.
EPOCHS = 5


# data download and preprocessing

keras.utils.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)
dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# Load simplebooks-92 train set and filter out short lines.
raw_train_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# Load simplebooks-92 validation set and filter out short lines.
raw_val_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)


def train_word_piece(raw_train_ds, filename):

    file_path = pathlib.Path(filename)

    if not file_path.exists():
        print ("File doesn't exists !")

        keras_hub.tokenizers.compute_word_piece_vocabulary(
            raw_train_ds,
            vocabulary_size=VOCAB_SIZE,
            lowercase=True,
            reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
            vocabulary_output_file=filename,
        )

    abs_path = os.path.abspath(file_path)
    vocab = keras.utils.get_file(origin="file://" + abs_path)

    '''
    vocab_lst = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip whitespace and newline characters, then append the word
            word = line.strip()
            if word: # Ensure the line is not empty
                vocab_lst.append(word)

    print (f"Tokens: {vocab_lst[100:110]}")
    '''
    
    return vocab


vocab = train_word_piece(raw_train_ds, "vocab_causal.txt")

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

start_packer = keras_hub.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

def preprocess(inputs):

    outputs  = tokenizer(inputs)
    features = start_packer(outputs)
    labels   = outputs

    return features, labels

train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
val_ds   = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)


# model construction

# Functional
def build_causal_llm():
    inputs = keras.layers.Input(shape=(None,), dtype="int32")

    embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=VOCAB_SIZE,
        sequence_length=SEQ_LEN,
        embedding_dim=EMBED_DIM,
        mask_zero=True,
    )
    x = embedding_layer(inputs)

    for _ in range(NUM_LAYERS):
        decoder_layer = keras_hub.layers.TransformerDecoder(
            num_heads=NUM_HEADS,
            intermediate_dim=FEED_FORWARD_DIM,
        )
        x = decoder_layer(x)  # Giving one argument only skips cross-attention.

    # Output.
    outputs = keras.layers.Dense(VOCAB_SIZE)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

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
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        
        super().__init__()

        self.att  = MultiHeadAttention(embed_dim, num_heads)
        self.ffn  = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        
        self.drp1 = layers.Dropout(dropout_rate)
        self.drp2 = layers.Dropout(dropout_rate)

        self.causal_mask = 1 - tf.linalg.band_part(tf.ones((SEQ_LEN, SEQ_LEN)), -1, 0)


    # Using post-LayerNorm as in the original paper.
    def call(self, x, training=False, mask=None):

        att_out = self.att(x, x, x, mask=self.causal_mask)
        att_out = self.drp1(att_out, training=training)
        out1    = self.ln1(x + att_out)

        ffn_out = self.ffn(out1)
        ffn_out = self.drp2(ffn_out, training=training)
        out2    = self.ln2(out1 + ffn_out)

        return out2


# using keras multiheadattention implementation
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):

        super().__init__()

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.drp1 = layers.Dropout(dropout_rate)
        self.drp2 = layers.Dropout(dropout_rate)

   # Using pre-layerNorm. keras.MultiHeadAttention is inconsistent with post-layerNorm for transformerDecoder.
    def call(self, x):

        x1      = self.ln1(x)
        att_out = self.att(x1, x1, use_causal_mask=True)
        att_out = self.drp1(att_out)
        out1    = x + att_out

        x2      = self.ln2(out1)
        ffn_out = self.ffn(x2)
        ffn_out = self.drp2(ffn_out)
        out2    = x2 + ffn_out

        return out2
    

@keras.saving.register_keras_serializable()
class Causal_LLM(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # from library
        #self.tok_and_pos = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=SEQ_LEN, embedding_dim=EMBED_DIM, mask_zero=True)
        #self.trs_dec     = [keras_hub.layers.TransformerDecoder(intermediate_dim=FEED_FORWARD_DIM, num_heads=NUM_HEADS) for _ in range(NUM_LAYERS)]

        # from local
        #self.tok_and_pos = TokenAndPositionEmbedding(VOCAB_SIZE, SEQ_LEN, EMBED_DIM)
        #self.trs_dec     = [TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM) for _ in range(NUM_LAYERS)]

        # from local scratch
        self.tok_and_pos  = TokenAndPositionEmbedding(VOCAB_SIZE, SEQ_LEN, EMBED_DIM)
        self.trs_dec      = [TransformerBlock_Sc(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM) for _ in range(NUM_LAYERS)]
        
        self.out =  keras.layers.Dense(VOCAB_SIZE)

        
    def call(self, x):
        
        x = self.tok_and_pos(x)

        for i in range(NUM_LAYERS):
            x = self.trs_dec[i](x)

        out = self.out(x)

        return out


#model = build_causal_llm()
model = Causal_LLM()

model.summary()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_hub.metrics.Perplexity(from_logits=True, mask_token_id=0)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"]) #perplexity])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

#model.save("causal_llm.keras")
#model = keras.models.load_model("causal_llm.keras", compile=False)


# Evaluation

prompt_tokens = start_packer(tokenizer([""]))
print (f"prompt -> {prompt_tokens}")

def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache


sampler = keras_hub.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # Start sampling immediately after the [BOS] token.
)

txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")
#print (f"len -> {len(txt[0].split())}")

sampler = keras_hub.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)

txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")
#print (f"len -> {len(txt[0].split())}")

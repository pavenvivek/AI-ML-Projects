import os
import keras
import keras_hub
import numpy as np

from keras import ops
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
NUM_HEADS = 3
NUM_LAYERS = 3
VOCAB_SIZE = 5000  # Limits parameters in model.
EPOCHS = 20


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

@keras.saving.register_keras_serializable()
class LLM_Seq_to_Seq(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tok_and_pos = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=SEQ_LEN, embedding_dim=EMBED_DIM, mask_zero=True)
        self.trs_dec     = [keras_hub.layers.TransformerDecoder(intermediate_dim=FEED_FORWARD_DIM, num_heads=NUM_HEADS) for _ in range(NUM_LAYERS)]

        self.out =  keras.layers.Dense(VOCAB_SIZE)

        
    def call(self, x):
        
        x = self.tok_and_pos(x)

        for i in range(NUM_LAYERS):
            x = self.trs_dec[i](x)

        out = self.out(x)

        return out


#model = build_causal_llm()
model = LLM_Seq_to_Seq()

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

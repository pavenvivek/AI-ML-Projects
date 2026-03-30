import os, sys

import keras
import keras_hub
from keras import layers
import tensorflow as tf

import pandas as pd
import numpy as np
import glob
from pprint import pprint


# Model parameters

SEQ_LENGTH = 128 #256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

NUM_LAYERS = 3
MODEL_DIM = 128 #256
INTERMEDIATE_DIM = 128 #512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5


# data preprocessing

vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
    special_tokens=["[MASK]"],
    special_tokens_in_strings=True
)

def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name) as f:
            for line in f:
                text_list.append(line)
    return text_list


def get_data_from_text_files(folder_name):
    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df


def get_masked_input_and_labels(encoded_texts):

    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15

    # Do not mask special tokens (0 for padding)
    inp_mask[encoded_texts <= 2] = False

    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[inp_mask_2mask] = (
        mask_token_id  # mask token is the last in the dict
    )

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 0.10)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


# Get mask token id for masked language model
mask_token_id = tokenizer(["[MASK]"]).numpy()[0][0]
VOCAB_SIZE = tokenizer.vocabulary_size()

def get_data():

    train_df = get_data_from_text_files("train")
    test_df = get_data_from_text_files("test")

    all_data = pd.concat([train_df, test_df], ignore_index=True)
    print (all_data)

    # Prepare data for masked language model
    x_all_review = tokenizer(all_data.review.values).numpy()
    x_masked_train, y_masked_labels, sample_weights = get_masked_input_and_labels(
        x_all_review
    )

    mlm_ds = tf.data.Dataset.from_tensor_slices(
        (x_masked_train, y_masked_labels, sample_weights)
    )
    mlm_ds = mlm_ds.shuffle(1000).batch(BATCH_SIZE)

    return mlm_ds



# model construction

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


@keras.saving.register_keras_serializable()    
class BertBackbone(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tok_and_pos = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=tokenizer.vocabulary_size(), sequence_length=SEQ_LENGTH, embedding_dim=MODEL_DIM)

        self.ln  = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
        self.drp = keras.layers.Dropout(rate=DROPOUT)

        # from library
        #self.trs_enc  = [keras_hub.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout=DROPOUT, layer_norm_epsilon=NORM_EPSILON) for _ in range(NUM_LAYERS)]

        # from local
        self.trs_enc = [TransformerBlock(MODEL_DIM, NUM_HEADS, INTERMEDIATE_DIM) for _ in range(NUM_LAYERS)]

        
    def call(self, x):

        x = self.tok_and_pos(x)
        x = self.drp(self.ln(x))

        for i in range(NUM_LAYERS):
            x = self.trs_enc[i](x)

        return x


class MaskedLanguageModel(keras.Model):

    def __init__(self, bert_backbone):
        super().__init__()

        self.bert_backbone = bert_backbone
        self.mlm_head = layers.Dense(VOCAB_SIZE, activation="softmax")
        
    def call(self, x):

        x = self.bert_backbone(x)
        x = self.mlm_head(x)
            
        return x
    

# model training and evaluation    

id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


class MaskedTextGenerator(keras.callbacks.Callback):
    
    def __init__(self, sample_tokens, top_k=5):

        self.sample_tokens = sample_tokens
        self.k = top_k

    def decode(self, tokens):

        return " ".join([id2token[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):

        return id2token[id]

    def on_epoch_end(self, epoch, logs=None):

        prediction = self.model.predict(self.sample_tokens)

        masked_index = np.where(self.sample_tokens == mask_token_id)
        masked_index = masked_index[1]
        mask_prediction = prediction[0][masked_index]

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1]
        values = mask_prediction[0][top_indices]
        
        for i in range(len(top_indices)):
            p = top_indices[i]
            v = values[i]
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(self.sample_tokens[0]),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)



if __name__ == "__main__":
    
    sample_tokens = tokenizer(["I have watched this [MASK] and it was awesome"])
    generator_callback = MaskedTextGenerator(sample_tokens.numpy())

    bert_backbone = BertBackbone()

    mlm_model = MaskedLanguageModel(bert_backbone)
    mlm_model.predict(sample_tokens)
    mlm_model.summary()

    mlm_ds = get_data()
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    mlm_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    mlm_model.fit(mlm_ds, epochs=EPOCHS, callbacks=[generator_callback])

    bert_backbone.save("bert_backbone_mlm.keras")


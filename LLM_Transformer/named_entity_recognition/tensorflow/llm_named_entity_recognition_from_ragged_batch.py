import os, sys

import keras
from keras import ops
import numpy as np
import tensorflow as tf
from keras import layers
from datasets import load_dataset
from collections import Counter


# hyperparameters

batch_size = 32
epochs     = 10
SEQ_LEN    = 128


# data download and preprocessing

conll_data = load_dataset("lhoestq/conll2003") #"conll2003")


def export_to_file(export_file_path, data):

    with open(export_file_path, "w") as f:

        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )


#os.mkdir("data")
#export_to_file("./data/conll_train.txt", conll_data["train"])
#export_to_file("./data/conll_val.txt", conll_data["validation"])

def make_tag_lookup_table():
    
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels

    return dict(zip(range(0, len(all_labels) + 1), all_labels))


mapping = make_tag_lookup_table()
print(mapping)

all_tokens = sum(conll_data["train"]["tokens"], [])
all_tokens_array = np.array(list(map(str.lower, all_tokens)))

counter = Counter(all_tokens_array)
print(len(counter))

num_tags = len(mapping)
vocab_size = 20000

# We only take (vocab_size - 2) most commons words from the training data since
# the `StringLookup` class uses 2 additional tokens - one denoting an unknown
# token and another one denoting a masking token
vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

# The StringLook class will convert tokens to token IDs
lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

train_data = tf.data.TextLineDataset("./data/conll_train.txt")
val_data = tf.data.TextLineDataset("./data/conll_val.txt")

print(list(train_data.take(1).as_numpy_iterator()))


def map_record_to_training_data(record):

    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1

    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
    
    tokens = tf.strings.lower(tokens)
    tokens = lookup_layer(tokens)
    
    return tokens


# We use `padded_batch` here because each record in the dataset has a
# different length.
train_dataset = (
    train_data
    .map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
    #.batch(batch_size)
)

val_dataset = (
    val_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
    #.batch(batch_size)
)


# model construction

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

    
class NERModel(keras.Model):
    
    def __init__(self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32):

        super().__init__()

        self.embedding_layer   = TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        #self.ffn = layers.Dense(ff_dim, activation="relu")
        self.out = layers.Dense(num_tags, activation="softmax")

        #self.drp1 = layers.Dropout(0.1)
        self.drp2 = layers.Dropout(0.1)

    def call(self, inputs, training=False):

        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        #x = self.drp1(x, training=training)
        #x = self.ffn(x)
        x = self.drp2(x, training=training)
        x = self.out(x)

        return x



ner_model = NERModel(num_tags, vocab_size, embed_dim=64, num_heads=4, ff_dim=128)


# model compilation and training

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=None
        )

        loss = loss_fn(y_true, y_pred)
        mask = ops.cast((y_true > 0), dtype="float32")
        loss = loss * mask

        return ops.sum(loss) / ops.sum(mask)


loss = CustomNonPaddingTokenLoss()

#tf.config.run_functions_eagerly(True)
ner_model.compile(optimizer="adam", loss=loss, metrics=["accuracy"], run_eagerly=True)

ner_model.fit(train_dataset, epochs=epochs)


# model evaluation

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    tokens = lowercase_and_convert_to_ids(tokens)
    return tokens

# Sample inference using the trained model
sample_input = tokenize_and_convert_to_ids(
    "eu rejects german call to boycott british lamb and chicken and duck"
)
sample_input = ops.reshape(sample_input, newshape=[1, -1])

output = ner_model.predict(sample_input)
prediction = np.argmax(output, axis=-1) #[0]
prediction = ops.reshape(prediction, [-1])
prediction = [mapping[i] for i in np.array(prediction)]

# eu -> B-ORG, german -> B-MISC, british -> B-MISC
print(prediction)

loss, acc = ner_model.evaluate(val_dataset) #.take(2))

#sys.exit(-1)

print (f"\nAccuracy: {acc}")

def calculate_metrics(dataset):
    all_true_tag_ids, all_predicted_tag_ids = [], []

    #i = 0
    for x, y in dataset:
        output = ner_model.predict(x, verbose=0)
        predictions = ops.argmax(output, axis=-1)
        predictions = ops.reshape(predictions, [-1])
        true_tag_ids = ops.reshape(y, [-1])

        mask = (true_tag_ids > 0) & (predictions > 0)
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predicted_tag_ids)

    all_true_tag_ids = np.concatenate(all_true_tag_ids)
    all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)

    predicted_tags = [mapping[tag] for tag in all_predicted_tag_ids]
    real_tags = [mapping[tag] for tag in all_true_tag_ids]

    correct = 0
    count = 0
    
    for i in range(0, len(predicted_tags)):
        if predicted_tags[i] == real_tags[i] and predicted_tags[i] != 'O':
            correct = correct + 1
            count = count + 1
        elif predicted_tags[i] != real_tags[i]:
            count = count + 1 

    print (f"Accuracy (non-O): {100 * (correct/count)}")


calculate_metrics(val_dataset)

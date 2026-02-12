import os,sys

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.cluster as cluster


BATCH_SIZE = 6
EPOCHS     = 2

NUM_TRAIN_BATCHES = 200
NUM_TEST_BATCHES = 75
AUTOTUNE = tf.data.experimental.AUTOTUNE


wiki_train = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/train.csv",
    batch_size=1,
    num_epochs=1,
)
wiki_test = tf.data.experimental.make_csv_dataset(
    "wikipedia-sections-triplets/test.csv",
    batch_size=1,
    num_epochs=1,
)


def preprocess(data):

    anc = data["Sentence1"]
    pos = data["Sentence2"]
    neg = data["Sentence3"]

    lbl = 0
        
    return ((anc, pos, neg), lbl)


wiki_train = (
    wiki_train
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
    .take(NUM_TRAIN_BATCHES)
    .prefetch(tf.data.AUTOTUNE)
)
wiki_test = (
    wiki_test.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
    .take(NUM_TEST_BATCHES)
    .prefetch(tf.data.AUTOTUNE)
)


j = 0
for i in wiki_train:
    print (f"{i}")
    j = j + 1

    if j == 3:
        break


#sys.exit(-1)

class BertEncoder(keras.Model):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.preprocessor  = keras_hub.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
        self.backbone      = keras_hub.models.BertBackbone.from_preset("bert_tiny_en_uncased")
        self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        

    def call(self, inputs):

        x = self.preprocessor(inputs)
        h = self.backbone(x)
        e = self.pooling_layer(h["sequence_output"], x["padding_mask"])

        return e


class TripletSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):

        super().__init__(**kwargs)

        self.encoder = encoder

    def call(self, inputs):

        anc, pos, neg = inputs

        #print (f"anc shape: {anc.shape}, anc: {anc}")
        #print (f"pos shape: {pos.shape}, pos: {pos}")
        
        ea = self.encoder(tf.squeeze(anc))
        ep = self.encoder(tf.squeeze(pos))
        en = self.encoder(tf.squeeze(neg))

        positive_dist = keras.ops.sum(keras.ops.square(ea - ep), axis=1)
        negative_dist = keras.ops.sum(keras.ops.square(ea - en), axis=1)

        positive_dist = keras.ops.sqrt(positive_dist)
        negative_dist = keras.ops.sqrt(negative_dist)

        output = keras.ops.stack([positive_dist, negative_dist], axis=0)

        return output

        
    def get_encoder(self):
        return self.encoder
    

class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        positive_dist, negative_dist = tf.unstack(y_pred, axis=0)

        losses = keras.ops.relu(positive_dist - negative_dist + self.margin)
        return keras.ops.mean(losses, axis=0)


bert_encoder = BertEncoder()
bert_encoder.summary()

bert_triplet_siamese = TripletSiamese(bert_encoder)
bert_triplet_siamese.summary()


bert_triplet_siamese.compile(
    loss=TripletLoss(),
    optimizer=keras.optimizers.Adam(2e-5),
    #jit_compile=False,
    run_eagerly=True
)

bert_triplet_siamese.fit(wiki_train, validation_data=wiki_test, epochs=EPOCHS)

print ("\nEvaluation:\n")
questions = [
    "What should I do to improve my English writting?",
    "How to be good at speaking English?",
    "How can I improve my English?",
    "How to earn money online?",
    "How do I earn money online?",
    "How to work and earn money through internet?",
]

#encoder = bert_triplet_siamese.get_encoder()
embeddings = bert_encoder(tf.constant(questions))

#print (f"embeddings -> {embeddings}")

kmeans = cluster.KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)

for i, label in enumerate(kmeans.labels_):
    print(f"sentence ({questions[i]}) belongs to cluster {label}")

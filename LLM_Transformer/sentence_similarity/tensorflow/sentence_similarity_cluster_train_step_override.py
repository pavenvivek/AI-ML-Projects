import os,sys

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.cluster as cluster


BATCH_SIZE = 6
EPOCHS     = 3

stsb_ds = tfds.load("glue/stsb",)
stsb_train, stsb_valid = stsb_ds["train"], stsb_ds["validation"]


def change_range(x):
    return (x / 2.5) - 1


def preprocess(data):

    s1 = data["sentence1"]
    s2 = data["sentence2"]
    lbl = [tf.cast(change_range(data["label"]), tf.float32)]
        
    return (s1, s2, lbl)


# set drop_remainder=True in batch to handle keras.ops.split in RegressionSiamese below. squeeze is removing
# all dimensions and sending flat (no array) input to backbone which is throwing error.
stsb_train = (
    stsb_train
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE, drop_remainder=True) 
    .prefetch(tf.data.AUTOTUNE)
)
stsb_valid = (
    stsb_valid.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

'''
j = 0
for i in stsb_train:
    print (f"i -> {i}")
    j = j + 1

    if j == 4:
        break

sys.exit(-1)
'''

'''
print("\nInput Samples:")
for x, y in stsb_train:
    for i, example in enumerate(x):
        print(f"sentence 1 : {example[0]} ")
        print(f"sentence 2 : {example[1]} ")
        print(f"similarity : {y[i]} \n")
    break
'''

class BertEncoder(keras.Model):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.preprocessor  = keras_hub.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
        self.backbone      = keras_hub.models.BertBackbone.from_preset("bert_tiny_en_uncased")
        self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        self.norm_layer    = keras.layers.UnitNormalization(axis=1)
        

    def call(self, inputs):

        x = self.preprocessor(inputs)
        h = self.backbone(x)
        e = self.pooling_layer(h["sequence_output"], x["padding_mask"])
        e = self.norm_layer(e)

        return e


class RegressionSiamese(keras.Model):
    def __init__(self, encoder, **kwargs):

        super().__init__(**kwargs)

        self.encoder = encoder

    def call(self, inputs):

        sen1, sen2 = inputs

        #print (f"inputs shape: {inputs.shape}, inputs: {inputs}")
        #sen1, sen2 = keras.ops.split(inputs, 2, axis=1)
        #print (f"sen1 -> {tf.squeeze(sen1)}, sen2 -> {sen2}")

        u = self.encoder(sen1) #tf.squeeze(sen1))
        v = self.encoder(sen2) #tf.squeeze(sen2))
        cosine_similarity_scores = keras.ops.matmul(u, keras.ops.transpose(v))

        return cosine_similarity_scores

    def get_encoder(self):
        return self.encoder


    def train_step(self, data):
        s1, s2, labels = data

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self([s1, s2], training=True)
            # Compute loss
            loss = self.compiled_loss(labels, outputs)

        # Compute gradients
        grads = tape.gradient(loss, self.trainable_variables)

        # Apply gradients (update weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metric(s)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, outputs)
                
        return {m.name: m.result() for m in self.metrics}


    def predict_step(self, data):
        s1, s2 = data

        # Forward pass
        outputs = self([s1, s2])

        return outputs

    
    def test_step(self, data):
        s1, s2, labels = data

        # Forward pass
        outputs = self([s1, s2])

        # Compute loss
        loss = self.compiled_loss(labels, outputs)

        # Update metric(s)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, outputs)

        return {m.name: m.result() for m in self.metrics}
    

def test_sample(bert_encoder):
    
    sentences = [
        "Today is a very sunny day.",
        "I am hungry, I will get my meal.",
        "The dog is eating his food.",
    ]
    query = ["The dog is enjoying his meal."]

    sentence_embeddings = bert_encoder(tf.constant(sentences))
    query_embedding = bert_encoder(tf.constant(query))

    cosine_similarity_scores = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
    for i, sim in enumerate(cosine_similarity_scores[0]):
        print(f"cosine similarity score between sentence {i+1} and the query = {sim} ")


        
bert_encoder = BertEncoder()
bert_regression_siamese = RegressionSiamese(bert_encoder)


print ("\nBefore training:")
test_sample(bert_encoder)
print ()

bert_regression_siamese.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(2e-5),
    run_eagerly=True,
)

bert_regression_siamese.fit(stsb_train, validation_data=stsb_valid, epochs=EPOCHS)

print ("\nAfter training:")
test_sample(bert_encoder)
print ()

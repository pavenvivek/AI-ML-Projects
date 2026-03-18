import os,sys

import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import sklearn.cluster as cluster


BATCH_SIZE = 6
EPOCHS     = 1


def change_range(x):
    return (x / 2.5) - 1


def preprocess(data):

    s1 = data["sentence1"]
    s2 = data["sentence2"]
    lbl = [tf.cast(change_range(data["label"]), tf.float32)]
        
    #return ([s1, s2], lbl) -> this format (array: [s1, s2]) doesn't split s1, s2 across batch dimension while the following format (tuple (s1, s2)) does
    return ((s1, s2), lbl)


def get_data():

    stsb_ds = tfds.load("glue/stsb",)
    stsb_train, stsb_valid = stsb_ds["train"], stsb_ds["validation"]

    # set drop_remainder=True in batch if using squeeze in RegressionSiamese below. squeeze is removing
    # all dimensions and sending flat (no array) input to backbone which is throwing error.
    stsb_train = (
        stsb_train
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE) #, drop_remainder=True) 
        .prefetch(tf.data.AUTOTUNE)
    )
    stsb_valid = (
        stsb_valid.
        map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE) #, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


    #j = 0
    #for i in stsb_train:
    #    print (f"i -> {i}")
    #    j = j + 1

    #    if j == 4:
    #        break

    return stsb_train, stsb_valid


@keras.saving.register_keras_serializable()
class BertEncoder(keras.Model):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.preprocessor  = keras_hub.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
        self.backbone      = keras_hub.models.BertBackbone.from_preset("bert_tiny_en_uncased")
        self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        self.norm_layer    = keras.layers.UnitNormalization(axis=1) # normalization needed to keep cosine similarity within range (0,1)
        

    def call(self, inputs):

        x = self.preprocessor(inputs)
        h = self.backbone(x)
        e = self.pooling_layer(h["sequence_output"], x["padding_mask"])
        e = self.norm_layer(e)

        return e


class RegressionSiamese(keras.Model):
    def __init__(self, encoder1, encoder2, **kwargs):

        super().__init__(**kwargs)

        self.encoder1 = encoder1
        self.encoder2 = encoder2

    def call(self, inputs):

        #print (f"inputs shape: {inputs.shape}, inputs: {inputs}")
        sen1, sen2 = inputs # keras.ops.split(inputs, 2, axis=1) -> use keras.ops.split if using array format in preprocess
        #print (f"sen1 -> {tf.squeeze(sen1)}, sen2 -> {sen2}")

        u = self.encoder1(sen1) #tf.squeeze(sen1))
        v = self.encoder2(sen2) #tf.squeeze(sen2))

        cosine_similarity_scores = keras.ops.matmul(u, keras.ops.transpose(v))

        return cosine_similarity_scores

    def get_encoder(self):
        return self.encoder


def test_sample(bert_encoder1, bert_encoder2, sentences, query):
    
    query_embedding = bert_encoder1(tf.constant(query))
    sentence_embeddings = bert_encoder2(tf.constant(sentences))

    cosine_similarity_scores = tf.matmul(query_embedding, tf.transpose(sentence_embeddings))
    for i, sim in enumerate(cosine_similarity_scores[0]):
        print(f"cosine similarity score between sentence {i+1} and the query = {sim} ")



if __name__ == "__main__":
        

    stsb_train, stsb_valid = get_data()

    bert_encoder1 = BertEncoder()
    bert_encoder2 = BertEncoder()
    bert_regression_siamese = RegressionSiamese(bert_encoder1, bert_encoder2)

    sentences = [
            "Today is a very sunny day.",
            "I am hungry, I will get my meal.",
            "The meal cost is high.",
            "The dog is playing in dirt.",
            "The furnitures are on sale.",
            "The dog is eating his food.",
        ]
    query = ["The dog is enjoying his meal."]

    print ("\nBefore training:")
    test_sample(bert_encoder1, bert_encoder2, sentences, query)
    print ()

    bert_regression_siamese.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(2e-5),
        run_eagerly=True,
    )

    bert_regression_siamese.fit(stsb_train, validation_data=stsb_valid, epochs=EPOCHS)



    print ("\nAfter training:")
    test_sample(bert_encoder1, bert_encoder2, sentences, query)
    print ()

    query = ["Food cost is increasing."]
    test_sample(bert_encoder1, bert_encoder2, sentences, query)
    print ()

    query = ["Temperature is high."]
    test_sample(bert_encoder1, bert_encoder2, sentences, query)

    #test_sample(bert_encoder2, bert_encoder1)
    #print ()

    bert_encoder1.save("sentence_transformer_query.keras")
    bert_encoder2.save("sentence_transformer_docs.keras")


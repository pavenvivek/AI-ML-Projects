import os, sys
import keras
import keras_hub
import numpy as np

from keras import ops
import pathlib, math
import tensorflow as tf

import keras, keras_hub
from keras import layers


# Parameters

# Preprocessing params.
BATCH_SIZE = 32
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
LEARNING_RATE = 5e-5
EPOCHS = 1

# LoRA-specific hyperparameters
RANK = 4
ALPHA = 32.0


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


# We use prefetch() to pre-compute preprocessed batches on the fly on our CPU.
train_data = (
    sst_train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
vald_data = (
    sst_val_ds.
    map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=1000)
    .batch(BATCH_SIZE)
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

backbone = keras_hub.models.Backbone.from_preset("bert_base_en_uncased", num_layers=5) #, hidden_dim=256) 

model = keras_hub.models.BertClassifier(
    backbone=backbone,
    activation="sigmoid",
    num_classes=1
)

backbone.summary()
model.summary()

'''
encoder_layer = backbone.get_layer(f"transformer_layer_0")
self_attention_layer = encoder_layer._self_attention_layer
query = self_attention_layer._query_dense
value = self_attention_layer._value_dense

print (f"model: \n{vars(encoder_layer)} \n\n{vars(self_attention_layer)}") # \n\n{vars(query)} \n\n{vars(value)}")
print (f"\nquery config: {query.get_config()}")
print (f"\nvalue config: {value.get_config()}")
'''


# create LoRA layers

class LoraLayer(keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        rank=8,
        alpha=32,
        trainable=False,
        **kwargs,
    ):
        # We want to keep the name of this layer the same as the original
        # dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]

        #print (f"original_layer_config: {original_layer_config}")
        
        kwargs.pop("name", None)

        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha

        self._scale = alpha / rank

        self._num_heads = original_layer_config["output_shape"][-2]
        self._hidden_dim = self._num_heads * original_layer_config["output_shape"][-1]

        # Layers.

        # Original dense layer.
        self.original_layer = original_layer
        # No matter whether we are training the model or are in inference mode,
        # this layer should be frozen.
        self.original_layer.trainable = False

        # LoRA dense layers.
        self.A = keras.layers.Dense(
            units=rank,
            use_bias=False,
            # Note: the original paper mentions that normal distribution was
            # used for initialization. However, the official LoRA implementation
            # uses "Kaiming/He Initialization".
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name=f"lora_A",
        )
        # B has the same `equation` and `output_shape` as the original layer.
        # `equation = abc,cde->abde`, where `a`: batch size, `b`: sequence
        # length, `c`: `hidden_dim`, `d`: `num_heads`,
        # `e`: `hidden_dim//num_heads`. The only difference is that in layer `B`,
        # `c` represents `rank`.
        self.B = keras.layers.EinsumDense(
            equation=original_layer_config["equation"],
            output_shape=original_layer_config["output_shape"],
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"lora_B",
        )

        #print (f"original_layer_config['equation'] : {original_layer_config['equation']}")
        #print (f"original_layer_config['output_shape'] : {original_layer_config['output_shape']}")

    def call(self, inputs):

        original_output = self.original_layer(inputs)
        A_out = self.A(inputs)
        B_out = self.B(self.A(inputs))

        if self.trainable:
            # If we are fine-tuning the model, we will add LoRA layers' output
            # to the original layer's output.
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output

        return original_output


def create_lora_layers(lora_model):

    for layer_idx in range(lora_model.backbone.num_layers):
        # Change query dense layer.
        decoder_layer = lora_model.backbone.get_layer(f"transformer_layer_{layer_idx}")
        self_attention_layer = decoder_layer._self_attention_layer
        # Allow mutation to Keras layer state.
        self_attention_layer._tracker.locked = False

        # Change query dense layer.
        self_attention_layer._query_dense = LoraLayer(
            self_attention_layer._query_dense,
            rank=RANK,
            alpha=ALPHA,
            trainable=True,
        )

        # Change value dense layer.
        self_attention_layer._value_dense = LoraLayer(
            self_attention_layer._value_dense,
            rank=RANK,
            alpha=ALPHA,
            trainable=True,
        )

    # need a dummy forward pass here for maintaining a valid chain of computation
    inputs = preprocessor(["LoRA is very useful for quick LLM finetuning"])
    inputs = {"token_ids" : inputs["token_ids"],
              "segment_ids" : inputs["segment_ids"],
              "padding_mask" : inputs["padding_mask"]}
    lora_model(inputs)

    for layer in lora_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())

        if len(lst_of_sublayers) == 1:  # "leaves of the model"
            if layer.name in ["lora_A", "lora_B"]:
                layer.trainable = True
            else:
                layer.trainable = False

    return lora_model


lora_model = create_lora_layers(model)
lora_model.summary()

#sys.exit(-1)

# model training

lora_model.compile(
    optimizer=keras.optimizers.AdamW(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lora_model.fit(
    train_data,
    #validation_data=vald_data,
    epochs=EPOCHS)


# merge weights

# In inference mode, "merge" the LoRA layers' weights into the original layer's weights
def add_weights_to_original_layer(lora_model):

    for layer_idx in range(lora_model.backbone.num_layers):
        self_attention_layer = lora_model.backbone.get_layer(
            f"transformer_layer_{layer_idx}"
        )._self_attention_layer

        # Merge query dense layer.
        query_lora_layer = self_attention_layer._query_dense

        A_weights = query_lora_layer.A.kernel  # (768, 4) (a, b)     #  b=4 -> Rank
        B_weights = query_lora_layer.B.kernel  # (4, 12, 64) (b, c, d)
        increment_weights = tf.einsum("ab,bcd->acd", A_weights, B_weights) * (ALPHA / RANK)
        query_lora_layer.original_layer.kernel.assign_add(increment_weights)

        #print (f"A_weights shape: {A_weights.shape}")
        #print (f"B_weights shape: {B_weights.shape}")
        #print (f"increment_weights shape: {increment_weights.shape}")

        # Merge value dense layer.
        value_lora_layer = self_attention_layer._value_dense

        A_weights = value_lora_layer.A.kernel  # (768, 4) (a, b)
        B_weights = value_lora_layer.B.kernel  # (4, 12, 64) (b, c, d)
        increment_weights = tf.einsum("ab,bcd->acd", A_weights, B_weights) * (ALPHA / RANK)
        value_lora_layer.original_layer.kernel.assign_add(increment_weights)

        # Put back in place the original layers with updated weights
        self_attention_layer._query_dense = query_lora_layer.original_layer
        self_attention_layer._value_dense = value_lora_layer.original_layer

    return lora_model

        
lora_model = add_weights_to_original_layer(lora_model)


# model evaluation

loss, acc = lora_model.evaluate(vald_data)

print (f"Testing loss: {loss}, accuracy: {acc}")


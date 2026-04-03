import os, sys

import keras_hub
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import math

#keras.mixed_precision.set_global_policy("mixed_float16")


# General hyperparameters
BATCH_SIZE = 10
NUM_BATCHES = 200
EPOCHS = 1  # Can be set to a higher value for better results
MAX_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 200


# LoRA-specific hyperparameters
RANK = 4
ALPHA = 32.0


# data preprocessing

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)

for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break

train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
train_ds = train_ds.take(NUM_BATCHES)


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
    lora_model(preprocessor(["LoRA is very useful for quick LLM finetuning"])[0])

    for layer in lora_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())

        if len(lst_of_sublayers) == 1:  # "leaves of the model"
            if layer.name in ["lora_A", "lora_B"]:
                layer.trainable = True
            else:
                layer.trainable = False

    return lora_model


# Load the original model.
preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=128,
)

lora_model = keras_hub.models.GPT2CausalLM.from_preset(
    "gpt2_base_en",
    preprocessor=preprocessor,
)

#gpt_backbone = keras_hub.models.Backbone.from_preset(
#    "gpt2_base_en",
#    num_layers=4
#)

#gpt_backbone.summary()

#lora_model = keras_hub.models.GPT2CausalLM(
#    backbone=gpt_backbone, #"gpt2_base_en",
#    preprocessor=preprocessor,
#)

lora_model = create_lora_layers(lora_model)
lora_model.summary()

optimizer = keras.optimizers.AdamW(learning_rate=5e-5)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

lora_model.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

lora_model.fit(
    train_ds,
    epochs=EPOCHS
)


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
        
# evaluation

output = lora_model.generate("I like to watch", max_length=200)
print (f"Test case 1: \n{output}")

output = lora_model.generate("I am planning to visit a ", max_length=200)
print (f"Test case 2: \n{output}")

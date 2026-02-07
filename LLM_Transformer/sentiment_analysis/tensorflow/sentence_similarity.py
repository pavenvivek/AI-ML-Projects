import numpy as np
import tensorflow as tf
import keras
import keras_hub
import tensorflow_datasets as tfds


# hyperparameters

BATCH_SIZE = 32
EPOCHS = 3

SEQ_LENGTH = 128
NUM_LAYERS = 3
MODEL_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 4


# data download

snli_train = tfds.load("snli", split="train[:20%]")
snli_val = tfds.load("snli", split="validation")
snli_test = tfds.load("snli", split="test")

# Here's an example of how our training samples look like, where we randomly select
# four samples:
sample = snli_test.batch(4).take(1).get_single_element()
print(f"sample: {sample}")

def filter_labels(sample):
    return sample["label"] >= 0

def split_labels(sample):
    x = (sample["hypothesis"], sample["premise"])
    y = sample["label"]
    return x, y


train_ds = (
    snli_train.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)
val_ds = (
    snli_val.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)
test_ds = (
    snli_test.filter(filter_labels)
    .map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
)

#--

# build the backbone and load into classifier

tokenizer = keras_hub.models.BertTokenizer.from_preset("bert_tiny_en_uncased",)
preprocessor = keras_hub.models.BertPreprocessor(tokenizer, sequence_length=SEQ_LENGTH)


backbone = keras_hub.models.BertBackbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_dim=MODEL_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    max_sequence_length=SEQ_LENGTH,
)

model = keras_hub.models.BertTextClassifier(
    backbone=backbone,
    preprocessor=preprocessor,
    num_classes=3,
)

#--

# or

#--

# build the classifier using pretrained weights

#model = keras_hub.models.BertClassifier.from_preset(
#    "bert_tiny_en_uncased", num_classes=3
#)

#--

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.evaluate(test_ds)




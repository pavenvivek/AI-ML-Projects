import os, sys

import time
import keras_hub
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd


# hyper parameters

BATCH_SIZE = 8
NUM_BATCHES = 600
EPOCHS = 1  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40


# data pereprocessing

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
train_df = pd.read_csv("hf://datasets/knkarthick/samsum/" + splits["train"])
vald_df = pd.read_csv("hf://datasets/knkarthick/samsum/" + splits["validation"])
test_df = pd.read_csv("hf://datasets/knkarthick/samsum/" + splits["test"])

#print (train_df["summary"].to_numpy())

train_df = tf.data.Dataset.from_tensor_slices(train_df.to_numpy())

def preprocess_train(data):

    idx, dialogue, summary = data[0], data[1], data[2]
    return {"encoder_text": dialogue, "decoder_text": summary}


train_ds = (
    train_df.map(preprocess_train)
    .batch(BATCH_SIZE)
)

#print (f"data: {next(iter(train_ds.take(1)))}")
train_ds = train_ds.take(NUM_BATCHES)

#sys.exit(-1)


# model training

preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

bart_lm.summary()

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
)

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

bart_lm.fit(train_ds, epochs=EPOCHS)


# model evaluation

dialogues = test_df["dialogue"].to_numpy()
ground_truth_summaries = test_df["summary"].to_numpy()
test_df  = tf.data.Dataset.from_tensor_slices(test_df.to_numpy())

def preprocess_test(data):

    idx, dialogue, summary = data[0], data[1], data[2]
    return dialogue

test_ds = (
    test_df.map(preprocess_test)
    .batch(BATCH_SIZE)
)

#print (f"data: {next(iter(test_ds.take(1)))}")
test_ds = test_ds.take(2)

#sys.exit(-1)

def generate_text(model, input_text, max_length=200):
    output = model.generate(input_text, max_length=max_length)

    return output


# Generate summaries.
generated_summaries = generate_text(
    bart_lm,
    test_ds,
    max_length=MAX_GENERATION_LENGTH,
)

print (f"generated_summary: {generated_summaries}")
print (f"len generated_summary: {len(generated_summaries)}")

idx = 0
for dialogue, generated_summary, ground_truth_summary in zip(dialogues[:5], generated_summaries[:5], ground_truth_summaries[:5]):

    print (f"_____Test Case: {idx+1}_____\n")
    print("Dialogue:", dialogue)
    print("Generated Summary:", generated_summary)
    print("Ground Truth Summary:", ground_truth_summary)
    print("__________________\n")

    idx = idx + 1


    

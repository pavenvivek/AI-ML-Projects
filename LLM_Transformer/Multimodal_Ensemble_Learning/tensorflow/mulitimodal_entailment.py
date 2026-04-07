from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import os, sys
import time

import keras
import keras_hub
import tensorflow as tf
from keras.utils import PyDataset


print (f"___program____ start")
start = time.time()

# model hyperparameters

BATCH_SIZE = 32
EPOCHS     = 1


# data download

image_base_path = keras.utils.get_file(
    "tweet_images",
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
    untar=True,
)

df = pd.read_csv(
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv"
).iloc[
    0:1000
]  # Resources conservation since these are examples and not SOTA
print(df.sample(10))

images_one_paths = []
images_two_paths = []


for idx in range(len(df)):
    current_row = df.iloc[idx]
    id_1 = current_row["id_1"]
    id_2 = current_row["id_2"]
    extentsion_one = current_row["image_1"].split(".")[-1]
    extentsion_two = current_row["image_2"].split(".")[-1]

    image_one_path = os.path.join(image_base_path, str(id_1) + f".{extentsion_one}")
    image_two_path = os.path.join(image_base_path, str(id_2) + f".{extentsion_two}")

    images_one_paths.append(image_one_path)
    images_two_paths.append(image_two_path)

df["image_1_path"] = images_one_paths
df["image_2_path"] = images_two_paths

# Create another column containing the integer ids of
# the string labels.
label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
df["label_idx"] = df["label"].apply(lambda x: label_map[x])

train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label"].values, random_state=42
)
# 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")


# data preprocessing

text_preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=128,
)
bert_input_features = ["padding_mask", "segment_ids", "token_ids"]

def preprocess_text(text):

    text_1, text_2 = text
    output = text_preprocessor([text_1, text_2])
    output = {
        feature: keras.ops.reshape(output[feature], [-1])
        for feature in bert_input_features
    }

    return output

def preprocess_image(image):

    image = resize(imread(image[0]), (128, 128))
    if image.shape[2] == 4:
        image = Image.fromarray((image.astype(np.uint8))).convert("RGB")
        
    return image

# --
# preprocess before training
'''
def extract_cols(df):
    
    img1 = df[["image_1_path"]].to_numpy()
    img2 = df[["image_2_path"]].to_numpy()
    text = df[["text_1", "text_2"]].to_numpy()
    label = df[["label_idx"]].to_numpy()

    return (img1, img2, text, label)


def get_data(df):

    img1, img2, text, labels = extract_cols(df)
    
    # image preprocessing
    img1 = np.apply_along_axis(preprocess_image, axis=1, arr=img1) 
    img2 = np.apply_along_axis(preprocess_image, axis=1, arr=img2) 
    
    # text preprocessing    
    text = np.apply_along_axis(preprocess_text, axis=1, arr=text) 
    text = np.array([[d["padding_mask"], d["segment_ids"], d["token_ids"]] for d in text])
    padding_mask = text[:, 0]
    segment_ids  = text[:, 1]
    token_ids    = text[:, 2]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "image_1": img1,
                "image_2": img2,
                "padding_mask": padding_mask,
                "segment_ids" : segment_ids,
                "token_ids" : token_ids
            },
            labels
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    return dataset


train_ds = get_data(train_df)
test_ds  = get_data(test_df)
validation_ds = get_data(val_df)
'''
#--

# or

#--
# preprocess during training

class UnifiedPyDataset(PyDataset):
    """A Keras-compatible dataset that processes a DataFrame for TensorFlow, JAX, and PyTorch."""

    def __init__(
        self,
        df,
        batch_size=32,
        workers=4,
        use_multiprocessing=False,
        max_queue_size=10,
        **kwargs,
    ):
        """
        Args:
            df: pandas DataFrame with data
            batch_size: Batch size for dataset
            workers: Number of workers to use for parallel loading (Keras)
            use_multiprocessing: Whether to use multiprocessing
            max_queue_size: Maximum size of the data queue for parallel loading
        """
        
        super().__init__(**kwargs)
        self.dataframe = df

        # image files
        self.image_1 = self.dataframe[["image_1_path"]].to_numpy()
        self.image_2 = self.dataframe[["image_2_path"]].to_numpy()

        # text files
        self.text = self.dataframe[["text_1", "text_2"]].to_numpy()

        # labels
        self.labels = self.dataframe[["label_idx"]].to_numpy()

        # general
        self.batch_size = batch_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def __getitem__(self, index):
        """
        Fetches a batch of data from the dataset at the given index.
        """

        # Return x, y for batch idx.
        low  = index * self.batch_size
        high = min(low + self.batch_size, len(self.image_1))

        batch_image_1 = self.image_1[low:high]
        batch_image_2 = self.image_2[low:high]
        batch_text    = self.text[low:high]
        batch_labels  = self.labels[low:high]

        # image preprocessing
        image_1 = np.apply_along_axis(preprocess_image, axis=1, arr=batch_image_1) 
        image_2 = np.apply_along_axis(preprocess_image, axis=1, arr=batch_image_2) 

        # text preprocessing
        text = np.apply_along_axis(preprocess_text, axis=1, arr=batch_text)
        text = np.array([[d["padding_mask"], d["segment_ids"], d["token_ids"]] for d in text])
        padding_mask = text[:, 0]
        segment_ids  = text[:, 1]
        token_ids    = text[:, 2]


        return (
            {
                "image_1": image_1,
                "image_2": image_2,
                "padding_mask": padding_mask,
                "segment_ids": segment_ids,
                "token_ids": token_ids,
            },
            batch_labels,
        )

    def __len__(self):
        """
        Returns the number of batches in the dataset.
        """
        return math.ceil(len(self.dataframe) / self.batch_size)


    
def prepare_dataset(dataframe):

    ds = UnifiedPyDataset(
        dataframe,
        batch_size=BATCH_SIZE,
        workers=4,
    )

    return ds


train_ds = prepare_dataset(train_df)
test_ds = prepare_dataset(test_df)
validation_ds = prepare_dataset(val_df)

#--


# model construction

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims, activation="gelu")(embeddings)

    '''
    for _ in range(num_projection_layers):
        x = keras.ops.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    '''
    
    return projected_embeddings


def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Preprocess the input image.
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
    preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)

    # Generate the embeddings for the images using the resnet_v2 model
    # concatenate them.
    embeddings_1 = resnet_v2(preprocessed_1)
    embeddings_2 = resnet_v2(preprocessed_2)
    embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model([image_1, image_2], outputs, name="vision_encoder")


def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT BackBone using KerasHub.
    bert = keras_hub.models.BertBackbone.from_preset(
        "bert_base_en_uncased", num_classes=3
    )

    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
    inputs = {
        feature: keras.Input(shape=(256,), dtype="int32", name=feature)
        for feature in bert_input_features
    }

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")


def create_multimodal_model(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    vision_trainable=False,
    text_trainable=False,
):
    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Receive the text as inputs.
    bert_input_features = ["padding_mask", "segment_ids", "token_ids"]
    text_inputs = {
        feature: keras.Input(shape=(256,), dtype="int32", name=feature)
        for feature in bert_input_features
    }
    text_inputs = list(text_inputs.values())
    # Create the encoders.
    vision_encoder = create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, vision_trainable
    )
    text_encoder = create_text_encoder(
        num_projection_layers, projection_dims, dropout_rate, text_trainable
    )

    # Fetch the embedding projections.
    vision_projections = vision_encoder([image_1, image_2])
    text_projections = text_encoder(text_inputs)

    # Concatenate the projections and pass through the classification layer.
    concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
    outputs = keras.layers.Dense(3, activation="softmax")(concatenated)
    return keras.Model([image_1, image_2, *text_inputs], outputs)


multimodal_model = create_multimodal_model()

#keras.utils.plot_model(multimodal_model, show_shapes=True)
multimodal_model.summary()


# model training and evaluation

multimodal_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], #run_eagerly=True # enabling run_eagerly throws memory insufficient issue
)

history = multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=EPOCHS) 

_, acc = multimodal_model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")

end = time.time()

print(f"Total Time Elapsed: {end - start:.2f}s")
print (f"____program____ end")



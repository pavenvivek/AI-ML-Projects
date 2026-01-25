import os, sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import dgl

from tensorflow import keras
from tensorflow.keras import layers
#from dgl.nn.tensorflow import GraphConv


# Note: to use dgl set the following flags
## export DGLBACKEND=tensorflow 
## export CUDA_LAUNCH_BLOCKING=1  # -> this flag is necessary due to an internal error in dgl which needs to be fixed. might depend on the cuda version.

# hyperparameters

hidden_units = [100] #, 32]
learning_rate = 0.01
dropout_rate = 0.2
num_epochs = 50
batch_size = 256
NUM_HEADS = 8
NUM_LAYERS = 3


# Data download and preprocessing

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")

print (f"data_dir: {data_dir}")


citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("Citations shape:", citations.shape)


#print(citations.sample(frac=1).head())

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"), sep="\t", names=column_names, header=None, 
)
print("Papers shape:", papers.shape)
print(papers.sample(5))


class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

#plt.figure(figsize=(10, 10))
#colors = papers["subject"].tolist()
#cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
#subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
#nx.draw_spring(cora_graph, node_size=15, node_color=subjects)
#plt.show()

train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

feature_names = list(set(papers.columns) - {"paper_id", "subject"})
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]


edges = citations[["source", "target"]].to_numpy().T
# Create a node features array of shape [num_nodes, num_features].
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
)
graph_info = (node_features, edges)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)

print (f"Edges -> {edges}")

#g = dgl.graph((edges[0], edges[1]))
#g = dgl.add_self_loop(g)

#sys.exit(-1)

# model construction


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations):
        messages = self.ffn_prepare(node_repesentations)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        num_nodes = node_repesentations.shape[0]
        aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        node_embeddings = self.update_fn(h)
        
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)

        return node_embeddings

    def call(self, inputs):

        node_repesentations, edges = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        neighbour_messages = self.prepare(neighbour_repesentations)
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)

        return self.update(node_repesentations, aggregated_messages)


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        #print (f"shape : {node_states.shape}, {edges.shape}")
        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        #print (f"** shape : {node_states_expanded.shape}")
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        #print (f"shape : {node_states_expanded.shape}, {node_states_transformed.shape}, {(tf.shape(edges)[0], -1)}")
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)
    

class GraphAttentionNetwork(layers.Layer): #keras.Model):
    def __init__(
        self,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]

    def call(self, inputs):
        node_states, edges = inputs
        edges = tf.stack((edges[0], edges[1]), axis=1)
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x

        return x

    
class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        dropout_rate=0.2,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        node_features, edges = graph_info
        self.node_features = node_features
        self.edges = edges

        # for DGL
        #self.graph = dgl.graph((edges[0], edges[1]))
        #self.graph = dgl.add_self_loop(self.graph)

        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")

        # from local GCN
        #self.conv1 = GraphConvLayer(hidden_units, dropout_rate, normalize, name="graph_conv1")
        #self.conv2 = GraphConvLayer(hidden_units, dropout_rate, normalize, name="graph_conv2")
        ##self.conv3 = GraphConvLayer(hidden_units, dropout_rate, normalize, name="graph_conv3")

        # from DGL GCN
        #self.conv1 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv1")
        #self.conv2 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv2")
        ##self.conv3 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv3")

        # from local GAT
        self.gnn_gat = GraphAttentionNetwork(hidden_units[0], NUM_HEADS, NUM_LAYERS, num_classes)
        
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.out = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)

        # for local GCN
        #x1 = self.conv1((x, self.edges))
        ##x = x1 + x   # -> skip-connections
        #x2 = self.conv2((x1, self.edges))
        ##x = x2 + x

        # for DGL GCN
        #x1 = self.conv1(graph=self.graph, feat=x)
        #x2 = self.conv2(graph=self.graph, feat=x1)
        ##x3 = self.conv3(graph=self.graph, feat=x2)

        # for local GAT
        x2 = self.gnn_gat((x, self.edges)) # self.node_features

        x = self.postprocess(x2)
        node_embeddings = tf.gather(x, input_node_indices)
        x = self.out(node_embeddings)

        return x
        

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

print("GNN output shape:", gnn_model(tf.constant([1, 10, 100])))

gnn_model.summary()

#sys.exit(-1)

# model training

x_train = train_data.paper_id.to_numpy()

gnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    run_eagerly=True
)

gnn_model.fit(
    x=x_train,
    y=y_train,
    epochs=50, #num_epochs,
    batch_size=batch_size,
    validation_split=0.15,
)

x_test = test_data.paper_id.to_numpy()
_, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

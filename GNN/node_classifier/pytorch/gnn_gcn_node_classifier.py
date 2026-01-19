import os, sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#import tensorflow as tf
import dgl

from tensorflow import keras
#from dgl.nn import GraphConv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch_scatter import scatter_mean


# Note: to use dgl set the following flags
## export DGLBACKEND=pytorch


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# hyperparameters

hidden_units = [100] #, 32]
learning_rate = 0.01
dropout_rate = 0.2
num_epochs = 50
batch_size = 256


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
#x_train = train_data[feature_names].to_numpy()
#x_test = test_data[feature_names].to_numpy()
x_train = train_data.paper_id.to_numpy()
x_test = test_data.paper_id.to_numpy()

# Create train and test targets as a numpy array.
y_train = train_data["subject"].to_numpy()
y_test = test_data["subject"].to_numpy()

print (f"x_train -> {x_train}")
print (f"y_train -> {y_train}")

class citation_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.int)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_data = citation_dataset(x_train, y_train)
train_dload = DataLoader(train_data, batch_size=batch_size)

test_data = citation_dataset(x_test, y_test)
test_dload = DataLoader(test_data, batch_size=1)


edges = citations[["source", "target"]].to_numpy().T
# Create a node features array of shape [num_nodes, num_features].
node_features = torch.tensor(papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=torch.float32)
graph_info = (node_features, edges)

print("Edges shape:", edges.shape)
print("Nodes shape:", node_features.shape)

print (f"Edges -> {edges}")

g = dgl.graph((edges[0], edges[1]))
g = dgl.add_self_loop(g)

#sys.exit(-1)



# model construction


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(nn.LazyBatchNorm1d())
        fnn_layers.append(nn.Dropout(p=dropout_rate))
        fnn_layers.append(nn.LazyLinear(units)) #, activation=nn.GELU))
        fnn_layers.append(nn.GELU())

    return nn.Sequential(*fnn_layers).to(device) #, name=name)


class GraphConvLayer(nn.Module):
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
        #print (f"neighbour_messages -> {neighbour_messages.shape}")
        #print (f"node_indices -> {node_indices.shape}")
        #print (f"num_nodes -> {num_nodes}")
        aggregated_message = scatter_mean(neighbour_messages, node_indices, dim=0, dim_size=num_nodes).to(device)
        #aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        h = torch.cat([node_repesentations, aggregated_messages], dim=1).to(device)
        node_embeddings = self.update_fn(h)
        
        #if self.normalize:
        #    node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)

        return node_embeddings

    def forward(self, inputs):

        node_repesentations, edges = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        node_indices = torch.tensor(node_indices).to(device)
        neighbour_indices = torch.tensor(neighbour_indices).to(device)

        #print (f"neighbour_indices -> {neighbour_indices.shape}")
        #print (f"node_repesentations -> {node_repesentations.shape}")
        
        neighbour_repesentations = torch.index_select(node_repesentations, 0, index=neighbour_indices).to(device)

        neighbour_messages = self.prepare(neighbour_repesentations)
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)

        return self.update(node_repesentations, aggregated_messages)


class GNNNodeClassifier(nn.Module):
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
        self.node_features = node_features.to(device)
        self.edges = edges

        # for DGL
        #self.graph = dgl.graph((edges[0], edges[1]))
        #self.graph = dgl.add_self_loop(self.graph)
        #self.graph = self.graph.to(device)
        
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")

        # from local
        self.conv1 = GraphConvLayer(hidden_units, dropout_rate, normalize) #, name="graph_conv1")
        self.conv2 = GraphConvLayer(hidden_units, dropout_rate, normalize) #, name="graph_conv2")
        ##self.conv3 = GraphConvLayer(hidden_units, dropout_rate, normalize) #, name="graph_conv3")

        # from DGL
        #self.conv1 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv1")
        #self.conv2 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv2")
        ##self.conv3 = GraphConv(hidden_units[-1], hidden_units[-1]) #, name="graph_conv3")
        
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.out = nn.LazyLinear(num_classes)

    def forward(self, input_node_indices):
        x = self.preprocess(self.node_features)

        # for local
        x1 = self.conv1((x, self.edges))
        ##x = x1 + x   # -> skip-connections
        x2 = self.conv2((x1, self.edges))
        ##x = x2 + x

        # for DGL
        #x1 = self.conv1(graph=self.graph, feat=x)
        #x2 = self.conv2(graph=self.graph, feat=x1)
        #x3 = self.conv3(graph=self.graph, feat=x2)
        
        x = self.postprocess(x2)
        node_embeddings = torch.index_select(x, 0, index=input_node_indices).to(device)
        x = self.out(node_embeddings)

        return x
        

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    #name="gnn_model",
).to(device)

dummy = torch.tensor([1, 10, 100]).to(device)
print("GNN output shape:", gnn_model(dummy))
print (gnn_model)
#gnn_model.summary()


optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
loss_fn   = nn.CrossEntropyLoss()


# model training


def train(train_dataloader, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0    
    for batch, (x, y) in enumerate(train_dataloader):

        x, y = x.to(device), y.to(device)

        # fordward pass
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)
           

print ("\nTraining: ")
Epochs = 20
for i in range(0, Epochs):
    loss = train(train_dload, gnn_model, optimizer, loss_fn)
    print (f"Epoch {i}/{Epochs}: Training loss -> {loss}")

    
# model testing
def test(test_dataloader, model, loss_fn):

    model.eval()

    with torch.no_grad():
        
        count = 0
        total_loss = 0
        acc = 0
        for x, y in test_dataloader:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            total_loss = total_loss + loss.item()
            count = count + 1

            if pred.argmax() == y:
                acc = acc + 1

        print (f"Testing loss: {total_loss/count}, accuracy: {acc/count}")
    

print ("\nTesting:")
test(test_dload, gnn_model, loss_fn)
    
    
# model prediction

def predict(model, count=10):
    model.eval()
    with torch.no_grad():

        i = 0

        for x, y in test_dload:

            if i == count:
                break

            x, y = x.to(device), y.to(device)

            pred = model(x)

            print (f"pred: {pred.argmax()}, y: {y}")
            i = i + 1

print ("\nSamples Prediction: ")
predict(gnn_model, 10)

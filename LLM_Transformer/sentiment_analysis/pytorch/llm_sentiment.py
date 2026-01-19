import os, sys
import numpy as np

#from keras import ops
import keras #, keras_hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets # huggingface
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig
from transformers import DataCollatorForLanguageModeling
from torchsummary import summary
import pandas as pd


# Data download

# Download pretraining data.
path = keras.utils.get_file(
    origin="https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/wikitext-103.tar.gz",  #"https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
    extract=True,
)

print (f"download path for wiki: {path}")

wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-103-raw/")

# Download finetuning data.
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


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# Parameters

# Preprocessing params.
PRETRAINING_BATCH_SIZE = 64 #128
FINETUNING_BATCH_SIZE = 32 #128 #32
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
PRETRAINING_LEARNING_RATE = 5e-4
PRETRAINING_EPOCHS = 2 #8
FINETUNING_LEARNING_RATE = 0.0001 #5e-5
FINETUNING_EPOCHS = 3


# Data Preprocessing

# Load SST-2.
sst_train_ds = pd.read_csv(
    sst_dir + "train.tsv", sep="\t"
)
sst_test_ds = pd.read_csv(
    sst_dir + "dev.tsv",sep="\t"
)

batch_count = int(len(sst_train_ds)/FINETUNING_BATCH_SIZE)
print (f"training_data len -> {len(sst_train_ds)}")
print (f"batch_count -> {batch_count}")

#print (f"sst_train -> {sst_train_ds}")
#print (f"sst_train -> {sst_train_ds.iloc[:,:-1]}")
#print (f"sst_train -> {sst_train_ds.iloc[:,-1:]}")

train_x, train_y = sst_train_ds.iloc[:,:-1], sst_train_ds.iloc[:,-1:]
test_x, test_y = sst_test_ds.iloc[:,:-1], sst_test_ds.iloc[:,-1:]


# Load the tokenizer for a specific pre-trained model (e.g., 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # vocab_file) #"bert-base-uncased")
#tokens = tokenizer(train_data[1], padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")
#tokens = tokenizer.tokenize(train_data[0], return_tensors="pt")

#print (f"tokens -> {tokens}")
print (f"vocab_size -> {tokenizer.vocab_size}")


class SST_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data.values.tolist() #torch.tensor(data, dtype=torch.float)
        self.label = np.array(label, dtype=np.float32) #.values.tolist() #torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tokenizer(self.data[idx], padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")["input_ids"].squeeze(), self.label[idx]


train_data = SST_dataset(train_x, train_y)
train_dload = DataLoader(train_data, batch_size=FINETUNING_BATCH_SIZE, shuffle=True) #

test_data = SST_dataset(test_x, test_y)
test_dload = DataLoader(test_data, batch_size=1, shuffle=True)



# model construction


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.tok_emb = nn.Embedding(tokenizer.vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(SEQ_LENGTH, hidden_size)

        
    def forward(self, x):

        tok_emb = self.tok_emb(x)
        pos_emb_inp = torch.arange(0, SEQ_LENGTH).to(device)
        out = self.tok_emb(x) + self.pos_emb(pos_emb_inp)
    
        return out


class LLM_Text(nn.Module):

    def __init__(self, hidden_size, num_layers):
        super().__init__()

        self.nlayers = num_layers
        self.hdim    = hidden_size

        self.tok_and_pos = TokenAndPositionEmbedding(hidden_size).to(device)

        # Bad Repr -> Very slow convergence of Accuracy (but doing good for batch size 1)
        #self.trans_enc = [nn.TransformerEncoderLayer(d_model=self.hdim, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_DIM, batch_first=True).to(device) for _ in range(self.nlayers)] 

        # Good Repr -> Faster convergence of accuracy
        self.trans_enc_lyr = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_DIM, batch_first=True) 
        self.trans_enc = nn.TransformerEncoder(self.trans_enc_lyr, num_layers=self.nlayers)

        # Good Repr -> Faster convergence of accuracy but more code
        #self.trans_enc1 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_DIM, batch_first=True)
        #self.trans_enc2 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_DIM, batch_first=True)
        #self.trans_enc3 = nn.TransformerEncoderLayer(d_model=self.hdim, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_DIM, batch_first=True)

        self.drp  = nn.Dropout(p=0.1)
        self.ln    = nn.LayerNorm(self.hdim)

        
    def forward(self, x):

        x = self.tok_and_pos(x)
        x = self.drp(self.ln(x))

        #for i in range(0, self.nlayers):
        #    x = self.trans_enc[i](x)

        x = self.trans_enc(x)
        
        return x



class LLM_Text_Classifier(nn.Module):

    def __init__(self, hidden_size, num_layers, model_llm_text):
        super().__init__()

        self.nlayers = num_layers
        self.hdim    = hidden_size

        self.llm_text = model_llm_text #LLM_Text(hidden_size, num_layers) #.to(device)
        
        # classification head
        self.out  = nn.LazyLinear(1)        
        self.sig  = nn.Sigmoid()


    def forward(self, x):

        x = self.llm_text(x)

        #x = self.sig(self.out(x["last_hidden_state"][:,0,:]))
        #x = self.sig(self.out(x["pooler_output"]))

        x = self.sig(self.out(x[:,0,:]))
        
        return x


# Load from local pre-trained model
model_llm_text = LLM_Text(MODEL_DIM, NUM_LAYERS).to(device)
#model_llm_text.load_state_dict(torch.load('./saved_model/model_llm_text_weights_final.pth', weights_only=True))

# Load model from HuggingFace
#configuration = BertConfig(hidden_size=MODEL_DIM, num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS, intermediate_size=INTERMEDIATE_DIM)
#configuration = BertConfig(hidden_size=768, num_hidden_layers=NUM_LAYERS) #, num_attention_heads=NUM_HEADS, intermediate_size=INTERMEDIATE_DIM)
#model_llm_text = BertModel(configuration, add_pooling_layer=True)

# Load pretrained model from HuggingFace
##configuration = BertConfig.from_pretrained("bert-base-uncased") #, hidden_size=MODEL_DIM, num_hidden_layers=NUM_LAYERS, num_attention_heads=NUM_HEADS, intermediate_size=INTERMEDIATE_DIM)
#configuration = BertConfig(hidden_size=768, num_hidden_layers=NUM_LAYERS) #, num_attention_heads=NUM_HEADS, intermediate_size=INTERMEDIATE_DIM)
#model_llm_text = BertModel.from_pretrained("bert-base-uncased", config=configuration) #(configuration, add_pooling_layer=True)


model = LLM_Text_Classifier(MODEL_DIM, NUM_LAYERS, model_llm_text).to(device)

print (model_llm_text)
print ("=============")
print (model)
#summary(model, input_size=(128, 256)) # need dummy input due to LazyLinear


#print (f"sample -> {train_data[:5]}")
#sample = train_data[:5]
#dummy_input, y = sample
#print (f"input -> {dummy_input}")

sample = train_x.values.tolist()[:5]
dummy_input = []
for i in range(0, 5):
    dummy_input = dummy_input + sample[i]

print (f"sample -> {dummy_input}")

#print (f"y -> {y}")
#print (f"tok out -> {tokenizer(dummy_input, return_tensors='pt')}")
dummy_input = tokenizer(dummy_input, truncation=True, padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")["input_ids"]
dummy_input = dummy_input.to(device)
val = model(dummy_input)
print (f"val -> {val}")
#print (f"val -> {val['last_hidden_state'][:,0,:]}")
#print (f"val -> {val[:,0,:]}")
print (f"val shape -> {val.shape}")
#print (f"val last_hidden_state shape -> {val['last_hidden_state'].shape}")
#print (f"val pooler_output shape -> {val['pooler_output'].shape}")

#sys.exit(-1)


optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNING_LEARNING_RATE)
loss_fn = nn.BCELoss() #nn.BCEWithLogitsLoss() # 


# model training

def train(train_dload, model, optimizer, loss_fn):

    model.train()

    loss_cml  = 0
    batch_cnt = 0
    for batch, (x,y) in enumerate(train_dload):

        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_cml  = loss_cml + loss.item()
        batch_cnt = batch_cnt + 1

    return (loss_cml/batch_cnt)


print ("\nTraining: ")
Epochs = 3
for i in range(0, Epochs):
    loss = train(train_dload, model, optimizer, loss_fn)
    print (f"Epoch {i+1}/{Epochs}: Training loss -> {loss}")

    
# model testing

def test(test_dload, model, loss_fn):

    model.eval()

    with torch.no_grad():
        count  = 0
        acc    = 0
        loss_v = 0
        i = 0
        for x, y in test_dload:

            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            loss_v = loss_v + loss.item()
            count = count + 1

            if pred.round() == y:
                acc = acc + 1

    print (f"Testing loss: {loss_v/count}, Accuracy: {acc/count}")


print ("\nTesting: ")
test(test_dload, model, loss_fn)


# prediction samples

def predict(model, count=10):
    model.eval()
    with torch.no_grad():

        i = 0
        for x, y in test_dload:

            if i == count:
                break

            x, y = x.to(device), y.to(device)
            pred = model(x)

            print (f"pred: {pred.round()}, prd : {pred}, y: {y}, val: {pred.round() == y}")
            i = i + 1

print ("\nSamples Prediction: ")
predict(model, count=10)

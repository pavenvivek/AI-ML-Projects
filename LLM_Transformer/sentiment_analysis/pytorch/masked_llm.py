import os, sys
import numpy as np

#from keras import ops
import keras #, keras_hub

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import datasets # huggingface
from transformers import AutoTokenizer, BertTokenizer
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

# Load the tokenizer for a specific pre-trained model (e.g., 'bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(vocab_file) #"bert-base-uncased")
#tokens = tokenizer(train_data[1], padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")
#tokens = tokenizer.tokenize(train_data[0], return_tensors="pt")

#print (f"tokens -> {tokens}")
print (f"vocab_size -> {tokenizer.vocab_size}")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, #mask_replace_prob=0, random_replace_prob=1
)

#masked_tokens = data_collator([tokens])

#print (f"masked_tokens -> {masked_tokens}")


class TextLineDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
        #with open(filepath, 'r') as f:
            self.lines_inp = f.readlines()
        
        #self.lines = [line.strip() for line in self.lines]
        self.lines = []
        
        for l in self.lines_inp:

            if len(l) > 100:
                self.lines.append(l)
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        #return self.lines[idx]

        tokens = tokenizer(self.lines[idx], truncation=True, padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")["input_ids"]
        masked_tokens = data_collator(tokens)

        return masked_tokens['input_ids'].squeeze(), masked_tokens['labels'].squeeze()

    
train_data = TextLineDataset(wiki_dir+"wiki.train.tokens")
validation_data = TextLineDataset(wiki_dir+"wiki.valid.tokens")

print(f"len training_dataset: {len(train_data)}")
print(f"training_dataset: {train_data[0]}")


train_dload = DataLoader(train_data, batch_size=PRETRAINING_BATCH_SIZE, shuffle=True) #, collate_fn=data_collator)
validation_dload = DataLoader(validation_data, batch_size=1) #, collate_fn=data_collator)

batch_count = int(len(train_data)/PRETRAINING_BATCH_SIZE)
print (f"batch_count -> {batch_count}")

'''
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
'''

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

class LLM_Text_MLM(nn.Module):

    def __init__(self, hidden_size, num_layers, model_llm_text):
        super().__init__()

        self.nlayers = num_layers
        self.hdim    = hidden_size

        self.llm_text = model_llm_text #LLM_Text(hidden_size, num_layers) #.to(device)

        # mlm head
        self.dense = nn.Linear(self.hdim, self.hdim)
        self.rlu   = nn.ReLU()
        self.ln    = nn.LayerNorm(self.hdim)
        self.decoder = nn.Linear(self.hdim, tokenizer.vocab_size, bias=True)
        self.bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))
        #self.sfx  = nn.Softmax(dim=1)
        

    def forward(self, x):

        x = self.llm_text(x)
        x = self.dense(x)
        x = self.rlu(x)
        x = self.ln(x)
        x = self.decoder(x)
        #x = self.sfx(x)        
        #x = self.sig(self.out(x[:,0,:]))
        
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
        x = self.sig(self.out(x[:,0,:]))
        
        return x

    
    

model_llm_text = LLM_Text(MODEL_DIM, NUM_LAYERS).to(device)
model = LLM_Text_MLM(MODEL_DIM, NUM_LAYERS, model_llm_text).to(device)

print (model)
#summary(model, input_size=(128, 256)) # need dummy input due to LazyLinear

'''
print (f"sample -> {train_data[:5]}")
sample = train_data[:5]
dummy_input, y = sample
#print (f"input -> {dummy_input}")
#print (f"y -> {y}")
#print (f"tok out -> {tokenizer(dummy_input, return_tensors='pt')}")
#dummy_input_tokens = tokenizer(dummy_input, truncation=True, padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")["input_ids"]
#dummy_input = data_collator(dummy_input)
#print (f"masked tokens -> {dummy_input}")
dummy_input = dummy_input.to(device)
val = model(dummy_input)
print (f"val -> {val}")
#print (f"val -> {val[:,0,:]}")
print (f"val shape -> {val.shape}")
'''

optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNING_LEARNING_RATE)
loss_fn = nn.BCELoss() #nn.BCEWithLogitsLoss() # 

#sys.exit(-1)

# model training

def train(train_dload, model, optimizer, loss_fn):

    model.train()

    i = 0
    for batch, (x,y) in enumerate(train_dload):

        #print (f"train: x -> {x}")
        #print (f"train: y -> {y}")
        
        x, y = x.to(device), y.to(device)

        # forward
        pred = model(x)
        #loss = loss_fn(pred, y)

        '''
        print (f"train: pred -> {pred}")
        print (f"train: y -> {y}")
        print (f"train: x shape -> {pred.shape}")
        print (f"train: y shape -> {y.shape}")
        '''
        
        pred_1 = pred.view(-1, tokenizer.vocab_size)
        y_1 = y.view(-1)

        '''
        print (f"* train: pred -> {pred_1}")
        print (f"* train: y -> {y_1}")
        print (f"* train: x shape -> {pred_1.shape}")
        print (f"* train: y shape -> {y_1.shape}")
        '''
        
        loss = nn.functional.cross_entropy(pred_1, y_1)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #i = i + 1
        #if i==5:
        #    return
        
        if batch%(int(batch_count/10)) == 0:
            print (f"Training loss: {loss.item()}")


# model testing

def test(test_dload, model, loss_fn):

    model.eval()

    with torch.no_grad():
        count  = 0
        cnt    = 0
        acc    = 0
        loss_v = 0
        i = 0
        for x, y in test_dload:

            x, y = x.to(device), y.to(device)

            pred = model(x)

            pred_1 = pred.view(-1, tokenizer.vocab_size)
            y_1 = y.view(-1)

            loss = nn.functional.cross_entropy(pred_1, y_1)
            #loss = loss_fn(pred, y)

            loss_v = loss_v + loss.item()
            count = count + 1


            for i in range(0, len(pred_1)):
                if y_1[i] != -100:
                    if pred_1[i].argmax() == y_1[i]:
                        acc = acc + 1

                    cnt = cnt + 1
                    #print (f"pred : {pred_1[i]}")
                    #print (f"pred argmax : {pred_1[i].argmax()}")
                    #print (f"y : {y_1[i]}")

            #sys.exit(-1)

    print (f"Testing loss: {loss_v/count}, Accuracy: {acc/cnt}")


# model run    
Epoch=15
for i in range(0, Epoch):
    print (f"Epoch: {i}")
    train(train_dload, model, optimizer, loss_fn)
    test(validation_dload, model, loss_fn)
    print (f"----------")

    torch.save(model_llm_text.state_dict(), './saved_model/model_llm_text_weights_2.pth')
    print (f"--- model saved ! ---\n")

torch.save(model_llm_text.state_dict(), './saved_model/model_llm_text_weights_final.pth')
print (f"--- final model saved ! ---\n")

#model_llm_text.load_state_dict(torch.load('./saved_model/model_llm_text_weights.pth', weights_only=True))
#model.load_state_dict(torch.load('./saved_model/model_llm_text_weights.pth', weights_only=True))

# prediction samples

model.eval()
with torch.no_grad():

    i = 0
    for x, y in validation_dload:

        if i == 10:
            break
        
        x, y = x.to(device), y.to(device)
        pred = model(x)

        pred_1 = pred.view(-1, tokenizer.vocab_size)
        y_1 = y.view(-1)

        for j in range(0, len(pred_1)):
            if y_1[j] != -100:
                #if pred_1[j].argmax() == y_1[j]:
                print (f"{i}: pred: {pred_1[j].argmax()}, y: {y_1[j]}")
        i = i + 1


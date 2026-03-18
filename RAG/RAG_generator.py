import os, sys
import keras
import keras_hub
import numpy as np

import shutil
from keras import ops
import pathlib
import tensorflow as tf

import keras, keras_hub, keras_nlp
from keras import layers

# imports for locally trained llm and sentence transformer usage
#from RAG_retriever1 import BertEncoder
#from RAG_retriever2 import BertEncoder
#from causal_llm_RAG import Causal_LLM, start_packer, tokenizer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List


# use CPU if the causalLLM causes memory not sufficient issue
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Parameters

def get_documents_from_db():
    documents = []
    metadata = []
    ids = []

    with open('./spells.txt', 'r', encoding='utf-8') as spells:
        i = 0
        for ln in spells:
            line = ln.split(" - ")
            documents.append(Document(ids=line[0], metadata={'spell' : line[0]}, page_content=ln))
            i = i + 1

    return documents


def retrive_embeddings_hf():

    # Pre-trained huggingface sentence transformer embeddings
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        encode_kwargs=encode_kwargs
    )

    return embeddings
 

def retrive_embeddings_local():

    # local sentence transformer embeddings

    class Custom_Embeddings:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, docs: List[str]) -> List[List[float]]:
            return self.model(tf.constant(docs)).numpy().tolist()

        def embed_query(self, query: str) -> List[float]:
            return self.model(tf.constant([query]))[0].numpy().tolist()

    sentence_transformer = keras.models.load_model("./sentence_transformer.keras", compile=False)
    embeddings = Custom_Embeddings(sentence_transformer)

    '''
    class Custom_Embeddings_2:
        def __init__(self, model_query, model_docs):
            self.model_query = model_query
            self.model_docs  = model_docs

        def embed_documents(self, docs: List[str]) -> List[List[float]]:
            return self.model_docs(tf.constant(docs)).numpy().tolist()

        def embed_query(self, query: str) -> List[float]:
            return self.model_query(tf.constant([query]))[0].numpy().tolist()


    sentence_transformer_query = keras.models.load_model("./sentence_transformer_query.keras", compile=False)
    sentence_transformer_docs  = keras.models.load_model("./sentence_transformer_docs.keras", compile=False)
    embeddings = Custom_Embeddings_2(sentence_transformer_query, sentence_transformer_docs)
    '''
    
    return embeddings


def retrieve_relevant_docs(documents, query, embeddings):
    
    vector_store = Chroma.from_documents(
        documents = documents,
        embedding = embeddings,
        collection_name = "rag_collection"
    )

    ret_docs = vector_store.similarity_search(query, k=3) # similarity_search_with_score # max_marginal_relevance_search

    docs = []
    print(f"\n###### Retrievd Documents #######\n")
    for i, doc in enumerate(ret_docs):
        print(f"Result {i+1}: {doc.page_content}")
        docs.append(doc.page_content)

    print()

    return docs


# pre-trained llm from kerashub
def llm_inference_kerashub(prompt):
    

    # load pretrained llm from kerashub
    
    #llm = keras_nlp.models.Phi3CausalLM.from_preset("phi3_mini_4k_instruct_en")
    #llm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en") #
    #llm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_4b_text")
    #llm = keras_hub.models.Llama3CausalLM.from_preset("llama3.2_instruct_3b") # Llama usage needs approval
    llm = keras_hub.models.Qwen3CausalLM.from_preset("qwen3_4b_en")

    #llm.backbone.summary()

    inference = llm.generate(prompt, max_length = 200)
    print(f"\n### Inference ###\n\n{inference}")


# local llm
def llm_inference_local(prompt):
    

    # load locally trained llm
    llm  = keras.models.load_model("./causal_llm_RAG.keras", compile=False)

    #llm.summary()


    prompt_tokens = start_packer(tokenizer([prompt]))
    #print (f"prompt -> {prompt_tokens}")

    prompt_tokens_lst = prompt_tokens[0].numpy().tolist()
    prompt_tokens_lst = [x for x in prompt_tokens_lst if x != 0]
    #print (f"prompt lst: {prompt_tokens_lst}")
    print (f"prompt len: {len(prompt_tokens_lst)}")

    def next(prompt, cache, index):
        logits = llm(prompt)[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache


    sampler = keras_hub.samplers.GreedySampler()
    output_tokens = sampler(
        next=next,
        prompt=prompt_tokens,
        index=len(prompt_tokens_lst), 
    )

    txt = tokenizer.detokenize(output_tokens)
    print(f"Greedy search generated text: \n\n{txt}\n")
    

if __name__ == "__main__":

    # Queries
    #query = "How do I build house?"
    #query = "How do I illuminate light?"
    query = "How to attack and damage the enemy in battle?"
    #query = "How to pet an elephant?"

    documents = get_documents_from_db()

    embeddings = retrive_embeddings_hf()
    #embeddings = retrive_embeddings_local()

    documents = retrieve_relevant_docs(documents, query, embeddings)    

    prompt = """spells: {documents} \nuser: {query} \nSuggest a spell based on the recommended spell and the user input. Do not give any additional explanation. Only return one spell.""" #Give clear step-by-step instructions. 

    prompt = prompt.format(documents=documents, query=query)
    print (f"\n### Prompt ###\n\n{prompt}\n")

    llm_inference_kerashub(prompt)
    #llm_inference_local(prompt)



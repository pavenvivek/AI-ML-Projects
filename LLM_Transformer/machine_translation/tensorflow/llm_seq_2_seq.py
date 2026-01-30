import keras_hub
import pathlib
import random
import os, sys

import keras
from keras import ops

import tensorflow.data as tf_data


# hyper-parameters

BATCH_SIZE = 64
EPOCHS = 20  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
SPA_VOCAB_SIZE = 15000

EMBED_DIM = 256
INTERMEDIATE_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 1


# data download and pre-processing

text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file) / "spa-eng" / "spa.txt"

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    eng = eng.lower()
    spa = spa.lower()
    text_pairs.append((eng, spa))


#for _ in range(5):
#    print(random.choice(text_pairs))


random.shuffle(text_pairs)
num_val_samples   = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs       = text_pairs[:num_train_samples]
val_pairs         = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs        = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

def train_word_piece(text_samples, vocab_size, reserved_tokens, filename):

    file_path = pathlib.Path(filename)

    if not file_path.exists():
        print ("File doesn't exists !")

        word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
        keras_hub.tokenizers.compute_word_piece_vocabulary(
            word_piece_ds.batch(1000).prefetch(2),
            vocabulary_size=vocab_size,
            reserved_tokens=reserved_tokens,
            vocabulary_output_file=filename, # if output filename is provided, this function writes the output to the file and returns None 
        )

    abs_path = os.path.abspath(file_path)
    vocab = keras.utils.get_file(origin="file://" + abs_path)

    '''
    vocab_lst = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip whitespace and newline characters, then append the word
            word = line.strip()
            if word: # Ensure the line is not empty
                vocab_lst.append(word)

    print (f"Tokens: {vocab_lst[100:110]}")
    '''
    
    return vocab


reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens, "eng_vocab.txt")

spa_samples = [text_pair[1] for text_pair in train_pairs]
spa_vocab = train_word_piece(spa_samples, SPA_VOCAB_SIZE, reserved_tokens, "spa_vocab.txt")

#print("English Tokens: ", eng_vocab[100:110])
#print("Spanish Tokens: ", spa_vocab[100:110])

eng_tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False
)
spa_tokenizer = keras_hub.tokenizers.WordPieceTokenizer(
    vocabulary=spa_vocab, lowercase=False
)

eng_input_ex = text_pairs[0][0]
eng_tokens_ex = eng_tokenizer(eng_input_ex) #.tokenize(eng_input_ex)
print("English sentence: ", eng_input_ex)
print("Tokens: ", eng_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    eng_tokenizer.detokenize(eng_tokens_ex),
)

spa_input_ex = text_pairs[0][1]
spa_tokens_ex = spa_tokenizer(spa_input_ex) #.tokenize(spa_input_ex)
print("Spanish sentence: ", spa_input_ex)
print("Tokens: ", spa_tokens_ex)
print(
    "Recovered text after detokenizing: ",
    spa_tokenizer.detokenize(spa_tokens_ex),
)


def preprocess_batch(eng, spa):
    eng = eng_tokenizer(eng)
    spa = spa_tokenizer(spa)

    # Pad `eng` to `MAX_SEQUENCE_LENGTH`.
    eng_start_end_packer = keras_hub.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=eng_tokenizer.token_to_id("[PAD]"),
    )
    eng = eng_start_end_packer(eng)

    # Add special tokens (`"[START]"` and `"[END]"`) to `spa` and pad it as well.
    spa_start_end_packer = keras_hub.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=spa_tokenizer.token_to_id("[START]"),
        end_value=spa_tokenizer.token_to_id("[END]"),
        pad_value=spa_tokenizer.token_to_id("[PAD]"),
    )
    spa = spa_start_end_packer(spa)

    #print (f"eng -> {eng}")
    #print (f"spa -> {spa}")
    #print (f"spa[:,:-1] -> {spa[:,:-1]}")
    #print (f"spa[:,1:] -> {spa[:,1:]}")

    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )

#print ("Sample input:")
#preprocess_batch([text_pairs[0][0], text_pairs[1][0]], [text_pairs[0][1], text_pairs[1][1]])


def make_dataset(pairs):

    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset   = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset   = dataset.batch(BATCH_SIZE)
    dataset   = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)

    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds   = make_dataset(val_pairs)

print (f"training data: ")
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

    print (f"enc: {inputs['encoder_inputs']}, dec: {inputs['decoder_inputs']}, target: {targets}")


# model construction

# Functional


def build_LLM_Seq_to_Seq():
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=ENG_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(encoder_inputs)


    for _ in range(NUM_LAYERS):
        x = keras_hub.layers.TransformerEncoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(x)

    #print (f"encoder x.shape -> {x.shape}")    
    encoder_outputs = x    
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    # Decoder
    decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")

    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=SPA_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(decoder_inputs)

    for _ in range(NUM_LAYERS):
        x = keras_hub.layers.TransformerDecoder(
            intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
        )(decoder_sequence=x, encoder_sequence=encoder_outputs) #encoded_seq_inputs)

    #print (f"decoder x.shape -> {x.shape}")    
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(SPA_VOCAB_SIZE, activation="softmax")(x)
    #print (f"decoder_outputs.shape: {decoder_outputs.shape}")

    decoder = keras.Model(
        [
            decoder_inputs,
            encoder_outputs,
        ],
        decoder_outputs,
    )

    transformer = keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="transformer",
    )

    return transformer


# Subclass

@keras.saving.register_keras_serializable()
class LLM_Seq_to_Seq(keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tok_and_pos_enc = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=ENG_VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM)
        self.tok_and_pos_dec = keras_hub.layers.TokenAndPositionEmbedding(vocabulary_size=SPA_VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM)

        self.trs_enc     = [keras_hub.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS) for _ in range(NUM_LAYERS)]
        self.trs_dec     = [keras_hub.layers.TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS) for _ in range(NUM_LAYERS)]

        self.out =  keras.layers.Dense(SPA_VOCAB_SIZE, activation="softmax")
        
        self.drp = keras.layers.Dropout(0.5)
        
    def call(self, inputs):

        enc_inp = inputs["encoder_inputs"]
        dec_inp = inputs["decoder_inputs"]
        
        x1 = self.tok_and_pos_enc(enc_inp)

        for i in range(NUM_LAYERS):
            x1 = self.trs_enc[i](x1)

        x2 = self.tok_and_pos_dec(dec_inp)
        
        for i in range(NUM_LAYERS):
            x2 = self.trs_dec[i](decoder_sequence=x2, encoder_sequence=x1)

        x   = self.drp(x2)
        out = self.out(x)

        return out


#transformer = build_LLM_Seq_to_Seq()
transformer = LLM_Seq_to_Seq()

transformer.summary()
print ("Dummy Input")
for inputs, targets in train_ds.take(1):
    output = transformer(inputs)
    print (f"output shape: {output.shape}")

#sys.exit(-1)

transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"] #, run_eagerly=True
)

transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

#transformer.save("llm_seq_to_seq.keras")
#transformer = keras.models.load_model("llm_seq_to_seq.keras", compile=False)


# Evaluation

def translate(input_sentences, sample_alg):

    batch_size = 1
    length     = MAX_SEQUENCE_LENGTH

    # Tokenize the encoder input.
    encoder_input_tokens = ops.convert_to_tensor(
        eng_tokenizer(input_sentences), sparse=False, ragged=False
    )
    if ops.shape(encoder_input_tokens)[1] < MAX_SEQUENCE_LENGTH:
        pads = ops.zeros(
            (1, MAX_SEQUENCE_LENGTH - ops.shape(encoder_input_tokens)[1]),
            dtype=encoder_input_tokens.dtype,
        )
        encoder_input_tokens = ops.concatenate([encoder_input_tokens, pads], 1)

        
    def next(prompt, cache, index):
        logits = transformer({"encoder_inputs": encoder_input_tokens, "decoder_inputs" : prompt})[:, index - 1, :]
        hidden_states = None

        return logits, hidden_states, cache

    
    # Build a prompt of length 40 with a start token and padding tokens.
    start  = ops.full((batch_size, 1), spa_tokenizer.token_to_id("[START]"))
    pad    = ops.full((batch_size, length - 1), spa_tokenizer.token_to_id("[PAD]"))
    prompt = ops.concatenate((start, pad), axis=-1)
    #print (f"prompt: {prompt}")


    if sample_alg == "Greedy":
        sampler = keras_hub.samplers.GreedySampler()
    elif sample_alg == "TopP": # TopP not working. Why?
        sampler = keras_hub.samplers.TopPSampler(p=0.3,k=100)
        
    output_tokens = sampler(
        next=next,
        prompt=prompt,
        stop_token_ids=[spa_tokenizer.token_to_id("[END]")],
        index=1,
    )
    
    generated_sentences = spa_tokenizer.detokenize(output_tokens)
    
    return generated_sentences


test_eng_texts = [pair[0] for pair in test_pairs]
for i in range(2):

    input_sentence = random.choice(test_eng_texts)

    translated = translate([input_sentence], "Greedy")[0]
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )

    print(f"** Example {i} **")
    print(input_sentence)
    print(translated)
    print()




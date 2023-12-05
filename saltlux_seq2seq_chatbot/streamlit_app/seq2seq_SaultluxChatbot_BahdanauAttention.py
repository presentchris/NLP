import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import matplotlib.pyplot as plt
import inspect
import json

import re
from tqdm import tqdm
from konlpy.tag import Okt


## Data 경로 설정
DATA_IN_PATH = './data_in/csv_short/'
DATA_OUT_PATH = './data_out/csv_short/'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIGS = 'data_configs.json'

## 파일 일기
index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUTS, 'rb'))
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUTS , 'rb'))
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS , 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))


MODEL_NAME = 'seq2seq_kor_Attention'
BATCH_SIZE = 2
MAX_SEQUENCE = 25
EPOCH = 30
UNITS = 1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1 
char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']
std_index = prepro_configs['std_symbol']
end_index = prepro_configs['end_symbol']
vocab_size = prepro_configs['vocab_size'] # ['<PAD>\n', '<SOS>\n', '<END>\n', ... ,

print(char2idx) # {'<PAD>': 0, '<SOS>': 1, '<END>': 2, '<UNK>': 3,
print(idx2char) # {'0': '<PAD>', '1': '<SOS>', '2': '<END>', '3': '<UNK>', '4': '혼자인게', '5':  숫자 결과를 문장으로 변환할 때
print(std_index) # <SOS>
print(end_index) # <END>
## Encoder
class Encoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz # 2
        self.enc_units = enc_units # 1024
        self.vocab_size = vocab_size # 111
        self.embedding_dim = embedding_dim # 256
        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        
    def call(self, x, hidden):
        x = self.embedding(x) # x shape = (20, 25)
        # x shape = (20, 25, 256)
        
        # w/o attention
        # _, state = self.gru(x, initial_state=hidden)

        # w attention
        output, state = self.gru(x, initial_state=hidden)        
        return output, state 
        # output = (20, 25, 1024), state shape = (20, 1024)
    
    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0], self.enc_units)) # inp shape = [20, 25], self.enc_units = 1024
    

    ## Bahdanu's Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units): # units = 1024
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) # decoder units
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values): # query: hidden from dec, values: enc_output
        # 스코어(score)계산을 위해 덧셈을 수행하고자 시간 축을 확장하여 아래의 과정을 수행합니다.
        query_with_time_axis = tf.expand_dims(query, 1) 
        # query_with_time_axis = (batch_size, 1, hidden size)
        # values = (batch_size, max_len, hidden size)
        
        # score는 (batch_size, max_length, 1)쌍으로 이루어져 있습니다.
        # score를 self.V에 적용하기 때문에 마지막 축에 1을 얻습니다.
        # self.V에 적용하기 전에 텐서는 (batch_size, max_length, units)쌍으로 이루어져 있습니다.
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values # weighted sum of values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
        

## Decoder
class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(self.vocab_size)
        
        # w attention
        self.attention = BahdanauAttention(self.dec_units)


     
    def call(self, x, hidden, enc_output):
        # hidden = query, enc_output = values
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        x = self.embedding(x) #  x dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32) shape =  (20, 1)
        # x shape = (20, 1, 256)
        
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # x shape = (batch_size, 1, embedding + context_vector)
        
        output, state = self.gru(x) # state = dec_hidden
        # output shape = (20, 1, 1024), state shape = (20, 1024)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)
        # x shape = (20, 111)
        
        return x, state, attention_weights
    
## seq2seq 모델
class seq2seq(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, 
                 batch_sz, end_token_idx = 2):
        super(seq2seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_sz)
        
    def call(self, x):
        inp, tar = x # [index_inputs, index_outputs]
        
        end_hidden = self.encoder.initialize_hidden_state(inp) # 20 x 25 (sentence x MAX_SEQUENCE)
        # end_hidden =  zeros (20, 1024 (enc_units)) 
        enc_output, enc_hidden = self.encoder(inp, end_hidden)  # enc_state (20, 1024) only last(hidden) state
        dec_hidden = enc_hidden

        predict_tokens = list()
        for t in range(0, tar.shape[1]): # tar.shape = (20, 25 (MAX_LENGTH))
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:, t], 1), tf.float32) # tar[:, t].shape = (20,  )
            # dec_input shape = (20, 1)
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            # predictions shape = (20, 111), dec_hidden = (20, 1024)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))
            # predict_tokens shape = (25, 20, 111)

        predicted = tf.stack(predict_tokens, axis=1) #   
        return predicted # tf.Tensor: shape=(20, 25, 111)
        
    def inference(self, x):
        inp = x
       
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        # enc_hidden_init size = (, 1024)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([char2idx[std_index]], 1)
        
        predict_tokens =  list()
        for t in range(0, MAX_SEQUENCE):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0]) # predictions shape =  (1, 111)
            
            if predict_token == self.end_token_idx:
                break

            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token], 0), tf.float32)
            
        return tf.stack(predict_tokens, axis=0).numpy()
    

## Loss Function
# optimizer = tf.keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction ='none')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
###########
# real = [1, 3]
# pred = [[0.05, 0.9, 0, 0.05], 
#           [0.1, 0.8, 0.05, 0.05]]
###########

def loss(real, pred): # real shape = (20, 25), pred shape = (20, 25, 111)
    mask = tf.math.logical_not(tf.math.equal(real, 0)) # real 값이 0 이면 True => tf.math.logical_not => False
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)



## 결과 확인
# model = seq2seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE, char2idx[end_index])
# model.compile(loss = loss, optimizer=keras.optimizers.Adam(1e-3))
# SAVE_FILE_NM = "weights.h5"
# model.load_weights(os.path.join(DATA_OUT_PATH, MODEL_NAME, SAVE_FILE_NM))


## 결과 확인
FILTERS = "([~.,!?\"':;)(])"
CHANGE_FILTER = re.compile(FILTERS)


PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"


def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        sequence_index = []
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]

        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length


model = seq2seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE, char2idx[end_index])
model.compile(loss = loss, optimizer=keras.optimizers.Adam(1e-3))
model.build([(20, 25), [20, 25]])
model.load_weights("./data_out/csv_short/seq2seq_kor_BahdanauAttention/weights_attention_pretrained.h5")

def answer(query):    
    
    test_index_inputs, _ = enc_processing([query], char2idx) 
    # print(test_index_inputs)
    predict_tokens = model.inference(test_index_inputs)
    # print(predict_tokens)
    answer = ' '.join([idx2char[str(t)] for t in predict_tokens])

    return answer

# query = "남자친구가 나 안 믿어줘"

# query = "남자친구 교회 데려가고 싶어"


# test_index_inputs, _ = enc_processing([query], char2idx) 
# print(test_index_inputs)

# predict_tokens = model.inference(test_index_inputs)
# print(predict_tokens)

# print(' '.join([idx2char[str(t)] for t in predict_tokens]))
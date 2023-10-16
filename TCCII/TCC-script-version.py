#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import sys
import os
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from sklearn.utils import shuffle
import nltk
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from keras.layers import Input, GRU, Dense, Embedding
from keras.utils import  pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from sklearn.model_selection import train_test_split

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# ### Loading the Dataset
# Here we load the data from the `dataset.csv` file (generated in the other script)

# In[2]:


arguments = sys.argv
filename = arguments[1] if len(arguments) > 1 else 'dataset-5k-1'
print("Working on: ", f'data/fraction/{filename}.csv')

# In[3]:


def load_data():
    return pd.read_csv(f'data/fraction/{filename}.csv')


# In[4]:


def load_vocabulary():
    vocabulary = list()
    with open('data/vocab-30k.txt', encoding='utf-8') as f:
        for line in f:
            vocabulary.append(line.strip())
    return vocabulary


# ### Data pre-processing
# Transform to lower, remove the new line and the punctuation

# In[5]:


def lower_data(data):
    return data.str.lower() 
    
def clean_data(data):
    return data.str.replace(',', ' , ')                \
                .str.replace('.',' . ', regex=False)  \
                .str.replace('?',' ? ', regex=False)   \
                .str.replace("''",' ', regex=False)   \
                .str.replace(r"(\s)'|'(\s)",' ', regex=True) \
                .str.replace(r"[^a-zA-Z0-9?'.,]+",' ',regex=True)

def get_data():
    data = load_data()
    for column in data.columns:    
        data[column] = lower_data(data[column])
        data[column] = clean_data(data[column])
    return shuffle(data)


# In[6]:


def remove_outliers(data):
    return data[(data['question'].str.len() < 100) & (data['answer'].str.len() < 200)]

def padd_data(data):
    data = data.assign(question = '<start> ' + data.question  + ' <end>')
    data = data.assign(answer = '<start> ' + data.answer  + ' <end>')
    return data


# ### Creating the dataset
# Removing the outliers and adding <start> and <end> for each question, awnser pair

# In[7]:


def create_dataset(num_examples):
    dataset = remove_outliers(get_data())
    dataset = padd_data(dataset)
    return dataset['question'].tolist(), dataset['answer'].tolist()


# ### Tokenizing 
# Tokenize the data, padd the sequence and create the vocabulary

# In[8]:


def load_tokenize(vocabulary):
    tokenizer = Tokenizer(filters='!"#$%&()*+-:;=@[\\]^_{|}~\t')
  
    # Convert sequences into internal vocab
    tokenizer.fit_on_texts(vocabulary)

    return tokenizer


# In[9]:


def tokenize(text, tokenizer):

    # Convert internal vocab to numbers
    tensor = tokenizer.texts_to_sequences(text)

    # Pad the tensors to assign equal length to all the sequences
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', truncating='post',maxlen=None)

    return tensor


# ### Load the clean and formated data 

# In[10]:


def load_dataset(num_examples=None):
 
    questions, answers = create_dataset(num_examples=None)
    vocabulary = load_vocabulary()
    
    #Create the tokenizer for inputs and outputs
    tokenizer = load_tokenize(vocabulary)
    
    questions_tensor = tokenize(questions, tokenizer)
    answers_tensor = tokenize(answers, tokenizer)

    return questions_tensor, answers_tensor, tokenizer


# In[11]:


questions_tensor, answers_tensor, tokenizer = load_dataset()


# ### Split in train and test
# Split 80% of the data to train and 20% for testing

# In[12]:


max_length_input, max_length_target = questions_tensor.shape[1], answers_tensor.shape[1]
input_train, input_test, target_train, target_test = train_test_split(questions_tensor, answers_tensor, test_size=0.2)

print("Train count:", len(input_train))
print("Test count:", len(input_test))


# ### Setting the hyperparameter

# In[13]:


BUFFER_SIZE = len(input_train)
BATCH_SIZE = 8
steps_per_epoch = len(input_train)//BATCH_SIZE
steps_per_epoch_test = len(input_test)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_input_size = len(tokenizer.word_index) + 1
vocab_target_size = len(tokenizer.word_index) + 1


# In[14]:


dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset_test = tf.data.Dataset.from_tensor_slices((input_test, target_test)).shuffle(BUFFER_SIZE)
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)

# In[15]:


example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


# In[16]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units

        # Embed the vocab to a dense embedding 
        self.embedding = Embedding(vocab_size, embedding_dim)

        # GRU Layer
        # glorot_uniform: Initializer for the recurrent_kernel weights matrix, 
        # used for the linear transformation of the recurrent state
        self.gru = GRU(self.encoder_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # Encoder network comprises an Embedding layer followed by a GRU layer
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # To initialize the hidden state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))


# In[17]:


encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


# In[18]:


# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # values shape == (batch_size, max_len, hidden size)

        # we are doing this to broadcast addition along the time axis to calculate the score
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[19]:


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


# In[20]:


# Decoder class
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # x shape == (batch_size, 1)
        # hidden shape == (batch_size, max_length)
        # enc_output shape == (batch_size, max_length, hidden_size)

        # context_vector shape == (batch_size, hidden_size)
        # attention_weights shape == (batch_size, max_length, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


# In[21]:


decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


# In[22]:


# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# Loss function
def loss_function(real, pred):

    # Take care of the padding. Not all sequences are of equal length.
    # If there's a '0' in the sequence, the loss is being nullified
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[23]:


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    # tf.GradientTape() -- record operations for automatic differentiation
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        # dec_hidden is used by attention, hence is the same enc_hidden
        dec_hidden = enc_hidden

        # <start> token is the initial decoder input
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):

            # Pass enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            # Compute the loss
            loss += loss_function(targ[:, t], predictions)

            # Use teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    # As this function is called per batch, compute the batch_loss
    batch_loss = (loss / int(targ.shape[1]))

    # Get the model's variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # Compute the gradients
    gradients = tape.gradient(loss, variables)

    # Update the variables of the model/network
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# In[24]:

@tf.function
def test_step(inp, targ, enc_hidden):

    test_loss = 0

    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # dec_hidden is used by attention, hence is the same enc_hidden
    dec_hidden = enc_hidden

    # <start> token is the initial decoder input
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):

        # Pass enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        # Compute the loss
        test_loss += loss_function(targ[:, t], predictions)

        # Use teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    # As this function is called per batch, compute the test_batch_loss
    test_batch_loss = (test_loss / int(targ.shape[1]))

    return test_batch_loss


# In[25]:


checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10)
checkpoint.restore(manager.latest_checkpoint)



# In[26]:


EPOCHS = 30


infoLog = open("info.txt", "a")
infoLog.write('File {}.csv\n'.format(filename))
infoLog.write('Starting {}\n'.format(datetime.now()))
infoLog.close()
        
        
# Training loop
with tf.device('/cpu:0'):


    for epoch in range(EPOCHS):

        # ============================= TRAIN PHASE ==================================

        # Initialize the hidden state
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        # Loop through the dataset
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

            # Call the train method
            batch_loss = train_step(inp, targ, enc_hidden)

            # Compute the loss (per batch)
            total_loss += batch_loss
            
        # Save (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            manager.save()

        # Save the the loss in a file
        lossLog = open("loss.txt", "a")
        lossLog.write('{:.4f}\n'.format(total_loss / steps_per_epoch))
        lossLog.close()
        
        
        # Output the loss observed until that epoch
        print('Train Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))


        # ============================= TEST PHASE ==================================

        # Initialize the hidden state
        enc_test_hidden = encoder.initialize_hidden_state() 
        total_test_loss = 0

        for (batch_test, (inp_test, targ_test)) in enumerate(dataset_test.take(steps_per_epoch_test)):

            # Call the test method
            batch_test_loss = test_step(inp_test, targ_test, enc_test_hidden)

            # Compute the loss (per batch)
            total_test_loss += batch_test_loss
        
        # Save the the loss in a file
        testLossLog = open("test_loss.txt", "a")
        testLossLog.write('{:.4f}\n'.format(total_test_loss / steps_per_epoch_test))
        testLossLog.close()            


    infoLog = open("info.txt", "a")
    infoLog.write('Last Train Execution Loss {:.4f}\n'.format(total_loss / steps_per_epoch))
    infoLog.write('Last Test Execution Loss {:.4f}\n'.format(total_test_loss / steps_per_epoch_test))
    infoLog.write('Ended {}\n'.format(datetime.now()))
    infoLog.write('-----------------------------------------------------------------------\n')
    infoLog.close()


import collections
import logging
import os
import pathlib
import re                      # regex regular expressions here 
import string
import sys
import os
import time

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

import data_preprocess 


def build_model(vocab_size, embedding_dim, rnn_units):       # A single Layered LSTM model seq2seq task here 
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(
            rnn_units, return_sequences=True),         # One could also try the Bidirectional LSTM in here :: 
        tf.keras.layers.Dense(vocab_size)        # Note that above, return sequences is True : meaning output at all time steps 
        # is considered here 
    ])
    return model


def train(file_path):

	ds = data_preprocess.data_preprocess(file_path)     # Get the dataset here 



	charset_size = len(char_array)
	embedding_dim = 256
	rnn_units = 512                      # It would return a sequence for each of the characters (64,40,512) thats it 

	tf.random.set_seed(1)

	model = build_model(
	    vocab_size = charset_size,
	    embedding_dim=embedding_dim,
	    rnn_units=rnn_units)

	# model.summary()

	model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True                                  # Because softmax is not used??
    ))

	model.fit(ds, epochs=20)

	return model 




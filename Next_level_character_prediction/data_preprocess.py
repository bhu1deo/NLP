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

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)


# Here the data preprocessing is done w.r.t Jules Verne's Mysterious Island txt file :: 

def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


def data_preprocess(file_path):
	with open(file_path, 'r', encoding='utf8') as fp:
    text=fp.read()
    
	start_indx = text.find('THE MYSTERIOUS ISLAND')
	end_indx = text.find('End of the Project Gutenberg')            # Rest of the material is logistics ignore it please 
	print(start_indx, end_indx)

	text = text[start_indx:end_indx]
	char_set = set(text)

	# Note that we should convert everything into tf datasets and batchdatasets in order to feed to the model here :: 

	chars_sorted = sorted(char_set)
	char2int = {ch:i for i,ch in enumerate(chars_sorted)}
	char_array = np.array(chars_sorted)

	text_encoded = np.array(
	    [char2int[ch] for ch in text],
	    dtype=np.int32)

	ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

	seq_length = 40                       # hyperparameter 
	chunk_size = seq_length + 1

	ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)

	ds_sequences = ds_chunks.map(split_input_target)     # Each batch would now be split in here 

	BATCH_SIZE = 64
	BUFFER_SIZE = 10000

	tf.random.set_seed(1)
	ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)# drop_remainder=True)

	return ds



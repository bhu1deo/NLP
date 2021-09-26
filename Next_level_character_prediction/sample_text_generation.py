# Load the trained model, get the start seed, and then generate the text 
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

import train



def sample(file_path,starting_str, 
           len_generated_text=500, 
           max_input_length=40,
           scale_factor=1.0):               # Samples the next step predictions from the model here:: 

	# file_path = 
    
	model = train.train(file_path)             # The Trained model here       

    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))

    generated_str = starting_str



    model.reset_states()                # the internal RNN states have been reset
    
    for i in range(len_generated_text):
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)

        scaled_logits = logits * scale_factor
        new_char_indx = tf.random.categorical(
            scaled_logits, num_samples=1)                       # Sample from this distribution in here
        
        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()        # index for new character in here    

        generated_str += str(char_array[new_char_indx])
        
        new_char_indx = tf.expand_dims([new_char_indx], 0)
        encoded_input = tf.concat(
            [encoded_input, new_char_indx],
            axis=1)
        encoded_input = encoded_input[:, -max_input_length:]             # append to the current input that is being fed to the model in here 

    return generated_str


if __name__ == '__main__':
	file_path = "D:/datasets/NLP_project/jules_verne_mysterious_island.txt"
	sample(file_path,"Above")


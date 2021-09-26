#!/usr/bin/env python
# coding: utf-8

# In[5]:


# We implement the Generalized Linformer in here :: As we did for the Generalized Transformer 
# This would be tested on the next level character prediction dataset in here :: 

# this would be used in char level prediction 
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


# In[6]:


def scaled_dot_product_attention_linformer(q, k, v, mask):        # autoreg mask padding of q,k,v is already done here
    
    # same as the transformer module in here :: 
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# In[7]:


# The MHA Block is very different from the standard Transformer in here :: 
# This accepts queries Keys and values here 

# Takes in values keys queries adn their masks and returns multi-head attention here :: 


class MultiHeadAttention_linformer(tf.keras.layers.Layer):             # So this is a multihead attention layer superclass is the Keras layer here 
    
    # Take in the padding mask passed to the transformer :: based on raw inputs here 
    
    # Embed -> Compress -> Mask The inputs would already be embedded :: One needs to compress and then use the padding mask here 
    
    # Make an autoreg mask if necessary :: decoder first attention block requires this 
    
    # The autoreg mask is required only in the Decoder First attention head and not anywhere else 
    
    # seq_len_k = seq_len_v required for dimension reduction in here :: k is the reduced dimension in here :: whether autoreg or not  
    
    def __init__(self, seq_len_k,d_model, num_heads,k,autoreg):
        super(MultiHeadAttention_linformer, self).__init__()
        self.num_heads = num_heads           # Learn rich features across multiple heads here 
        # the embedding dimension vector is broken down into num_heads and then the attention map for each dimension 
        # is computed here 
        self.d_model = d_model               # This is either the Embedding Layer dimension of the transformed model dimension here 

        assert d_model % self.num_heads == 0          # The number of heads should be a factor of d_model here 

        self.depth = d_model // self.num_heads        # Number of attention heads here 

        self.wq = tf.keras.layers.Dense(d_model)      # The three transformations here 
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
        self.k = k                # The output compression dimension here 
        
        # The Compression is not that easy here :: 
        
        self.init = tf.keras.initializers.GlorotNormal()

        self.conv_layer_k = tf.keras.layers.Conv1D(
            self.d_model, seq_len_k-(self.k-1), strides=1, padding='valid'
        )                 # Num filters, kernel size 

        self.lin_layer_k = tf.keras.layers.Dense(d_model,kernel_initializer = self.init)
        
        self.conv_layer_v = tf.keras.layers.Conv1D(
            d_model, seq_len_k-(self.k-1), strides=1, padding='valid'
        )
        # The 1D conv layer can also change the num_filters but here it is kept the same 
        # As we don't want to change the embedding dimension here 
        

        self.lin_layer_v = tf.keras.layers.Dense(d_model,kernel_initializer = self.init)
        
        self.autoreg = autoreg                 # Chooses whether an autoregressive mask is required or not 

        
        
    def create_autoreg_mask(self,batch_dim,num_heads,query_len,k):
        # This is of the size of the attention matrix which would be generated here :: approximate mask 
        temp = tf.ones([batch_dim,num_heads,query_len,k],tf.float32)               # Our mask would be NxK here N>K K is a hyperparameter here 

        lowertrig = temp - tf.linalg.band_part(temp,-1,0)       # Lower triangular autoreg maks here
        
        return lowertrig                # The autoreg mask here 
    
    # In the scaled dot product attention mechanism :: the autoreg mask is used to add large negative 
    # value to the computed attention weights so take care of the 1's and 0's in the autoreg mask here 
    # The future values should be masked not the previous values hence the subtraction here 
    
        
        

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).           d_model is split here 
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth) # Maybe done for parallelizibility here 
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))          # self.depth has already been calculated in the constructor here 
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, query_mask,key_mask,value_mask):                # Forward pass here ::  The mask here is the padding mask 
        # autoreg mask is used in scaled dot product attention :: it is generated and then passed to the scaled dot product 
        # attention module here 
        
        batch_size,query_len = tf.shape(q)[0],tf.shape(q)[1]
        
        # The Embedded vectors are passed in here not the raw tokens :: but the padding mask 
        # is computed on the raw tokens 
        
        
        # Each of the masks is BxSxD  :: mask are computed based on raw token values and then passed here 
        # Sometimes here the mask would be None :: for ex. the Decoder second attention head doesn't require any masks 
        
        if(query_mask is not None):
            q = q*query_mask
        if(key_mask is not None):
            k = k*key_mask
        
        if(value_mask is not None):
            v = v*value_mask                 # Note that this is elementwise multiplication in here O(NxD) for one item 
         

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
#         print(k.shape)
#         print(v.shape)
        
        k,v = self.lin_layer_k(self.conv_layer_k(k)),self.lin_layer_v(self.conv_layer_v(v))             # Dimension reduction here 
        
        # Note that the head splitting is done after the conv1D operation of compression here 
        
#         print("Key\n")
#         print(k.shape)
#         print("Value\n")
#         print(v.shape)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)           # After summing up here 
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)         # Parallelizable forget the first 2 dimesnions here
        
        
        if(self.autoreg):
            autoreg_mask = self.create_autoreg_mask(batch_size,self.num_heads,query_len,self.k)
        else:
            autoreg_mask = None
            

        # Okay One More problem here :: 
        
        
        scaled_attention, attention_weights = scaled_dot_product_attention_linformer(
            q, k, v, autoreg_mask)
        
        # So hereon the num_heads is also treated as the batch dimension here 
        
        
        
        
        # Wherever we have a mask there we give large negative output to the softmax :: so that it's output is 0 
        
        

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# In[8]:





# In[9]:


# So use this MHA and scaled dot product attention 

# Then do normal encoder-decoder transformer thing here :: 

# dff is the internal layer num of nodes here 

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)     # intermediate embedding size here 
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64,100, 50, 256))).shape        # The first three dimensions are teated as a batch here !!!   


# In[10]:


# The Encoder is NOT autoregressive :: Only the first decoder layer is autoregressive in here :: 
# The Padding mask needs to be passed :: dimension is BxseqlenxD :: as this is self attention in here :: 
# seq_len_k = seq_len_v is also taken in as an argument here :: 

class EncoderLayer_linformer(tf.keras.layers.Layer):
    def __init__(self, seq_len_k,d_model, num_heads,compressing_dim, dff, enc_autoreg, rate=0.1):
        super(EncoderLayer_linformer, self).__init__()

        self.mha = MultiHeadAttention_linformer(seq_len_k,d_model, num_heads,compressing_dim,autoreg=enc_autoreg)
        self.ffn = point_wise_feed_forward_network(d_model, dff)           # Multi head attention followed by pointwise feed-forward neural network here 

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):                # This is self attention here :: only mask is padding mask in here 
        # which is computed on the raw tokens in the transformer class 
        
        # Mask needs to be constructed on raw tokens before the embedding happens here  
        # For the encoder query key and value are all the same :: mask is basically the padding mask here
        # v,k,q in MHA here :: 

        attn_output, _ = self.mha(x, x, x, mask, mask, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
#         print(attn_output.shape)
#         print(x.shape)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)     # dropout followed by residual connection followed by layer normalization here 

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# In[12]:


class DecoderLayer_linformer(tf.keras.layers.Layer):
    
    def __init__(self, seq_len_k,seq_len_enc,d_model, num_heads,compressing_dim, dff, rate=0.1):
        super(DecoderLayer_linformer, self).__init__()

        self.mha1 = MultiHeadAttention_linformer(seq_len_k,d_model, num_heads,compressing_dim,autoreg=True)   # autoreg + padding here 
        self.mha2 = MultiHeadAttention_linformer(seq_len_enc,d_model, num_heads,compressing_dim,autoreg=False)  # only padding here 

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,mask):        # This mask is just the padding mask in here ::
        
        
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        
        # dimensions of x need to be seen here 

        # masked multihead self attention with lookahead mask to preserve autoregressive nature here 
#         print("attn1\n")
        attn1, attn_weights_block1 = self.mha1(x, x, x, mask,mask,mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # The complete encoder output is taken as an input here :: only padding mask here :: complete encoder output used for 
        # computing the attention weights here :: The encoder output would act as the key and value here 
        # this would mean passing it's sequence length as input argument here 
        
#         print(out1.shape)
#         print(enc_output.shape)
#         print("attn2\n")
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None,None,None)  # (batch_size, target_seq_len, d_model)
        
        
        
        # Here v,k,q is passed hence target_length need not be equal to the input sequence length here 
        # Decoder output acts a query always 

        
        
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


# In[14]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# In[15]:


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],         # np.newaxis is simply used to increase the dimension here 
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)          # cast the tensor into a new type here 


# In[17]:



class Encoder_linformer(tf.keras.layers.Layer):
    
    def __init__(self, seq_len_k,num_layers, d_model, num_heads,compress_dim, dff, input_vocab_size,enc_autoreg,
               maximum_position_encoding, rate=0.1):
        
        super(Encoder_linformer, self).__init__()

        self.d_model = d_model                         # Embedding dimension here 
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)                # As done in the original model here 
        
        # Positional encoding same as the adding an external embedding here 

        self.enc_layers = [EncoderLayer_linformer(seq_len_k,d_model, num_heads,compress_dim, dff,enc_autoreg, rate)
                           for _ in range(num_layers)]                            # N encoder layers here 
#         (self, seq_len_k,d_model, num_heads,compressing_dim, dff, rate=0.1)

        self.dropout = tf.keras.layers.Dropout(rate)     

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]       # Note that in the encoder the query key and the value are all the same 
        

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))                 # Why is root(depth) multiplied here ?? 
#         print(x.shape)
#         print(self.pos_encoding.shape)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)              # One by one passed through the layers here 

        return x  # (batch_size, input_seq_len, d_model)
    
# The compressing dimension and the seq length need to be passed from the 
# Transformer class here 

    


# In[18]:



class Decoder_linformer(tf.keras.layers.Layer):
    
    def __init__(self, seq_len_k,seq_len_enc,num_layers, d_model, num_heads,compress_dim, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        
        super(Decoder_linformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer_linformer(seq_len_k, seq_len_enc, d_model, num_heads,compress_dim, dff, rate=0.1)
                           for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           mask):
        
        # One should note that only the first decoder attention head has a requirement of mask :: and here 
        # query key value is the same so only one mask argument would suffice 
        # the second attention head we do not require a mask :: 
        
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
#         x, enc_output, training,mask
#         print("Start decoder layer here \n")
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# In[19]:


# Now the transformer class for the Linformer in here :: 

class Transformer_linformer(tf.keras.Model):
    # The encoder and the decoder input sequence lengths are the same as the max sequence length across the dataset 
    
    def __init__(self, seq_len_enc_inp,seq_len_dec_inp,compress_dim,num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, encoder_autoreg, pe_input, pe_target, rate=0.1):
        
        # The pe vars are associated with the Positional encoding here :: 
        
        super(Transformer_linformer, self).__init__()
        
#         (self, seq_len_k,num_layers, d_model, num_heads,compress_dim, dff, input_vocab_size,
#                maximum_position_encoding, rate=0.1)      for the Encoder here

        self.encoder = Encoder_linformer(seq_len_enc_inp,num_layers, d_model, num_heads,compress_dim, dff,
                                 input_vocab_size, encoder_autoreg, pe_input, rate)
    
        # Now for the decoder here :: 
        # (self, seq_len_k,seq_len_enc,num_layers, d_model, num_heads,compress_dim, dff, target_vocab_size,
#                maximum_position_encoding, rate=0.1):

        self.decoder = Decoder_linformer(seq_len_dec_inp,seq_len_enc_inp,num_layers, d_model, num_heads,compress_dim, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)   # Logits over the target vocabulary size here 
        
        # The encoder output seq len used by the decoder is also = to the max inp seq len here 
        

        
    # The two padding masks and then the input and the target here 
    
    def call(self, inp, tar, training, enc_padding_mask,
            dec_padding_mask):      # No need of the lookahead mask here :: 
        # as the autoregressive nature is taken care of in the MHA itself here :: 
        
#         print("Encoder\n")
        # x training mask (self, x, training, mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # Encoder just requires the padding mask as it is not autoregressive :: and loss should be 0 for 0 padded tokens here
        
#         print("Decoder\n")

        # dec_output.shape == (batch_size, tar_seq_len, d_model) (self, x, enc_output, training, mask):
        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, dec_padding_mask)
        
#         print("Final logits\n")

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# In[20]:





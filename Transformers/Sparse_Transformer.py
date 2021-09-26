# We implement the general transformer architecture in here similar to the linformer :: 
# The q,k,v would be masked before feeding to the trans :: and the autoreg would be done in the scaled dot prod attention 
# Later convert this into a .py file and then imprort wherever you want 


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
from tf_slice_assign import slice_assign
import time 

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)

def scaled_dot_product_attention(q, k, v, layout_length, mask):        # autoreg mask, padding of q,k,v is already done here
    
    batch,num_heads,seq_len_queries,query_dim = q.shape        # Query and Key Dims are same :: Key and Value seq lengths are same 
    
    batch,num_heads,seq_len_keys,key_dim = k.shape

    batch,num_heads,seq_len_val,val_dim = v.shape
    
    k_transpose = tf.transpose(k, perm=[0, 1, 3, 2])
    
    value_sparse = tf.Variable(tf.zeros((batch,num_heads,layout_length,val_dim), dtype=tf.float32),trainable=True)
    
    # layout_length*num_heads = seq_len_keys :: for every head then the values also need to be 
    # truncated as per the layout indices here 
    
    factored_key = tf.Variable(tf.zeros((batch,num_heads,query_dim,layout_length), dtype=tf.float32),trainable=True)
    
    
    for i,h in enumerate(range(num_heads)):
        # do slice assignment to the factored keys here :: 
        factored_key = slice_assign(factored_key, k_transpose[:,:,:,i*layout_length:(i+1)*layout_length],slice(0, None, 1),slice(0, None, 1),slice(0, None, 1),slice(0,None,1))
        value_sparse = slice_assign(value_sparse, v[:,:,i*layout_length:(i+1)*layout_length,:],slice(0, None, 1),slice(0, None, 1),slice(0, None, 1),slice(0,None,1))

    # Now our BxHxdxlayout_length factorised key matrix is ready here the factored key and the value seq lengths MUST be the same 
    # here 
    
    matmul_qk = tf.matmul(q, factored_key, transpose_b=False)  # (..., seq_len_q, layout_length)
    
    

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)             # Root d scaling would still be the same here 
    
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)          # Note that this is the autoreg mask

     
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, layout_length) # The softmax is now computed over the layout length instaed of the seq_len_k here 
    

    output = tf.matmul(attention_weights, value_sparse)  # (..., seq_len_q, value_dim)          # The layout length parameter would be consumed here 

    return output, attention_weights

# This accepts queries Keys and values here 

# Takes in values keys queries adn their masks and returns multi-head attention here :: 


class MultiHeadAttention(tf.keras.layers.Layer):             # So this is a multihead attention layer superclass is the Keras layer here 
    
    def __init__(self,d_model, num_heads,autoreg):
        
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads           # Learn rich features across multiple heads here 
         
        self.d_model = d_model               # This is either the Embedding Layer dimension of the transformed model dimension here 

        assert d_model % self.num_heads == 0          # The number of heads should be a factor of d_model here 

        self.depth = d_model // self.num_heads        # Number of attention heads here 

        self.wq = tf.keras.layers.Dense(d_model)      # The three transformations here 
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
        self.autoreg = autoreg                 # Chooses whether an autoregressive mask is required or not 
        
        
    def create_autoreg_mask(self,batch_dim,num_heads,query_len,key_len):
        # This is of the size of the attention matrix which would be generated here 
        temp = tf.ones([batch_dim,num_heads,query_len,key_len],tf.float32)               # Our mask would be NxK here N>K K is a hyperparameter here 

        lowertrig = temp - tf.linalg.band_part(temp,-1,0)       # Lower triangular autoreg maks here
        
        return lowertrig                # The autoreg mask here 
        

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
        key_len = tf.shape(k)[1]
    
        if(query_mask is not None):            # These are the padding masks in here 
            q = q*query_mask
        if(key_mask is not None):
            k = k*key_mask
        
        if(value_mask is not None):
            v = v*value_mask                 # Note that this is elementwise multiplication in here O(NxD) for one item 
         

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # First we should be able to do it for the non-autoregressive case here :: 
        # Uniform layout length for each head here ::
                    
        layout_length = k.shape[2]//self.num_heads            # Uniform layout being done here 
        
        
        if(self.autoreg):
            autoreg_mask = self.create_autoreg_mask(batch_size,self.num_heads,query_len,layout_length)
        else:
            autoreg_mask = None
            
            
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, layout_length,autoreg_mask)           # This is non-autoreg. attention layer here 
        
    

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)     # intermediate embedding size here 
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads,autoreg=False)
        self.ffn = point_wise_feed_forward_network(d_model, dff)           # Multi head attention followed by pointwise feed-forward neural network here 

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)           # Batchnorm 

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)          # Dropout 

    def call(self, x, training, mask):                # This is self attention here :: only mask is padding mask in here 
        
        attn_output, _ = self.mha(x, x, x, mask, mask, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)     # dropout followed by residual connection followed by layer normalization here 

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
    
class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self,d_model, num_heads, dff, rate=0.1):
        
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads,autoreg=True)   # autoreg + padding here 
        self.mha2 = MultiHeadAttention(d_model, num_heads,autoreg=False)  # only padding here 

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,mask):        # This mask is just the padding mask in here ::
        
    
        attn1, attn_weights_block1 = self.mha1(x, x, x, mask,mask,mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # the second attention block doesn't have any masks in here :: 
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, None,None,None)  # (batch_size, target_seq_len, d_model)
        
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],         # np.newaxis is simply used to increase the dimension here 
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32) 

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        
        super(Encoder, self).__init__()

        self.d_model = d_model                         # Embedding dimension here 
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)                # As done in the original model here 
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]                            # N encoder layers here 

        self.dropout = tf.keras.layers.Dropout(rate)     

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]       # Note that in the encoder the query key and the value are all the same 
        

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))                 # Why is root(depth) multiplied here ?? 

        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)              # One by one passed through the layers here 

        return x  
    
class Decoder(tf.keras.layers.Layer):
    
    def __init__(self,num_layers,d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate=0.1)
                           for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
           mask):

        
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights
    
class Sparse_Transformer(tf.keras.Model):
    
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
        
        
        super(Sparse_Transformer, self).__init__()
        

    
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, rate)

    
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)   # Logits over the target vocabulary size here 
                
    
    def call(self, inp, tar, training, enc_padding_mask,
            dec_padding_mask):      # No need of the lookahead mask here :: 

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, dec_padding_mask)
        

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    
    
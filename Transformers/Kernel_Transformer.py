# Kernel Autoregressive Transformers here :: 
from tf_slice_assign import slice_assign              # used in slice assignment in here 
# Unfortunately this only works for uint8 datatype here :: have to try tf scatter update 
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



import tensorflow as tf

from tf_slice_assign import slice_assign



class MultiHeadAttention_KT(tf.keras.layers.Layer):             # So this is a multihead attention layer superclass is the Keras layer here 
    
    def __init__(self, d_model, num_heads,autoreg):
        super(MultiHeadAttention_KT, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model               # This is either the Embedding Layer dimension of the transformed model dimension here 
        self.autoreg = autoreg               # if autoreg then the process is a bit diffreent in here :: 
        
        assert d_model % self.num_heads == 0          # The number of heads should be a factor of d_model here 

        self.depth = d_model // self.num_heads        # Number of attention heads here 

        self.wq = tf.keras.layers.Dense(d_model)      # The three transformations here 
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
        self.feature_map = tf.keras.layers.ReLU(negative_slope = 0.1)
        
#         self.feature_map = tf.keras.layers.ELU(alpha=1.0)           # ELU feature map in here ::
        
    # Once everything needs to be insured to be in float32 in here ::
    
    def compute_scaled_attention(self,q,k,v):
        if(self.autoreg):             # We have to compute the autorgressive context vector in here 
            # store each of the num and denom terms in a matrix in here :: using slice assignment 
            batch,num_heads,seq_len_queries,query_dim = q.shape        # Query and Key Dims are same :: Key and Value seq lengths are same 
    
            batch,num_heads,seq_len_keys,key_dim = k.shape

            batch,num_heads,seq_len_val,val_dim = v.shape
            
            k_transpose = tf.transpose(k,perm=(0,1,3,2))
            
            # For each query we have to store a matrix :: note that query outer dimensions need to be neglected 
            # Our outer matrix is hence :: batch,num_heads,seq_len_queries,query_dim,val_dim
            outer = tf.Variable(tf.zeros((batch,num_heads,seq_len_queries,query_dim,val_dim), dtype=tf.float32),trainable=True)
            sum_cols = tf.Variable(tf.zeros((batch,num_heads,seq_len_queries,query_dim), dtype=tf.float32),trainable=True)
            # Both the sum and the outer product matrices need to be computed via slice assignment here ::
            # Here the seq_len_queries is coming in the length parameter as we are doing for each query separately 
            
            for i in range(seq_len_keys):
                row = k_transpose[:,:,:,i]            # B x n_h x k_d
                row_expanded = tf.expand_dims(row,axis=-1)

                value_row = v[:,:,i,:]

                value_row_expanded = tf.expand_dims(value_row,axis=2)
                
                outer_prod = tf.matmul(row_expanded,value_row_expanded)       # Bxn_hxk_dxv_d
                
                if(i==0):            # first pass here 
                    outer_prod = tf.expand_dims(outer_prod,axis=2)
                    outer = slice_assign(outer, outer_prod,slice(0, None, 1),slice(0, None, 1),slice(0, 1, 1),slice(0,None,1),slice(0,None,1))
                    row = tf.expand_dims(row,axis=2)
                    sum_cols = slice_assign(sum_cols,row,slice(0, None, 1),slice(0, None, 1),slice(0, 1, 1),slice(0,None,1))
                    
                else:                 # Use the previous stored value to get the new value here 
                    
                    
                    outer_prod = outer[:,:,i-1,:,:]+outer_prod
                    outer_prod = tf.expand_dims(outer_prod,axis=2)
                    new_sum = sum_cols[:,:,i-1,:]+row
                    new_sum = tf.expand_dims(new_sum,axis=2)
                    outer = slice_assign(outer, outer_prod,slice(0, None, 1),slice(0, None, 1),slice(0, 1, 1),slice(0,None,1),slice(0,None,1))
                    sum_cols = slice_assign(sum_cols,new_sum,slice(0, None, 1),slice(0, None, 1),slice(0, 1, 1),slice(0,None,1))
                    
            sum_cols = tf.expand_dims(sum_cols,axis=-1) 
            
            # Because of dimension mismatch between q and outer we have to do this here :: 
            q_expand = tf.expand_dims(q,axis=3)
    
    
            num = tf.squeeze(tf.matmul(q_expand,outer),axis=3)   
        
            denom = tf.squeeze(tf.matmul(q_expand,sum_cols),axis=-1)              # (B,S,D) --- (B,D,1) === (B,S,1) 
        


            net = num/denom
            
            
            return net                 # for now in here :: 
        else:
            # store the whole num and denom terms in one big matrix in here :: 
            batch,num_heads,seq_len_queries,query_dim = q.shape
    
            batch,num_heads,seq_len_keys,key_dim = k.shape

            batch,num_heads,seq_len_val,val_dim = v.shape
            
            k_transpose = tf.transpose(k,perm=(0,1,3,2))          # Batchxnum_headsxkey_dimxseq_len_key
            
            
            
            for i in range(seq_len_keys):            # we have to do individually as it is an outer product in here 
                
                if(i==0):
                    row = k_transpose[:,:,:,0]
                    row = tf.expand_dims(row,axis=-1)
                    
                    value_row = v[:,:,0,:]

                    value_row = tf.expand_dims(value_row,axis=2)


                    outer = tf.matmul(row,value_row)
                else:
                    row = k_transpose[:,:,:,i]
                    row = tf.expand_dims(row,axis=-1)
                    
                    value_row = v[:,:,i,:]

                    value_row = tf.expand_dims(value_row,axis=2)


                    outer+= tf.matmul(row,value_row)
                    
            # outer is :: BxNxkey_dimxval_dim
            sum_col = tf.math.reduce_sum(k_transpose,axis=-1)

            sum_col = tf.expand_dims(sum_col,axis=-1)          # Batchxnum_headsxkey_dimx1
            
            numerator = tf.matmul(q,outer)             # BxNxSqxVal_dim

            denominator = tf.matmul(q,sum_col)

            net = numerator/denominator             # Again batch and num_heads treated as outer in here :: 
            
            return net 
            
            
            
            
    def apply_feature_map(self,query,key):
        
        # apply feature maps on query and key and then return them here 
        
        return self.feature_map(query),self.feature_map(key)           
        
        
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
    
        if(query_mask is not None):             # Do padding masking explicitly here :: 
            q = q*query_mask
        if(key_mask is not None):
            k = k*key_mask
        
        if(value_mask is not None):
            v = v*value_mask                 # Note that this is elementwise multiplication in here O(NxD) for one item 
         

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q,k = self.apply_feature_map(q,k)               # Feature Map is Applied here 

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        scaled_attention = self.compute_scaled_attention(q,k,v)               # BxHxSxD then later merge 
            
            
            
        # The computed context matrix should be merged along the number of heads dimension in here :: so the below remains same :: 
        

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
    
    



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)     # intermediate embedding size here 
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64,100, 50, 256))).shape


# In[47]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model, num_heads, dff, rate=0.1,autoregressive=False):     # Generally the Encoder is NOT Autoregressive : but in some cases it might just be required to use it as one!!!
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention_KT(d_model, num_heads,autoreg=autoregressive)
        self.ffn = point_wise_feed_forward_network(d_model, dff)           # Multi head attention followed by pointwise feed-forward neural network here 

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)           # Batchnorm 

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)          # Dropout 

    def call(self, x, training, mask):                # This is self attention here :: only mask is padding mask in here 
        
        attn_output = self.mha(x, x, x, mask, mask, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)     # dropout followed by residual connection followed by layer normalization here 

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# In[49]:


class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self,d_model, num_heads, dff, rate=0.1):
        
        super(DecoderLayer, self).__init__()
        # Automatically the autoregressive is set to Treu to make the Decoder first attention module as autoregressive 
        self.mha1 = MultiHeadAttention_KT(d_model, num_heads,autoreg=True)   # autoreg + padding here 
        self.mha2 = MultiHeadAttention_KT(d_model, num_heads,autoreg=False)  # only padding here 

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,mask):        # This mask is just the padding mask in here ::
        
    
        attn1 = self.mha1(x, x, x, mask,mask,mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

    
        attn2 = self.mha2(
            enc_output, enc_output, out1, None,None,None)  # (batch_size, target_seq_len, d_model)
        
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


# In[51]:


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


# In[52]:


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


# In[56]:


class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, encoder_autoreg, rate=0.1):
        
        super(Encoder, self).__init__()

        self.d_model = d_model                         # Embedding dimension here 
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)                # As done in the original model here 
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, autoregressive = encoder_autoreg)
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


# In[65]:


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
#         attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training,
                                                 mask)

#             attention_weights[f'decoder_layer{i+1}_block1'] = block1
#             attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x


# In[66]:


class Kernel_Transformer(tf.keras.Model):
    
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, encoder_autoreg, pe_input, pe_target, rate=0.1,):
        
        
        super(Kernel_Transformer, self).__init__()
        

    
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, pe_input, encoder_autoreg, rate)

    
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)   # Logits over the target vocabulary size here 
        
        self.enc_autoreg = encoder_autoreg 
                
    
    def call(self, inp, tar, training, enc_padding_mask,
            dec_padding_mask):      # No need of the lookahead mask here :: 
        
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        
        dec_output = self.decoder(
            tar, enc_output, training, dec_padding_mask)          # Only padding, decoder first attention is auto autoregressive here
        

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output



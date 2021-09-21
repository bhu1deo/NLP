"""
The Transformer model using time2vec embeddings:


"""

import copy
from copy import deepcopy
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
import numpy as np
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import torch.nn as nn 
T = 16
N = 81
d_model_decoder = 24

class Transformer(Module):
    def __init__(self, d_model_enc: int = N, d_model_dec: int = d_model_decoder, nhead_enc: int = int(np.sqrt(N)), nhead_dec: int = 8, num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}        # cuda and float32 ?? 

        super(Transformer, self).__init__()

        if custom_encoder is not None:        # We won't work with custom modules here 
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model_enc, nhead_enc, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            
            encoder_norm = LayerNorm(d_model_enc, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model_dec, nhead_dec, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model_dec, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model_enc = d_model_enc
        self.d_model_dec = d_model_dec
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec

        self.batch_first = batch_first         # Here batch_first means that in the input which convention is followed 
        # If the Batch Dimension comes first, then batch_first is True here 
        
        self.Wtv1 = nn.Linear(1,d_model_decoder//2,bias=True)
        self.Wtv2 = nn.Linear(1,d_model_decoder//2,bias=True)
        
        self.W1 = nn.Linear(d_model_dec,1)       # 100 hidden units in here 
        
        self.W2 = nn.Linear(100,1)             # For the final single Dimensional output here 
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        
    # The autoregressive attention mask for the Decoder needs to be explicitly provided here :: 
    # Only target mask of appropriate size needs to be specified here ::
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")         # I think we have batch_first as True as we have the Batch-dimension at the start 
            # source and target batches need to be kept the same here  

        # if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        # No the above is NOT necessary here we have different length for the source and the target here 

        # The Encoder doesn't need any mask here :: The different masks need to be seen here :: src is the driving time series and the tgt
        
        # We have not used any embedding here :: and we also have only a scalar feature here 
        # Let's try time to vec here :: This would give us an additional dimension as well as 
        # hopefully circumvent the layernorm problem which we were facing earlier 
        
        self.compo1 = self.Wtv1(tgt)
        
        self.compo2 = torch.sin(self.Wtv2(tgt))
        
        tgt = torch.cat((self.compo1,self.compo2), -1)            # 10D features instead here 
        
        
        
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)      # additive mask for the source sequence and key padding mask here 
#         memory.to(device)
#         print(memory)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
    
#         print(output)
#         output.to(device)
        
        # pass through a final linear layer here : 
        
        output = (self.leaky_relu(self.W1(output)))
        
        
        # Return only the last time step output here : it would be used for prediction 
        
        return output[:,-1,:]

# They do not require a class instance creation. So, they are not dependent on the state of the object.

# The difference between a static method and a class method is:

# Static method knows nothing about the class and just deals with the parameters. IND OF THE CLASS HERE 
# Class method works with the class since its parameter is always the class itself.

    # If in the decoder we have a Tx1 input as the target then our self attention matrix would be of size TxT
    # We would require a TxT attention mask here 

    @staticmethod         # Use the below method to generate the target mask here 
    def generate_square_subsequent_mask(sz: int) -> Tensor:         # results in an upper triangular matrix here 
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)  # torch.full creates a square matrix of -infinity values here unmasked pos. filled with 0 here 
        # So here basically diagonal is a boolean :: if 1 then 0 is kept on the diagonal 
        # If 0 then -inf would be kept, this mask should be added to the computed attention matrix during sa in decoder 

    def _reset_parameters(self):

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):           # encoder_layer is of type TransformerEncoderLayer object type
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
#         output.to(torch.device('cuda:0'))

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers) 
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt       # Initially note that loss would be computed with a shifted version of this target here 

        for mod in self.layers:                # pass thorugh each of the decoder layers here 
            output = mod(output, memory, tgt_mask=tgt_mask,       # HAS TO BE PRESENT HERE 
                         memory_mask=memory_mask,            # None 
                         tgt_key_padding_mask=tgt_key_padding_mask,      # None 
                         memory_key_padding_mask=memory_key_padding_mask)   # None 
            
        # Our Second MHA in the Decoder should be non-autoregressive as the complete context vector from the 
        # Encoder is available to us here :: so memory mask should be non here 

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):           # each layer of the encoder here 
    __constants__ = ['batch_first', 'norm_first']

    # Note that here d_model_enc needs to be passed as it's different from the decoder 

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):        # Either explicitly provide an activation function here 
        # or string version of it 
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src           # Source seq here 
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)       # self-attention 
            x = x + self._ff_block(self.norm2(x))            # FFNN 
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm2(x + self._ff_block(x))
#             x = x + self._ff_block(x)

        return x

    # self-attention block :: Only self attention here 
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         print(x.shape)
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]                 # Here Self Attention is a MHA block :: We are taking the output and neglecting the attention weights here 

        return self.dropout1(x)                          # Okay in this self attention the attn_mask should be None 
        # as it's not autoregressive here :: key_padding mask is also None as we DO NOT have zero padding of the inputs 
        # like in NMT 

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']
    # Note d_model is different for the Encoder and the Decoder here :: nhead is 1 for the decoder as the input size is 1 

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,             # This means layernorm would be done afterwards here :: 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,kdim=N,vdim=N,
                                                 **factory_kwargs)
        # For the Second MultiHead Attention Layer the Keys and Values have different 
        # dimension as compared to the query hence kdim needs to be specified here 
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:                                   # we have this :: norm layer follwed by residual and all 
            

            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
#             x = x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
#             x = x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
            x = self.norm3(x + self._ff_block(x))
#             x = x + self._ff_block(x)
#         print("\n An iterate complete here")
        
        return x
    # So in the decoder we not only have the Self Attention but also the multi-head attention the self attention is masked 
    # The second attention is not self attention and is also not masked

    # self-attention block          This NEEDS masking : autoreg mask :: padding mask NOT required here 
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,          # This should be the autoregressive attention mask here 
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block      No mask whatsoever here  
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         print(x.shape)
#         print(mem.shape)
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]                 # Decoder query and encoder context vector  
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):                                        # Yeah Modulelist afterall here 
    return ModuleList([deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



"""
Scaled Dot Product Attention replaces the Attention mechanism described in the paper here: 
"""

import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import numpy as np 


# Equations (8) will change here 
class InputAttentionEncoder_scaled_dot_product(nn.Module):          # Scaled Dot Product attention here 
    def __init__(self, N, M, T, stateful=False):
        # We change some of the equations here 
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)        # He is using a single layer here 
        
        self.W_e = nn.Linear(2*self.M, self.T)                      # 2M -> 16 here          2M = 128
        self.U_e = nn.Linear(self.T, self.T, bias=False)            # 16 -> 16 here 
        self.v_e = nn.Linear(self.T, 1, bias=False)                 # 16 -> 1 here 
    
    def forward(self, inputs):                                             # Batchsize x T x N  here      : 128 x 16 x 81 here  
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).cuda()          # Hidden states for all batches and all time steps here 


        
        #initiale hidden states and cell states here 
        h_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()             # Batchsize x M here : this is for one timestep here 
        s_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()
        
        for t in range(self.T):                    # Here he has done vectorization for all N one shot :: no double for loops 
            
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)          # Used to compute the attention weights here becomes 2M here 

            
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)            # 128 x 81 x 16  :: 128 here is the batchsize  N is 81 here repeated N times herein 
            
            y = self.U_e(inputs.permute(0, 2, 1))          # Here Last is made T and the second is made N BxNxT
            
#             z = torch.tanh(x + y)
#             print(x.shape,y.shape)
            
            e_k_t = torch.sum(torch.matmul(x,y.permute(0,2,1)),dim=-1)                    # 128 x 81 after squeezing here :: before it would have been 128 x 81 x 1 here 
#             print(e_k_t.shape)
            
            alpha_k_t = F.softmax(e_k_t, dim=1)         # N dimension for every t here 
#             print(alpha_k_t.shape)
            
            
            weighted_inputs = alpha_k_t * inputs[:, t, :]     # Here this is elementwise multiplication :: scaling by softmax weights 
            
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))     # Only hidden states :: output would be just be a MLP of this here    
            
            encoded_inputs[:, t, :] = h_tm1                   # h_tm1 is the hidden state at one time instant here:: 
        return encoded_inputs                           # Batchsize x T x M Here The M dimensional hidden vectors for T time steps have been returned 




# Equations (12) will change here 
class TemporalAttentionDecoder_scaled_dot_product(nn.Module):      # Here we use scaled dot product attention instead of the scheme used by the author 
    def __init__(self, M, P, T, stateful=False):
        
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful
        
        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)      # Input size = 1? 
        
        #equation 12 matrices
        self.W_d = nn.Linear(2*self.P, self.M)                # Note that earlier in the encoder this was 2M -> T Here it's 2P -> M 
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias = False)
        
        #equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)
        
        #equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)
        
    def forward(self, encoded_inputs, y):         # BXTxM here 
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()

        for t in range(self.T):         # Here also we want to do for all the T encoder time steps in one shot 
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)         # Just as earlier here 
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1) # Note that each of the T encoded inputs is M dimensional due to M units in the Encoder layer here 
            # BxTx2P -> BxTxM 
            y1 = self.U_d(encoded_inputs)           # BxTxM -> BxTxM here 
#             z1 = torch.tanh(x1 + y1)                # BxTx1 here 
            l_i_t = torch.sum(torch.matmul(x1,y1.permute(0,2,1)),dim=-1) 

#             print(l_i_t.shape)
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            
#             print(beta_i_t.shape)
            beta_i_t = torch.unsqueeze(beta_i_t,dim=-1)
            
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1) # Note that this is NOT autoregressive as thought earlier :: 

            
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)        # Append the context vector and create a new input here 
            y_tilda_t = self.w_tilda(y_c_concat) 

            
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))


        
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)        # Note that the other intermediate hidden and cell states don't matter here 


        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1                              # Returning the last output (T+1)th output only here 

        
class DARNN_scaled_dot_product(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder_scaled_dot_product(N, M, T, stateful_encoder).cuda()
        self.decoder = TemporalAttentionDecoder_scaled_dot_product(M, P, T, stateful_decoder).cuda()
    def forward(self, X_history, y_history):
        out = self.decoder(self.encoder(X_history), y_history)
        return out


# Quick Test here: 

# model = DARNN_scaled_dot_product(81, 64, 64, 16).cuda()         # self, N, M, P, T, stateful_encoder=False, stateful_decoder=False
# opt = torch.optim.Adam(model.parameters(), lr=0.001)
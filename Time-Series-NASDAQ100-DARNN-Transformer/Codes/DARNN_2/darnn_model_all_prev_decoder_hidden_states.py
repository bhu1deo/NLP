
import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import numpy as np 

"""
Use all the previous Decoder States to make the desired time step prediction here: The Encoder remains the same here 
"""
class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)        # He is using a single layer here 
        # This is because x_tilda_t is of size N and M is the number of LSTM units here 
        
        #equation 8 matrices
        
        # linear layer operates on the last dimension IMPORTANT 
        # T is the time length of each of the driving time series here 
        self.W_e = nn.Linear(2*self.M, self.T)                      # 2M -> 16 here          2M = 128
        self.U_e = nn.Linear(self.T, self.T, bias=False)            # 16 -> 16 here 
        self.v_e = nn.Linear(self.T, 1, bias=False)                 # 16 -> 1 here 
    
    def forward(self, inputs):                                             # Batchsize x T x N  here      : 128 x 16 x 81 here  
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).cuda()          # Hidden states for all batches and all time steps here 


        
        #initiale hidden states and cell states here 
        h_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()             # Batchsize x M here : this is for one timestep here 
        s_tm1 = torch.zeros((inputs.size(0), self.M)).cuda()
        
        for t in range(self.T):                    # Here he has done vectorization for all N one shot :: no double for loops 
            #concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)          # Used to compute the attention weights here becomes 2M here 

            # print("\n Inputs shape")
            # print(inputs.shape)

            # print(self.W_e.shape)
            # print("\n hc_concat here :: ")
            # print(h_c_concat.shape)
            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)            # 128 x 81 x 16  :: 128 here is the batchsize  N is 81 here repeated N times herein 
            # BxNxT 
            # print("\n x here::")
            # print(x.shape)
            y = self.U_e(inputs.permute(0, 2, 1))          # Here Last is made T and the second is made N BxNxT
            # For all N this is done in one shot here :: we have to operate on T hence T is put at the last dimension 
            # print("\n y here ::")
            # print(y.shape)
            z = torch.tanh(x + y)
            # print("\n Before e_k shape here :: ")
            # print(self.v_e(z).shape)
            e_k_t = torch.squeeze(self.v_e(z))                    # 128 x 81 after squeezing here :: before it would have been 128 x 81 x 1 here 
            # Above is BxN here 
            # e_k_t represents for the t_th time instant hence BxNx1 -> BxN after squeezing 

            # So what he is doing is that for all the driving series he is computing the ekt for all the series at once and the first term is the same that's why he has to repeat 
        
            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)         # N dimension for every t here 
            
            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :]     # Here this is elementwise multiplication :: scaling by softmax weights 
            # Remember input size was N :: BxN weighted_inputs

            #calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))     # Only hidden states :: output would be just be a MLP of this here    
            
            encoded_inputs[:, t, :] = h_tm1                   # h_tm1 is the hidden state at one time instant here:: 
        return encoded_inputs                           # Batchsize x T x M Here The M dimensional hidden vectors for T time steps have been returned 


# Okay here he has done this forward pass for all the time steps T 
# 


class TemporalAttentionDecoder_prev_decoder_states(nn.Module):      # Here we can use the previous decoder states also to predict the output
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
        
        # All the Hidden decoder states to generate a context vector here :: 
        self.W_dec = nn.Linear(self.P*(self.T+1), self.P)
        
    def forward(self, encoded_inputs, y):         # BXTxM here 
        
        #initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).cuda()
        
        dt_list = torch.zeros((encoded_inputs.size(0), self.P)).cuda()         # The decoder states concatenated here 

        for t in range(self.T):         # Here also we want to do for all the T encoder time steps in one shot 
            #concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)         # Just as earlier here 
            #print(d_s_prime_concat)
            #temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1) # Note that each of the T encoded inputs is M dimensional due to M units in the Encoder layer here 
            # BxTx2P -> BxTxM 
            y1 = self.U_d(encoded_inputs)           # BxTxM -> BxTxM here 
            z1 = torch.tanh(x1 + y1)                # BxTx1 here 
            l_i_t = self.v_d(z1)

            # Earlier we were doing for the N driving time series, here we are doing for the T (all T) 
            # all the encoder states are reweighted here 
            
            
            #normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            
#             print(beta_i_t.shape)
            
            #create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1) # Note that this is NOT autoregressive as thought earlier :: 
            # All timesteps of the driving series are used probably because we are going to prediction for T+1 :: see the data for clarifications 

            
            # concatenate c_t and y_t Note that here y_t is a scalar because our target is just one column the NASDAQ value 
            # 
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)        # Append the context vector and create a new input here 
            # create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat) 
            # Here after passing through the linear layer, 
            # we get a 1D output here :: because our input to the decoder is 1D here 

            
            #calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
            
            dt_list = torch.cat((dt_list,d_tm1),-1)

            # Here we just used the T hidden states of the decoder instead like earlier we did N 
            # driving time series 

        
        #concatenate context vector at step T and hidden state at step T
        
        d_tm1 = self.W_dec(dt_list)
        
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)        # Note that the other intermediate hidden and cell states don't matter here 

        # One could use the previous hidden state combination also to compute the next output maybe?? 

        #calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1                              # Returning the last output (T+1)th output only here 

        
# Do a Quick test in here: 


class DARNN_prev_decoder_states(nn.Module):
    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder).cuda()
        self.decoder = TemporalAttentionDecoder_prev_decoder_states(M, P, T, stateful_decoder).cuda()
    def forward(self, X_history, y_history):
        out = self.decoder(self.encoder(X_history), y_history)
        return out


# model = DARNN_prev_decoder_states(81, 64, 64, 16).cuda()         # self, N, M, P, T, stateful_encoder=False, stateful_decoder=False
# opt = torch.optim.Adam(model.parameters(), lr=0.001)

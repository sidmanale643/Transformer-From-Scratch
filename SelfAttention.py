import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self , d_model):
        super().__init__()
        self.d_model = d_model 
 
        self.w_q = nn.Linear(self.d_model , self.d_model)
        self.w_k = nn.Linear(self.d_model , self.d_model)
        self.w_v = nn.Linear(self.d_model , self.d_model)
        
    def forward(self , inputs):
               
        Q = self.w_q(inputs)
        K = self.w_k(inputs)
        V = self.w_v(inputs)
             
        attention_scores = torch.matmul(Q , K.transpose(-2 , -1)) 
        attention_weights = torch.softmax(attention_scores / np.sqrt(self.d_model) , dim = -1)              
        context_vector = torch.matmul(attention_weights , V)
      
        return context_vector


# batch_size = 3
# n_seq = 10
# d_model = 512
# input_matrix = torch.randn(batch_size , n_seq , d_model)
# print(input_matrix.shape)

# SDPA = SelfAttention(d_model)

# context_vector = SDPA(input_matrix)
# print(context_vector.shape)
# print(context_vector)
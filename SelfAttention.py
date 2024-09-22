import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self , d_model):
        super().__init__()
        self.d_model = d_model #is the dimension of embeddings (columns of input matrix)
        #Eg: d_model -> 512
        #matrix shape is (512 , 512)
        #w_q , w_k , w_v are all linear transformation matrices to transform the input matrix to Query , Key and Value matrices
        self.w_q = nn.Linear(self.d_model , self.d_model)
        self.w_k = nn.Linear(self.d_model , self.d_model)
        self.w_v = nn.Linear(self.d_model , self.d_model)
        
    def forward(self , inputs):
        #Eg: B = 3 , N = 10 (rows of the input matrix)
        #inputs shape -> (batch_size(B) , n_sequences(N) , d_model))
        #N is the number of sequences
                     
        Q = self.w_q(inputs)
        K = self.w_k(inputs)
        V = self.w_v(inputs)
      
        # Q , K , V all have shapes ( 3, 10 , 512)
              
        attention_scores = torch.matmul(Q , K.transpose(-2 , -1)) #since shape of Q,K is (B , N , d_model) -> transpose the last and the second last dimensions-> (3 , 10 , 512) X (3 , 512 , 10) -> (3,10, 10) 
        #(B , N , N)      
        #scaled dot product attention
        attention_weights = torch.softmax(attention_scores / np.sqrt(self.d_model) , dim = -1)#dim =-1 applies softmax to the last dimension
                                                                                              #scale with the square root to of d_model to avoid gradients getting too large with softmax              
        context_vector = torch.matmul(attention_weights , V)
        #final shape is -> (3 , 10, 10) X (3 , 10 , 512) -> (3 , 10 , 512)
        return context_vector


batch_size = 3
n_seq = 10
d_model = 512
input_matrix = torch.randn(batch_size , n_seq , d_model)
print(input_matrix.shape)

SDPA = SelfAttention(d_model)

context_vector = SDPA(input_matrix)
print(context_vector.shape)
print(context_vector)
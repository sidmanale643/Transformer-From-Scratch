import numpy as np
import torch
import torch.nn as nn

class MHA(nn.Module): #beneficial because each head has seperate transformation matrices which helps them focus on different parts of sequence since all the outputs from all the heads are going to concatenated
    def __init__(self , d_model , n_heads , mask : None ):
        super().__init__()
        
        assert d_model % n_heads == 0 #remainder should be 0 
        
        self.d_model =  d_model  #dimension of embeddings
        self.n_heads = n_heads  #number of heads for MultiHeadAttention
        self.d_k = self.d_model // self.n_heads #since the dimensions are split for n heads , we divide d_model by num of heads to see how many dimensions each head recieves
        self.scale =  torch.sqrt(self.d_k)
        self.mask = mask
        #linear transformation matrices to project the input matrix in Q , K , V matrices
        self.w_q = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model , self.d_model, bias = False)
        
    def forward(self , input):
      
        batch_size , n_seq  = input.size()[0] , input.size()[1]
        
        #project input matrices
        Q = self.w_q(input)
        K = self.w_k(input)
        V = self.w_v(input)
        
        #d_k = d_model // number of heads in MHA 
        #interpret the view as every heads gets a portion of embeddings i.e it gets all the rows but columns are split ( d_k = d_model // n_heads -> eg: 512 // 4 = 128) -> each heads gets n_seq rows and d_k columns
        #transpose to switch n_seq with heads dimension , so all the computations are performerd along the head dimension
        Q = Q.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2) 
        K = K.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2)
        V = V.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2)
        
        #Dot Product Attention scaled by square root of d_k 
        attention_scores = torch.matmul(Q , K.transpose(-2 , -1)) / self.scale
       
        
        if self.mask is not None:
            attention_scores = attention_scores.masked_fill(self.mask == 0 , -1e9)
            
        attention_weights = torch.softmax(attention_scores , dim = -1)
        out = torch.matmul(attention_weights , V )
            
        #concatenate outputs from all the heads
        #when elements data structure are stored in continous memory locations , the memory is said to be contiguous
        #operations like "transpose" can cause the memory to become Non-Contiguous and operations like "view()"" require the memory to be Contiguous
        #having Contiguous memory is better for optimization and vectorization
        out = out.transpose(1 , 2).contiguous().view(batch_size , n_seq , self.d_model)
        
        #final output projection
        attention_output = self.w_o(out)
        
        return attention_output

batch_size = 3
n_seq = 10
d_model = 512
n_heads = 8

input = torch.randn( batch_size , n_seq , d_model)
mha = MHA(d_model , n_heads)
context_vector = mha(input)
print(context_vector.shape)


        

    
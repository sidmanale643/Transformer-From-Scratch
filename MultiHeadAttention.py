import numpy as np
import torch
import torch.nn as nn

class MHA(nn.Module): 
    def __init__(self , d_model , n_heads  ):
        super().__init__()
        
        assert d_model % n_heads == 0 
        
        self.d_model =  d_model  
        self.n_heads = n_heads  
        self.d_k = self.d_model // self.n_heads 
        self.scale =  torch.sqrt(torch.tensor(self.d_k))
      
        self.w_q = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model , self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model , self.d_model, bias = False)
        
    def forward(self , Q , K , V , mask):
      
        batch_size , n_seq  = Q.size()[0] , Q.size()[1]
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2) 
        K = K.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2)
        V = V.view(batch_size , n_seq , self.n_heads , self.d_k ).transpose(1 , 2)
        
        attention_scores = torch.matmul(Q , K.transpose(-2 , -1)) / self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores , dim = -1)
        out = torch.matmul(attention_weights , V )
        out = out.transpose(1 , 2).contiguous().view(batch_size , n_seq , self.d_model)
        attention_output = self.w_o(out)
        
        return attention_output



        

    
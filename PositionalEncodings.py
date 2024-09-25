import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from Embeddings import InputEmbeddings

torch.manual_seed(123)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, n_seq : int, dropout: float) ->None:
        super().__init__()
        self.d_model = d_model 
        self.n_seq = n_seq
        self.dropout = nn.Dropout(dropout) 
        pe = torch.zeros(n_seq, d_model)
        position = torch.arange(0, n_seq, dtype = torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
         x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
         return self.dropout(x) 
     
# embeddings  = InputEmbeddings(100 , 10)
# pose = PositionalEncoding(10 , 5 , 0.01)
# tokenizer_out = torch.tensor([0 , 55 , 68 , 69 , 80])

# ip_emb = embeddings(tokenizer_out)
# pos_emb = pose(ip_emb)
# print(ip_emb)
# print(pos_emb)
import torch
import torch.nn as nn
from Embeddings import InputEmbeddings
from AddNorm import AddNorm
from MultiHeadAttention import MHA
from FeedForwardNetwork import FFN
from PositionalEncodings import PositionalEncoding

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
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores , dim = -1)
        out = torch.matmul(attention_weights , V )
        out = out.transpose(1 , 2).contiguous().view(batch_size , n_seq , self.d_model)
        attention_output = self.w_o(out)
        
        return attention_output
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        
        self.mha = MHA(self.d_model, self.n_heads , self.mask )
        self.ffn = FFN(self.d_model, self.dropout)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        
    def forward(self, x , mask):
        attention_out = self.mha(x , x , x  ,mask)
        norm1 = self.add_norm1(x, attention_out)
        ffn_out = self.ffn(norm1)
        norm2 = self.add_norm2(norm1, ffn_out)
        return norm2

class Encoder(nn.Module):
    def __init__(self,  n_seq, vocab_size, d_model, n_heads, dropout , n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_seq = n_seq
        self.dropout = dropout
        self.input_embeddings = InputEmbeddings(self.vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model, self.n_seq, self.dropout)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        
    def forward(self, x , mask):
        input_embeddings = self.input_embeddings(x)
        x = self.pos_emb(input_embeddings)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        
        return x


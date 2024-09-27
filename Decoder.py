import torch
import torch.nn as nn
from MultiHeadAttention import MHA
from FeedForwardNetwork import FFN
from AddNorm import AddNorm
from Embeddings import InputEmbeddings
from PositionalEncodings import PositionalEncoding

class DecoderBlock(nn.Module):
    def __init__(self ,d_model, n_heads, dropout ):
        super().__init__()
            
        self.d_model = d_model
        self.n_heads = n_heads
    
        self.dropout = dropout
        
        self.mmha1 = MHA(d_model , n_heads)
        self.mha = MHA(d_model , n_heads )
        self.ffn = FFN(d_model , dropout) 
        self.norm1 = AddNorm()
        self.norm2 = AddNorm()
        self.norm3 = AddNorm()
                
    def forward(self , x , mask , encoder_out ,):
        mha1_out = self.mmha1(x , x , x , mask)
        norm_1 = self.norm1(x , mha1_out)
        cross_attn = self.mha(encoder_out , encoder_out , norm_1 , mask)
        norm_2 = self.norm2(norm_1 , cross_attn)
        ffn_1 = self.ffn(norm_2)
        norm_3 = self.norm3(norm_2 , ffn_1)
        return norm_3
     
class Decoder(nn.Module):
    def __init__(self ,n_seq , vocab_size , d_model, n_heads, dropout , n_blocks):
        super().__init__()
        
        self.n_seq = n_seq
        self.vocab_size = vocab_size
        self.d_model =  d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
       
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])
        self.input_embeddings = InputEmbeddings(vocab_size , d_model)
        self.pos_encodings = PositionalEncoding(d_model , n_seq , dropout)
       
    def forward(self , x , encoder_out , mask):
        input_embeddings = self.input_embeddings(x)
        
        x = self.pos_encodings(input_embeddings)
        
        for decoder_block in self.decoder_blocks:
            x =  decoder_block(x , encoder_out , mask)
       
        return x 
        
import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Moudule):
    def __init__(self ,n_seq, vocab_size, d_model, n_heads, dropout, n_blocks ):
        super().__init__()
        
        self.encoder = Encoder( n_seq, vocab_size, d_model, n_heads, dropout, n_blocks)
        self.decoder = Decoder( n_seq, vocab_size, d_model, n_heads, dropout, n_blocks)
        self.fc = nn.Linear(d_model , vocab_size)
                
    def forward(self , x , src_mask , tgt_mask):
        encoder_out  = self.encoder(x ,src_mask)
        decoder_out = self.decoder(x , encoder_out ,src_mask , tgt_mask)
        
        out = self.fc(decoder_out)
        
        return out

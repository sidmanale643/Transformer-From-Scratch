import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_seq, d_model, n_heads, n_encoder_blocks, n_decoder_blocks, dropout):
        super().__init__()
        
        self.encoder = Encoder(n_seq, vocab_size, d_model, n_heads, dropout, n_encoder_blocks)
        self.decoder = Decoder(n_seq, vocab_size, d_model, n_heads, dropout, n_decoder_blocks)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_tgt_mask):
      
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)
        
        out = self.fc_out(decoder_out)
        
        return out

model_params = {
    'vocab_size': 10000, 
    'n_seq': 20,         
    'd_model': 512,      
    'n_heads': 8,        
    'n_encoder_blocks': 6, 
    'n_decoder_blocks': 6, 
    'dropout': 0.1,
    'pad_token_id': 10000}

model = Transformer(
    vocab_size=model_params['vocab_size'],
    n_seq=model_params['n_seq'],
    d_model=model_params['d_model'],
    n_heads=model_params['n_heads'],
    n_encoder_blocks=model_params['n_encoder_blocks'],
    n_decoder_blocks=model_params['n_decoder_blocks'],
    dropout=model_params['dropout']
)

def create_src_mask(src):
    src_mask = (src != model_params['pad_token_id']).unsqueeze(1).unsqueeze(2)
    return src_mask

def create_tgt_mask(tgt):
    tgt_seq_len = tgt.size(1)
    tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)).bool()
    return tgt_mask

def create_src_tgt_mask(src, tgt):
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)
    src_tgt_mask = (src != model_params['pad_token_id']).unsqueeze(1).unsqueeze(2).expand(-1, -1, tgt_seq_len, -1)
    return src_tgt_mask


import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self , vocab_size : int , d_model : int):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size , d_model)
    
    def forward(self , x):
        embeddings = self.embeddings(x.long()) * math.sqrt(self.d_model)
        return embeddings

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
        x = x + (self.pe[:, :x.shape[1], :])
         
        return self.dropout(x) 
    
class MHA(nn.Module): 
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        assert d_model % n_heads == 0 
        
        self.d_model = d_model  
        self.n_heads = n_heads  
        self.d_k = self.d_model // self.n_heads 
        self.scale = math.sqrt(torch.tensor(self.d_k))
      
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        
    def forward(self, Q, K, V, mask=None):
        batch_size, n_seq = Q.size(0), Q.size(1)
        
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)
        
        Q = Q.view(batch_size, n_seq, self.n_heads, self.d_k).transpose(1, 2) 
        K = K.view(batch_size, n_seq, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, n_seq, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_seq, self.d_model)
        attention_output = self.w_o(out)
        
        return attention_output
       
class FFN(nn.Module):
    def __init__(self , d_model , dropout):
        super().__init__()
        
        self.layer1 = nn.Linear(d_model , 4 * d_model)
        self.layer2 = nn.Linear(4 * d_model , d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    
class AddNorm(nn.Module):
    def __init__(self , eps = 1e-5):
        super().__init__()

        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1))    
        self.beta = nn.Parameter(torch.zeros(1))   
        
    def forward(self , x , sub_layer_out ):
        X = x + sub_layer_out
        
        mean = torch.mean(X , dim = -1 , keepdim= True)
        var = torch.var(X , dim = -1 , keepdim= True)
        norm =  self.gamma * ((X - mean)/torch.sqrt(var + self.eps))  + self.beta
        return norm
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.mha = MHA(self.d_model, self.n_heads)
        self.ffn = FFN(self.d_model, self.dropout)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        
    def forward(self, x, src_mask):
        attention_out = self.mha(x, x, x, src_mask)
        norm1 = self.add_norm1(x, attention_out)
        ffn_out = self.ffn(norm1)
        norm2 = self.add_norm2(norm1, ffn_out)
        return norm2

class Encoder(nn.Module):
    def __init__(self, n_seq, vocab_size, d_model, n_heads, dropout, n_blocks):
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
        
    def forward(self, x, src_mask):
        input_embeddings = self.input_embeddings(x)
        x = self.pos_emb(input_embeddings)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.mmha1 = MHA(d_model, n_heads)
        self.mha = MHA(d_model, n_heads)
        self.ffn = FFN(d_model, dropout) 
        self.norm1 = AddNorm()
        self.norm2 = AddNorm()
        self.norm3 = AddNorm()
                
    def forward(self, x, encoder_out, tgt_mask, src_tgt_mask):
        mha1_out = self.mmha1(x, x, x, tgt_mask)
        norm_1 = self.norm1(x, mha1_out)
        cross_attn = self.mha(norm_1, encoder_out, encoder_out, src_tgt_mask)
        norm_2 = self.norm2(norm_1, cross_attn)
        ffn_1 = self.ffn(norm_2)
        norm_3 = self.norm3(norm_2, ffn_1)
        return norm_3
     
class Decoder(nn.Module):
    def __init__(self, n_seq, vocab_size, d_model, n_heads, dropout, n_blocks):
        super().__init__()
        
        self.n_seq = n_seq
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])
        self.input_embeddings = InputEmbeddings(vocab_size, d_model)
        self.pos_encodings = PositionalEncoding(d_model, n_seq, dropout)
       
    def forward(self, x, encoder_out, tgt_mask, src_tgt_mask):
        input_embeddings = self.input_embeddings(x)
        x = self.pos_encodings(input_embeddings)
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_out, tgt_mask, src_tgt_mask)
       
        return x
    
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

src_input = torch.randint(0, model_params['vocab_size'], (2, model_params['n_seq']))  
tgt_input = torch.randint(0, model_params['vocab_size'], (2, model_params['n_seq']))  

src_mask = create_src_mask(src_input)
tgt_mask = create_tgt_mask(tgt_input)
src_tgt_mask = create_src_tgt_mask(src_input, tgt_input)

output = model(src_input, tgt_input, src_mask, tgt_mask, src_tgt_mask)
print(output.shape)  


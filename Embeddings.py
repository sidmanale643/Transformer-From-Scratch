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
    
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self , d_model , dropout):
        super().__init__()
        
        self.layer1 = nn.Linear(d_model , 4 * d_model)
        self.layer2 = nn.Linear(4 * d_model , d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self , x):
        
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    

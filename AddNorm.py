import torch
import torch.nn as nn

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
    

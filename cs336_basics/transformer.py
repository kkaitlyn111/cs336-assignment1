import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = np.sqrt(2/(in_features + out_features))
        weights = torch.nn.init.trunc_normal_(torch.zeros([in_features, out_features]), mean=0, std=std, a=-3*std, b=3*std)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights.T, x, "in_features out_features, ... in_features -> ... out_features")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=True)
        
    def set(self, weights):
        self.weights = torch.nn.Parameter(weights.to(device=self.device, dtype=self.dtype))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

        


        
    

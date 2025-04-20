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

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad = True)
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        rms = torch.sqrt(norm ** 2 / self.d_model + self.eps)
        x_normed = x / rms
        result = einsum(self.weights, x_normed, "d, b s d -> b s d")
        
        return result.to(in_dtype)
        

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.d_ff = d_ff

    def forward(self, x):
        t1 = einsum(x, self.W1.weights, "b s d_model, d_ff d_model -> b s d_ff")
        t2 = torch.sigmoid(t1) * t1
        t3 = einsum(x, self.W3.weights, "b s d_model, d_ff d_model -> b s d_ff")
        t4 = t2 * t3
        result = einsum(t4, self.W2.weights, "b s d_ff, d_model d_ff -> b s d_model")
        return result


        
    

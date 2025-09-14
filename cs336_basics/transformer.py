from re import X
import torch
from einops import rearrange, einsum, reduce, repeat
import torch.nn as nn
import numpy as np


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        std = torch.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=std, a= -3.0 * std, b = 3.0 * std)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean = 0, std = 1, a=-3, b=3)

    def forward(self, token_ids: torch.LongTensor) ->  torch.Tensor:
        return self.embedding_matrix[token_ids] # wait how does pytorch handle this wtf

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.eps = eps # not a learnable param
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMSa = torch.sqrt((reduce(x**2, '... d_model -> ...', 'sum') + self.eps)/self.d_model)
        RMSa = rearrange(RMSa, '... -> ... 1')
        RMSNorma = x / RMSa
        RMSNorma = RMSNorma * self.g
        return RMSNorma.to(in_dtype)

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
    
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.W2 = nn.Parameter(torch.randn(d_model, d_ff))
        self.W3 = nn.Parameter(torch.randn(d_ff, d_model))
        std = np.sqrt(2.0 / (d_ff + d_model))
        nn.init.trunc_normal_(self.W1, mean=0, std=std, a= -3.0 * std, b = 3.0 * std)
        nn.init.trunc_normal_(self.W2, mean=0, std=std, a= -3.0 * std, b = 3.0 * std)
        nn.init.trunc_normal_(self.W3, mean=0, std=std, a= -3.0 * std, b = 3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(x, self.W1, '... d_model, dff d_model -> ... dff')
        w3x = einsum(x, self.W3, '... d_model, dff d_model -> ... dff')
        result = SiLU(w1x) * w3x
        result = einsum(result, self.W2, '... dff, d_model dff -> ... d_model')
        return result
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "d_k must be even for RoPE"

        k = torch.arange(0, d_k//2, device=device)
        k = rearrange(k, 'd_k -> 1 d_k')
        i = torch.arange(max_seq_len, device=device)
        i = rearrange(i, 'n -> n 1')

        denom = theta**(2*k/d_k) # shape 1 d_k
        theta_ik = i / denom # shape (n 1) / (1 d_k), broadcasts, ouputs (n d_k)
        cos_ik = torch.cos(theta_ik)
        sin_ik = torch.sin(theta_ik)

        self.register_buffer('cos_ik', cos_ik, persistent=False)
        self.register_buffer('sin_ik', sin_ik, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_ik[token_positions]
        sin = self.sin_ik[token_positions]

        x_paired = rearrange(x, '... seq_len (d_half two) -> ... seq_len d_half two', two=2)
        x_even = x_paired[..., 0]
        x_odd = x_paired[..., 1]

        y_even = x_even * cos - x_odd * sin # ... seq d_half
        y_odd = x_even * sin + x_odd * cos # ... seq d_half

        # stack into new dim, then interleave
        y = torch.stack([y_even, y_odd], dim=-1) # ... seq d_half 2
        y = rearrange(y, '... seq d_half two -> ... seq (d_half two)', two=2) # ... seq d_k

        return y

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    val, idx = torch.max(x, dim=dim, keepdim=True)
    x = x - val
    denom = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    # import code; code.interact(local=locals())
    softmax = torch.exp(x) / denom
    return softmax


# Q ... n d_k
# K ... m d_k
# V ... m d_v
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask) -> torch.Tensor:
    n = Q.size(dim=-2)
    d_k = Q.size(dim=-1)
    m = K.size(dim=-2)
    d_v = V.size(dim=-1)
    device = Q.device

    inner_result = einsum(Q, K, '... n d_k, ... m d_k -> ... n m') / np.sqrt(d_k)
    # i = torch.arange(n)
    # j = torch.arange(m)
    # mask = j <= i
    

    scores = inner_result.masked_fill(~mask, float('-inf'))
    final = einsum(softmax(scores), V, '... n m, ... m d_v -> ... n d_v')
    return final


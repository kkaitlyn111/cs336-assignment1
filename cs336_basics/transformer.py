from re import X
import torch
from einops import rearrange, einsum, reduce, repeat
import torch.nn as nn
import numpy as np
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        std = np.sqrt(2.0 / (in_features + out_features))
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
        self.W1 = Linear(in_features=d_ff, out_features=d_model)
        self.W2 = Linear(in_features=d_model, out_features=d_ff)
        self.W3 = Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.W1(x)
        w3x = self.W3(x)
        result = SiLU(w1x) * w3x
        result = self.W2(result)
        return result
    
class RoPE(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float | None = 1e-5, device: torch.device | None = None):
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
        cos = self.cos_ik[token_positions] # this works bc (n d_k) indexed by (n)
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
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    n = Q.size(dim=-2)
    d_k = Q.size(dim=-1)
    m = K.size(dim=-2)
    d_v = V.size(dim=-1)
    device = Q.device

    scale = 1.0 / math.sqrt(d_k)
    scores = einsum(Q, K, '... n d_k, ... m d_k -> ... n m') * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    final = einsum(softmax(scores, dim=-1), V, '... n m, ... m d_v -> ... n d_v') # softmax all key scores (m dim), for each query (n)
    return final

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, apply_rope: bool | None = False, token_positions: torch.Tensor | None = None):
        super().__init__()

        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model not divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.Wo = Linear(in_features=d_model, out_features=d_model)
        self.Wq = Linear(in_features=d_model, out_features=d_model)
        self.Wk = Linear(in_features=d_model, out_features=d_model)
        self.Wv = Linear(in_features=d_model, out_features=d_model)

        self.apply_rope = apply_rope

        self.theta = theta
        self.token_positions = token_positions
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        device = x.device
        dtype = x.dtype
        n = x.size(dim=-2)

        Q = self.Wq(x) # [... n d_model] [d_model d_model] -> [n d_model]
        K = self.Wk(x) # [... n d_model] [d_model d_model] -> [n d_model]
        V = self.Wv(x) # [... n d_model] [d_model d_model] -> [n d_model]

        # move head dimension to front bc it's treated like a batch basically!!
        Qi = rearrange(Q, '... n (h d_k) -> ... h n d_k', h = self.num_heads)
        Ki = rearrange(K, '... n (h d_k) -> ... h n d_k', h = self.num_heads)
        Vi = rearrange(V, '... n (h d_k) -> ... h n d_k', h = self.num_heads)
        # each of these is [... h n d_k]

        if self.apply_rope:
            rope = RoPE(d_k = self.d_k, max_seq_len = n, theta=self.theta, device=device)
            # token_positions must be of shape [... seq_len] for RoPE
            if self.token_positions is None:
                self.token_positions = torch.arange(n)
            Qi = rope(Qi, self.token_positions)
            Ki = rope(Ki, self.token_positions)

        i = torch.arange(n, device=device, dtype=dtype).unsqueeze(1) # [n, 1]
        j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0) # [1, n]
        mask = j <= i # [n, n]
        mask.to(device=device, dtype=dtype)

        # for masked_fill, the shapes have to be broadcastable to each other
        # currently scores is [... n n] and [n n], so that works out fine
        attn = scaled_dot_product_attention(Qi, Ki, Vi, mask) # has shape [... h n d_k]

        # concat all the heads together to just get [... n d_model]
        # (h d_k) order matters 
        concat = rearrange(attn, '... h n d_k -> ... n (h d_k)')

        result = self.Wo(concat)

        return result








        




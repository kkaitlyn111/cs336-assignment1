import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, LongTensor

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

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.d_k = d_k

        if d_k % 2 != 0:
            raise ValueError("Input dimension must be even")
        
        position_indices = torch.arange(max_seq_len, device=device)
        dim_indices = torch.arange(0, d_k, 2, device=device)
        freq = 1.0 / (theta ** (dim_indices.float() / d_k))

        angles = torch.outer(position_indices, freq)

        cos_sin = torch.zeros(max_seq_len, d_k // 2, 2, device=device)
        cos_sin[:, :, 0] = torch.cos(angles)
        cos_sin[:, :, 1] = torch.sin(angles)

        self.register_buffer('cos_sin_matrix', cos_sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # rearrange input into pairs
        x_pairs = rearrange(x, "... (split1 split2) -> ... split1 split2", 
                         split1=self.d_k // 2, split2=2)
        
        # rotate
        cos_vals = self.cos_sin_matrix[token_positions, :, 0]
        sin_vals = self.cos_sin_matrix[token_positions, :, 1]

        rotated = torch.zeros_like(x_pairs)
        rotated[..., 0] = cos_vals * x_pairs[..., 0] - sin_vals * x_pairs[..., 1]
        rotated[..., 1] = sin_vals * x_pairs[..., 0] + cos_vals * x_pairs[..., 1]

        # reshape back to original dimension
        result = rotated.reshape(x.shape)

        return result
    
def softmax(tensor: torch.Tensor, dim: int):
    max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
    
    # trick to ensure numerical stability
    exp_tensor = torch.exp(tensor - max_vals)
    
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)

    softmax_output = exp_tensor / sum_exp
    return softmax_output

def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"], K: Float[Tensor, " ... keys d_k"], V: Float[Tensor, " ... values d_v"], mask: Float[Tensor, " ... queries keys"] | None = None,) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    attention_scores = torch.einsum("...qd,...kd->...qk", Q, K)
    attention_scores = attention_scores / np.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == False, -1e9)
    
    attention_weights = softmax(attention_scores, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", attention_weights, V)
    
    return output

    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None, apply_rope: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.theta = theta
        assert d_model % num_heads == 0
        self.d_v = d_model // num_heads
        self.d_k = self.d_v

        self.Q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.O_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:

        Q_x = self.Q_proj.forward(x)
        K_x = self.K_proj.forward(x)
        V_x = self.V_proj.forward(x)

        Q = rearrange(Q_x, "... batch max_seq_len (n_h d_k) -> ... batch n_h max_seq_len d_k", n_h = self.num_heads)
        K = rearrange(K_x, "... batch max_seq_len (n_h d_k) -> ... batch n_h max_seq_len d_k", n_h = self.num_heads)
        V = rearrange(V_x, "... batch max_seq_len (n_h d_v) -> ... batch n_h max_seq_len d_v", n_h = self.num_heads)

        if self.rope != None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).to(torch.bool)
        mask = ~mask 

        result = scaled_dot_product_attention(Q, K, V, mask=mask)

        result = rearrange(result, "... n_h max_seq_len d_k -> ... max_seq_len (n_h d_k)")

        result = self.O_proj.forward(result)
        
        return result

        



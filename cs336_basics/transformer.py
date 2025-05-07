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
        weights = torch.nn.init.trunc_normal_(
<<<<<<< HEAD
            torch.zeros([out_features, in_features], device=device, dtype=dtype), 
=======
            torch.zeros([in_features, out_features], device=device, dtype=dtype), 
>>>>>>> ebe672e7df2604172e5fc64531dc1d0a3eeaa5d3
            mean=0, std=std, a=-3*std, b=3*std
        )
        self.weight = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # always use float32 for the weights, regardless of input dtype - change later ?
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=torch.float32), requires_grad=True)
        
    def set(self, weights):
        self.weight = torch.nn.Parameter(weights.to(device=self.device, dtype=torch.float32))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=torch.float32), requires_grad=True)
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        rms = torch.sqrt(norm ** 2 / self.d_model + self.eps)
        x_normed = x / rms
        result = einsum(self.weight, x_normed, "d, b s d -> b s d")
        
        return result.to(in_dtype)
        

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.d_ff = d_ff

    def forward(self, x):
        t1 = self.w1(x)
        t3 = self.w3(x)
        t2 = torch.sigmoid(t1) * t1
        t4 = t2 * t3
        result = self.w2(t4)
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
        # x shape: [batch_size, num_heads, seq_len, d_k]
        # token_positions shape: [batch_size, seq_len]
        
        # rearrange input into pairs
        x_pairs = rearrange(x, "... (split1 split2) -> ... split1 split2", 
                         split1=self.d_k // 2, split2=2)
        
        # Get cos and sin values for each position
        # cos_sin_matrix shape: [max_seq_len, d_k//2, 2]
        # token_positions shape: [batch_size, seq_len]
        cos_vals = self.cos_sin_matrix[token_positions, :, 0]  # [batch_size, seq_len, d_k//2]
        sin_vals = self.cos_sin_matrix[token_positions, :, 1]  # [batch_size, seq_len, d_k//2]
        
        # Add head dimension for broadcasting
        cos_vals = cos_vals.unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]
        sin_vals = sin_vals.unsqueeze(1)  # [batch_size, 1, seq_len, d_k//2]

        # Apply rotation
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

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None and apply_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # project queries, keys, and values
        Q_x = self.q_proj(x)
        K_x = self.k_proj(x)
        V_x = self.v_proj(x)

        # reshape for multi-head attention
        Q = rearrange(Q_x, "b s (n_h d_k) -> b n_h s d_k", n_h=self.num_heads)
        K = rearrange(K_x, "b s (n_h d_k) -> b n_h s d_k", n_h=self.num_heads)
        V = rearrange(V_x, "b s (n_h d_v) -> b n_h s d_v", n_h=self.num_heads)

        # apply RoPE if enabled
        if self.rope is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)
            
        # creating attention mask here
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).to(torch.bool)
        mask = ~mask 
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        result = scaled_dot_product_attention(Q, K, V, mask=mask)

        # reshape back and project
        result = rearrange(result, "b n_h s d_k -> b s (n_h d_k)")
        result = self.output_proj(result)
        
        return result

        
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                max_seq_len: int = None,
                theta: float = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__();
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.device = device
        self.dtype = dtype

        self.theta = theta
        self.max_seq_len = max_seq_len

        eps: float = 1e-5

        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype, apply_rope=True)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm attention
        h = self.ln1(x)
        h = self.attn(h)
        x = x + h
        
        # pre-norm feedforward
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        
        return x


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int,
                max_seq_len: int = None,
                theta: float = None,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.device = device
        self.dtype = dtype

        self.theta = theta
        # Use context_length as max_seq_len if not provided
        self.max_seq_len = max_seq_len if max_seq_len is not None else context_length

        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, self.max_seq_len, theta, device, dtype))

        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    # def load_state_dict(self, state_dict):
    #     # token embeddings
    #     self.token_embeddings.weight.data = state_dict['token_embeddings.weight']
        
    #     # layers
    #     for i in range(self.num_layers):
    #         layer = self.layers[i]
    #         layer.ln1.weight.data = state_dict[f'layers.{i}.ln1.weight']
    #         layer.attn.q_proj.weight.data = state_dict[f'layers.{i}.attn.q_proj.weight']
    #         layer.attn.k_proj.weight.data = state_dict[f'layers.{i}.attn.k_proj.weight']
    #         layer.attn.v_proj.weight.data = state_dict[f'layers.{i}.attn.v_proj.weight']
    #         layer.attn.output_proj.weight.data = state_dict[f'layers.{i}.attn.output_proj.weight']
    #         layer.ln2.weight.data = state_dict[f'layers.{i}.ln2.weight']
    #         layer.ffn.w1.weight.data = state_dict[f'layers.{i}.ffn.w1.weight']
    #         layer.ffn.w2.weight.data = state_dict[f'layers.{i}.ffn.w2.weight']
    #         layer.ffn.w3.weight.data = state_dict[f'layers.{i}.ffn.w3.weight']
        
    #     # final layer norm and lm head
    #     self.ln_final.weight.data = state_dict['ln_final.weight']
    #     self.lm_head.weight.data = state_dict['lm_head.weight']

    def forward(self, x: torch.Tensor):
        # token embeddings
        x = self.token_embeddings(x)
        
        # apply transformer layers
        for i in range(self.num_layers):
            x = self.layers[i](x)
        
        # final layer norm
        x = self.ln_final(x)
        
        # LM head
        x = self.lm_head(x)
        
        return x



        






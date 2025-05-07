import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from jaxtyping import Float, Int
import numpy.typing as npt
from typing import IO, BinaryIO
from torch import Tensor, LongTensor
import time
import math
import os

def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:

     # flatten batch and sequence dimensions if needed
    if inputs.dim() > 2:
        inputs = inputs.reshape(-1, inputs.size(-1))
        targets = targets.reshape(-1)
    
    # numerical stability: subtract max value
    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs = inputs - inputs_max
    
    # calc softmax probs
    exp_inputs = torch.exp(inputs)
    exp_inputs_sum = exp_inputs.sum(dim=-1)
    
    batch_indices = torch.arange(inputs.shape[0])
    target_inputs = inputs[batch_indices, targets]
    
    losses = torch.log(exp_inputs_sum) - target_inputs
    
    return losses.mean()

def get_cosine_lr(t: int, alpha_min: float, alpha_max: float, Tw: int, Tc: int) -> float:
    if t < Tw:
        alpha_t = t / Tw * alpha_max
    elif t <= Tc:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - Tw)/(Tc - Tw)*math.pi))*(alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t

def gradient_clipping(params: list[torch.Tensor], M: float, eps: float = 1e-6) -> None:
    grads = [p.grad.flatten() for p in params if p is not None and p.grad is not None]
    all_grads = torch.cat(grads, dim=0)
    norm = torch.norm(all_grads)

    if norm >= M:
        for p in params:
            if p is not None and p.grad is not None:
                p.grad *= M/(norm + eps)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes], run_info: dict = None):
    saved = {}
    saved['model'] = model.state_dict()
    saved['optimizer'] = optimizer.state_dict()
    saved['iteration'] = iteration 
    if run_info is not None:
        saved['run_info'] = run_info
    torch.save(saved, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    saved = torch.load(src)
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    return saved['iteration']
    







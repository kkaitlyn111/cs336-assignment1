import torch
from einops import rearrange, einsum, reduce, repeat
import torch.nn as nn
import numpy as np
import math
from cs336_basics.transformer import softmax
import torch.optim as optim
from collections.abc import Callable, Iterable
from typing import Optional, Tuple, IO, BinaryIO
import os


# input preds [batch ... vocab], targets [batch ... vocab]
# average across batch
# canceled out log/exp terms in formula
def cross_entropy_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.float:
    maxes = torch.amax(predicted, dim=-1, keepdim=True) # subtract largest elem across vocab size (last dim)
    predicted = predicted - maxes
    sumexps = torch.exp(predicted).sum(dim=-1)
    gathered = predicted.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    losses = torch.log(sumexps) - gathered
    return losses.mean()

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'lr went negative {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] # get state associated with p
                t = state.get('t', 0) # get iteration number
                grad = p.grad.data    # get gradient
                p.data -= lr / math.sqrt(t+1) * grad  # update the weight tensor in place!!
                state['t'] = t + 1    # increment iteration count
        return loss

class AdamW(optim.Optimizer):
    def __init__(self, params, lr: torch.float, betas: Tuple[torch.float], eps: torch.float, weight_decay: torch.float):
        defaults = {'lr': lr}
        super().__init__(params, defaults)

        self.eps = eps
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.lr = lr
        self.weight_decay = weight_decay

        self.state = dict()
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {'m': torch.zeros_like(p.data), 'v': torch.zeros_like(p.data)}

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p] # get state associated with p
                t = state.get('t', 1) # get iteration number, start from 1
                grad = p.grad.data    # get gradient

                state['m'] = self.beta1 * state['m'] + (1-self.beta1) * grad     # update first moment estimate
                state['v'] = self.beta2 * state['v'] + (1-self.beta2) * grad * grad   # update second moment estimate
                lr_t = self.lr * math.sqrt(1-math.pow(self.beta2, t)) / (1-math.pow(self.beta1, t))   # adjusted alpha for iteration t
                p.data -= lr_t * state['m'] / (torch.sqrt(state['v']) + self.eps)   # update weights
                p.data -= self.lr * self.weight_decay * p.data   # apply weight decay
                state['t'] = t + 1    # increment iteration count
        return loss

def cosine_lr_schedule(t: int, alpha_max: torch.float, alpha_min: torch.float, warmup_steps: torch.float, cosine_steps: torch.float) -> torch.float:
    if t < warmup_steps:
        return t / (warmup_steps) * alpha_max
    if t > cosine_steps:
        return alpha_min
    return alpha_min + (1 + math.cos((t - warmup_steps)/(cosine_steps - warmup_steps) * math.pi))/2 * (alpha_max - alpha_min)


def gradient_clipping(params: Iterable[torch.nn.Parameter], M: torch.float, eps: torch.float = 1e-6):
    # compute global norm of all gradients
    total_norm = 0.0
    for param in params:
        if param.grad is None:
            continue
        param_norm = torch.norm(param.grad, p=2)
        total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    
    if total_norm > M:
        clip_coef = M / (total_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad *= clip_coef


# x is an integer array of token id's
def data_loader(x: torch.LongTensor, batch_size: int, context_length: int, device_str: str):

    device = torch.device(device_str)
    inputs = []
    targets = []

    for _ in range(batch_size):
        start = np.random.choice(len(x) - context_length)
        inputs.append(x[start : start + context_length])
        targets.append(x[start + 1 : start + context_length + 1])

    inputs = np.array(inputs)
    targets = np.array(targets)
    
    return (torch.LongTensor(inputs, device=device), torch.LongTensor(targets, device=device))
        
def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    obj = {
        'model': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'it': iteration
    }
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer_state'])
    return obj['it']

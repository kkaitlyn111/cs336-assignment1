import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, LongTensor
import time
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch.optim import Optimizer

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate. 
  
            for p in group["params"]: 
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 	
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
def training_loop():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters. 
        loss = (weights**2).mean() # Compute a scalar loss value. 
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients. 
        opt.step() # Run optimizer step.

	

class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.1):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                    
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # state initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize first moment (momentum) - m in the algorithm
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment (variance) - v in the algorithm
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                # get current internal state
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                # update first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m = β₁m + (1-β₁)g
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = β₂v + (1-β₂)g²
                
                # compute bias-corrected lr
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # compute adjusted lr
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # apply weight decay -- * this is decoupled from gradient update
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss


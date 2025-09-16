import torch
from einops import rearrange, einsum, reduce, repeat
import torch.nn as nn
import numpy as np
import math
from cs336_basics.transformer import softmax


# input preds [batch ... vocab], targets [batch ... vocab]
# average across batch
# canceled out log/exp terms in formula
def cross_entropy_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.float:
    val, _ = torch.max(predicted, dim=-1, keepdim=True) # subtract largest elem across vocab size (last dim)
    predicted = predicted - val
    sumexps = reduce(torch.exp(predicted), 'batch ... vocab -> batch ...', 'sum')
    target = target.unsqueeze(1)
    index = repeat(target, 'batch 1 -> batch vocab', vocab=predicted.size(dim=0))
    gathered = torch.gather(predicted, dim=-1, index=index)[:, 0]
    losses = torch.log(sumexps) - gathered
    loss = reduce(losses, 'batch ... -> 1 ...', 'mean')
    return loss


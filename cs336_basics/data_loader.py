import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, LongTensor
import time
import math


def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.LongTensor, torch.LongTensor]:
    inputs = []
    next_tokens = []
    for i in range(batch_size):
        start = np.random.choice(len(dataset) - context_length)
        inputs.append(dataset[start:start+context_length])
        next_tokens.append(dataset[start+1:start+context_length+1])
    
    return (torch.LongTensor(np.array(inputs),device=device), torch.LongTensor(np.array(next_tokens),device=device))


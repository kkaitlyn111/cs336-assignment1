import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int
import numpy.typing as npt
from torch import Tensor, LongTensor
import time
import math
from typing import Tuple


def data_loader(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.LongTensor, torch.LongTensor]:
    inputs = []
    next_tokens = []
    for i in range(batch_size):
        start = np.random.choice(len(dataset) - context_length)
        inputs.append(dataset[start:start+context_length])
        next_tokens.append(dataset[start+1:start+context_length+1])
    
    return (torch.LongTensor(np.array(inputs),device=device), torch.LongTensor(np.array(next_tokens),device=device))


def load_batch(data: np.memmap, batch_size: int, context_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
   load data from memory map array of token IDs
   returns (inputs, targets), moves to device
    """
    max_start = len(data) - context_length - 1
    
    # random sample of starting positions
    starts = np.random.randint(0, max_start, size=batch_size)
    
    inputs = []
    targets = []
    
    for start in starts:

        input_seq = data[start:start + context_length]
        # shifted by 1
        target_seq = data[start + 1:start + context_length + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    
    return inputs, targets


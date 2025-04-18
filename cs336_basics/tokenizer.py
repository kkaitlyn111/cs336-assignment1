import regex as re
from typing import *
from tqdm import tqdm
import logging
import os
from typing import BinaryIO
from .train_bpe import *

class Tokenizer:
    def __init__(self, special_tokens: list[str] = None):
        self.special_tokens = special_tokens
        self.vocab = {}

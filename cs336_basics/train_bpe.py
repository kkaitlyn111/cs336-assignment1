import regex as re
from typing import *
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_NUM_TOKENS = 256


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], progress_bar: bool = True) -> list[dict[int, bytes], list[tuple[bytes, bytes]]]:

    logging.info("Initializing vocabulary")

    # initialize the vocabulary with 256 bytes

    vocab = {i: bytes([i]) for i in range(DEFAULT_NUM_TOKENS)}

    # add special tokens to vocab

    vocab.update({DEFAULT_NUM_TOKENS + i: special.encode("UTF-8") for i, special in enumerate(special_tokens)})

    pretoken_freq = read_txt_file(input_path, special_tokens)
    merges = {}
        
    return vocab, merges


def read_txt_file(input_path: str, special_tokens: list[str]):
    """
    reads file, removes special tokens, counts token frequencies
    returns pretoken freq table
    """
    with open(input_path, "r") as file:
        txt = file.read()

    logging.info(f"Read input file (length of text is {len(txt)})")
    
    for special in special_tokens:
        txt = txt.replace(special, "")

    logging.info(f"Removed {len(special_tokens)} unique special tokens from text)")

    pretokenized = pretokenize(txt)
    freq_table = {
        tuple(bytes([b]) for b in pretoken.encode("utf-8")): pretokenized[pretoken] # tuple of bytes, where even one-byte pretokens are stored as indiv bytes
        for pretoken in pretokenized
    }
    return freq_table

def pretokenize(txt: str) -> dict[str, int]:
    """
    pretokenizes based on GPT-2 regex
    returns dict {pretoken : freq}
    """
    logging.info("Counting pretoken frequencies")
    pretokens = (match.group(0) for match in re.finditer(PRETOKENIZE_REGEX, txt))
    counter = Counter(pretokens)
    logging.info(f"{sum(counter.values())} pretokens found")
    return counter
    

    
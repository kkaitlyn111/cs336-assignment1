import regex as re
from typing import *
from tqdm import tqdm
import logging
from .pretokenization_example import find_chunk_boundaries
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_NUM_TOKENS = 256


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = None, progress_bar: bool = True) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    logging.info("Initializing vocabulary")

    # initialize the vocabulary with 256 bytes

    vocab = {i: bytes([i]) for i in range(DEFAULT_NUM_TOKENS)}

    # add special tokens to vocab

    vocab.update({DEFAULT_NUM_TOKENS + i: special.encode("UTF-8") for i, special in enumerate(special_tokens)})

    pretoken_freq = read_txt_file(input_path, num_workers, special_tokens)
    pair_freq = initialize_pair_frequency(pretoken_freq)

    initial_vocab_size = len(vocab)
    logging.info("Training BPE (merging)")
    progbar = tqdm(total=vocab_size-initial_vocab_size)
    merges = []
    while(len(vocab) < vocab_size):
        most_freq = max(pair_freq, key=lambda k: (pair_freq[k], k))
        merges.append(most_freq)

        # update vocab table
        
        new_vocab_id = max(vocab.keys()) + 1
        new_token = b"".join(most_freq)
        vocab[new_vocab_id] = new_token

        # update pretoken and pair tables
        new_pretoken_freq = {}
        for pretoken, freq in pretoken_freq.items():
            for i in range(0, len(pretoken) - 1):
                pair = [pretoken[i], pretoken[i+1]]
                if pair == most_freq:
                        pretoken = pretoken[0:i] + new_token + pretoken[i+2:] # replace previous pair with single merged token

                        # update pair table
                        prev_pair = [pretoken[i-1], pretoken[i]] if i > 0 else None
                        next_pair = [pretoken[i+1], pretoken[i+2]] if i+2 < len(pretoken) else None
                        
                        if prev_pair:
                            pair_freq[prev_pair] -= freq
                            new_left_pair = [pretoken[i-1], new_token]
                            pair_freq[new_left_pair] = pair_freq.get(new_left_pair, 0) + freq
                            
                        if next_pair:
                            pair_freq[prev_pair] -= freq
                            new_right_pair = [new_token, pretoken[i+2]]
                            pair_freq[new_right_pair] = pair_freq.get(new_right_pair, 0) + freq
                        
                        pair_freq[most_freq] -= freq    
                        
                new_pretoken_freq[pretoken] = freq
        pretoken_freq = new_pretoken_freq
        progbar.update(len(vocab) - initial_vocab_size - progbar.n)

    progbar.close() 
    return vocab, merges


def read_txt_file(input_path: str, num_workers: int, special_tokens: list[str]):
    """
    reads file and chunks based on valid special token boundaries 
    pretokenizes each chunk separately: removes special tokens, counts pretoken frequencies
    returns final pretoken freq table (sum of all chunks)
    """
    logging.info(f"Processing file with {num_workers} workers")
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, "<|endoftext|>".encode("utf-8"))
        
        chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        print("CHUNK ARGS",chunk_args)
        
        full_freq_table = Counter()
        with Pool(num_workers) as pool:
            results = pool.starmap(process_chunk, chunk_args)
            for local_freq_table in results:
                full_freq_table.update(local_freq_table)

        logging.info(f"Final pretokens: {len(full_freq_table)} unique entries")
        return full_freq_table

def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]):
    """
    Process a single chunk of the file.
    """
    split_pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    freq_table = Counter()
    
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            for piece in re.split(split_pattern, chunk):
                if piece in special_tokens:
                    continue
                pretokenized = pretokenize(piece)
                for pretoken, count in pretokenized.items():
                    key = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    freq_table[key] += count
    except Exception as e:
        logging.error(f"Error processing chunk {start}-{end}: {str(e)}")
        raise
    
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

def initialize_pair_frequency(pretoken_freq: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """
    takes in pretoken occurence table
    returns a pair frequency table
    we'll initialize the whole thing only once, and do modifications afterward
    """
    logging.info("Count all pair frequencies to initialize pair freq table")
    pair_freq = {}
    for pretoken, freq in pretoken_freq.items():
        for i in range(0, len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i+1])
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq
    return pair_freq


    
    
    
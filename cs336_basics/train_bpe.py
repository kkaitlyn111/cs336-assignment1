import regex as re
from typing import *
from tqdm import tqdm
import logging
from .pretokenization_example import find_chunk_boundaries
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool
import multiprocessing
import time
import pickle


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_NUM_TOKENS = 256


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = None, progress_bar: bool = True, pretoken_file: str = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    logger = logging.getLogger(__name__)
    total_start_time = time.time()
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    logger.info(f"Using {num_workers} worker processes")
    
    # vocab initialization
    vocab = {i: bytes([i]) for i in range(DEFAULT_NUM_TOKENS)}
    vocab.update({DEFAULT_NUM_TOKENS + i: special.encode("UTF-8") for i, special in enumerate(special_tokens)})

    # pretokenization or load from file
    if pretoken_file is not None:
        logger.info(f"Loading pretoken frequency table from {pretoken_file}")
        with open(pretoken_file, "rb") as f:
            pretoken_freq = pickle.load(f)
        logger.info(f"Loaded {len(pretoken_freq)} unique pretokens from file")
        pretokenization_time = 0.0
    else:
        logger.info("Reading and pretokenizing input file")
        pretoken_start_time = time.time()
        pretoken_freq = read_txt_file(input_path, num_workers, special_tokens)
        pretokenization_time = time.time() - pretoken_start_time
        logger.info(f"Pretokenization completed in {pretokenization_time:.2f} seconds")
    
    # pair frequency initialization
    pair_freq = initialize_pair_frequency(pretoken_freq)

    initial_vocab_size = len(vocab)
    target_merges = vocab_size - initial_vocab_size
    logger.info(f"Starting BPE training with {target_merges} merges to perform")
    
    merges = []
    merge_start_time = time.time()
    merge_times = []
    merge_stats = {
        'find_max_pair': [],
        'update_pretoken_freq': []
    }
    
    pbar = tqdm(total=target_merges, desc="Training BPE", disable=not progress_bar)
    merge_num = 0
    try:
        while(len(vocab) < vocab_size):
            # logger.info(f"Starting merge {merge_num+1}/{target_merges} (vocab size: {len(vocab)}, pairs: {len(pair_freq)})")
            merge_iter_start = time.time()
            # time finding the most frequent pair
            find_max_start = time.time()
            most_freq = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
            find_max_time = time.time() - find_max_start
            # logger.info(f"Found most frequent pair in {find_max_time:.2f} seconds: {most_freq}")
            merge_stats['find_max_pair'].append(find_max_time)
            merges.append(most_freq)
            new_vocab_id = max(vocab.keys()) + 1
            new_token = b"".join(most_freq)
            vocab[new_vocab_id] = new_token

            # time updating pretoken and pair tables
            update_pretoken_start = time.time()
            new_pretoken_freq = {}
            pretoken_count = 0
            pair_updates = 0
            
            for pretoken, freq in pretoken_freq.items():
                pretoken_count += 1
                new_pretoken = list(pretoken)
                i = 0
                modified = False
                while i < len(new_pretoken) - 1:
                    pair = (new_pretoken[i], new_pretoken[i+1])
                    if pair == most_freq:
                        pair_updates += 1
                        # first, decrease frequency of all affected pairs
                        if i > 0:
                            old_pair = (new_pretoken[i-1], new_pretoken[i])
                            pair_freq[old_pair] = pair_freq.get(old_pair, 0) - freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]
                        
                        if i < len(new_pretoken) - 2:
                            old_pair = (new_pretoken[i+1], new_pretoken[i+2])
                            pair_freq[old_pair] = pair_freq.get(old_pair, 0) - freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]
                        
                        # replace the pair with the merged token
                        new_pretoken[i:i+2] = [new_token]
                        modified = True
                        
                        # update frequencies of new pairs
                        if i > 0:
                            new_pair = (new_pretoken[i-1], new_token)
                            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                        
                        if i < len(new_pretoken) - 1:
                            new_pair = (new_token, new_pretoken[i+1])
                            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                    else:
                        i += 1
                
                if modified:
                    new_pretoken_freq[tuple(new_pretoken)] = freq
                else:
                    new_pretoken_freq[pretoken] = freq
            
            update_time = time.time() - update_pretoken_start
            merge_stats['update_pretoken_freq'].append((update_time, pretoken_count, pair_updates))
            #logger.info(f"Updated pretoken frequencies in {update_time:.2f} seconds (pretoken_count={pretoken_count}, pair_updates={pair_updates})")
            del pair_freq[most_freq]
            pretoken_freq = new_pretoken_freq
            merge_time = time.time() - merge_iter_start
            #logger.info(f"Completed merge {merge_num+1} in {merge_time:.2f} seconds")
            merge_num += 1
            pbar.update(1)
    
        total_merge_time = time.time() - merge_start_time
        logger.info(f"\nBPE training completed in {total_merge_time:.2f} seconds")
        
        logger.info("\nWithin Merge Statistics:")
        

        find_max_total = sum(merge_stats['find_max_pair'])
        find_max_percentage = (find_max_total / total_merge_time) * 100
        logger.info("Find max pair:")
        logger.info(f"  Total time: {find_max_total:.2f}s ({find_max_percentage:.1f}% of merge time)")
        logger.info(f"  Avg time: {find_max_total/len(merges):.6f}s")
        

        update_stats = merge_stats['update_pretoken_freq']
        update_total = sum(t[0] for t in update_stats)
        update_percentage = (update_total / total_merge_time) * 100
        logger.info("Update pretoken frequencies:")
        logger.info(f"  Total time: {update_total:.2f}s ({update_percentage:.1f}% of merge time)")
        logger.info(f"  Avg time: {update_total/len(merges):.6f}s")
        
    finally:
        pbar.close()
    
    total_time = time.time() - total_start_time
    logger.info("\nTotal Time Breakdown:")
    logger.info(f"Pretokenization: {pretokenization_time:.2f}s ({pretokenization_time/total_time*100:.1f}%)")
    logger.info(f"Merging process: {total_merge_time:.2f}s ({total_merge_time/total_time*100:.1f}%)")
    logger.info(f"Total time: {total_time:.2f}s")
    
    return vocab, merges


def read_txt_file(input_path: str, num_workers: int, special_tokens: list[str]):
    """
    reads file and chunks based on valid special token boundaries 
    pretokenizes each chunk separately: removes special tokens, counts pretoken frequencies
    returns final pretoken freq table (sum of all chunks)
    """
    logger = logging.getLogger(__name__)
    total_start_time = time.time()
    
    # use the first special token for chunking if available, otherwise use default
    split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
    split_token_bytes = split_token.encode("utf-8")
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token_bytes)
        logger.info(f"Found {len(boundaries)-1} chunks using '{split_token}' as delimiter")
        
        chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        full_freq_table = Counter()
        chunk_times = []
        chunk_stats = {
            'pretokenize_time': [],
            'encoding_time': []
        }
        
        with Pool(num_workers) as pool:
            # add a progress bar for chunk processing
            with tqdm(total=len(chunk_args), desc="Pretokenizing chunks") as pbar:
                for local_freq_table, chunk_time, stats in pool.starmap(process_chunk, chunk_args):
                    full_freq_table.update(local_freq_table)
                    chunk_times.append(chunk_time)
                    for key in chunk_stats:
                        chunk_stats[key].extend(stats[key])
                    pbar.update(1)

        # log detailed pretokenization statistics
        logger.info("\nPretokenization Statistics:")

        total_time = time.time() - total_start_time
        logger.info(f"Total pretokenization time: {total_time:.2f} seconds")
        logger.info(f"Final pretokens: {len(full_freq_table)} unique entries")
        return full_freq_table

def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]):
    """
    process a single chunk of the file
    """
    logger = logging.getLogger(__name__)
    chunk_start_time = time.time()
    #logger.info(f"Starting process_chunk: {start} to {end}")
    split_pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    freq_table = Counter()
    
    stats = {
        'pretokenize_time': [],
        'encoding_time': []
    }
    
    try:
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            pieces = re.split(split_pattern, chunk)
            for piece in pieces:
                if piece in special_tokens:
                    continue
                    
                # time pretokenization
                pretoken_start = time.time()
                pretokenized = pretokenize(piece)
                stats['pretokenize_time'].append(time.time() - pretoken_start)
                
                # time encoding and frequency counting
                encoding_start = time.time()
              
                for pretoken, count in pretokenized.items():
                    key = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    freq_table[key] += count
                stats['encoding_time'].append(time.time() - encoding_start)
        #logger.info(f"Finished process_chunk: {start} to {end}")
    except Exception as e:
        logger.error(f"Exception in process_chunk ({start} to {end}): {e}")
        raise
    
    chunk_time = time.time() - chunk_start_time
    return freq_table, chunk_time, stats

def pretokenize(txt: str) -> dict[str, int]:
    """
    pretokenizes based on GPT-2 regex
    returns dict {pretoken : freq}
    """
    pretokens = (match.group(0) for match in re.finditer(PRETOKENIZE_REGEX, txt))
    counter = Counter(pretokens)
    return counter

def initialize_pair_frequency(pretoken_freq: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    """
    takes in pretoken occurence table
    returns a pair frequency table
    we'll initialize the whole thing only once, and do modifications afterward
    """
    pair_freq = {}
    for pretoken, freq in pretoken_freq.items():
        for i in range(0, len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i+1])
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq
    return pair_freq


    
    
    
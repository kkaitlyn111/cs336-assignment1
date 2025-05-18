import os
import torch
from cs336_basics.train_bpe import train_bpe
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.pretokenization_example import pretokenize_file_only
import time
import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

def process_lines(args):
    start, end, train_path, vocab_path, merges_path, special_tokens = args
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    ids = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            ids.extend(tokenizer.encode(line))
    return ids

def generate_owt():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    

    train_path = "/data/a1-basics/owt_train.txt"
    valid_path = "/data/a1-basics/owt_valid.txt"
    vocab_path = "/data/c-kaitwang/owt_vocab.pkl"
    merges_path = "/data/c-kaitwang/owt_merges.pkl"
    pretokens_train_path = "/data/c-kaitwang/owt_train_pretokens.npy"
    pretokens_valid_path = "/data/c-kaitwang/owt_valid_pretokens.npy"
    pretokens_internal_path = "/data/c-kaitwang/owt_internal.pkl"

    # train_path = "/home/c-kaitwang/cs336_basics/tinystoriesexample.txt"
    # valid_path = "/home/c-kaitwang/cs336_basics/tinystoriesexample.txt"
    # vocab_path = "/data/c-kaitwang/example_vocab.pkl"
    # merges_path = "/data/c-kaitwang/example_merges.pkl"
    # pretokens_train_path = "/data/c-kaitwang/example_train_pretokens.npy"
    # pretokens_valid_path = "/data/c-kaitwang/example_valid_pretokens.npy"
    # pretokens_internal_path = "/data/c-kaitwang/example_internal.pkl"

    # train_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt"
    # valid_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt"
    # vocab_path = "/data/c-kaitwang/tinystories_vocab_test.pkl"
    # merges_path = "/data/c-kaitwang/tinystories_merges_test.pkl"
    # pretokens_train_path = "/data/c-kaitwang/tinystories_pretokens_test.npy"
    # pretokens_valid_path = "/data/c-kaitwang/tinystories_valid_pretokens_test.npy"
    # pretokens_internal_path = "/data/c-kaitwang/example_internal.pkl"



    
    # Parameters for BPE training
    
    vocab_size = 30000
    special_tokens = ["<|endoftext|>"]

    # first train tokenizer using train set

    #print("First pretokenize and save to file")
    #pretokenize_file_only(
    #    input_path=train_path,
    #    pkl_output_path=pretokens_internal_path,
    #    max_workers = 64,
    #    special_tokens=special_tokens
    #)

    print("Now BPE training...")
    vocab, merges = train_bpe(input_path = train_path, vocab_size = vocab_size, special_tokens = special_tokens, pretokens_internal_path=pretokens_internal_path)


    # vocab, merges = train_bpe(
    #     input_path=train_path,
    #     vocab_size=vocab_size,
    #     special_tokens=special_tokens,
    #     num_workers=16,  
    #     progress_bar=True,
    #     pretoken_file=pretokens_internal_path
    # )

    print("\nSaving vocabulary and merges...")
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    tokenizer.save(vocab_path, merges_path)

    # print("\nLoading vocabulary and merges from files...")
    # tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    # print("Loaded tokenizer from files.")
    

    print("\nGenerating training pretoken token IDs using trained tokenizer with progress bar...")

    # count total lines for progress bar
    with open(train_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    max_lines = total_lines  # process up to this many lines
    num_workers = min(64, multiprocessing.cpu_count())
    
    # split line indices into chunks
    chunk_size = (max_lines + num_workers - 1) // num_workers
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, max_lines), train_path, vocab_path, merges_path, special_tokens)
        for i in range(num_workers)
    ]

    token_ids = []
    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=max_lines, desc="Tokenizing", unit="lines") as pbar:
            for ids in pool.imap(process_lines, chunks):
                token_ids.extend(ids)
                pbar.update(chunk_size)

    np.save(pretokens_train_path, np.array(token_ids, dtype=np.int32))
    print(f"Saved pretoken token IDs to {pretokens_train_path}")

    print("\nDone! Files have been regenerated:")
    print(f"- Vocabulary: {vocab_path}")
    print(f"- Merges: {merges_path}")
    print(f"- Pretokens: {pretokens_train_path}")

        

    tokens = np.load(pretokens_train_path)
    print(tokens[0:100])  # Print the first 100 tokens
    print(tokenizer.decode(tokens[0:1000]))


    print("\nGenerating validation pretoken token IDs using trained tokenizer with progress bar...")

        # count total lines for progress bar
    with open(valid_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    max_lines = total_lines  # process up to this many lines

    token_ids = []
    with open(valid_path, 'r', encoding='utf-8') as f, tqdm(total=max_lines, desc="Tokenizing", unit="lines") as pbar:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            ids = tokenizer.encode(line)
            token_ids.extend(ids)
            pbar.update(1)

    np.save(pretokens_valid_path, np.array(token_ids, dtype=np.int32))
    print(f"Saved pretoken token IDs to {pretokens_valid_path}")

    print("\nDone! Files have been regenerated:")
    print(f"- Vocabulary: {vocab_path}")
    print(f"- Merges: {merges_path}")
    print(f"- Pretokens: {pretokens_valid_path}")

        

    tokens = np.load(pretokens_valid_path)
    print(tokens[0:100])  # Print the first 100 tokens
    print(tokenizer.decode(tokens[0:1000]))




if __name__ == "__main__":
    generate_owt() 

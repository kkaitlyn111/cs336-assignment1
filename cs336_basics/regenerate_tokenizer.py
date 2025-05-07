import os
import torch
from cs336_basics.train_bpe import train_bpe
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.pretokenization_example import pretokenize_file_only
import time
import numpy as np
import pickle

import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()

def regenerate_tokenizer():
    # Check for CUDA availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Paths for the large TinyStories dataset
    train_path = "/data/a1-basics/TinyStoriesV2-GPT4-train.txt"
    valid_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt"
    vocab_path = "/data/c-kaitwang/tinystories_vocab.pkl"
    merges_path = "/data/c-kaitwang/tinystories_merges.pkl"
    pretokens_path = "/data/c-kaitwang/tinystories_pretokens.npy"
    pretokens_internal_path = "/data/c-kaitwang/tinystories_pretokens.pkl"

    # train_path = "/data/a1-basics/owt_train.txt"
    # vocab_path = "/data/c-kaitwang/owt_vocab.pkl"
    # merges_path = "/data/c-kaitwang/owt_merges.pkl"
    # pretokens_path = "/data/c-kaitwang/owt_pretokens.npy"
    # pretokens_internal_path = "/data/c-kaitwang/owt_pretokens.pkl"
    
    # Parameters for BPE training
    
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    if 0: 

        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

        tokenizer.pretokenize_file(input_path = train_path, output_path = pretokens_path)

        print("\nDone! Files have been regenerated:")
        print(f"- Vocabulary: {vocab_path}")
        print(f"- Merges: {merges_path}")
        print(f"- Pretokens: {pretokens_path}")
    

    if 1:

        print("First pretokenize and save to file")
        pretokenize_file_only(
            input_path=train_path,
            pkl_output_path=pretokens_internal_path,
            max_workers = 16,
            special_tokens=special_tokens
        )
        
        print("Now BPE training...")
        vocab, merges = train_bpe(
            input_path=train_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            num_workers=16,  
            progress_bar=True,
            pretoken_file=pretokens_internal_path
        )
        
        print("\nSaving vocabulary and merges...")
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
        tokenizer.save(vocab_path, merges_path)

        # Generate pretoken token IDs using the trained tokenizer
        print("\nGenerating pretoken token IDs using trained tokenizer...")
        with open(train_path, 'r', encoding='utf-8') as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        np.save(pretokens_path, np.array(token_ids, dtype=np.int32))
        


        # with open(pretokens_internal_path, 'rb') as f:
        #     pretoken_freq = pickle.load(f)
        #     print(f"Type of pretoken_freq: {type(pretoken_freq)}")
        #     print(f"Number of unique pretokens: {len(pretoken_freq)}")
        #     print("Sample pretokens and their frequencies:")
        #     for i, (pretoken, freq) in enumerate(pretoken_freq.items()):
        #         print(f"  Pretoken {i}: {pretoken} (freq: {freq})")
        #         if i >= 4:  # print only the first 5
        #             break
            
        #     #
        # all_token_ids = []
        # for pretoken, freq in pretoken_freq.items():
        #     text = b''.join(pretoken).decode('utf-8', errors='ignore')
        #     token_ids = tokenizer.encode(text)
        #     all_token_ids.extend(token_ids * freq)
        # np.save(pretokens_path, np.array(all_token_ids, dtype=np.int32))
        print(f"Saved pretoken token IDs to {pretokens_path}")

        print("\nDone! Files have been regenerated:")
        print(f"- Vocabulary: {vocab_path}")
        print(f"- Merges: {merges_path}")
        print(f"- Pretokens: {pretokens_path}")

if __name__ == "__main__":
    regenerate_tokenizer() 
import torch
import numpy as np
import os
from cs336_basics.training_loop import parse_args, main
from cs336_basics.simple_tokenizer import Tokenizer

def test_with_tinystories():
    args = parse_args()
    
    args.train_path = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    args.valid_path = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt" 
    args.vocab_path = "tinystories_vocab.pkl"
    args.merges_path = "tinystories_merges.pkl"
    args.pretokens_path = "/Users/kaitlynwang/assignment1-basics/data/tinystories_pretokens.npy" 
    
    args.vocab_size = 10000
    args.context_length = 128
    args.batch_size = 4
    args.d_model = 64
    args.num_heads = 4
    args.num_layers = 2
    args.d_ff = 128
    args.learning_rate = 5.18e-4
    args.max_steps = 5000
    args.max_seq_len = 256  # must be greater than context_length
    args.min_loss_threshold = 2
    
    args.device = "mps"
    args.use_compile = True
    args.use_memmap = True
    args.use_parallel_pretokenize = True
    
    print("Starting training test with TinyStories...")
    print(f"Max steps set to: {args.max_steps}")
    main(args) 

if __name__ == "__main__":
    test_with_tinystories() 
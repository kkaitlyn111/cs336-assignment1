import torch
import numpy as np
import os
from cs336_basics.training_loop import parse_args, main
from cs336_basics.simple_tokenizer import Tokenizer

def test_with_tinystories():
    args = parse_args()
    
    args.train_path = "data/TinyStoriesExample.txt"
    args.valid_path = "data/TinyStoriesExample.txt"  # using same file for validation
    args.vocab_path = "tinystories_vocab.pkl"
    args.merges_path = "tinystories_merges.pkl"
    args.pretokens_path = "data/TinyStoriesExample_pretokens.npy" 
    
    args.vocab_size = 10000
    args.context_length = 128
    args.batch_size = 4
    args.d_model = 64
    args.num_heads = 4
    args.num_layers = 2
    args.d_ff = 128
    args.learning_rate = 1e-4
    args.max_steps = 200
    args.max_seq_len = 256  # must be greater than context_length
    
    args.device = "mps"
    args.use_compile = False  
    args.use_memmap = False 
    args.use_parallel_pretokenize = False 
    
    print("Starting training test with TinyStories...")
    print(f"Max steps set to: {args.max_steps}")
    main(args) 

if __name__ == "__main__":
    test_with_tinystories() 
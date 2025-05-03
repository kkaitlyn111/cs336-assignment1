import os
import torch
from cs336_basics.train_bpe import train_bpe
from cs336_basics.simple_tokenizer import Tokenizer

def regenerate_tokenizer():
    # Check for CUDA availability
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Paths for the large TinyStories dataset
    train_path = "/data/a1-basics/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "/data/c-kaitwang/tinystories_vocab.pkl"
    merges_path = "/data/c-kaitwang/tinystories_merges.pkl"
    pretokens_path = "/data/c-kaitwang/tinystories_pretokens.npy"
    
    # Parameters for BPE training
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    print("Starting BPE training...")
    vocab, merges = train_bpe(
        input_path=train_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=4,  # Using 4 workers for stability
        progress_bar=True
    )
    
    print("\nSaving vocabulary and merges...")
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    tokenizer.save(vocab_path, merges_path)
    
    print("\nGenerating pretokens...")
    tokenizer.pretokenize_file(
        input_path=train_path,
        output_path=pretokens_path,
        use_parallel=True
    )
    
    print("\nDone! Files have been regenerated:")
    print(f"- Vocabulary: {vocab_path}")
    print(f"- Merges: {merges_path}")
    print(f"- Pretokens: {pretokens_path}")

if __name__ == "__main__":
    regenerate_tokenizer() 
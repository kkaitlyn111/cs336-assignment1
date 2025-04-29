import psutil
import os
import time
import multiprocessing
from cs336_basics.train_bpe import train_bpe
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.simple_tokenizer import Tokenizer 

# Example usage:
if __name__ == "__main__":
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    print("Processors:", multiprocessing.cpu_count())

    tokenizer = Tokenizer(max_workers=3)
    
    # Start tracking memory and time
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Train on TinyStories dataset with smaller vocabulary size for testing
    tokenizer.train(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        #input_path="data/owt_train.txt",
        vocab_size=10000, 
        #vocab_size=32000,
        special_tokens=["<|endoftext|>"]
    )
    
    # Calculate training time and memory usage
    end_time = time.time()
    end_memory = get_memory_usage()
    training_time = (end_time - start_time) / 60  # Convert to minutes for smaller test
    memory_used = end_memory - start_memory
    
    # Save the trained tokenizer
    tokenizer.save("tinystories_vocab.pkl", "tinystories_merges.pkl")
    
    # Analyze vocabulary
    longest_token = max(tokenizer.vocab.values(), key=len)
    longest_token_length = len(longest_token)
    
    # Print detailed results
    print("\nTraining Results:")
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Longest token length: {longest_token_length} bytes")
    print(f"Longest token: {longest_token}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of merges: {tokenizer.get_merges_count()}")
    
    # Example encoding/decoding
    text = "Hello, world!"
    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)
    
    # Example streaming encoding
    texts = ["Hello", " world", "!"]
    streamed_ids = list(tokenizer.encode_iterable(texts))
    
    print(f"\nExample Results:")
    print(f"Original text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded_text}")
    print(f"Streamed token IDs: {streamed_ids}") 
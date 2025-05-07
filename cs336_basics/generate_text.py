import torch
import sys
from cs336_basics.transformer import TransformerLM
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.decoder import generate_text

# Model and tokenizer hyperparameters (from training_loop.py)

vocab_size = 10000
context_length = 128
d_model = 128
num_heads = 2
num_layers = 1
d_ff = 256
max_steps = 2000
max_seq_len = 128 # must be greater than context_length

rope_theta = 10000
device = torch.device("cuda")

# 1. Load your trained model checkpoint
checkpoint_path = "/data/c-kaitwang/checkpoints/final_model.pt"

# 2. Initialize the model
model = TransformerLM(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    vocab_size=vocab_size,
    context_length=context_length,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    theta=rope_theta,
    device=device
)

# Load state_dict from checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
print(checkpoint.get("run_info", "No run info found"))
model.load_state_dict(checkpoint['model'])
model.eval()


vocab_path = "/data/c-kaitwang/tinystories_vocab.pkl"
merges_path = "/data/c-kaitwang/tinystories_merges.pkl"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


prompt = "Once upon a time there was a little boy named "


output = generate_text(
    model, tokenizer, prompt,
    new_tokens_limit=256,   
    temperature=0.8,       # increase -> more creative 
    top_p=0.9      # increase -> more diverse
)

print("Generated text:\n", output)

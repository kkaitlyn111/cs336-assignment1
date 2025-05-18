import torch
import sys
from cs336_basics.transformer import TransformerLM
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.decoder import generate_text

# model and tokenizer hyperparameters (from training_loop.py)

vocab_size = 30000
context_length = 256
d_model = 512
num_heads = 16
num_layers = 4
d_ff = 1344
max_steps = 5000
max_seq_len = 512 # must be greater than context_length

rope_theta = 10000

device = torch.device("cuda")

# load trained model checkpoint -- best model or final model
checkpoint_path = "/data/c-kaitwang/checkpoints/best_model.pt"

# initialize the model
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

model = model.to(device)

# load state_dict from checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)
print(checkpoint.get("run_info", "No run info found"))
model.load_state_dict(checkpoint['model'])
model.eval()


vocab_path = "/data/c-kaitwang/owt_vocab.pkl"
merges_path = "/data/c-kaitwang/owt_merges.pkl"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


prompt = "Education and learning are essential for children's "


output = generate_text(
    model, tokenizer, prompt,
    new_tokens_limit=256,   
    temperature=0.9,       # increase -> more creative 
    top_p=0.85      # increase -> more diverse
)

print("Generated text:\n", output)

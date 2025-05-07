import pdb
import torch
from cs336_basics.transformer import TransformerLM

def test_model_forward():
    # Example model parameters (adjust as needed)
    d_model = 512
    num_heads = 16
    d_ff = 1344
    vocab_size = 10000
    context_length = 32
    num_layers = 4
    max_seq_len = 32
    theta = 10000
    device = torch.device("cpu")

    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        theta=theta,
        device=device
    )

    # Create a dummy input batch
    batch_size = 2
    dummy_input = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long, device=device)
    pdb.set_trace()  # Set a breakpoint here
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, context_length, vocab_size), "Logits shape mismatch!"

if __name__ == "__main__":
    test_model_forward()
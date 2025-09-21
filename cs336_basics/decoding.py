import torch
import torch.nn as nn
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.training import load_checkpoint, AdamW


def softmax_temp_scaling(x: torch.Tensor, dim: int, temp: float | None):

    if temp is None:
        temp = 1.0

    if temp == 0.0:
        argmax_idx = x.argmax(dim=dim, keepdim=True)
        return torch.zeros_like(x).scatter_(dim, argmax_idx, 1.0)

    # clamp to 1e-6
    temp = max(float(temp), 1e-6)
    x = x - torch.amax(x, dim=dim, keepdim=True)
    ex = torch.exp(x / temp)
    denom = ex.sum(dim=dim, keepdim=True)
    return ex / denom


def generate(model: nn.Module, tokenizer: Tokenizer, prompt: str, max_tokens: int = 256, temperature: float | None = None, top_p: float | None= None, device ='cuda'):

    input_ids = tokenizer.encode(prompt)  # list[str]
    generated_ids = input_ids # list[str]
    stop_token = tokenizer.encode("<|endoftext|>")[0]
    context_length = model.context_length
    
    model.to(device).eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            if len(generated_ids) > context_length:
                input_ids = generated_ids[-context_length:]
            else:
                input_ids = generated_ids
            
            input = torch.tensor(input_ids, dtype=torch.long, device=device)
            logits = model(input) # logits is of shape [batch ... seqlen vocab]
            next_token = logits[..., -1, :] # take the last token in the seqlen dimension, which is the new token
            # [batch ... vocab]
            probs = softmax_temp_scaling(next_token, dim=-1, temp=temperature)  # converted to probs
            # [batch ... vocab]

            if top_p is not None and top_p < 1.0:
                vals, indices = torch.sort(probs, dim=-1, descending=True)  # sorting vocab size dimension by prob, descending
                cumulative = torch.cumsum(vals, dim=-1)
                mask = cumulative > top_p
                mask = torch.roll(mask, shifts=1, dims=-1)  # shifts to keep the boundary token
                mask[..., 0] = False  # makes sure we have at least 1 prob that's kept
                vals = vals.masked_fill(mask, 0.0)
                vals = vals / vals.sum(dim=-1, keepdim=True)
                sampled_sorted = torch.multinomial(vals, num_samples=1)
                next_token_id = torch.gather(indices, dim=-1, index=sampled_sorted).item()
                
            else: 
                next_token_id = torch.multinomial(probs, num_samples=1).item() # shape [batch 1] where the value is the actual token ID
            
            generated_ids.append(next_token_id)

            if next_token_id == stop_token:
                break

        return tokenizer.decode(generated_ids)


if __name__ == '__main__':
    model = TransformerLM(vocab_size=10000,context_length=256,d_model=512,num_heads=16,num_layers=4, d_ff=1344, theta=10000, device='cuda',dtype=torch.float32)
    # optimizer = AdamW(params=model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.1, eps=1e-8)
    load_checkpoint('/juice5b/scr5b/kaitwang/cs336/a1/models/best_9900', model)


    data_path = "/juice5b/scr5b/kaitwang/cs336/data"
    vocab_path = f"{data_path}/tinystories_vocab.pkl"
    merges_path = f"{data_path}/tinystories_merges.pkl"
    special_tokens = ["<|endoftext|>"]


    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    new_tokens = generate(model, tokenizer, 'Once upon a time there was a rat named', max_tokens=1000, temperature=0.7, top_p = 0.9)
    print(new_tokens)
import torch
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM

def decode(model: TransformerLM, tokenizer: Tokenizer, prompt: str, new_tokens_limit: int = 32, temperature: float = 0.7, top_p: float = 0.9):

    stop_token = tokenizer.encode("<|endoftext|>")[0]
    input_ids = tokenizer.encode(prompt)
    device = model.device
    context_length = model.context_length
    
    # initialize generation
    generated_ids = input_ids.copy()
    model.eval()
    
    with torch.no_grad():
        for _ in range(new_tokens_limit):
            # prepare input
            if len(generated_ids) > context_length:
                input_ids = generated_ids[-context_length:]
            else:
                input_ids = generated_ids
            
            # convert to tensor and get model output
            input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
            logits = model(input_tensor)
            
            # get next token probabilities
            next_token_logits = logits[0, -1, :]
            
            # apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # append to generated sequence
            generated_ids.append(next_token)
            
            # stop if we hit the end token
            if next_token == stop_token:
                break
    
    return tokenizer.decode(generated_ids)

def generate_text(model: TransformerLM, tokenizer: Tokenizer, prompt: str, **kwargs):
    return decode(model, tokenizer, prompt, **kwargs)

    

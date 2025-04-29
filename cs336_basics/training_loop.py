import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import wandb
from tqdm import tqdm
from cs336_basics.training import cross_entropy_loss, get_cosine_lr, gradient_clipping, save_checkpoint
from cs336_basics.optimizers import AdamW, SGD
from cs336_basics.data_loader import data_loader
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument("--train_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--valid_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/tinystories_merges.pkl")
    parser.add_argument("--pretokens_path", type=str)

    # data loading params
    parser.add_argument("--vocab_size", type=int, default = 10000)
    parser.add_argument("--context_length", type=int, default = 256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_memmap", type=bool, default=True)

    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--pad_token", type=str, default="<|pad|>")

    # model params
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=None)

    # training and optimizer params
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gradient_clip_M", type=float, default=0.01)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=100000)

    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--save_best_only", type=bool, default=True)

    # device 
    parser.add_argument("--device", type=str, default="mps")
    
    return parser.parse_args()

def load_data_memmap(file_path, dtype=np.int32):
    """Load data using memory mapping for efficient memory usage."""
    return np.memmap(file_path, dtype=dtype, mode='r')

def train_step(model, optimizer, train_data, args, device):
    inputs, targets = data_loader(train_data, args.batch_size, args.context_length, device)
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), args.gradient_clip_M)
    optimizer.step()
    
    return loss.item()

def evaluate(model, valid_data, args, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = data_loader(valid_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
    model.train()
    return loss.item()

def main():
    args = parse_args()
    
    # Setup
    wandb.init(project=args.wandb_project, config=vars(args))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load data and model
    tokenizer = Tokenizer()
    tokenizer.load(args.vocab_path, args.merges_path)
    
    # Load data using memory mapping if enabled
    if args.use_memmap:
        print("Loading data using memory mapping...")
        train_data = load_data_memmap(args.pretokens_path)
        valid_data = load_data_memmap(args.valid_path)
    else:
        print("Loading data into memory...")
        train_data = np.load(args.pretokens_path)
        valid_data = np.load(args.valid_path)
    
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
        device=device
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        betas=(args.beta1, args.beta2)
    )
    
    # Training loop
    best_val_loss = float('inf')
    for step in tqdm(range(args.max_steps)):
        # Training
        train_loss = train_step(model, optimizer, train_data, args, device)
        
        # Learning rate scheduling
        lr = get_cosine_lr(
            step,
            args.min_lr,
            args.max_lr,
            args.max_steps // 10,  # warmup steps
            args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Logging
        if step % args.log_freq == 0:
            wandb.log({
                "train/loss": train_loss,
                "train/lr": lr,
                "train/step": step
            })
        
        # Evaluation and checkpointing
        if step % args.eval_freq == 0:
            val_loss = evaluate(model, valid_data, args, device)
            wandb.log({
                "val/loss": val_loss,
                "val/step": step
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    os.path.join(args.checkpoint_dir, "best_model.pt")
                )
        
        # Periodic checkpointing
        if not args.save_best_only and step % args.checkpoint_freq == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt")
            )
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        args.max_steps,
        os.path.join(args.checkpoint_dir, "final_model.pt")
    )
    
    wandb.finish()

if __name__ == "__main__":
    main()





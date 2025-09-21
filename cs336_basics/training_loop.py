import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.training import data_loader, save_checkpoint, load_checkpoint, gradient_clipping, cosine_lr_schedule, AdamW, cross_entropy_loss
import argparse
import wandb
from dataclasses import asdict
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)



parser = argparse.ArgumentParser()

data_path = "/juice5b/scr5b/kaitwang/cs336/data"

# config, seed
parser.add_argument(
    '--seed', type=int, default=42
)
parser.add_argument(
    '--wandb_project', type=str, default='cs336_a1'
)
parser.add_argument(
    '--wandb_run_name', type=str, default=None
)
parser.add_argument(
    '--model_save_path', type=str, default=f'/juice5b/scr5b/kaitwang/cs336/a1/models'
)

# tokenizer loading, data loading
parser.add_argument(
    '--vocab_path', type=str, default=f"{data_path}/tinystories_vocab.pkl")
parser.add_argument(
    '--merges_path', type=str, default=f"{data_path}/tinystories_merges.pkl")
parser.add_argument(
    '--train_path', type=str, default=f"{data_path}/tinystories_train_tokenIDs.npy")
parser.add_argument(
    '--valid_path', type=str, default=f"{data_path}/tinystories_valid_tokenIDs.npy")


# systems stuff
parser.add_argument(
    '--device', type=torch.device, default='cuda')
parser.add_argument(
    '--dtype', type=torch.dtype, default=torch.float32)
parser.add_argument(
    '--compile', type=bool, default=False)

# model definition
parser.add_argument(
    '--batch_size', type=int, default=16
)
parser.add_argument(
    '--vocab_size', type=int, default=10000
)
parser.add_argument(
    '--context_length', type=int, default=256
)
parser.add_argument(
    '--d_model', type=int, default=512
)
parser.add_argument(
    '--num_layers', type=int, default=4
)
parser.add_argument(
    '--num_heads', type=int, default=16
)
parser.add_argument(
    '--d_ff', type=int, default=1344
)
parser.add_argument(
    '--theta', type=int, default=10000
)

# optimizer
parser.add_argument(
    '--lr', type=float, default=1e-5
)
parser.add_argument(
    '--max_lr', type=float, default=1e-3
)
parser.add_argument(
    '--min_lr', type=float, default=1e-6
)
parser.add_argument(
    '--beta1', type=float, default=0.9
)
parser.add_argument(
    '--beta2', type=float, default=0.999
)
parser.add_argument(
    '--eps', type=float, default=1e-8
)
parser.add_argument(
    '--weight_decay', type=float, default=0.1
)
parser.add_argument(
    '--grad_clip_M', type=float, default=0.1
)


# training controls
parser.add_argument(
    '--num_steps', type=int, default=10000
)
parser.add_argument(
    '--warmup_steps', type=int, default=500
)


# validation controls
parser.add_argument(
    '--eval_freq', type=int, default=50
)
parser.add_argument(
    '--valid_batches', type=int, default=100
)


def evaluate(model, valid_data, args):
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for _ in range(args.valid_batches):
            inputs, targets = data_loader(valid_data, args.batch_size, args.context_length, args.device)
            preds = model(inputs)
            loss = cross_entropy_loss(preds, targets)
            total_loss += loss.item()
    model.train()

    return total_loss / args.valid_batches
  


def load_dataset(args) -> (Tokenizer, np.memmap, np.memmap):
    special_tokens = ['<|endoftext|>']
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=special_tokens)
    train_data = np.load(args.train_path, mmap_mode='r')
    valid_data = np.load(args.valid_path, mmap_mode='r')
    return tokenizer, train_data, valid_data

def train(args):
    tokenizer, train_data, valid_data = load_dataset(args)

    model = TransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, theta=args.theta, device=args.device, dtype=args.dtype)
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)
    
    if args.device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)

    best_loss = None
    t0 = time.time()
    def wallclock(): return time.time() - t0

    wandb.define_metric('step')
    wandb.define_metric('train/*', step_metric='step')
    wandb.define_metric('val/*', step_metric='step')
    wandb.define_metric('val/loss', step_metric='step', summary='min')
    wandb.define_metric('train/loss', step_metric='step', summary='last')

    for step in range(args.num_steps):
        inputs, targets = data_loader(x=train_data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
        
        lr_t = cosine_lr_schedule(
            t=step,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            warmup_steps=args.warmup_steps,
            cosine_steps=args.num_steps - args.warmup_steps
        )
        # optimizer.lr = lr_t
        for g in optimizer.param_groups:
            g['lr'] = lr_t

        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        if args.grad_clip_M is not None:
            gradient_clipping(model.parameters(), args.grad_clip_M)
        optimizer.step()

        wandb.log({
            'step': step,
            'train/loss': loss.item(),
            'time/wallclock_s': wallclock(),
            'lr': lr_t
        })

        if step % args.eval_freq == 0:
            val_loss = evaluate(model=model, valid_data = valid_data, args=args)
            if (best_loss is None or (val_loss < best_loss and best_loss - val_loss >= 0.01)): 
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, step, f'{args.model_save_path}/best_model')
                
            wandb.log({
                'step': step,
                'val/loss': val_loss,
                'time/wallclock_s': wallclock(),
                'lr': lr_t
            })


    wandb.run.summary['final/best_loss'] = best_loss
    wandb.run.summary['final/train_loss'] = loss

    wandb.finish()

    

if __name__ == '__main__':

    args = parser.parse_args()
    logging.info(args)

    config = vars(args)

    wandb.init(
        project=args.wandb_project,
        config=config,
        name=args.wandb_run_name,
        job_type='train'
    )
    train(args)






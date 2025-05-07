import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import wandb
from tqdm import tqdm
from cs336_basics.training import cross_entropy_loss, get_cosine_lr, gradient_clipping, save_checkpoint
from cs336_basics.optimizers import AdamW, SGD
from cs336_basics.data_loader import load_batch
from cs336_basics.simple_tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.logger import ExperimentLogger
import time

def parse_args():
    parser = argparse.ArgumentParser()
    
    # data paths
    parser.add_argument("--train_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default = "/Users/kaitlynwang/assignment1-basics/tinystories_merges.pkl")
    parser.add_argument("--pretokens_train_path", type=str, default="/data/c-kaitwang/tinystories_pretokens.npy", help="Path to pretokenized training data")
    parser.add_argument("--pretokens_valid_path", type=str, default="/data/c-kaitwang/tinystories_valid_pretokens.npy", help="Path to pretokenized validation data")
    parser.add_argument("--reuse_pretokens", action="store_true", default = True, help="Reuse existing pretokenized data if available")

    # data loading params
    parser.add_argument("--vocab_size", type=int, default = 10000)
    parser.add_argument("--context_length", type=int, default = 256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_memmap", type=bool, default=False)
    parser.add_argument("--use_parallel_pretokenize", type=bool, default=True)  # Default to parallel for full dataset

    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--pad_token", type=str, default="<|pad|>")

    # model params
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--max_seq_len", type=int, default=256)

    # training and optimizer params
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--gradient_clip_M", type=float, default=5.0)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--min_loss_threshold", type=int, default=2)

    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="/data/c-kaitwang/checkpoints")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--checkpoint_freq", type=int, default=1000)
    parser.add_argument("--save_best_only", type=bool, default=True)
    parser.add_argument("--experiment_name", type=str, default=None)

    # device and compilation
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use_compile", type=bool, default=True)
    
    return parser.parse_args()

def load_data_memmap(file_path, dtype=np.int32):
    """Load data using memory mapping for efficient memory usage."""
    print(f"Loading data from {file_path} using memory mapping...")
    return np.memmap(file_path, dtype=dtype, mode='r')

def load_data_regular(file_path, dtype=np.int32):
    """Load data into regular memory."""
    print(f"Loading data from {file_path} into regular memory...")
    data = np.load(file_path)
    print(f"Loaded {len(data)} tokens")
    return data

def train_step(model, optimizer, train_data, args, device):
    inputs, targets = load_batch(train_data, args.batch_size, args.context_length, device)
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), args.gradient_clip_M)
    optimizer.step()
    
    return loss.item()

def evaluate(model, valid_data, args, device, n_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        max_start = len(valid_data) - args.context_length - 1
        for _ in range(n_batches):
            start = np.random.randint(0, max_start)
            input_seq = valid_data[start:start + args.context_length]
            target_seq = valid_data[start + 1:start + args.context_length + 1]
            inputs = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(target_seq, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main(args=None):
    if args is None:
        args = parse_args()
    
    # setup logger
    logger = ExperimentLogger(
        project_name=args.wandb_project,
        experiment_name=args.experiment_name,
        config=vars(args)
    )
    
    # setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    # only enable TF32‐style "high" matmul precision on cuda, never on mps
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")
    
    # load tokenizer from files
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)


    s = "Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, “Wow, that is a really amazing vase! Can I buy it?” The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!” So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn’t believe how lucky Ben was. And that’s how Ben found an amazing vase in the store!"
    ids = tokenizer.encode(s)
    print(tokenizer.decode(ids))

    # reuse or create fresh pretokenized data
    if os.path.exists(args.pretokens_train_path):
        if args.reuse_pretokens:
            print(f"Reusing existing pretokenized training data from: {args.pretokens_train_path}")
            pretokenize_needed = False
        else:
            print(f"Existing pretokenized training data found but fresh tokenization requested")
    else:
        pretokenize_needed = True

    if pretokenize_needed:
        print(f"Creating fresh pretokenized training data...")
        tokenizer.pretokenize_file(
            args.train_path, 
            args.pretokens_train_path, 
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized training data to: {args.pretokens_train_path}")

    # Validation pretokenization (optional, only if not present)
    if not os.path.exists(args.pretokens_valid_path):
        print(f"Creating fresh pretokenized validation data...")
        tokenizer.pretokenize_file(
            args.valid_path,
            args.pretokens_valid_path,
            use_parallel=args.use_parallel_pretokenize
        )
        print(f"Saved fresh pretokenized validation data to: {args.pretokens_valid_path}")

    # load data based on the specified method
    if not args.use_memmap:
        print("Loading data into regular memory...")
        train_data = load_data_regular(args.pretokens_train_path)
        valid_data = load_data_regular(args.pretokens_valid_path)
    else:
        print("Loading data using memory mapping...")
        train_data = load_data_memmap(args.pretokens_train_path)
        valid_data = load_data_memmap(args.pretokens_valid_path)
    
    # create model
    model = TransformerLM(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        theta=args.rope_theta,
        device=device
    )
    model = model.to(device) 
    
    print("Created model.")

    print("Max token ID in valid_data:", valid_data.max())
    print("Min token ID in valid_data:", valid_data.min())
    print("Model vocab size:", model.vocab_size)

    print("Max token ID in train_data:", train_data.max())
    print("Min token ID in train_data:", train_data.min())
    print("Model vocab size:", model.vocab_size)
    
    # compile model if enabled
    if args.use_compile:
        print("Compiling model for better performance...")
        model = torch.compile(model, backend="aot_eager")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        betas=(args.beta1, args.beta2)
    )
    
    print("Created optimizer, starting training loop")
    print(f"Will run for {args.max_steps} steps")
    
    # training loop
    best_val_loss = float('inf')
    patience = 50  # number of evaluations without improvement before stopping
    no_improvement_count = 0
    min_loss_threshold = args.min_loss_threshold  # stop if loss gets below this threshold
    
    # Stability tracking
    val_losses = []  # Keep track of recent validation losses
    stability_window = 5  # Number of recent validation losses to check
    stability_threshold = 0.01  # Maximum allowed relative change between consecutive losses
    
    # Divergence detection
    divergence_threshold = 100.0  # Loss threshold for divergence
    last_losses = []  # Keep track of recent losses
    window_size = 10  # Increased from 5 to 10 for more stability
    min_increase = 0.1  # Minimum relative increase to consider as divergence
    
    pbar = tqdm(range(args.max_steps), desc="Training")
    for step in pbar:
        # training
        train_loss = train_step(model, optimizer, train_data, args, device)
        
        # Check for divergence
        last_losses.append(train_loss)
        if len(last_losses) > window_size:
            last_losses.pop(0)
        
        # If loss is too high or increasing rapidly, consider it diverged
        if train_loss > divergence_threshold:
            print(f"\nTraining diverged at step {step} with loss {train_loss:.4f}")
            return {
                'final_val_loss': float('inf'),
                'best_val_loss': float('inf'),
                'final_step': step,
                'diverged': True,
                'divergence_reason': 'loss_too_high'
            }
        
        # Check for consistent increase with minimum threshold
        if len(last_losses) == window_size:
            increases = [last_losses[i+1] - last_losses[i] for i in range(len(last_losses)-1)]
            relative_increases = [inc / last_losses[i] for i, inc in enumerate(increases)]
            if all(inc > min_increase for inc in relative_increases):
                print(f"\nTraining diverged at step {step} with loss {train_loss:.4f}")
                return {
                    'final_val_loss': float('inf'),
                    'best_val_loss': float('inf'),
                    'final_step': step,
                    'diverged': True,
                    'divergence_reason': 'loss_increasing'
                }
        
        # LR scheduling
        lr = get_cosine_lr(
            step,
            args.min_lr,
            args.max_lr,
            args.max_steps // 20,  # warmup steps 5% of max_steps
            args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # logging
        if step % args.log_freq == 0:
            logger.log_train_step(train_loss, step)
            pbar.write(f"Step {step}: Train Loss = {train_loss:.4f}")
        
        # eval and checkpointing
        if step % args.eval_freq == 0:
            val_loss = evaluate(model, valid_data, args, device, n_batches=10)
            logger.log_validation(val_loss, step)
            pbar.write(f"Step {step}: Val Loss = {val_loss:.4f}")
            
            # track validation losses for stability check
            val_losses.append(val_loss)
            if len(val_losses) > stability_window:
                val_losses.pop(0)
            
            # check for both low loss and stability
            if val_loss < min_loss_threshold and len(val_losses) == stability_window:
                #relative changes between consecutive losses
                relative_changes = [abs(val_losses[i+1] - val_losses[i]) / val_losses[i] 
                                 for i in range(len(val_losses)-1)]
                # check if all changes are below threshold
                if all(change < stability_threshold for change in relative_changes):
                    pbar.write(f"\nEarly stopping at step {step} - reached minimum loss threshold of {min_loss_threshold} with stable loss")
                    break
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                run_info = {
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "wandb_run_name": getattr(logger, "experiment_name", None)
                }
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    os.path.join(args.checkpoint_dir, "best_model.pt"),
                    run_info=run_info
                )
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    pbar.write(f"\nEarly stopping at step {step} - no improvement for {patience} evaluations")
                    break
        
        # periodic checkpointing
        if not args.save_best_only and step % args.checkpoint_freq == 0:
            run_info = {
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "wandb_run_name": getattr(logger, "experiment_name", None)
            }
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(args.checkpoint_dir, f"checkpoint_{step}.pt"),
                run_info=run_info
            )
    
    # save final model
    run_info = {
        "save_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "wandb_run_name": getattr(logger, "experiment_name", None)
    }
    save_checkpoint(
        model,
        optimizer,
        step,  # use actual step instead of max_steps
        os.path.join(args.checkpoint_dir, "final_model.pt"),
        run_info=run_info
    )
    
    # finish logging
    logger.finish()
    
    # Return final metrics
    return {
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'final_step': step,
        'diverged': False
    }

if __name__ == "__main__":
    main()
    





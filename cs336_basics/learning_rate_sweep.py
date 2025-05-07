import torch
import numpy as np
from pathlib import Path
import json
import os
from training_loop import main, parse_args
from logger import ExperimentLogger
import wandb
from tqdm import tqdm
import time

def run_lr_sweep(
    min_test_lr=1e-5,
    max_test_lr=1e-3,
    num_runs=5,
    project_name="transformer-lm-lr-sweep"
):
    """
    Run a LR sweep with exponentially spaced learning rates using existing training loop
    """
    # exponentially spaced learning rates
    learning_rates = np.exp(np.linspace(np.log(min_test_lr), np.log(max_test_lr), num_runs))
    
    results = {
        'learning_rates': learning_rates.tolist(),
        'final_losses': [],
        'best_losses': [],
        'final_steps': [],
        'diverged': [],
        'divergence_reasons': []
    }
    
    # results directory
    results_dir = Path("lr_sweep_results")
    results_dir.mkdir(exist_ok=True)
    

    for lr in tqdm(learning_rates, desc="Running learning rate sweep"):
        # override LR args
        args = parse_args()
        args.learning_rate = lr
        args.max_lr = lr
        args.min_lr = 10e-5
        args.wandb_project = project_name
        args.experiment_name = f"lr_{lr:.2e}"  # Add learning rate to experiment name
        
        # base tinystories configs
        args.train_path = "/data/a1-basics/TinyStoriesV2-GPT4-train.txt"
        args.valid_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt" 
        args.vocab_path = "/data/c-kaitwang/tinystories_vocab.pkl"
        args.merges_path = "/data/c-kaitwang/tinystories_merges.pkl"
        args.pretokens_path = "/data/c-kaitwang/tinystories_pretokens.npy" 
        
        args.vocab_size = 10000
        args.context_length = 256
        args.batch_size = 64
        args.d_model = 512
        args.num_heads = 16
        args.num_layers = 4
        args.d_ff = 1344
        args.max_steps = 10000
        args.max_seq_len = 512 # must be greater than context_length
        args.min_loss_threshold = 1.45
        
        # adjust eval freq to get validation points
        args.eval_freq = args.max_steps // 50  # evaluate more frequently
        args.log_freq = args.max_steps // 50
        
        args.device = "cuda"
        args.use_compile = False
        args.use_memmap = True
        args.use_parallel_pretokenize = True
        
        try:
            metrics = main(args)
            
            results['final_losses'].append(metrics['final_val_loss'])
            results['best_losses'].append(metrics['best_val_loss'])
            results['final_steps'].append(metrics['final_step'])
            results['diverged'].append(metrics['diverged'])
            results['divergence_reasons'].append(metrics.get('divergence_reason', None))
            
            if metrics['diverged']:
                print(f"Run with lr={lr} diverged: {metrics.get('divergence_reason', 'unknown reason')}")
            
        except Exception as e:
            print(f"Run with lr={lr} failed with error: {str(e)}")
            results['final_losses'].append(float('inf'))
            results['best_losses'].append(float('inf'))
            results['final_steps'].append(0)
            results['diverged'].append(True)
            results['divergence_reasons'].append('error')
    
    # save results dir
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # initialize sweep logger after all runs are complete
    sweep_logger = ExperimentLogger(
        project_name=project_name,
        experiment_name="lr_sweep",
        config={
            'min_test_lr': min_test_lr,
            'max_test_lr': max_test_lr,
            'num_runs': num_runs,
            'results': results
        }
    )
    
    # log all results at once
    for lr, final_loss, diverged in zip(results['learning_rates'], results['final_losses'], results['diverged']):
        if not diverged:  # Only log non-diverged runs
            sweep_logger.log_validation(final_loss, lr)
    
    sweep_logger.finish()
    
    # summary
    print("\nLearning Rate Sweep Summary:")
    print("----------------------------")
    for lr, final_loss, best_loss, diverged, reason in zip(
        results['learning_rates'], 
        results['final_losses'], 
        results['best_losses'],
        results['diverged'],
        results['divergence_reasons']
    ):
        status = "DIVERGED" if diverged else "SUCCESS"
        if diverged:
            print(f"LR: {lr:.2e} - {status} ({reason})")
        else:
            print(f"LR: {lr:.2e} - {status} - Final Loss: {final_loss:.4f}, Best Loss: {best_loss:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_lr_sweep(
        # try wide range to get it to diverge
        #min_lr=1e-8,  
        #max_lr=1e-1,  
        #num_runs=6,   
        #project_name="transformer-lm-lr-sweep-v5" 

        # try to get a good result
        min_test_lr=5e-5,  
        max_test_lr=5e-2,  
        num_runs=8,   
        project_name="transformer-lm-lr-sweep-cluster-v3-bigmodel-8" 

        #min_lr=1e-5,  
        #max_lr=1e-2, 
        #num_runs=8,   # try 8 different learning rates
        #project_name="transformer-lm-lr-sweep-v3"  
    ) 
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

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))  # Listen on all interfaces, port 5678
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # Pause execution until debugger is attached

def run_batch_sweep(
    min_test_batch=1,
    max_test_batch=4096,
    num_runs=4,
    project_name="transformer-lm-batch-sweep"
):
    """
    Run a batch size sweep from 1 to GPU memory limit using existing training loop. 
    """

    # batch_sizes = [1, 64, 128, 1024, 4096, 10000]
    #batch_sizes = [1, 64, 128, 1024, 4096, 10000]
    batch_sizes = [256]
    # try to get to GPU limit

    #learning_rates = np.exp(np.linspace(np.log(min_test_batch), np.log(max_test_batch), num_runs))
    
    results = {
        'batch_sizes': batch_sizes,
        'final_losses': [],
        'best_losses': [],
        'final_steps': [],
        'diverged': [],
        'divergence_reasons': []
    }
    
    # results directory
    results_dir = Path("batch_sweep_results")
    results_dir.mkdir(exist_ok=True)
    

    for batch in tqdm(batch_sizes, desc="Running batch sweep"):
        # override LR args
        args = parse_args()
        args.batch_size = batch
        args.wandb_project = project_name
        args.experiment_name = f"batch_{batch}"  # Add learning rate to experiment name
        
        # base tinystories configs
        # args.train_path = "/data/a1-basics/TinyStoriesV2-GPT4-train.txt"
        # args.valid_path = "/data/a1-basics/TinyStoriesV2-GPT4-valid.txt" 
        args.vocab_path = "/data/c-kaitwang/owt_vocab.pkl"
        args.merges_path = "/data/c-kaitwang/owt_merges.pkl"
        args.pretokens_train_path = "/data/c-kaitwang/owt_train_pretokens.npy" 
        args.pretokens_valid_path = "/data/c-kaitwang/owt_valid_pretokens.npy" 
        args.reuse_pretokens = True
        
        args.vocab_size = 30000
        args.context_length = 256
        args.d_model = 512
        args.num_heads = 16
        args.num_layers = 4
        args.d_ff = 1344
        args.max_steps = 5000
        args.max_seq_len = 512 # must be greater than context_length
        args.min_loss_threshold = 1.45

        args.learning_rate=5e-2
        args.max_lr = 1e-1
        args.min_lr = 1e-2
        
        # adjust eval freq to get validation points
        args.eval_freq = args.max_steps // 50  # evaluate more frequently
        args.log_freq = args.max_steps // 50
        
        args.device = "cuda"
        args.use_compile = False
        args.use_memmap = False
        args.use_parallel_pretokenize = True
        
        try:
            metrics = main(args)
            
            results['final_losses'].append(metrics['final_val_loss'])
            results['best_losses'].append(metrics['best_val_loss'])
            results['final_steps'].append(metrics['final_step'])
            results['diverged'].append(metrics['diverged'])
            results['divergence_reasons'].append(metrics.get('divergence_reason', None))
            
            if metrics['diverged']:
                print(f"Run with batch={batch} diverged: {metrics.get('divergence_reason', 'unknown reason')}")
            
        except Exception as e:
            print(f"Run with batch={batch} failed with error: {str(e)}")
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
        experiment_name="batch_sweep",
        config={
            'min_batch': min_test_batch,
            'max_batch': max_test_batch,
            'num_runs': num_runs,
            'results': results
        }
    )
    
    # log all results at once
    for batch, final_loss, diverged in zip(results['batch_sizes'], results['final_losses'], results['diverged']):
        if not diverged:  # Only log non-diverged runs
            sweep_logger.log_validation(final_loss, batch)
    
    sweep_logger.finish()
    
    # summary
    print("\nBatch Size Sweep Summary:")
    print("----------------------------")
    for batch, final_loss, best_loss, diverged, reason in zip(
        results['batch_sizes'], 
        results['final_losses'], 
        results['best_losses'],
        results['diverged'],
        results['divergence_reasons']
    ):
        status = "DIVERGED" if diverged else "SUCCESS"
        if diverged:
            print(f"BATCH SIZE: {batch} - {status} ({reason})")
        else:
            print(f"BATCH SIZE: {batch} - {status} - Final Loss: {final_loss:.4f}, Best Loss: {best_loss:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_batch_sweep(
        # try wide range to get it to diverge
        #min_lr=1e-8,  
        #max_lr=1e-1,  
        #num_runs=6,   
        #project_name="transformer-lm-lr-sweep-v5" 

        # try to get a good result
        min_test_batch=1,  
        max_test_batch=10000,  
        num_runs=5,   
        project_name="transformer-lm-owt-test" 

        #min_lr=1e-5,  
        #max_lr=1e-2, 
        #num_runs=8,   # try 8 different learning rates
        #project_name="transformer-lm-lr-sweep-v3"  
    ) 

    # Uncomment to run batch size sweep
    # results = run_batchsize_sweep(
    #     min_batch=8,
    #     max_batch=1024,
    #     num_runs=6,
    #     project_name="transformer-lm-batchsize-sweep-v1"
    # ) 
import wandb
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs"
    ):
        """
        initialize experiment logger
        
        args:
            config: a dictionary of configs
            log_dir: directory to save logs
        """
        self.start_time = time.time()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.run = wandb.init(
            project=project_name,
            name=experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=config or {}
        )
        
        # save config locally
        self.config_path = os.path.join(log_dir, f"{self.run.name}_config.json")
        with open(self.config_path, 'w') as f:
            json.dump(config or {}, f, indent=2)
        
        self.metrics = {
            'train': {'loss': [], 'steps': [], 'time': []},
            'val': {'loss': [], 'steps': [], 'time': []}
        }
    
    def log_train_step(self, loss: float, step: int):
        """log training metrics for a single step"""
        current_time = time.time() - self.start_time
        
        # update local metrics and log to wandb
        self.metrics['train']['loss'].append(loss)
        self.metrics['train']['steps'].append(step)
        self.metrics['train']['time'].append(current_time)
        
        wandb.log({
            'train/loss': loss,
            'train/wallclock_time': current_time
        }, step=step)
    
    def log_validation(self, loss: float, step: int):
        """log validation metrics"""
        current_time = time.time() - self.start_time
        
        # update local metrics and log to wandb
        self.metrics['val']['loss'].append(loss)
        self.metrics['val']['steps'].append(step)
        self.metrics['val']['time'].append(current_time)
        
        wandb.log({
            'val/loss': loss,
            'val/wallclock_time': current_time
        }, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """log hyperparameters to wandb"""
        wandb.config.update(hyperparams)
    
    def save_metrics(self):
        """save metrics to local file"""
        metrics_path = os.path.join(self.log_dir, f"{self.run.name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def finish(self):
        """finish the experiment and save all data"""
        self.save_metrics()
        wandb.finish()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """get a summary of the experiment"""
        return {
            'name': self.run.name,
            'config': self.run.config,
            'train_steps': len(self.metrics['train']['steps']),
            'val_steps': len(self.metrics['val']['steps']),
            'total_time': time.time() - self.start_time,
            'best_val_loss': min(self.metrics['val']['loss']) if self.metrics['val']['loss'] else None
        }

def create_experiment_log(experiments: list[Dict[str, Any]], output_file: str = "experiment_log.md"):
    """
    create a single file documenting all experiments
    """
    with open(output_file, 'w') as f:
        f.write("# Experiment Log\n\n")
        
        for exp in experiments:
            f.write(f"## {exp['name']}\n\n")
            f.write("### Configuration\n")
            f.write("```json\n")
            f.write(json.dumps(exp['config'], indent=2))
            f.write("\n```\n\n")
            
            f.write("### Results\n")
            f.write(f"- Training Steps: {exp['train_steps']}\n")
            f.write(f"- Validation Steps: {exp['val_steps']}\n")
            f.write(f"- Total Time: {exp['total_time']:.2f} seconds\n")
            f.write(f"- Best Validation Loss: {exp['best_val_loss']:.4f}\n\n")
            
            f.write("### Learning Curves\n")
            f.write(f"![Learning Curves]({exp['name']}_metrics.png)\n\n")
            
            f.write("---\n\n") 
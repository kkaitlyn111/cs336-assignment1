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
        Initialize the experiment logger.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Optional name for this specific experiment
            config: Dictionary of experiment configuration
            log_dir: Directory to save local logs
        """
        self.start_time = time.time()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize wandb
        self.run = wandb.init(
            project=project_name,
            name=experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S"),
            config=config or {}
        )
        
        # Save config locally
        self.config_path = os.path.join(log_dir, f"{self.run.name}_config.json")
        with open(self.config_path, 'w') as f:
            json.dump(config or {}, f, indent=2)
        
        # Initialize metrics
        self.metrics = {
            'train': {'loss': [], 'steps': [], 'time': []},
            'val': {'loss': [], 'steps': [], 'time': []}
        }
    
    def log_train_step(self, loss: float, step: int):
        """Log training metrics for a single step."""
        current_time = time.time() - self.start_time
        
        # Update local metrics
        self.metrics['train']['loss'].append(loss)
        self.metrics['train']['steps'].append(step)
        self.metrics['train']['time'].append(current_time)
        
        # Log to wandb
        wandb.log({
            'train/loss': loss,
            'train/step': step,
            'train/wallclock_time': current_time
        })
    
    def log_validation(self, loss: float, step: int):
        """Log validation metrics."""
        current_time = time.time() - self.start_time
        
        # Update local metrics
        self.metrics['val']['loss'].append(loss)
        self.metrics['val']['steps'].append(step)
        self.metrics['val']['time'].append(current_time)
        
        # Log to wandb
        wandb.log({
            'val/loss': loss,
            'val/step': step,
            'val/wallclock_time': current_time
        })
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters to wandb."""
        wandb.config.update(hyperparams)
    
    def save_metrics(self):
        """Save metrics to local file."""
        metrics_path = os.path.join(self.log_dir, f"{self.run.name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def finish(self):
        """Finish the experiment and save all data."""
        self.save_metrics()
        wandb.finish()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment."""
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
    Create a markdown file documenting all experiments.
    
    Args:
        experiments: List of experiment summaries
        output_file: Path to save the markdown file
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
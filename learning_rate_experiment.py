import torch
from cs336_basics.optimizers import SGD

def run_experiment(lr):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    
    for t in range(10):  # Run for 10 iterations
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
    
    return losses

# Run experiments with different learning rates
learning_rates = [1e1, 1e2, 1e3]
results = {}

for lr in learning_rates:
    print(f"\nRunning experiment with learning rate: {lr}")
    losses = run_experiment(lr)
    results[lr] = losses
    print(f"Losses: {losses}") 
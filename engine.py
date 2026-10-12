import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader

def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int = 0,
    use_wandb: bool = False
) -> None:
    """Trains the model for one epoch"""
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward Pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss_val:>7f} | [{current:>5d}/{size:>5d}]")
            
            if use_wandb:
                global_step = epoch * len(dataloader) + batch
                wandb.log({
                    "Train/Loss": loss_val,
                    "epoch": epoch,
                    "global_step": global_step
                })

def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    epoch: int = 0,
    use_wandb: bool = False
) -> dict:
    """Evaluate the model's performance on the test set."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")

    if use_wandb:
        wandb.log({
            "Test/Accuracy": accuracy,
            "Test/Loss": test_loss,
            "epoch": epoch
        })
    
    return {
        "final_accuracy": accuracy,
        "final_test_loss": test_loss
    }
import os
import time
import argparse
import torch
from torch import nn, optim
import wandb

from dataset import get_dataloaders
from model import NeuralNetwork
from engine import train, test
from utils import save_experiment

def main():
    # 1. Setup Configuration via Argparse
    parser = argparse.ArgumentParser(description="Train FashionMNIST Model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # 2. Setup Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # 3. Initialize Data Pipeline
    print("Loading data...")
    train_dataloader, test_dataloader, _ = get_dataloaders(args.batch_size)

    # 4. Initialize Model, Loss, and Optimizer
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Initialize Weights & Biases
    run_name = f"fashion_mnist_E{args.epochs}_B{args.batch_size}_LR{args.lr}"
    wandb.init(
        project="fashion-mnist-mastery",
        name=run_name,
        config={
            "architecture": "MLP",
            "dataset": "FashionMNIST",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        }
    )
    
    # MAGIC FEATURE: Watch the model to track gradients and parameter distributions!
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # 5. Execute Training Loop
    print(f"Starting training for {args.epochs} epochs with batch size {args.batch_size}...")
    
    # Start Tracking Time
    start_time = time.time()
    final_metrics = {}
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer, device, epoch, use_wandb=True)
        final_metrics = test(test_dataloader, model, criterion, device, epoch, use_wandb=True)
        scheduler.step()

    # End Tracking Time
    end_time = time.time()
    training_duration = end_time - start_time

    print("Training Complete!")
    
    # Combine our hyperparameters with our actual results
    experiment_data = {
        "architecture": "Multi-Layer Perceptron (MLP)",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "final_accuracy": final_metrics["final_accuracy"],
        "final_test_loss": final_metrics["final_test_loss"],
        "training_time_seconds": round(training_duration, 2)
    }

    # Save using our new utility function
    model_name = f"fashion_mnist_E{args.epochs}_B{args.batch_size}"
    save_experiment(
        model_state=model.state_dict(),
        model_name=model_name,
        metrics=experiment_data
    )
    
    # Upload the Model to the Cloud
    cloud_model_path = os.path.join("models", f"{model_name}.pth")
    wandb.save(cloud_model_path)
    
    # Close and run
    wandb.finish()

if __name__ == "__main__":
    main()
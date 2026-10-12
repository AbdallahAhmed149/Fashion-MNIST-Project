import torch
from torch import nn
import matplotlib.pyplot as plt
from typing import List
import json
import os

def load_checkpoint(model: nn.Module, path: str, device: str) -> nn.Module:
    """Loads a saved model state dictionary."""
    model.load_state_dict(torch.load(path, map_location=device))

    # CRITICAL: Always set the model to evaluation mode after loading for inference
    model.eval()
    print(f"Model successfully loaded from {path}")
    return model

def predict_single_image(
    model: nn.Module,
    image: torch.Tensor,
    actual_label: int,
    class_names: List[str],
    device: str
) -> None:
    """Runs inference on a single image and plots the result."""
    model.eval()
    with torch.no_grad():
        # Ensure the image has a batch dimension: [1, C, H, W]
        x = image.unsqueeze(0).to(device)
        
        # Get raw logits and find the index of the highest value
        logits = model(x)
        predicted_idx = logits[0].argmax(0).item()
        
        predicted_name = class_names[predicted_idx]
        actual_name = class_names[actual_label]
        
        print(f"\n--- Prediction Results ---")
        print(f'Predicted: "{predicted_name}"')
        print(f'Actual: {actual_name}')
        
        # Visual confirmation
        plt.figure(figsize=(4, 4))
        
        # Squeeze removes the channel/batch dimensions so matplotlib can read the 28x28 grid
        plt.imshow(image.cpu().squeeze(), cmap="gray")
        color = "green" if predicted_idx == actual_label else "red"
        plt.title(f"Prediction: {predicted_name} | Actual: {actual_name}", color=color)
        plt.axis("off")
        plt.show()

def save_experiment(model_state: dict, model_name: str, metrics: dict) -> None:
    """Saves the model weights and JSON file with the performance metrices."""
    
    # Define directories
    model_dir = "models"
    metrics_dir = "metrics"
    
    # Create folders if missing
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Define file paths
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    metrics_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    
    # Save the PyTorch model
    torch.save(model_state, model_path)
    
    # Save the metrics as JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\n--- Experiment Saved ---")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
import torch

from dataset import get_dataloaders
from model import NeuralNetwork
from utils import load_checkpoint, predict_single_image

def main():
    # 1. Setup Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # 2. Define the classes for FashionMNIST
    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    
    # We pass batch_size=1 just to initialize the loader, but we extract the raw dataset
    _, _, test_data = get_dataloaders(batch_size=1)
    
    # Grab the very first image and its label
    image, label = test_data[0][0], test_data[0][1]
    
    # 3. Initialize the Model and Load the Weights
    model = NeuralNetwork().to(device)
    model_path = r"models\fashion_mnist_E5_B64.pth"
    
    try:
        model = load_checkpoint(model, model_path, device)
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Did you run main.py first?")
        return

    # 5. Run the prediction
    predict_single_image(model, image, label, classes, device)

if __name__ == "__main__":
    main()
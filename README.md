# Fashion-MNIST Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights%20%26%20Biases](https://img.shields.io/badge/Experiment%20Tracking-W%26B-FFBE00?logo=weightsandbiases&logoColor=111111)](https://wandb.ai/)
[![License](https://img.shields.io/github/license/AbdallahAhmed149/Fashion-MNIST-Project)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/AbdallahAhmed149/Fashion-MNIST-Project)

A PyTorch-based image-classification project that trains a fully connected neural network to recognize clothing categories from the Fashion-MNIST dataset. The project provides a complete workflow for dataset preparation, model training, evaluation, experiment tracking, checkpoint storage, and single-image inference.

## Overview

Fashion-MNIST contains 28×28 grayscale images representing ten clothing categories. This repository implements a reproducible baseline machine-learning workflow that transforms the raw dataset into efficient PyTorch data loaders, trains a multi-layer perceptron, evaluates classification accuracy and loss, records experiments with Weights & Biases, and persists trained model weights for later inference.

The project is intentionally modular: data loading, model definition, training/evaluation logic, experiment persistence, and prediction are separated into focused Python modules. This makes the codebase suitable for learning, experimentation, and extending toward more advanced architectures such as convolutional neural networks.

## Key Features & Metrics

### Core features

- **Automated dataset acquisition:** Downloads and prepares the Fashion-MNIST training and test sets through `torchvision`.
- **Configurable training pipeline:** Supports command-line configuration for batch size, number of epochs, and learning rate.
- **MLP image classifier:** Uses a regularized multi-layer perceptron with ReLU activations and dropout.
- **Evaluation and checkpointing:** Reports test loss and accuracy, saves model weights as `.pth`, and stores run metrics as JSON.
- **Experiment observability:** Integrates with Weights & Biases for loss, accuracy, hyperparameter, gradient, and parameter tracking.
- **Single-image inference:** Loads a saved checkpoint and visualizes the predicted and actual Fashion-MNIST labels.

### Quantitative metrics

The training script records the following metrics for every experiment:

- **Final test accuracy** — calculated on the complete Fashion-MNIST test set.
- **Final test loss** — calculated using cross-entropy loss.
- **Training duration** — measured in seconds for the configured run.
- **Experiment configuration** — architecture, batch size, epochs, and learning rate.

The repository does not currently commit a fixed benchmark result, so accuracy, loss, and training-time values are generated when `main.py` is executed and saved under `metrics/`. This avoids presenting environment-dependent results as universal guarantees.

## System Architecture

The project follows a modular, layered training design:

```text
Fashion-MNIST Dataset
        │
        ▼
 dataset.py
  transforms + DataLoader
        │
        ├──────────────► model.py
        │                 MLP classifier
        │
        ▼
 main.py ─────────────► engine.py
 configuration          train/evaluate loops
        │                       │
        ├──��───────────► Weights & Biases
        │                 metrics and gradients
        │
        ▼
 utils.py
 checkpoints + JSON metrics
        │
        ▼
 predict.py
 single-image inference and visualization
```

### Design principles

- **Separation of concerns:** Dataset preparation, model architecture, training logic, persistence, and inference are isolated in separate modules.
- **Reproducible experiments:** Training parameters are explicit command-line arguments and are persisted with the resulting metrics.
- **Hardware awareness:** The training and inference scripts automatically select an available PyTorch accelerator and fall back to CPU when necessary.
- **Regularization:** Dropout layers are used to reduce overfitting in the fully connected classifier.
- **Observability by default:** Training and evaluation metrics can be logged to Weights & Biases, while local JSON artifacts provide a durable record of each run.

## Tech Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Deep learning | PyTorch |
| Dataset and preprocessing | Torchvision, Fashion-MNIST |
| Data loading | `torch.utils.data.DataLoader` |
| Optimization | Adam optimizer and step learning-rate scheduler |
| Loss function | Cross-entropy loss |
| Experiment tracking | Weights & Biases (`wandb`) |
| Visualization | Matplotlib |
| Model artifacts | PyTorch `.pth` checkpoints and JSON metrics |
| License | MIT License |

## Project Structure

```text
.
├── data/                 # Downloaded Fashion-MNIST files
├── metrics/              # JSON metrics generated by training
├── models/               # Saved PyTorch model checkpoints
├── dataset.py            # Dataset download, transforms, and data loaders
├── engine.py             # Training and evaluation loops
├── model.py              # NeuralNetwork model definition
├── main.py               # Configurable training entry point
├── predict.py            # Single-image inference entry point
├── utils.py              # Checkpoint, metrics, and visualization utilities
├── LICENSE               # MIT License
└── README.md
```

## Getting Started

### Prerequisites

- Python **3.9 or newer**
- `pip` and `venv`
- Internet access for the initial Fashion-MNIST download
- Optional: a CUDA-, MPS-, or other PyTorch-supported accelerator
- Optional: a Weights & Biases account for remote experiment tracking

### 1. Clone the repository

```bash
git clone https://github.com/AbdallahAhmed149/Fashion-MNIST-Project.git
cd Fashion-MNIST-Project
```

### 2. Create and activate a virtual environment

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the required Python packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision matplotlib wandb
```

> For a production or team environment, pin the installed versions in a `requirements.txt` or `pyproject.toml` file after validating them on the target hardware.

### 4. Configure Weights & Biases

`main.py` initializes a Weights & Biases run and logs training metrics. Authenticate before training:

```bash
wandb login
```

Alternatively, set the API key through an environment variable:

#### macOS/Linux

```bash
export WANDB_API_KEY="your_api_key"
```

#### Windows PowerShell

```powershell
$env:WANDB_API_KEY = "your_api_key"
```

If you do not want to use remote tracking, remove or disable the `wandb.init`, `wandb.watch`, `wandb.log`, `wandb.save`, and `wandb.finish` calls in the training workflow before running locally.

## Usage

### Train the model

Run the default configuration: five epochs, batch size 64, and learning rate `1e-3`.

```bash
python main.py
```

Customize the run with command-line arguments:

```bash
python main.py --epochs 10 --batch_size 128 --lr 0.0005
```

During training, the script will:

1. Download the dataset into `data/` if it is not already available.
2. Select an available accelerator or use the CPU.
3. Train the MLP using Adam and cross-entropy loss.
4. Evaluate the model after each epoch.
5. Save model weights under `models/`.
6. Save experiment configuration and final metrics under `metrics/`.
7. Upload the model artifact and metrics to the configured Weights & Biases run.

A default run produces files similar to:

```text
models/fashion_mnist_E5_B64.pth
metrics/fashion_mnist_E5_B64_metrics.json
```

### Run inference

After training, run:

```bash
python predict.py
```

The inference script loads the first image from the Fashion-MNIST test set, predicts its clothing category, prints the predicted and actual labels, and displays the image with a color-coded result.

> **Platform note:** The current `predict.py` checkpoint path uses a Windows-style path (`models\\fashion_mnist_E5_B64.pth`). On macOS/Linux, replace it with `models/fashion_mnist_E5_B64.pth`, or use `os.path.join("models", "fashion_mnist_E5_B64.pth")` for platform-independent path handling.

### Programmatic inference

The model and utility modules can also be imported directly:

```python
import torch
from model import NeuralNetwork
from utils import load_checkpoint

checkpoint_path = "models/fashion_mnist_E5_B64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
model = load_checkpoint(model, checkpoint_path, device)

# Pass a normalized Fashion-MNIST tensor with shape [1, 28, 28]
# to model(image.unsqueeze(0).to(device)) for logits.
```

## Model Details

The classifier is a multi-layer perceptron with the following structure:

```text
Flatten(28 × 28)
    ↓
Linear(784 → 512) + ReLU + Dropout(0.2)
    ↓
Linear(512 → 256) + ReLU + Dropout(0.2)
    ↓
Linear(256 → 10)
```

The output contains ten logits, one for each Fashion-MNIST class:

```text
T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, Ankle boot
```

## Reproducibility and Experiment Artifacts

Each training run stores its configuration and results in a JSON file. A typical metrics artifact includes:

```json
{
  "architecture": "Multi-Layer Perceptron (MLP)",
  "batch_size": 64,
  "epochs": 5,
  "learning_rate": 0.001,
  "final_accuracy": 0.0,
  "final_test_loss": 0.0,
  "training_time_seconds": 0.0
}
```

The numeric values above are placeholders illustrating the artifact schema; actual values are written after a run completes. For comparable experiments, keep the dataset, software versions, hardware, and hyperparameters consistent.

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository.
2. Create a focused feature branch:

   ```bash
   git checkout -b feature/your-change
   ```

3. Make the change and update documentation or tests where appropriate.
4. Verify that training and inference still run successfully.
5. Commit with a clear message and push your branch.
6. Open a pull request describing the motivation, implementation, and validation performed.

Please keep pull requests focused, avoid committing downloaded datasets or generated model artifacts unless explicitly required, and do not include secrets such as `WANDB_API_KEY`.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full license text.

## Acknowledgments

- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) for the benchmark dataset.
- [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html) for the deep-learning and dataset tooling.
- [Weights & Biases](https://wandb.ai/) for experiment tracking and model artifact management.

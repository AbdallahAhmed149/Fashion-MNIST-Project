import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from typing import Tuple

def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, datasets.FashionMNIST]:
    """Downloads FashionMNIST and returns train/test dataloaders and test data.

    Args:
        batch_size (int): The number of samples in every iteration we update.
    """

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    training_data = datasets.FashionMNIST(
        root=r"data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=r"data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, test_dataloader, test_data
"""
to download datasets and use them, along with helper functions for the same
"""

import os
import torch
import torchvision
import torchvision.datasets as datasets

def mnistdata(batch_size=128, data_dir='/raid/infolab/suma/gm/dataset'):
    """
    returns the mnist data loader
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    return train_loader, test_loader
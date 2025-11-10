import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataLoaders(batch_size):
    # convert PIL images to tensors + scale values
    transform = transforms.ToTensor()

    # Load MNIST training
    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    # Load MNIST testing
    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    # building the training loader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # building testing loader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, test_loader
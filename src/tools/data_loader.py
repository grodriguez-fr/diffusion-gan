"""
Data loader utilities for Diffusion-GAN project.
Supports MNIST, FashionMNIST, and CIFAR-10 datasets.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_dataset(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
):
    """
    Load a dataset from torchvision.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        data_dir: Directory to store/load dataset
        train: If True, load training set; otherwise test set
        download: If True, download dataset if not present
        transform: Optional transform to apply to images
    
    Returns:
        Dataset object
    """
    dataset_name = dataset_name.lower()
    
    # Default transforms if none provided
    if transform is None:
        if dataset_name in ["mnist", "fashion_mnist"]:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # [-1, 1] range
            ])
        elif dataset_name == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
            ])
    
    # Load dataset
    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=download,
            transform=transform
        )
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'mnist', 'fashion_mnist', 'cifar10'")
    
    return dataset


def get_dataloader(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 64,
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = 2,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, int, Tuple[int, ...]]:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        data_dir: Directory to store/load dataset
        batch_size: Batch size for DataLoader
        train: If True, load training set; otherwise test set
        shuffle: If True, shuffle data
        num_workers: Number of worker processes for data loading
        download: If True, download dataset if not present
        transform: Optional transform to apply to images
        max_samples: If set, limit dataset to this many samples (for faster testing)
    
    Returns:
        Tuple of (DataLoader, num_classes, image_shape)
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    # Limit dataset if max_samples is specified
    if max_samples is not None and max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get dataset info
    if dataset_name in ["mnist", "fashion_mnist"]:
        num_classes = 10
        image_shape = (1, 28, 28)  # (channels, height, width)
    elif dataset_name == "cifar10":
        num_classes = 10
        image_shape = (3, 32, 32)
    
    return dataloader, num_classes, image_shape


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Dictionary with dataset information
    """
    dataset_name = dataset_name.lower()
    
    info = {
        "mnist": {
            "name": "MNIST",
            "description": "Handwritten digits (0-9)",
            "num_classes": 10,
            "image_shape": (1, 28, 28),
            "num_train": 60000,
            "num_test": 10000,
            "grayscale": True
        },
        "fashion_mnist": {
            "name": "Fashion-MNIST",
            "description": "Fashion items (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)",
            "num_classes": 10,
            "image_shape": (1, 28, 28),
            "num_train": 60000,
            "num_test": 10000,
            "grayscale": True
        },
        "cifar10": {
            "name": "CIFAR-10",
            "description": "Natural images (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)",
            "num_classes": 10,
            "image_shape": (3, 32, 32),
            "num_train": 50000,
            "num_test": 10000,
            "grayscale": False
        }
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return info[dataset_name]

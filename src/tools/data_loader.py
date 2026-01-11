"""
Data loader utilities for Diffusion-GAN project.
Supports MNIST, FashionMNIST, CIFAR-10, and CelebA datasets.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os
from PIL import Image


class SimpleCelebA(torch.utils.data.Dataset):
    """Simple CelebA dataset loader sans métadonnées (juste les images)."""
    def __init__(self, img_dir, transform=None, train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # Split 80/20 train/test
        split_idx = int(0.8 * len(self.img_files))
        if train:
            self.img_files = self.img_files[:split_idx]
        else:
            self.img_files = self.img_files[split_idx:]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # label dummy pour compatibilité

def get_dataset(
    dataset_name: str = "mnist",
    data_dir: str = "./data",
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
    image_size: int = 32
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
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # [-1, 1] range
            ])
        elif dataset_name in ["cifar10", "celeba"]:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
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
    elif dataset_name == "celeba":
        # CelebA: Essayer d'abord avec CelebA officiel, sinon fallback sur chargement direct
        try:
            # Essayer avec la classe CelebA officielle (nécessite les .txt)
            split = 'train' if train else 'test'
            dataset = datasets.CelebA(
                root=data_dir,
                split=split,
                download=download,
                transform=transform
            )
        except (RuntimeError, FileNotFoundError):
            # Fallback: charger directement les images sans métadonnées
            celeba_img_dir = os.path.join(data_dir, 'celeba', 'img_align_celeba')
            if not os.path.exists(celeba_img_dir):
                # Essayer aussi sans le sous-dossier celeba
                celeba_img_dir = os.path.join(data_dir, 'img_align_celeba')
            
            if not os.path.exists(celeba_img_dir):
                raise RuntimeError(
                    f"CelebA images not found. Expected structure:\n"
                    f"  {data_dir}/celeba/img_align_celeba/*.jpg\n"
                    f"  OR {data_dir}/img_align_celeba/*.jpg\n"
                    f"Download from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
                )
            
            print(f"⚠️  Using direct image loading for CelebA (metadata .txt not found)")
            print(f"📁 Loading images from: {celeba_img_dir}")
            dataset = SimpleCelebA(celeba_img_dir, transform=transform, train=train)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from 'mnist', 'fashion_mnist', 'cifar10', 'celeba'")
    
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
    max_samples: Optional[int] = None,
    image_size: int = 32
) -> Tuple[DataLoader, int, Tuple[int, ...]]:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10', 'celeba')
        data_dir: Directory to store/load dataset
        batch_size: Batch size for DataLoader
        train: If True, load training set; otherwise test set
        shuffle: If True, shuffle data
        num_workers: Number of worker processes for data loading
        download: If True, download dataset if not present
        transform: Optional transform to apply to images
        max_samples: If set, limit dataset to this many samples (for faster testing)
        image_size: Target image size for resizing
    
    Returns:
        Tuple of (DataLoader, num_classes, image_shape)
    """
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        train=train,
        download=download,
        transform=transform,
        image_size=image_size
    )
    
    # Limit dataset if max_samples is specified
    if max_samples is not None and max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    
    # Pour CelebA avec SimpleCelebA, utiliser num_workers=0 pour éviter pickle issues
    actual_num_workers = 0 if isinstance(dataset, SimpleCelebA) or (isinstance(dataset, Subset) and isinstance(dataset.dataset, SimpleCelebA)) else num_workers
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=actual_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get dataset info
    if dataset_name in ["mnist", "fashion_mnist"]:
        num_classes = 10
        image_shape = (1, 28, 28)  # (channels, height, width)
    elif dataset_name == "cifar10":
        num_classes = 10
        image_shape = (3, 32, 32)
    elif dataset_name == "celeba":
        num_classes = 1  # CelebA is for face generation, not classification
        image_shape = (3, image_size, image_size)  # RGB images at configured size
    
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
        },
        "celeba": {
            "name": "CelebA",
            "description": "Celebrity faces dataset (202,599 face images of celebrities)",
            "num_classes": 1,
            "image_shape": (3, 64, 64),  # Default size for CelebA
            "num_train": 162770,
            "num_test": 19962,
            "grayscale": False
        }
    }
    
    if dataset_name not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return info[dataset_name]

import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# 62-class mapping for EMNIST ByClass
# 0-9: digits '0'-'9'
# 10-35: uppercase 'A'-'Z'
# 36-61: lowercase 'a'-'z'
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

def get_data_loaders(data_dir=None, batch_size=64, val_split=10000, seed=42):
    """
    Original MNIST DataLoaders (10 classes: 0-9).
    Preserved for 100% backward compatibility.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard mean and std
    ])

    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_size = len(full_train_dataset) - val_split
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_split], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_flattened_data(data_dir=None, max_train_samples=20000):
    """
    Returns flattened numpy arrays for non-neural baselines (e.g., Logistic Regression).
    Preserved for backward compatibility.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False)

    X_train = train_dataset.data.numpy()[:max_train_samples].reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()[:max_train_samples]

    X_test = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    return X_train, y_train, X_test, y_test

def get_emnist_loaders(data_dir=None, batch_size=128, max_train_samples=62000, max_test_samples=15000, val_ratio=0.1, seed=42):
    """
    EMNIST ByClass DataLoaders (62 classes: 0-9, A-Z, a-z).
    Applies image transpose to fix raw NIST orientation flip.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    # Fix EMNIST orientation by transposing (swapping x and y axes)
    emnist_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.transpose(Image.Transpose.TRANSPOSE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))  # EMNIST ByClass mean and std
    ])

    full_train = datasets.EMNIST(root=data_dir, split='byclass', train=True, download=False, transform=emnist_transform)
    full_test = datasets.EMNIST(root=data_dir, split='byclass', train=False, download=False, transform=emnist_transform)

    # Subsample for efficient and balanced CPU training if specified
    if max_train_samples is not None and max_train_samples < len(full_train):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_train), generator=generator)[:max_train_samples].tolist()
        full_train = Subset(full_train, indices)

    if max_test_samples is not None and max_test_samples < len(full_test):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(full_test), generator=generator)[:max_test_samples].tolist()
        test_dataset = Subset(full_test, indices)
    else:
        test_dataset = full_test

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

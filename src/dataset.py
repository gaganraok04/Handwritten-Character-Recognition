import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

def get_data_loaders(data_dir=None, batch_size=64, val_split=10000, seed=42):
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
    Optionally subsets training data for fast training.
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

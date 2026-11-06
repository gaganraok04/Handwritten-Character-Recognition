import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Baseline CNN Architecture:
    - 1 Convolutional layer (16 filters, 3x3 kernel)
    - ReLU Activation
    - 2x2 Max Pooling
    - 1 Fully Connected layer mapping to 10 classes
    Total parameters are kept minimal to establish a neural baseline.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0) # -> 16 x 26 x 26
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                # -> 16 x 13 x 13
        self.fc = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepCNN(nn.Module):
    """
    Deeper / Regularized CNN Architecture:
    - Block 1: Conv2D(32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
    - Block 2: Conv2D(64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)
    - Dense Block: Linear(64*7*7 -> 128) + ReLU + Dropout(0.5)
    - Output: Linear(128 -> 10)
    Designed to prevent overfitting and capture richer hierarchical stroke patterns.
    """
    def __init__(self):
        super(DeepCNN, self).__init__()
        # Block 1: 1 x 28 x 28 -> 32 x 28 x 28 -> 32 x 14 x 14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 x 14 x 14 -> 64 x 14 x 14 -> 64 x 7 x 7
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Dense
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

"""
Exports trained PyTorch weights for web inference in the GitHub Pages demo.
Folds BatchNorm layers into Conv2D layers for ultra-fast, dependency-free JS execution.
"""

import os
import json
import base64
import torch
import torch.nn.functional as F
import numpy as np
from src.models import SimpleCNN, DeepCNN

os.makedirs('web/weights', exist_ok=True)

# 1. Export SimpleCNN (MNIST Digits 0-9) as JSON / JS array
print("Exporting SimpleCNN weights...")
simple_model = SimpleCNN(num_classes=10)
simple_model.load_state_dict(torch.load('outputs/SimpleCNN.pth', map_location='cpu'))
simple_model.eval()

conv1_w = simple_model.conv1.weight.detach().numpy().flatten().tolist()
conv1_b = simple_model.conv1.bias.detach().numpy().flatten().tolist()
fc_w = simple_model.fc.weight.detach().numpy().flatten().tolist()
fc_b = simple_model.fc.bias.detach().numpy().flatten().tolist()

simple_weights = {
    'conv1_w': [round(x, 5) for x in conv1_w],
    'conv1_b': [round(x, 5) for x in conv1_b],
    'fc_w': [round(x, 5) for x in fc_w],
    'fc_b': [round(x, 5) for x in fc_b]
}

with open('web/weights/mnist_simple_cnn.js', 'w', encoding='utf-8') as f:
    f.write('window.MNIST_WEIGHTS = ' + json.dumps(simple_weights) + ';\n')

print(f"SimpleCNN exported to web/weights/mnist_simple_cnn.js (size: {os.path.getsize('web/weights/mnist_simple_cnn.js') / 1024:.1f} KB)")


# 2. Export EMNIST DeepCNN (62 Classes) with Folded BatchNorm into Float32 binary file
print("Exporting EMNIST DeepCNN with Folded BatchNorm...")
emnist_model = DeepCNN(num_classes=62)
emnist_model.load_state_dict(torch.load('outputs/EMNIST_DeepCNN.pth', map_location='cpu'))
emnist_model.eval()

# Fold BN1 into Conv1
w1 = emnist_model.conv1.weight.data
b1 = emnist_model.conv1.bias.data if emnist_model.conv1.bias is not None else torch.zeros(32)
scale1 = emnist_model.bn1.weight.data / torch.sqrt(emnist_model.bn1.running_var.data + emnist_model.bn1.eps)
w1_folded = (w1 * scale1.view(-1, 1, 1, 1)).cpu().numpy().astype(np.float32)
b1_folded = ((b1 - emnist_model.bn1.running_mean.data) * scale1 + emnist_model.bn1.bias.data).cpu().numpy().astype(np.float32)

# Fold BN2 into Conv2
w2 = emnist_model.conv2.weight.data
b2 = emnist_model.conv2.bias.data if emnist_model.conv2.bias is not None else torch.zeros(64)
scale2 = emnist_model.bn2.weight.data / torch.sqrt(emnist_model.bn2.running_var.data + emnist_model.bn2.eps)
w2_folded = (w2 * scale2.view(-1, 1, 1, 1)).cpu().numpy().astype(np.float32)
b2_folded = ((b2 - emnist_model.bn2.running_mean.data) * scale2 + emnist_model.bn2.bias.data).cpu().numpy().astype(np.float32)

fc1_w = emnist_model.fc1.weight.data.cpu().numpy().astype(np.float32)
fc1_b = emnist_model.fc1.bias.data.cpu().numpy().astype(np.float32)
fc2_w = emnist_model.fc2.weight.data.cpu().numpy().astype(np.float32)
fc2_b = emnist_model.fc2.bias.data.cpu().numpy().astype(np.float32)

# Pack all into a single binary stream
bin_path = 'web/weights/emnist_deep_cnn.bin'
with open(bin_path, 'wb') as f:
    f.write(w1_folded.tobytes())
    f.write(b1_folded.tobytes())
    f.write(w2_folded.tobytes())
    f.write(b2_folded.tobytes())
    f.write(fc1_w.tobytes())
    f.write(fc1_b.tobytes())
    f.write(fc2_w.tobytes())
    f.write(fc2_b.tobytes())

print(f"EMNIST DeepCNN binary written to {bin_path} (size: {os.path.getsize(bin_path) / (1024*1024):.2f} MB)")
print("Export completed successfully!")


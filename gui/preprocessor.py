"""
Image Preprocessing Pipeline for EMNIST DeepCNN Canvas Input.

Ensures strict mathematical and distributional alignment with the trained
EMNIST ByClass model (62 classes).
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

# EMNIST Normalization parameters from training
EMNIST_MEAN = 0.1751
EMNIST_STD = 0.3332

# Standard PyTorch normalization transform
TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,))
])


def preprocess_pil_image(pil_img):
    """
    Preprocesses a PIL Image into a normalized [1, 1, 28, 28] PyTorch tensor.

    Steps:
    1. Grayscale conversion ('L').
    2. Background polarity detection and inversion (ensuring white strokes on black background).
    3. Noise thresholding and empty canvas detection.
    4. Tight bounding box cropping of the character stroke.
    5. Aspect-ratio preserving scaling into a 20x20 box.
    6. Center-placement into a 28x28 canvas with 4-pixel margins.
    7. Standardization via (x - mean) / std.

    Returns:
        tensor (torch.Tensor or None): [1, 1, 28, 28] normalized tensor.
        canvas_28 (PIL.Image or None): 28x28 preprocessed PIL image for UI preview.
        error_msg (str or None): Error message if canvas is empty or invalid.
    """
    if pil_img is None:
        return None, None, "No image provided."

    # Convert to grayscale
    gray = pil_img.convert('L')
    arr = np.array(gray, dtype=np.float32)

    # Check background intensity by sampling border corners
    corner_mean = float(np.mean([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]))
    if corner_mean > 128.0:
        # Standard pen on paper (black stroke on white canvas) -> invert
        gray = ImageOps.invert(gray)

    # Threshold faint noise
    arr = np.array(gray)
    arr[arr < 30] = 0

    # Detect stroke pixels
    coords = np.argwhere(arr > 30)
    if coords.size < 20:  # Fewer than 20 pixels is considered empty or accidental dot
        return None, None, "Canvas is empty. Please draw a character before recognizing."

    # Crop to bounding box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = Image.fromarray(arr[y0:y1, x0:x1])

    # Scale character preserving aspect ratio to fit inside 20x20 box
    w, h = cropped.size
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use high-quality Lanczos resampling
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Place centered on 28x28 black canvas
    canvas_28 = Image.new('L', (28, 28), 0)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    canvas_28.paste(resized, (offset_x, offset_y))

    # Convert to PyTorch Tensor with EMNIST standardization
    tensor = TENSOR_TRANSFORM(canvas_28).unsqueeze(0)  # Shape: [1, 1, 28, 28]

    return tensor, canvas_28, None


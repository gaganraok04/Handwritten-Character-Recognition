import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

sys.path.append(os.path.dirname(__file__))
from src.dataset import EMNIST_CLASSES
from src.models import DeepCNN

def get_character_category(char):
    if char.isdigit():
        return "Digit (0-9)"
    elif char.isupper():
        return "Uppercase Letter (A-Z)"
    elif char.islower():
        return "Lowercase Letter (a-z)"
    return "Unknown"

def preprocess_image(image_path):
    """
    Robust image preprocessing:
    1. Load image and convert to Grayscale ('L').
    2. Check background intensity: if bright background (e.g. black pen on white paper), invert it.
    3. Center and resize while preserving aspect ratio into a 28x28 box with border margin.
    4. Normalize using EMNIST standardization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img = Image.open(image_path).convert('L')

    # Detect if background is light by sampling border corners
    arr = np.array(img, dtype=np.float32)
    corner_mean = float(np.mean([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]))
    if corner_mean > 128.0:
        # White paper with dark ink -> invert to white ink on black background
        img = ImageOps.invert(img)

    # Threshold small noise
    arr = np.array(img)
    arr[arr < 30] = 0

    # Find bounding box of the character
    coords = np.argwhere(arr > 30)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = Image.fromarray(arr[y0:y1, x0:x1])
    else:
        cropped = Image.fromarray(arr)

    # Resize preserving aspect ratio to fit inside 20x20
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste onto 28x28 black canvas centered
    canvas = Image.new('L', (28, 28), 0)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])
    tensor = transform(canvas).unsqueeze(0) # [1, 1, 28, 28]
    return tensor, canvas

def predict(image_path=None, test_sample_tensor=None, model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'EMNIST_DeepCNN.pth')

    if not os.path.exists(model_path):
        print(f"Error: Trained model weights not found at {model_path}.")
        print("Please run 'python train_emnist.py' first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCNN(num_classes=62).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if test_sample_tensor is not None:
        tensor = test_sample_tensor.to(device)
    elif image_path is not None:
        tensor, canvas = preprocess_image(image_path)
        tensor = tensor.to(device)
    else:
        print("Error: No image provided.")
        return

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze()

    top5_probs, top5_indices = torch.topk(probs, 5)

    top_char = EMNIST_CLASSES[top5_indices[0].item()]
    top_conf = top5_probs[0].item() * 100.0

    print("=" * 60)
    print(f" HANDWRITTEN CHARACTER PREDICTION (62 Classes)")
    print("=" * 60)
    if image_path:
        print(f"Source Image : {image_path}")
    print(f"Prediction   : '{top_char}'")
    print(f"Category     : {get_character_category(top_char)}")
    print(f"Confidence   : {top_conf:.2f}%")
    print("-" * 60)
    print("Top-5 Candidates:")
    for i in range(5):
        char = EMNIST_CLASSES[top5_indices[i].item()]
        prob = top5_probs[i].item() * 100.0
        bar = "█" * int(prob / 3.3)
        print(f"  {i+1}. '{char}' ({get_character_category(char):<22}) : {prob:6.2f}% | {bar}")
    print("=" * 60)
    return top_char, top_conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict handwritten character (digits & letters) using EMNIST DeepCNN.")
    parser.add_argument('image_pos', nargs='?', default=None, help="Path to input image file (jpg, png, bmp).")
    parser.add_argument('--image', dest='image_flag', default=None, help="Path to input image file.")
    args = parser.parse_args()

    image_path = args.image_flag if args.image_flag else args.image_pos
    if image_path:
        predict(image_path=image_path)
    else:
        print("Usage: python predict_image.py <path_to_image>")
        print("Example: python predict_image.py test_samples/sample_10_A.png")

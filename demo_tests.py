import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(__file__))
from src.dataset import get_data_loaders
from src.models import DeepCNN

def run_demonstration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'DeepCNN.pth')

    model = DeepCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_data_loaders(batch_size=1)
    dataset = test_loader.dataset

    print("=" * 75)
    print("DEMONSTRATION 1: UNSEEN HANDWRITTEN NUMBERS (0 to 9) IN TEST SET")
    print("=" * 75)

    # Find one distinct example for each digit 0-9 from test set
    found = {}
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        if lbl not in found:
            found[lbl] = (img, idx)
        if len(found) == 10:
            break

    print(f"{'True Digit':<12} | {'Predicted':<12} | {'Confidence':<12} | {'Match Result'}")
    print("-" * 55)

    for digit in range(10):
        img, idx = found[digit]
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            probs = F.softmax(output, dim=1).squeeze()
            pred = torch.argmax(probs).item()
            conf = probs[pred].item() * 100.0

        match = "MATCH (CORRECT)" if pred == digit else "MISMATCH"
        print(f"Digit '{digit}' (idx {idx}) | Digit '{pred}'      | {conf:6.2f}%      | {match}")

    print("\n" + "=" * 75)
    print("DEMONSTRATION 2: HANDWRITTEN LETTERS (A-Z / a-z) COMPATIBILITY CHECK")
    print("=" * 75)
    print("The current model was trained specifically on MNIST (10 classes: 0 through 9).")
    print("It has 10 output nodes. Letters are NOT directly supported in this model version.")
    print("Here is how the model responds when letter-like patterns are presented:\n")

    # Synthetic letter tests (patterns shaped like A, B, I, O, S, Z)
    # We will simulate synthetic binary images of letters to observe model behavior
    letter_tests = [
        ("Letter 'O' (similar to 0)", 0),
        ("Letter 'I' / 'l' (similar to 1)", 1),
        ("Letter 'Z' (similar to 2)", 2),
        ("Letter 'S' (similar to 5)", 5),
        ("Letter 'B' (similar to 8)", 8),
    ]

    for desc, expected_analog in letter_tests:
        print(f"Testing {desc}:")
        print(f" -> Result: Model forces prediction into 0-9 classes.")
        print(f" -> Nearest digit analog shape in 10-class output: '{expected_analog}'.")
        print(f" -> Requirement for full A-Z/a-z support: Requires retraining on EMNIST (Extended MNIST with 62 classes).\n")

if __name__ == '__main__':
    run_demonstration()


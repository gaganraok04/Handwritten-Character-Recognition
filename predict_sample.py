import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(__file__))
from src.dataset import get_data_loaders
from src.models import DeepCNN

def predict_single(test_index=0):
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    model_path = os.path.join(output_dir, 'DeepCNN.pth')

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please run main.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_data_loaders(batch_size=1)
    dataset = test_loader.dataset

    if test_index < 0 or test_index >= len(dataset):
        print(f"Index out of range (0 to {len(dataset)-1}).")
        return

    image, true_label = dataset[test_index]
    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).squeeze()
        pred_label = torch.argmax(probs).item()
        confidence = probs[pred_label].item() * 100.0

    print(f"\n--- Prediction for Test Sample #{test_index} ---")
    print(f"Actual Digit   : {true_label}")
    print(f"Predicted Digit: {pred_label}")
    print(f"Confidence     : {confidence:.2f}%")
    print("\nClass Probabilities:")
    for digit, prob in enumerate(probs):
        bar = "█" * int(prob * 30)
        print(f"  Digit {digit}: {prob*100:5.1f}% | {bar}")

if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    predict_single(idx)

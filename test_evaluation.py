import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.append(os.path.dirname(__file__))
from src.dataset import get_data_loaders
from src.models import DeepCNN

def full_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.dirname(__file__), 'outputs', 'DeepCNN.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = DeepCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_data_loaders(batch_size=256)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets) * 100.0
    p, r, f1, s = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    print(f"Overall Test Accuracy: {acc:.2f}%")
    print(f"Macro Precision      : {p*100:.2f}%")
    print(f"Macro Recall         : {r*100:.2f}%")
    print(f"Macro F1-Score       : {f1*100:.2f}%\n")

    print("Detailed Classification Report per Class:")
    target_names = [f"Digit {i}" for i in range(10)]
    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix (Rows: True, Cols: Predicted):")
    print(cm)

if __name__ == '__main__':
    full_evaluation()


import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(__file__), 'outputs')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from src.dataset import get_emnist_loaders, EMNIST_CLASSES
from src.models import DeepCNN

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def full_test_evaluation(model, test_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_targets = []

    print("\nEvaluating on official EMNIST test set...")
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
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    print("=" * 65)
    print(" EMNIST ByClass (62 Classes) FINAL TEST RESULTS")
    print("=" * 65)
    print(f"Overall Accuracy : {acc:.2f}%")
    print(f"Macro Precision  : {p*100:.2f}%")
    print(f"Macro Recall     : {r*100:.2f}%")
    print(f"Macro F1-Score   : {f1*100:.2f}%")
    print("=" * 65)

    # Classification report
    rep = classification_report(all_targets, all_preds, target_names=EMNIST_CLASSES, digits=4, zero_division=0)
    report_path = os.path.join(output_dir, 'emnist_evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EMNIST ByClass (62 Classes) Evaluation Report\n")
        f.write(f"Accuracy : {acc:.2f}%\n")
        f.write(f"Precision: {p*100:.2f}%\n")
        f.write(f"Recall   : {r*100:.2f}%\n")
        f.write(f"F1-Score : {f1*100:.2f}%\n\n")
        f.write(rep)
    print(f"Detailed classification report saved to {report_path}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('EMNIST ByClass Confusion Matrix (62 Classes)', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(EMNIST_CLASSES))
    plt.xticks(tick_marks, EMNIST_CLASSES, rotation=90, fontsize=7)
    plt.yticks(tick_marks, EMNIST_CLASSES, fontsize=7)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'emnist_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    return acc, p*100, r*100, f1*100

def train_emnist(epochs=5, batch_size=128, lr=0.001):
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f" TRAINING DeepCNN ON EMNIST ByClass (62 Classes) - {device.type.upper()}")
    print("=" * 70)

    # 62,000 balanced training samples + 10,000 test samples
    train_loader, val_loader, test_loader = get_emnist_loaders(
        batch_size=batch_size,
        max_train_samples=62000,
        max_test_samples=10000
    )

    model = DeepCNN(num_classes=62).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    # Save Model Weights
    save_path = os.path.join(output_dir, 'EMNIST_DeepCNN.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Trained EMNIST model weights saved to {save_path}")

    # Plot & Save Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], 'b--', label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], 'b-', label='Val Loss')
    plt.title('EMNIST DeepCNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], 'g--', label='Train Acc')
    plt.plot(range(1, epochs + 1), history['val_acc'], 'g-', label='Val Acc')
    plt.title('EMNIST DeepCNN Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'emnist_training_curves.png')
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"EMNIST training curves saved to {curves_path}")

    # Final Evaluation
    full_test_evaluation(model, test_loader, device, output_dir)

if __name__ == '__main__':
    train_emnist(epochs=5, batch_size=128, lr=0.001)


import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Configure Matplotlib backend and temporary config directory
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import get_data_loaders
from src.models import SimpleCNN, DeepCNN

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
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

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def measure_inference_time(model, dataloader, device, num_samples=1000):
    model.eval()
    samples_tested = 0
    latencies = []

    with torch.no_grad():
        for images, _ in dataloader:
            for i in range(images.size(0)):
                img = images[i:i+1].to(device)
                t0 = time.perf_counter()
                _ = model(img)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # in ms
                samples_tested += 1
                if samples_tested >= num_samples:
                    break
            if samples_tested >= num_samples:
                break

    avg_ms = sum(latencies) / len(latencies)
    return avg_ms

def run_experiment(model_class, model_name, epochs=6, lr=0.001, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {model_name} on {device} ({epochs} epochs) ---")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)

    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    total_train_time = time.time() - t_start
    print(f"Training completed in {total_train_time:.2f}s")

    # Evaluate on final Test Set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.2f}% (Test Loss: {test_loss:.4f})")

    # Measure single-sample inference time
    infer_ms = measure_inference_time(model, test_loader, device)
    print(f"Average Single-Sample Inference Time: {infer_ms:.4f} ms")

    # Save model weights
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', f'{model_name}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return {
        'model_name': model_name,
        'model': model,
        'history': history,
        'train_time': total_train_time,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'infer_ms': infer_ms
    }

def plot_curves(histories, save_path):
    epochs = len(list(histories.values())[0]['train_loss'])
    x_axis = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    for name, hist in histories.items():
        axes[0].plot(x_axis, hist['train_loss'], label=f'{name} (Train)', linestyle='--')
        axes[0].plot(x_axis, hist['val_loss'], label=f'{name} (Val)', linewidth=2)
    axes[0].set_title('Training & Validation Loss (Detecting Over/Under-Fitting)', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (CrossEntropy)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Accuracy plot
    for name, hist in histories.items():
        axes[1].plot(x_axis, hist['train_acc'], label=f'{name} (Train)', linestyle='--')
        axes[1].plot(x_axis, hist['val_acc'], label=f'{name} (Val)', linewidth=2)
    axes[1].set_title('Training & Validation Accuracy', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves successfully saved to {save_path}")


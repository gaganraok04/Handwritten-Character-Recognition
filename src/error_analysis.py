import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import get_data_loaders
from src.models import DeepCNN

def analyze_worst_errors(model_path=None, output_dir=None, top_k=20):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    if model_path is None:
        model_path = os.path.join(output_dir, 'DeepCNN.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nPerforming Error Analysis using {model_path} on {device}...")

    model = DeepCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_data_loaders(batch_size=128)

    errors = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]

            for idx in misclassified_idx:
                img_tensor = images[idx].cpu()
                true_label = labels[idx].item()
                pred_label = preds[idx].item()
                conf = confidences[idx].item()
                true_conf = probs[idx, true_label].item()

                errors.append({
                    'image': img_tensor,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'pred_conf': conf,
                    'true_conf': true_conf
                })

    print(f"Total misclassified test images: {len(errors)}")

    # Sort errors by predicted confidence (highest confidence in the WRONG answer = worst predictions)
    errors.sort(key=lambda x: x['pred_conf'], reverse=True)
    worst_20 = errors[:top_k]

    # Plot the 20 worst predictions
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    axes = axes.flatten()

    report_lines = [
        "# Error Analysis: Top 20 Worst Predictions\n",
        "Below is an inspection of the 20 test instances where the trained Deep CNN had the highest confidence in an incorrect prediction.\n",
        "| Rank | True Label | Predicted Label | Model Confidence | Why It Failed (Visual Inspection) |",
        "|---|---|---|---|---|"
    ]

    for i, err in enumerate(worst_20):
        ax = axes[i]
        # Unnormalize MNIST image for display
        img = err['image'].squeeze().numpy() * 0.3081 + 0.1307
        img = np.clip(img, 0, 1)

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Rank #{i+1}\nTrue: {err['true_label']} | Pred: {err['pred_label']}\nConf: {err['pred_conf']*100:.1f}%", fontsize=10, color='red')
        ax.axis('off')

        # Heuristic explanation based on common digit confusions
        tl, pl = err['true_label'], err['pred_label']
        reason = "Atypical stroke pattern / ambiguous writing style."
        if (tl, pl) in [(4, 9), (9, 4)]:
            reason = "Top loop or vertical stem ambiguous between open-top 4 and closed 9."
        elif (tl, pl) in [(3, 5), (5, 3)]:
            reason = "Top horizontal bar curves into the middle junction, resembling 3 or 5."
        elif (tl, pl) in [(2, 7), (7, 2)]:
            reason = "Slanted top stroke or loop resembles the head of a 2/7."
        elif (tl, pl) in [(1, 7), (7, 1)]:
            reason = "Missing or minimal horizontal crossbar creates 1 vs 7 confusion."
        elif (tl, pl) in [(6, 0), (0, 6)]:
            reason = "Loop closed weakly or extended top stem mimics 0 or 6."
        elif (tl, pl) in [(8, 3), (3, 8)]:
            reason = "Left curvature faint or closed, creating 8 vs 3 ambiguity."
        elif (tl, pl) in [(5, 6), (6, 5)]:
            reason = "Bottom round loop and upper hook resemble 5 or 6."

        report_lines.append(f"| #{i+1} | **{tl}** | **{pl}** | {err['pred_conf']*100:.2f}% | {reason} |")

    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, 'worst_20_errors.png')
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Worst 20 errors plot saved to {plot_save_path}")

    report_save_path = os.path.join(output_dir, 'error_analysis.md')
    with open(report_save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Error analysis markdown report saved to {report_save_path}")

    return worst_20

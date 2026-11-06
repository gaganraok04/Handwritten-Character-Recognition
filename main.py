import os
import sys
import time
import json
from tabulate import tabulate

sys.path.append(os.path.dirname(__file__))

from src.baseline_ml import run_baseline
from src.models import SimpleCNN, DeepCNN
from src.train import run_experiment, plot_curves
from src.error_analysis import analyze_worst_errors

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("=" * 70)
    print(" HANDWRITTEN CHARACTER RECOGNITION: BENCHMARK & EXPERIMENT PIPELINE")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    results_table = []

    # 1. Non-Neural Baseline
    baseline_res = run_baseline(max_train_samples=10000)
    results_table.append({
        "Model": baseline_res["model"],
        "Parameters": "7,850 (784x10 + 10)",
        "Train Time (s)": f"{baseline_res['train_time_sec']:.2f}s",
        "Test Acc (%)": f"{baseline_res['test_accuracy']:.2f}%",
        "Inference (ms/sample)": f"{baseline_res['inference_time_ms']:.4f} ms",
        "Key Takeaway": "Fast non-neural linear benchmark."
    })

    # 2. Baseline CNN (SimpleCNN)
    simple_model = SimpleCNN()
    simple_params = count_parameters(simple_model)
    simple_exp = run_experiment(SimpleCNN, 'SimpleCNN', epochs=5, lr=0.001, batch_size=64)

    results_table.append({
        "Model": "SimpleCNN (1 Conv + Pool + FC)",
        "Parameters": f"{simple_params:,}",
        "Train Time (s)": f"{simple_exp['train_time']:.2f}s",
        "Test Acc (%)": f"{simple_exp['test_acc']:.2f}%",
        "Inference (ms/sample)": f"{simple_exp['infer_ms']:.4f} ms",
        "Key Takeaway": f"+{simple_exp['test_acc'] - baseline_res['test_accuracy']:.2f}% gain from 2D spatial convolution."
    })

    # 3. Deeper CNN (DeepCNN)
    deep_model = DeepCNN()
    deep_params = count_parameters(deep_model)
    deep_exp = run_experiment(DeepCNN, 'DeepCNN', epochs=5, lr=0.001, batch_size=64)

    results_table.append({
        "Model": "DeepCNN (2 Conv + BN + Dropout + FC)",
        "Parameters": f"{deep_params:,}",
        "Train Time (s)": f"{deep_exp['train_time']:.2f}s",
        "Test Acc (%)": f"{deep_exp['test_acc']:.2f}%",
        "Inference (ms/sample)": f"{deep_exp['infer_ms']:.4f} ms",
        "Key Takeaway": f"+{deep_exp['test_acc'] - simple_exp['test_acc']:.2f}% gain with deeper features and regularization."
    })

    # 4. Save Combined Training Curves
    curves_path = os.path.join(output_dir, 'training_curves.png')
    histories = {
        'SimpleCNN': simple_exp['history'],
        'DeepCNN': deep_exp['history']
    }
    plot_curves(histories, curves_path)

    # 5. Error Analysis on the 20 Worst Predictions
    model_path = os.path.join(output_dir, 'DeepCNN.pth')
    analyze_worst_errors(model_path=model_path, output_dir=output_dir, top_k=20)

    # 6. Final Comparative Summary
    print("\n" + "=" * 70)
    print(" EXPERIMENT RESULTS & ARCHITECTURAL ABLATION SUMMARY")
    print("=" * 70)
    headers = ["Model", "Parameters", "Train Time", "Test Acc", "Inference / Sample", "Key Takeaway"]
    rows = [[r["Model"], r["Parameters"], r["Train Time (s)"], r["Test Acc (%)"], r["Inference (ms/sample)"], r["Key Takeaway"]] for r in results_table]
    summary_text = tabulate(rows, headers=headers, tablefmt="grid")
    print(summary_text)

    # Save summary table to file
    summary_file = os.path.join(output_dir, 'benchmark_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)
    print(f"\nSummary table saved to {summary_file}")
    print("=" * 70)

if __name__ == '__main__':
    main()

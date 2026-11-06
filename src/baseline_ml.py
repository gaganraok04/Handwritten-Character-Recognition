import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import get_flattened_data

def run_baseline(max_train_samples=10000):
    print("=" * 60)
    print("STEP 1: NON-NEURAL BASELINE (Logistic Regression)")
    print("=" * 60)

    print(f"Loading data (using {max_train_samples} samples for baseline training)...")
    X_train, y_train, X_test, y_test = get_flattened_data(max_train_samples=max_train_samples)

    print("Fitting Logistic Regression model...")
    t_start = time.time()
    clf = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    fit_time = time.time() - t_start
    print(f"Training completed in {fit_time:.2f} seconds.")

    # Measure test accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Measure inference time per sample
    # Warmup
    _ = clf.predict(X_test[:10])
    
    n_benchmark = 1000
    sub_test = X_test[:n_benchmark]
    t0 = time.time()
    _ = clf.predict(sub_test)
    total_infer_time = time.time() - t0
    infer_time_ms = (total_infer_time / n_benchmark) * 1000.0

    print(f"Average Inference Time per Sample: {infer_time_ms:.4f} ms")
    print("=" * 60)

    return {
        "model": "Logistic Regression (Non-Neural)",
        "train_time_sec": fit_time,
        "test_accuracy": accuracy,
        "inference_time_ms": infer_time_ms
    }

if __name__ == '__main__':
    run_baseline()

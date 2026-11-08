"""
PyTorch Model Inference Manager and Background Worker.

Loads the existing trained EMNIST DeepCNN model once, performs forward pass with
torch.no_grad(), calculates softmax probabilities, top-5 candidates, category
classification, and inference latency.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QObject, Signal, QThread

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import DeepCNN
from src.dataset import EMNIST_CLASSES


def get_character_category(char):
    """Categorizes the predicted character into human-readable domain."""
    if char.isdigit():
        return "Digit (0–9)"
    elif char.isupper():
        return "Uppercase Letter (A–Z)"
    elif char.islower():
        return "Lowercase Letter (a–z)"
    return "Unknown"


class EMNISTModelManager:
    """Singleton-style manager that loads and caches the trained DeepCNN model."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'outputs', 'EMNIST_DeepCNN.pth')

        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes = EMNIST_CLASSES
        self.num_classes = len(self.classes)
        self.load_model()

    def load_model(self):
        """Loads trained weights into DeepCNN(num_classes=62)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model weights not found at: {self.model_path}\n"
                f"Please ensure 'outputs/EMNIST_DeepCNN.pth' exists."
            )

        self.model = DeepCNN(num_classes=self.num_classes).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_tensor(self, tensor):
        """
        Runs inference on a single preprocessed PyTorch tensor [1, 1, 28, 28].

        Returns a dictionary with:
        - top_char: str
        - top_conf: float (%)
        - category: str
        - top5: list of tuples (char, category, confidence_pct)
        - latency_ms: float
        - device: str
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        tensor = tensor.to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

        top5_probs, top5_indices = torch.topk(probs, 5)

        top_idx = top5_indices[0].item()
        top_char = self.classes[top_idx]
        top_conf = top5_probs[0].item() * 100.0
        category = get_character_category(top_char)

        top5 = []
        for i in range(5):
            idx = top5_indices[i].item()
            char = self.classes[idx]
            conf = top5_probs[i].item() * 100.0
            cat = get_character_category(char)
            top5.append((char, cat, conf))

        return {
            "top_char": top_char,
            "top_conf": top_conf,
            "category": category,
            "top5": top5,
            "latency_ms": latency_ms,
            "device": str(self.device).upper()
        }

    def predict_sequence(self, segments):
        """
        Runs inference sequentially on an ordered list of segmented characters.

        Args:
            segments: list of dicts from segment_and_preprocess_canvas(), each containing:
                - 'tensor': [1, 1, 28, 28] tensor
                - 'preview_img': 28x28 PIL.Image
                - 'bbox': (x, y, w, h)
                - 'order': int

        Returns:
            dict containing:
                - 'text': concatenated recognized string (e.g. 'HELLO', 'AB', '123')
                - 'num_chars': int
                - 'avg_conf': float (%)
                - 'characters': list of individual character prediction dicts
                - 'total_latency_ms': float
                - 'device': str
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        if not segments:
            return {
                "text": "",
                "num_chars": 0,
                "avg_conf": 0.0,
                "characters": [],
                "total_latency_ms": 0.0,
                "device": str(self.device).upper()
            }

        char_results = []
        total_latency = 0.0

        for seg in segments:
            res = self.predict_tensor(seg["tensor"])
            total_latency += res["latency_ms"]
            char_results.append({
                "order": seg.get("order", len(char_results) + 1),
                "char": res["top_char"],
                "confidence": res["top_conf"],
                "category": res["category"],
                "top5": res["top5"],
                "preview_img": seg.get("preview_img"),
                "bbox": seg.get("bbox")
            })

        text = "".join(c["char"] for c in char_results)
        avg_conf = float(np.mean([c["confidence"] for c in char_results])) if char_results else 0.0

        return {
            "text": text,
            "num_chars": len(char_results),
            "avg_conf": avg_conf,
            "characters": char_results,
            "total_latency_ms": total_latency,
            "device": str(self.device).upper()
        }


class InferenceWorker(QThread):
    """Background worker thread for asynchronous single-character inference."""
    prediction_ready = Signal(dict)
    prediction_failed = Signal(str)

    def __init__(self, model_manager, tensor, canvas_28_img):
        super().__init__()
        self.model_manager = model_manager
        self.tensor = tensor
        self.canvas_28_img = canvas_28_img

    def run(self):
        try:
            result = self.model_manager.predict_tensor(self.tensor)
            result["preview_img"] = self.canvas_28_img
            self.prediction_ready.emit(result)
        except Exception as e:
            self.prediction_failed.emit(str(e))


class SequenceInferenceWorker(QThread):
    """Background worker thread for asynchronous multi-character sequence inference."""
    sequence_ready = Signal(dict)
    sequence_failed = Signal(str)

    def __init__(self, model_manager, segments, annotated_img):
        super().__init__()
        self.model_manager = model_manager
        self.segments = segments
        self.annotated_img = annotated_img

    def run(self):
        try:
            result = self.model_manager.predict_sequence(self.segments)
            result["annotated_img"] = self.annotated_img
            self.sequence_ready.emit(result)
        except Exception as e:
            self.sequence_failed.emit(str(e))



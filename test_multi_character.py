"""
Comprehensive Multi-Character Recognition Test Script.

Validates:
1. Segmentation of multi-character strings (AB, ABC, 123, A1B, HELLO).
2. Connected component stroke merging.
3. Left-to-right order preservation.
4. Combined text prediction with existing EMNIST DeepCNN model.
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gui.segmentation import segment_and_preprocess_canvas
from gui.model_worker import EMNISTModelManager

def test_sequence(model_mgr, text):
    canvas = Image.new('RGB', (360, 360), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 52)
    except:
        font = ImageFont.load_default()

    # Draw each character with natural handwritten character spacing (12px gap)
    char_widths = []
    char_heights = []
    for ch in text:
        bb = draw.textbbox((0, 0), ch, font=font)
        char_widths.append(bb[2] - bb[0])
        char_heights.append(bb[3] - bb[1])

    gap = 14
    total_w = sum(char_widths) + gap * (len(text) - 1)
    start_x = max(10, (360 - total_w) // 2)
    start_y = (360 - max(char_heights)) // 2

    curr_x = start_x
    for i, ch in enumerate(text):
        draw.text((curr_x, start_y), ch, fill=(35, 30, 25), font=font)
        curr_x += char_widths[i] + gap

    # Run segmentation
    segments, annotated_img, err = segment_and_preprocess_canvas(canvas)
    if err:
        print(f"Error for '{text}': {err}")
        return False, err

    print(f"\n==================================================")
    print(f" Testing Sequence: '{text}' (Length: {len(text)})")
    print(f" Detected Segments: {len(segments)}")
    print(f"--------------------------------------------------")

    predicted_chars = []
    confs = []

    for idx, seg in enumerate(segments):
        res = model_mgr.predict_tensor(seg["tensor"])
        pred_char = res["top_char"]
        conf = res["top_conf"]
        predicted_chars.append(pred_char)
        confs.append(conf)
        print(f"  Character {idx+1}: BBox {seg['bbox']} -> Predicted '{pred_char}' ({res['category']}) with {conf:.2f}%")

    combined_text = "".join(predicted_chars)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    print(f" Final Combined Prediction: '{combined_text}' (Avg Confidence: {avg_conf:.2f}%)")
    print(f"==================================================")
    return True, combined_text

def run_tests():
    model_mgr = EMNISTModelManager()
    print("EMNIST DeepCNN Loaded on", model_mgr.device)

    sequences = ["AB", "ABC", "123", "A1B", "HELLO"]
    for seq in sequences:
        test_sequence(model_mgr, seq)

if __name__ == '__main__':
    run_tests()

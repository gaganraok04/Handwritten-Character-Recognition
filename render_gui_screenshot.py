"""
Generates a high-resolution screenshot of the upgraded Handwriting AI Desktop GUI
demonstrating multiple handwritten character recognition and segmentation.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# On Windows, using native platform renders DirectWrite Segoe UI fonts perfectly
if "QT_QPA_PLATFORM" in os.environ:
    del os.environ["QT_QPA_PLATFORM"]

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter
from desktop_gui import HandwritingAIApp
from gui.segmentation import segment_and_preprocess_canvas

def capture_screenshot():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = HandwritingAIApp()
    window.resize(1060, 800)
    window.show()

    # Add a prior single-character recognition into history to showcase history tracking
    sample_path = "test_samples/sample_7_7.png"
    if os.path.exists(sample_path):
        sample_img = Image.open(sample_path).convert('L').resize((28, 28))
        window.add_history_entry("7", "Digit (0–9)", 99.65, 0.72, sample_img)

    # Draw natural multi-character string "ABC" onto canvas
    canvas_pil = Image.new('RGB', (360, 360), (255, 255, 255))
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except:
        font = ImageFont.load_default()

    text = "ABC"
    char_widths = []
    char_heights = []
    for ch in text:
        bb = draw.textbbox((0, 0), ch, font=font)
        char_widths.append(bb[2] - bb[0])
        char_heights.append(bb[3] - bb[1])

    gap = 20
    total_w = sum(char_widths) + gap * (len(text) - 1)
    start_x = max(20, (360 - total_w) // 2)
    start_y = (360 - max(char_heights)) // 2

    curr_x = start_x
    for i, ch in enumerate(text):
        draw.text((curr_x, start_y), ch, fill=(45, 40, 35), font=font)
        curr_x += char_widths[i] + gap

    # Paint onto canvas QImage
    qimg_sample = QImage(canvas_pil.tobytes(), 360, 360, 360 * 3, QImage.Format.Format_RGB888)
    painter = QPainter(window.canvas.image)
    painter.drawImage(0, 0, qimg_sample)
    painter.end()
    window.canvas.has_strokes = True
    window.canvas.update()

    # Run segmentation & sequence prediction
    segments, annotated_img, err = segment_and_preprocess_canvas(canvas_pil)
    if not err and segments:
        result = window.model_manager.predict_sequence(segments)
        result["annotated_img"] = annotated_img
        window.on_sequence_success(result)

    # Allow Qt layout pass
    app.processEvents()

    # Grab window pixmap
    pixmap = window.grab()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "desktop_gui_screenshot.png")
    pixmap.save(out_path, "PNG")

    # Also save to conversation artifacts directory for user viewing
    artifact_path = r"C:\Users\GAGAN RAO K\.gemini\antigravity\brain\c29de611-6852-4cb6-b1e6-57989b87cd8b\desktop_gui_screenshot.png"
    pixmap.save(artifact_path, "PNG")
    print(f"Redesigned screenshot saved to: {out_path} and {artifact_path}")

if __name__ == '__main__':
    capture_screenshot()

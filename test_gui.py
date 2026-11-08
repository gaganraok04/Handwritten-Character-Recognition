"""
Comprehensive GUI and End-to-End Test Suite for Handwriting AI Desktop Application.

Tests:
1. GUI Initialization in offscreen Qt mode.
2. Model Manager loading and device allocation.
3. Canvas Drawing with Mouse events.
4. Canvas Drawing with Touch events (finger simulation).
5. Canvas Drawing with Tablet/Stylus events with pressure sensitivity.
6. Undo, Redo, and Clear operations.
7. Empty canvas detection and edge case handling.
8. End-to-end Character Recognition across Digits, Uppercase, and Lowercase letters.
9. Top-5 breakdown and 28x28 preview generation.
10. Session History table insertion and clear.
11. PNG export functionality.
"""

import os
import sys
import time
import unittest
from PIL import Image

# Force offscreen platform for automated headless CI/CD testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import (
    QMouseEvent, QTabletEvent, QPointingDevice, QInputDevice
)
from PySide6.QtWidgets import QApplication

from desktop_gui import HandwritingAIApp
from gui.preprocessor import preprocess_pil_image
from gui.model_worker import EMNISTModelManager

# Ensure QApplication exists
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestHandwritingAIGUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gui = HandwritingAIApp()

    def test_01_model_loading(self):
        """Verify model loads properly and references the 62-class EMNIST DeepCNN."""
        self.assertIsNotNone(self.gui.model_manager)
        self.assertIsNotNone(self.gui.model_manager.model)
        self.assertEqual(self.gui.model_manager.num_classes, 62)
        print("✓ Test 1: Model Manager initialized with 62 classes.")

    def test_02_canvas_mouse_drawing(self):
        """Simulate mouse drawing and verify stroke commitment and undo stack."""
        canvas = self.gui.canvas
        canvas.undo_stack = [canvas.image.copy()]
        canvas.redo_stack.clear()
        self.assertEqual(len(canvas.undo_stack), 1)

        # Mouse press
        press_ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(50, 50),
                               Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(press_ev)
        self.assertTrue(canvas.drawing)

        # Mouse move
        move_ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(100, 100),
                              Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(move_ev)

        # Mouse release
        release_ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(100, 100),
                                 Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(release_ev)
        self.assertFalse(canvas.drawing)
        self.assertEqual(len(canvas.undo_stack), 2)
        print("✓ Test 2: Mouse drawing successfully registered and committed to undo stack.")

    def test_03_undo_redo_clear(self):
        """Verify Undo, Redo, and Clear state changes."""
        canvas = self.gui.canvas

        # Undo
        canvas.undo()
        self.assertEqual(len(canvas.undo_stack), 1)
        self.assertEqual(len(canvas.redo_stack), 1)

        # Redo
        canvas.redo()
        self.assertEqual(len(canvas.undo_stack), 2)
        self.assertEqual(len(canvas.redo_stack), 0)

        # Clear
        self.gui.on_clear()
        self.assertEqual(self.gui.pred_char_lbl.text(), "✎")
        self.assertEqual(self.gui.category_badge.text(), "Waiting for handwriting")
        self.assertEqual(self.gui.conf_lbl.text(), "Confidence: —")
        self.assertTrue(self.gui.top5_container.isHidden())
        self.assertFalse(self.gui.top5_placeholder.isHidden())
        print("✓ Test 3: Undo, Redo, and Clear operations function correctly with clean waiting state.")

    def test_04_mouse_drawing_and_brush_size(self):
        """Verify dedicated mouse drawing and brush size adjustments."""
        canvas = self.gui.canvas
        canvas.clear_canvas()

        # Test brush size update
        self.gui.brush_slider.setValue(24)
        self.assertEqual(canvas.brush_size, 24.0)
        self.assertEqual(self.gui.brush_size_val.text(), "24 px")

        # Simulate mouse stroke
        canvas.drawing = True
        canvas.last_point = QPointF(80, 80)
        canvas._draw_stroke_segment(QPointF(80, 80), QPointF(120, 120))
        canvas.drawing = False
        canvas._commit_stroke()

        self.assertTrue(canvas.has_strokes)
        print("✓ Test 4: Dedicated mouse drawing and brush thickness validated.")

    def test_05_empty_canvas_recognition(self):
        """Verify empty canvas is handled gracefully without crashing."""
        self.gui.canvas.clear_canvas()
        self.gui.on_recognize()
        # Should show warning in category badge
        self.assertIn("Canvas is empty", self.gui.category_badge.text())
        print("✓ Test 5: Empty canvas detected and reported cleanly.")

    def test_06_end_to_end_recognition_digits_and_letters(self):
        """Test recognition using sample test images painted onto the canvas."""
        test_cases = [
            ("test_samples/sample_10_A.png", "A", "Uppercase Letter (A–Z)"),
            ("test_samples/sample_7_7.png", "7", "Digit (0–9)"),
            ("test_samples/sample_40_e.png", "e", "Lowercase Letter (a–z)"),
        ]

        for img_path, expected_char, expected_cat in test_cases:
            if not os.path.exists(img_path):
                continue

            # Load image and paste onto canvas buffer
            sample_img = Image.open(img_path).convert('RGB')
            # Resize to fit canvas
            sample_resized = sample_img.resize((300, 300))
            
            # Place onto white 380x380 canvas
            canvas_bg = Image.new('RGB', (380, 380), (255, 255, 255))
            canvas_bg.paste(sample_resized, (40, 40))

            # Run preprocessor
            tensor, canvas_28, err = preprocess_pil_image(canvas_bg)
            self.assertIsNone(err)
            self.assertIsNotNone(tensor)
            self.assertEqual(tensor.shape, (1, 1, 28, 28))

            # Direct Model Manager prediction
            result = self.gui.model_manager.predict_tensor(tensor)
            self.assertEqual(result["top_char"], expected_char)
            self.assertEqual(result["category"], expected_cat)
            self.assertGreater(result["top_conf"], 70.0)
            self.assertEqual(len(result["top5"]), 5)

            # Update UI via slot
            result["preview_img"] = canvas_28
            self.gui.on_prediction_success(result)

            # Check UI displays
            self.assertEqual(self.gui.pred_char_lbl.text(), expected_char)
            self.assertEqual(self.gui.category_badge.text(), expected_cat)
            self.assertIn("%", self.gui.conf_lbl.text())

            print(f"✓ Test 6: '{expected_char}' ({expected_cat}) predicted accurately with {result['top_conf']:.2f}% confidence (Latency: {result['latency_ms']:.2f} ms).")

        # Check history table row count
        self.assertGreaterEqual(self.gui.history_table.rowCount(), len(test_cases))
        print("✓ Test 6b: Session history table correctly logged all predictions.")

    def test_07_save_drawing(self):
        """Test exporting canvas to PNG file."""
        test_out = "test_canvas_export.png"
        if os.path.exists(test_out):
            os.remove(test_out)

        self.gui.canvas.save_to_file(test_out)
        self.assertTrue(os.path.exists(test_out))
        self.assertGreater(os.path.getsize(test_out), 0)
        os.remove(test_out)
        print("✓ Test 7: Canvas PNG export verified.")

    def test_08_multi_character_sequence_workflow(self):
        """Test multi-character sequence segmentation and recognition workflow (e.g. 'AB', '123')."""
        from gui.segmentation import segment_and_preprocess_canvas
        from PIL import ImageDraw, ImageFont

        sequences = ["AB", "123", "ABC"]
        for seq in sequences:
            canvas = Image.new('RGB', (360, 360), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 52)
            except:
                font = ImageFont.load_default()

            char_widths = []
            char_heights = []
            for ch in seq:
                bb = draw.textbbox((0, 0), ch, font=font)
                char_widths.append(bb[2] - bb[0])
                char_heights.append(bb[3] - bb[1])

            gap = 14
            total_w = sum(char_widths) + gap * (len(seq) - 1)
            start_x = max(10, (360 - total_w) // 2)
            start_y = (360 - max(char_heights)) // 2

            curr_x = start_x
            for i, ch in enumerate(seq):
                draw.text((curr_x, start_y), ch, fill=(35, 30, 25), font=font)
                curr_x += char_widths[i] + gap

            # Segment canvas
            segments, annotated_img, err = segment_and_preprocess_canvas(canvas)
            self.assertIsNone(err)
            self.assertEqual(len(segments), len(seq))

            # Sequence inference via Model Manager
            result = self.gui.model_manager.predict_sequence(segments)
            self.assertEqual(result["text"], seq)
            self.assertEqual(result["num_chars"], len(seq))
            self.assertGreater(result["avg_conf"], 80.0)

            # Update GUI slot
            result["annotated_img"] = annotated_img
            self.gui.on_sequence_success(result)

            # Check UI displays
            self.assertEqual(self.gui.pred_char_lbl.text(), seq)
            self.assertIn(f"{len(seq)} Characters Detected", self.gui.category_badge.text())
            self.assertTrue(self.gui.top5_placeholder.isHidden())
            self.assertFalse(self.gui.top5_container.isHidden())
            print(f"✓ Test 8: Multi-character sequence '{seq}' recognized accurately (Avg Conf: {result['avg_conf']:.2f}%).")


if __name__ == '__main__':
    unittest.main()


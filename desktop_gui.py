"""
Handwriting AI - Premium Desktop Application
=============================================
Warm Grey, Beige & Brown Theme (#F0EFEC, #DED8CF, #B9AA98, #897765, #39342F)
Powered by PySide6 (Qt) and PyTorch EMNIST DeepCNN (62 Classes)
"""

import os
import sys
import datetime
from PIL import Image

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage, QKeySequence, QShortcut, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFrame, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QStatusBar,
    QGraphicsDropShadowEffect
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from gui.styles import APP_STYLESHEET
from gui.canvas import DrawingCanvas
from gui.preprocessor import preprocess_pil_image
from gui.segmentation import segment_and_preprocess_canvas
from gui.model_worker import (
    EMNISTModelManager,
    InferenceWorker,
    SequenceInferenceWorker
)


def apply_subtle_shadow(widget):
    """Adds a soft, high-end desktop card drop shadow."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(12)
    shadow.setColor(QColor(57, 52, 47, 18))  # Soft warm charcoal shadow
    shadow.setOffset(0, 2)
    widget.setGraphicsEffect(shadow)


class HandwritingAIApp(QMainWindow):
    """Main Application Window for Handwriting AI Desktop GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HANDWRITING AI — Intelligent Character Recognition & Segmentation")
        self.resize(1060, 810)
        self.setMinimumSize(990, 750)

        # Initialize Model Manager
        self.model_manager = None
        self.init_model()

        # Build UI Layout
        self.init_ui()

        # Apply Global Warm Styling
        self.setStyleSheet(APP_STYLESHEET)

        # Setup Keyboard Shortcuts
        self.setup_shortcuts()

        # Initial waiting state
        self.set_waiting_state()

        # Initial Undo / Redo button states
        self.on_undo_status(False)
        self.on_redo_status(False)

    def init_model(self):
        """Loads the trained EMNIST DeepCNN model once."""
        try:
            self.model_manager = EMNISTModelManager()
            self.model_status_text = f"EMNIST DeepCNN • Ready • {self.model_manager.device.type.upper()}"
        except Exception as e:
            self.model_manager = None
            self.model_status_text = f"Model Load Error: {str(e)}"

    def init_ui(self):
        """Constructs the responsive desktop layout."""
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(22, 18, 22, 16)
        main_layout.setSpacing(14)

        # ---------------------------------------------------------
        # TOP HEADER CARD
        # ---------------------------------------------------------
        header_card = QFrame()
        header_card.setProperty("class", "CardFrame")
        apply_subtle_shadow(header_card)
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(20, 14, 20, 14)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(3)

        app_title = QLabel("HANDWRITING AI")
        app_title.setObjectName("AppTitle")

        app_subtitle = QLabel("Intelligent Character Recognition & Segmentation System Using Deep CNN and OpenCV")
        app_subtitle.setObjectName("AppSubtitle")

        title_vbox.addWidget(app_title)
        title_vbox.addWidget(app_subtitle)
        header_layout.addLayout(title_vbox)
        header_layout.addStretch()

        # Model Status Badge
        self.badge_label = QLabel(self.model_status_text)
        self.badge_label.setObjectName("ModelStatusBadge")
        header_layout.addWidget(self.badge_label)

        main_layout.addWidget(header_card)

        # ---------------------------------------------------------
        # MAIN 2-COLUMN WORKSPACE
        # ---------------------------------------------------------
        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(14)

        # Left Column: Mouse Drawing Canvas & Controls
        left_card = self.create_drawing_panel()
        apply_subtle_shadow(left_card)
        workspace_layout.addWidget(left_card, 5)

        # Right Column: Recognition Result & Probabilities
        right_card = self.create_analytics_panel()
        apply_subtle_shadow(right_card)
        workspace_layout.addWidget(right_card, 5)

        main_layout.addLayout(workspace_layout, 1)

        # ---------------------------------------------------------
        # BOTTOM COMPACT RECOGNITION HISTORY
        # ---------------------------------------------------------
        history_card = self.create_history_panel()
        apply_subtle_shadow(history_card)
        main_layout.addWidget(history_card)

        # Minimal Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Draw any letter or digit using your mouse.")

    # -------------------------------------------------------------
    # Left Column: Mouse Drawing Canvas
    # -------------------------------------------------------------
    def create_drawing_panel(self):
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # Section Header
        hdr_layout = QVBoxLayout()
        hdr_layout.setSpacing(2)
        sec_title = QLabel("Drawing Canvas")
        sec_title.setProperty("class", "CardTitle")
        instruction = QLabel("Draw characters, words (AB, HELLO), or digits (123) using your mouse.")
        instruction.setProperty("class", "CardDescription")
        hdr_layout.addWidget(sec_title)
        hdr_layout.addWidget(instruction)
        layout.addLayout(hdr_layout)

        # Drawing Canvas Box (Centered)
        canvas_center_box = QHBoxLayout()
        canvas_center_box.addStretch()

        self.canvas_container = QWidget()
        self.canvas_container.setObjectName("CanvasContainer")
        c_layout = QVBoxLayout(self.canvas_container)
        c_layout.setContentsMargins(1, 1, 1, 1)

        self.canvas = DrawingCanvas(self, canvas_size=360)
        self.canvas.can_undo_changed.connect(self.on_undo_status)
        self.canvas.can_redo_changed.connect(self.on_redo_status)
        c_layout.addWidget(self.canvas)

        canvas_center_box.addWidget(self.canvas_container)
        canvas_center_box.addStretch()
        layout.addLayout(canvas_center_box)

        # Pen Thickness Slider Row
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(2, 2, 2, 2)
        slider_lbl = QLabel("Pen Thickness:")
        slider_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #39342F;")
        slider_layout.addWidget(slider_lbl)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(6, 36)
        self.brush_slider.setValue(18)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        slider_layout.addWidget(self.brush_slider)

        self.brush_size_val = QLabel("18 px")
        self.brush_size_val.setFixedWidth(40)
        self.brush_size_val.setStyleSheet("font-size: 12px; font-weight: 700; color: #897765;")
        slider_layout.addWidget(self.brush_size_val)
        layout.addLayout(slider_layout)

        # Action Buttons Row 1: Recognize Text (Brown), Recognize Single (Taupe) & Clear (Beige)
        btn_layout_1 = QHBoxLayout()
        btn_layout_1.setSpacing(8)

        self.recognize_text_btn = QPushButton("Recognize Text")
        self.recognize_text_btn.setObjectName("PrimaryButton")
        self.recognize_text_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recognize_text_btn.clicked.connect(self.on_recognize_text)
        btn_layout_1.addWidget(self.recognize_text_btn, 2)

        self.recognize_btn = QPushButton("Recognize Single")
        self.recognize_btn.setObjectName("TaupeButton")
        self.recognize_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recognize_btn.clicked.connect(self.on_recognize)
        btn_layout_1.addWidget(self.recognize_btn, 2)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("ClearButton")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.on_clear)
        btn_layout_1.addWidget(self.clear_btn, 1)

        layout.addLayout(btn_layout_1)

        # Action Buttons Row 2: Undo, Redo (Taupe) & Save PNG (Outlined Brown)
        btn_layout_2 = QHBoxLayout()
        btn_layout_2.setSpacing(8)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setObjectName("TaupeButton")
        self.undo_btn.clicked.connect(self.canvas.undo)
        btn_layout_2.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setObjectName("TaupeButton")
        self.redo_btn.clicked.connect(self.canvas.redo)
        btn_layout_2.addWidget(self.redo_btn)

        self.save_btn = QPushButton("Save PNG")
        self.save_btn.setObjectName("OutlinedBrownButton")
        self.save_btn.clicked.connect(self.on_save_drawing)
        btn_layout_2.addWidget(self.save_btn)

        layout.addLayout(btn_layout_2)

        return card

    # -------------------------------------------------------------
    # Right Column: Recognition Result Panel
    # -------------------------------------------------------------
    def create_analytics_panel(self):
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        # Section Header
        hdr_layout = QVBoxLayout()
        hdr_layout.setSpacing(2)
        sec_title = QLabel("Recognition Result")
        sec_title.setProperty("class", "CardTitle")
        sec_desc = QLabel("Real-time DeepCNN classification output and probabilities.")
        sec_desc.setProperty("class", "CardDescription")
        hdr_layout.addWidget(sec_title)
        hdr_layout.addWidget(sec_desc)
        layout.addLayout(hdr_layout)

        # Top Result Sub-Card (Soft Beige Background)
        hero_card = QFrame()
        hero_card.setStyleSheet("background-color: #EDE8E1; border: 1px solid #DDD7CF; border-radius: 8px;")
        hero_layout = QHBoxLayout(hero_card)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(16)

        # 1. Large Character Box (Waiting State / Actual Predicted Character)
        self.pred_char_lbl = QLabel()
        self.pred_char_lbl.setObjectName("LargeCharDisplay")
        self.pred_char_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_layout.addWidget(self.pred_char_lbl)

        # 2. Main Stat Details (Category, Confidence, Latency)
        meta_vbox = QVBoxLayout()
        meta_vbox.setSpacing(3)
        meta_vbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.category_badge = QLabel("Waiting for handwriting")
        self.category_badge.setObjectName("CategoryBadge")
        meta_vbox.addWidget(self.category_badge)

        self.conf_lbl = QLabel("Confidence: —")
        self.conf_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #39342F;")
        meta_vbox.addWidget(self.conf_lbl)

        self.latency_lbl = QLabel("Inference latency: —")
        self.latency_lbl.setStyleSheet("font-size: 11px; color: #897765; font-weight: 500;")
        meta_vbox.addWidget(self.latency_lbl)

        hero_layout.addLayout(meta_vbox)
        hero_layout.addStretch()

        # 3. Preview Box (28x28 single crop or annotated sequence bounding boxes)
        preview_vbox = QVBoxLayout()
        preview_vbox.setSpacing(4)
        preview_vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preview_title = QLabel("PREVIEW")
        self.preview_title.setStyleSheet("font-size: 10px; font-weight: 700; color: #897765; letter-spacing: 0.4px;")
        preview_vbox.addWidget(self.preview_title)

        self.preview_box = QLabel()
        self.preview_box.setObjectName("ProcessedPreviewBox")
        self.preview_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_vbox.addWidget(self.preview_box)

        hero_layout.addLayout(preview_vbox)
        layout.addWidget(hero_card)

        # Probabilities & Character Breakdown Section
        layout.addSpacing(2)
        self.breakdown_title = QLabel("Top 5 Probabilities")
        self.breakdown_title.setProperty("class", "SubSectionTitle")
        layout.addWidget(self.breakdown_title)

        # Placeholder label when no predictions have been made
        self.top5_placeholder = QLabel("Draw characters and click 'Recognize Text' to view predictions.")
        self.top5_placeholder.setStyleSheet("font-size: 12px; color: #897765; font-style: italic; padding: 12px 2px;")
        layout.addWidget(self.top5_placeholder)

        # Container for actual prediction / breakdown rows (hidden before recognition)
        self.top5_container = QWidget()
        top5_layout = QVBoxLayout(self.top5_container)
        top5_layout.setContentsMargins(0, 0, 0, 0)
        top5_layout.setSpacing(5)

        self.top5_widgets = []
        for i in range(8):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setSpacing(10)
            row.setContentsMargins(0, 1, 0, 1)

            char_lbl = QLabel(f"{i+1}.  —")
            char_lbl.setFixedWidth(56)
            char_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #39342F;")

            cat_lbl = QLabel("—")
            cat_lbl.setFixedWidth(120)
            cat_lbl.setStyleSheet("font-size: 12px; color: #897765; font-weight: 500;")

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)

            pct_lbl = QLabel("0.00%")
            pct_lbl.setFixedWidth(54)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #39342F;")

            row.addWidget(char_lbl)
            row.addWidget(cat_lbl)
            row.addWidget(bar, 1)
            row.addWidget(pct_lbl)

            top5_layout.addWidget(row_widget)
            self.top5_widgets.append((char_lbl, cat_lbl, bar, pct_lbl))

        layout.addWidget(self.top5_container)

        # Clean System Information Footer
        layout.addSpacing(4)
        info_divider = QFrame()
        info_divider.setFrameShape(QFrame.Shape.HLine)
        info_divider.setStyleSheet("color: #DDD7CF; background-color: #DDD7CF; border: none; height: 1px;")
        layout.addWidget(info_divider)

        device_name = self.model_manager.device.type.upper() if self.model_manager else "CPU"
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(2, 4, 2, 2)

        meta_items = [
            ("Model", "DeepCNN (62 Classes)"),
            ("Accuracy", "83.32% Test Acc"),
            ("Device", f"PyTorch {device_name}")
        ]

        for k, v in meta_items:
            item_box = QHBoxLayout()
            item_box.setSpacing(4)
            kl = QLabel(k + ":")
            kl.setProperty("class", "MetaLabel")
            vl = QLabel(v)
            vl.setProperty("class", "MetaValue")
            item_box.addWidget(kl)
            item_box.addWidget(vl)
            footer_row.addLayout(item_box)
            if k != meta_items[-1][0]:
                sep = QLabel("•")
                sep.setStyleSheet("color: #B9AA98; font-size: 10px;")
                footer_row.addWidget(sep)

        footer_row.addStretch()
        layout.addLayout(footer_row)
        layout.addStretch()

        return card

    # -------------------------------------------------------------
    # Bottom Section: Compact Recognition History Panel
    # -------------------------------------------------------------
    def create_history_panel(self):
        card = QFrame()
        card.setProperty("class", "CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(8)

        # Header Row
        h_row = QHBoxLayout()
        h_title = QLabel("Recent Predictions")
        h_title.setProperty("class", "CardTitle")
        h_row.addWidget(h_title)
        h_row.addStretch()

        self.clear_hist_btn = QPushButton("Clear History")
        self.clear_hist_btn.setObjectName("ClearButton")
        self.clear_hist_btn.setStyleSheet("font-size: 11px; padding: 3px 10px;")
        self.clear_hist_btn.clicked.connect(self.on_clear_history)
        h_row.addWidget(self.clear_hist_btn)
        layout.addLayout(h_row)

        # Compact History Table (92px height)
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels([
            "#", "Preview", "Predicted", "Category", "Confidence", "Timestamp"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setFixedHeight(92)
        self.history_table.setIconSize(QSize(22, 22))

        layout.addWidget(self.history_table)
        return card

    # -------------------------------------------------------------
    # Waiting State Management
    # -------------------------------------------------------------
    def set_waiting_state(self):
        """Sets the result panel into a clean waiting state before input."""
        # Clean waiting-state icon in soft taupe
        self.pred_char_lbl.setText("✎")
        self.pred_char_lbl.setStyleSheet("font-size: 34px; color: #B9AA98; font-weight: 400;")

        # Badges & labels
        self.category_badge.setText("Waiting for handwriting")
        self.conf_lbl.setText("Confidence: —")
        self.latency_lbl.setText("Inference latency: —")

        # Processed preview box with subtle beige background and placeholder text
        self.preview_title.setText("PREVIEW")
        self.preview_box.clear()
        self.preview_box.setText("Preview")
        self.preview_box.setStyleSheet(
            "font-size: 11px; font-weight: 700; color: #897765; background-color: #DED8CF; border: 1px solid #DDD7CF; border-radius: 6px;"
        )

        # Hide empty prediction rows before recognition
        self.breakdown_title.setText("Top 5 Probabilities")
        self.top5_container.setVisible(False)
        self.top5_placeholder.setVisible(True)

    # -------------------------------------------------------------
    # Keyboard Shortcuts
    # -------------------------------------------------------------
    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self.on_recognize_text)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, self.on_recognize_text)
        QShortcut(QKeySequence("Shift+Return"), self, self.on_recognize)
        QShortcut(QKeySequence("Shift+Enter"), self, self.on_recognize)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.on_clear)
        QShortcut(QKeySequence("Ctrl+Delete"), self, self.on_clear)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.canvas.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self.canvas.redo)
        QShortcut(QKeySequence.StandardKey.Save, self, self.on_save_drawing)

    # -------------------------------------------------------------
    # Event Handlers & Slots
    # -------------------------------------------------------------
    def on_brush_size_changed(self, val):
        self.canvas.set_brush_size(val)
        self.brush_size_val.setText(f"{val} px")

    def on_undo_status(self, can_undo):
        self.undo_btn.setEnabled(can_undo)

    def on_redo_status(self, can_redo):
        self.redo_btn.setEnabled(can_redo)

    def on_clear(self):
        self.canvas.clear_canvas()
        self.set_waiting_state()
        self.status_bar.showMessage("Canvas cleared. Ready for next character.")

    def on_clear_history(self):
        self.history_table.setRowCount(0)
        self.status_bar.showMessage("Recognition history cleared.")

    def on_save_drawing(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Character Drawing", "handwritten_drawing.png", "PNG Images (*.png)"
        )
        if file_path:
            success = self.canvas.save_to_file(file_path)
            if success:
                self.status_bar.showMessage(f"Drawing saved to: {file_path}", 4000)
            else:
                QMessageBox.warning(self, "Save Error", "Failed to save drawing to the specified file.")

    def on_recognize(self):
        """Processes single character strokes and dispatches inference worker."""
        if self.model_manager is None:
            QMessageBox.critical(self, "Model Error", "EMNIST DeepCNN model is not loaded.")
            return

        pil_img = self.canvas.get_pil_image()
        tensor, canvas_28, err = preprocess_pil_image(pil_img)
        if err is not None:
            self.status_bar.showMessage(f"Notice: {err}", 3500)
            self.category_badge.setText(err)
            return

        self.recognize_btn.setEnabled(False)
        self.recognize_btn.setText("Recognizing...")
        self.recognize_text_btn.setEnabled(False)
        self.status_bar.showMessage("Running DeepCNN single character inference...")

        self.worker = InferenceWorker(self.model_manager, tensor, canvas_28)
        self.worker.prediction_ready.connect(self.on_prediction_success)
        self.worker.prediction_failed.connect(self.on_prediction_error)
        self.worker.start()

    def on_recognize_text(self):
        """Segments multiple characters/words from canvas and recognizes sequence in order."""
        if self.model_manager is None:
            QMessageBox.critical(self, "Model Error", "EMNIST DeepCNN model is not loaded.")
            return

        pil_img = self.canvas.get_pil_image()
        segments, annotated_img, err = segment_and_preprocess_canvas(pil_img)
        if err is not None:
            self.status_bar.showMessage(f"Notice: {err}", 3500)
            self.category_badge.setText(err)
            return

        self.recognize_text_btn.setEnabled(False)
        self.recognize_text_btn.setText("Recognizing...")
        self.recognize_btn.setEnabled(False)
        self.status_bar.showMessage(f"Segmented {len(segments)} character(s). Running DeepCNN inference...")

        self.seq_worker = SequenceInferenceWorker(self.model_manager, segments, annotated_img)
        self.seq_worker.sequence_ready.connect(self.on_sequence_success)
        self.seq_worker.sequence_failed.connect(self.on_sequence_error)
        self.seq_worker.start()

    def on_prediction_success(self, result):
        """Updates UI components with single character inference results."""
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("Recognize Single")
        self.recognize_text_btn.setEnabled(True)
        self.recognize_text_btn.setText("Recognize Text")

        top_char = result["top_char"]
        top_conf = result["top_conf"]
        category = result["category"]
        top5 = result["top5"]
        latency = result["latency_ms"]
        preview_28 = result["preview_img"]

        # 1. Hero Card Display (Single Character)
        self.pred_char_lbl.setText(top_char)
        self.pred_char_lbl.setStyleSheet("font-size: 58px; font-weight: 800; color: #39342F;")
        self.category_badge.setText(category)
        self.conf_lbl.setText(f"Confidence: {top_conf:.2f}%")
        self.latency_lbl.setText(f"Inference latency: {latency:.2f} ms ({result['device']})")

        # 2. Show and Populate Top 5 Candidates
        self.breakdown_title.setText("Top 5 Probabilities")
        self.top5_placeholder.setVisible(False)
        self.top5_container.setVisible(True)

        for i in range(len(self.top5_widgets)):
            lbl_char, lbl_cat, bar, lbl_pct = self.top5_widgets[i]
            row_w = lbl_char.parentWidget()
            if i < len(top5):
                c, cat, conf = top5[i]
                lbl_char.setText(f"{i+1}.  '{c}'")
                lbl_cat.setText(cat)
                bar.setValue(int(conf * 10))
                lbl_pct.setText(f"{conf:.2f}%")
                row_w.setVisible(True)
            else:
                row_w.setVisible(False)

        # 3. 28x28 Processed Preview Box
        self.preview_title.setText("PROCESSED (28×28)")
        if preview_28 is not None:
            self.preview_box.setText("")
            img_rgba = preview_28.convert('RGBA')
            qimg = QImage(img_rgba.tobytes(), 28, 28, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            self.preview_box.setPixmap(pix)
            self.preview_box.setStyleSheet("background-color: #1E1B18; border: 2px solid #DED8CF; border-radius: 6px;")

        # 4. Add Entry to Session History
        self.add_history_entry(top_char, category, top_conf, latency, preview_28)
        self.status_bar.showMessage(f"Predicted '{top_char}' with {top_conf:.2f}% confidence in {latency:.2f} ms.")

    def on_prediction_error(self, err_msg):
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("Recognize Single")
        self.recognize_text_btn.setEnabled(True)
        self.recognize_text_btn.setText("Recognize Text")
        QMessageBox.critical(self, "Inference Error", f"Model prediction failed:\n{err_msg}")
        self.status_bar.showMessage("Inference failed.")

    def on_sequence_success(self, result):
        """Updates UI components with multi-character sequence inference results."""
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("Recognize Single")
        self.recognize_text_btn.setEnabled(True)
        self.recognize_text_btn.setText("Recognize Text")

        text = result["text"]
        num_chars = result["num_chars"]
        avg_conf = result["avg_conf"]
        characters = result["characters"]
        latency = result["total_latency_ms"]
        annotated_img = result.get("annotated_img")

        # 1. Hero Card Display (Adjust font size dynamically for sequence length)
        self.pred_char_lbl.setText(text)
        if len(text) <= 2:
            self.pred_char_lbl.setStyleSheet("font-size: 52px; font-weight: 800; color: #39342F;")
        elif len(text) <= 4:
            self.pred_char_lbl.setStyleSheet("font-size: 38px; font-weight: 800; color: #39342F;")
        else:
            self.pred_char_lbl.setStyleSheet("font-size: 28px; font-weight: 800; color: #39342F;")

        if num_chars == 1:
            self.category_badge.setText(f"{characters[0]['category']}")
            self.conf_lbl.setText(f"Confidence: {avg_conf:.2f}%")
        else:
            self.category_badge.setText(f"Sequence • {num_chars} Characters Detected")
            self.conf_lbl.setText(f"Avg Confidence: {avg_conf:.2f}%")

        self.latency_lbl.setText(f"Total latency: {latency:.2f} ms ({result['device']}) • {num_chars} char{'s' if num_chars != 1 else ''}")

        # 2. Show Preview (Annotated segmentation preview with numbered boxes or 28x28)
        if num_chars == 1 and characters and characters[0].get("preview_img") is not None:
            self.preview_title.setText("PROCESSED (28×28)")
            preview_28 = characters[0]["preview_img"]
            img_rgba = preview_28.convert('RGBA')
            qimg = QImage(img_rgba.tobytes(), 28, 28, QImage.Format.Format_RGBA8888)
            pix = QPixmap.fromImage(qimg).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            self.preview_box.setPixmap(pix)
            self.preview_box.setStyleSheet("background-color: #1E1B18; border: 2px solid #DED8CF; border-radius: 6px;")
        elif annotated_img is not None:
            self.preview_title.setText(f"SEGMENTS ({num_chars})")
            rgb_img = annotated_img.convert('RGB')
            w, h = rgb_img.size
            qimg = QImage(rgb_img.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(74, 74, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_box.setPixmap(pix)
            self.preview_box.setStyleSheet("background-color: #FFFFFF; border: 2px solid #DED8CF; border-radius: 6px;")

        # 3. Breakdown / Candidates Section
        self.top5_placeholder.setVisible(False)
        self.top5_container.setVisible(True)

        if num_chars == 1 and characters:
            # Single character inside sequence: show its Top 5
            self.breakdown_title.setText("Top 5 Probabilities")
            top5 = characters[0]["top5"]
            for i in range(len(self.top5_widgets)):
                lbl_char, lbl_cat, bar, lbl_pct = self.top5_widgets[i]
                row_w = lbl_char.parentWidget()
                if i < len(top5):
                    c, cat, conf = top5[i]
                    lbl_char.setText(f"{i+1}.  '{c}'")
                    lbl_cat.setText(cat)
                    bar.setValue(int(conf * 10))
                    lbl_pct.setText(f"{conf:.2f}%")
                    row_w.setVisible(True)
                else:
                    row_w.setVisible(False)
        else:
            # Multi-character: show breakdown of each detected character in order
            self.breakdown_title.setText(f"Detected Characters Breakdown ({num_chars} Characters)")
            for i in range(len(self.top5_widgets)):
                lbl_char, lbl_cat, bar, lbl_pct = self.top5_widgets[i]
                row_w = lbl_char.parentWidget()
                if i < len(characters):
                    c = characters[i]
                    lbl_char.setText(f"#{c['order']}. '{c['char']}'")
                    lbl_cat.setText(c['category'])
                    bar.setValue(int(c['confidence'] * 10))
                    lbl_pct.setText(f"{c['confidence']:.2f}%")
                    row_w.setVisible(True)
                else:
                    row_w.setVisible(False)

        # 4. Add Entry to Session History
        hist_cat = f"Sequence ({num_chars} chars)" if num_chars > 1 else characters[0]["category"]
        hist_preview = characters[0].get("preview_img") if characters else None
        self.add_history_entry(text, hist_cat, avg_conf, latency, hist_preview)

        self.status_bar.showMessage(
            f"Recognized '{text}' ({num_chars} character{'s' if num_chars != 1 else ''}) with {avg_conf:.2f}% avg confidence in {latency:.2f} ms."
        )

    def on_sequence_error(self, err_msg):
        self.recognize_btn.setEnabled(True)
        self.recognize_btn.setText("Recognize Single")
        self.recognize_text_btn.setEnabled(True)
        self.recognize_text_btn.setText("Recognize Text")
        QMessageBox.critical(self, "Sequence Inference Error", f"Model prediction failed:\n{err_msg}")
        self.status_bar.showMessage("Sequence inference failed.")

    def add_history_entry(self, char, category, confidence, latency, preview_pil):
        """Appends a new record to the session history table."""
        row_idx = self.history_table.rowCount()
        self.history_table.insertRow(row_idx)

        # Index
        item_idx = QTableWidgetItem(str(row_idx + 1))
        item_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_table.setItem(row_idx, 0, item_idx)

        # Thumbnail
        if preview_pil is not None:
            img_rgba = preview_pil.convert('RGBA')
            qimg = QImage(img_rgba.tobytes(), 28, 28, QImage.Format.Format_RGBA8888)
            icon = QIcon(QPixmap.fromImage(qimg))
            thumb_item = QTableWidgetItem(icon, "")
            thumb_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row_idx, 1, thumb_item)

        # Predicted Character
        item_char = QTableWidgetItem(f"'{char}'")
        item_char.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_table.setItem(row_idx, 2, item_char)

        # Category
        item_cat = QTableWidgetItem(category)
        self.history_table.setItem(row_idx, 3, item_cat)

        # Confidence
        item_conf = QTableWidgetItem(f"{confidence:.2f}%")
        item_conf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_table.setItem(row_idx, 4, item_conf)

        # Timestamp
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        item_time = QTableWidgetItem(time_str)
        item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_table.setItem(row_idx, 5, item_time)

        self.history_table.scrollToBottom()


def main():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("Handwriting AI")
    app.setStyle("Fusion")

    window = HandwritingAIApp()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

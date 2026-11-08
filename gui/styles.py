"""
Premium Warm Grey, Beige & Brown Theme for Handwriting AI Desktop Application.

Exact Palette Specifications:
- Main Background:    #F0EFEC
- Content Cards:      #FFFFFF
- Secondary Surfaces: #DED8CF
- Taupe:              #B9AA98
- Primary Brown:      #897765 (Hover: #736353, Pressed: #615243)
- Dark Text:          #39342F
- Subtle Borders:     #DDD7CF
"""

APP_STYLESHEET = """
/* Global Window & Fonts */
QMainWindow, QWidget#CentralWidget {
    background-color: #F0EFEC;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    color: #39342F;
}

/* Premium Card Containers */
QFrame.CardFrame {
    background-color: #FFFFFF;
    border: 1px solid #DDD7CF;
    border-radius: 9px;
}

/* Header Section */
QLabel#AppTitle {
    font-size: 19px;
    font-weight: 700;
    color: #39342F;
    letter-spacing: -0.3px;
}

QLabel#AppSubtitle {
    font-size: 12px;
    font-weight: 500;
    color: #897765;
}

/* Model Status Badge (Soft Beige Pill) */
QLabel#ModelStatusBadge {
    background-color: #DED8CF;
    color: #39342F;
    border: 1px solid #DDD7CF;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
}

/* Section Headings */
QLabel.CardTitle {
    font-size: 14px;
    font-weight: 700;
    color: #39342F;
    letter-spacing: -0.2px;
}

QLabel.CardDescription {
    font-size: 11px;
    font-weight: 400;
    color: #897765;
}

QLabel.SubSectionTitle {
    font-size: 12px;
    font-weight: 700;
    color: #39342F;
    letter-spacing: -0.1px;
}

/* Canvas Container with subtle warm-grey border */
QWidget#CanvasContainer {
    background-color: #FFFFFF;
    border: 1px solid #DDD7CF;
    border-radius: 7px;
}

/* Base Push Button */
QPushButton {
    font-size: 12px;
    font-weight: 600;
    border-radius: 6px;
    padding: 7px 14px;
    border: 1px solid #DDD7CF;
    background-color: #FFFFFF;
    color: #39342F;
}

QPushButton:hover {
    background-color: #EDE8E1;
    border-color: #B9AA98;
}

QPushButton:pressed {
    background-color: #DED8CF;
}

QPushButton:disabled {
    background-color: #F0EFEC;
    color: #B9AA98;
    border-color: #DDD7CF;
}

/* 1. Recognize Button (Muted Brown Background with White Text) */
QPushButton#PrimaryButton {
    background-color: #897765;
    color: #FFFFFF;
    border: 1px solid #736353;
    font-size: 13px;
    font-weight: 700;
    padding: 8px 18px;
}

QPushButton#PrimaryButton:hover {
    background-color: #736353;
    border-color: #615243;
}

QPushButton#PrimaryButton:pressed {
    background-color: #615243;
}

/* 2. Clear Button (Light Beige Background) */
QPushButton#ClearButton {
    background-color: #DED8CF;
    color: #39342F;
    border: 1px solid #DDD7CF;
    font-weight: 600;
}

QPushButton#ClearButton:hover {
    background-color: #D2CBC0;
    border-color: #B9AA98;
}

QPushButton#ClearButton:pressed {
    background-color: #C5BCB0;
}

/* 3. Undo & Redo Buttons (Subtle Taupe Styling) */
QPushButton#TaupeButton {
    background-color: #EDE8E1;
    color: #39342F;
    border: 1px solid #B9AA98;
}

QPushButton#TaupeButton:hover {
    background-color: #DED8CF;
    border-color: #897765;
}

/* 4. Save PNG Button (Outlined Brown Button) */
QPushButton#OutlinedBrownButton {
    background-color: #FFFFFF;
    color: #897765;
    border: 1px solid #897765;
    font-weight: 600;
}

QPushButton#OutlinedBrownButton:hover {
    background-color: #EDE8E1;
    border-color: #736353;
    color: #736353;
}

/* Pen Thickness Slider */
QSlider::groove:horizontal {
    height: 4px;
    background: #DDD7CF;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #897765;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 2px solid #897765;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #DED8CF;
    border-color: #736353;
}

/* Large Predicted Character / Sequence Display Box */
QLabel#LargeCharDisplay {
    background-color: #FFFFFF;
    border: 1px solid #DDD7CF;
    border-radius: 8px;
    min-width: 86px;
    max-width: 220px;
    min-height: 86px;
    max-height: 86px;
    padding: 0 10px;
}

/* Category Badge (Light Beige) */
QLabel#CategoryBadge {
    background-color: #DED8CF;
    color: #39342F;
    border: 1px solid #DDD7CF;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
}

/* Confidence Value Display */
QLabel#ConfidenceValue {
    font-size: 17px;
    font-weight: 800;
    color: #39342F;
    letter-spacing: -0.3px;
}

/* Processed Preview Box (28x28 or Annotated Segmentation Preview) */
QLabel#ProcessedPreviewBox {
    background-color: #DED8CF;
    border: 1px solid #DDD7CF;
    border-radius: 6px;
    min-width: 76px;
    max-width: 86px;
    min-height: 76px;
    max-height: 86px;
}

/* Taupe-Brown Probability Progress Bars */
QProgressBar {
    background-color: #EDE8E1;
    border: 1px solid #DDD7CF;
    border-radius: 3px;
    min-height: 8px;
    max-height: 8px;
    text-align: right;
}

QProgressBar::chunk {
    background-color: #B9AA98;
    border-radius: 2px;
}

/* Metadata Labels */
QLabel.MetaLabel {
    font-size: 11px;
    font-weight: 600;
    color: #897765;
}

QLabel.MetaValue {
    font-size: 11px;
    font-weight: 600;
    color: #39342F;
}

/* Clean Compact History Table */
QTableWidget {
    background-color: #FFFFFF;
    color: #39342F;
    border: 1px solid #DDD7CF;
    border-radius: 6px;
    gridline-color: #EDE8E1;
    font-size: 11px;
    selection-background-color: #EDE8E1;
    selection-color: #39342F;
}

QTableWidget::item {
    color: #39342F;
    padding: 4px 6px;
    border-bottom: 1px solid #EDE8E1;
}

QHeaderView::section {
    background-color: #EDE8E1;
    color: #897765;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    border: none;
    border-bottom: 1px solid #DDD7CF;
    padding: 5px 6px;
}

/* Minimal Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #F0EFEC;
    width: 6px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #B9AA98;
    min-height: 20px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #897765;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Bottom Status Bar */
QStatusBar {
    background-color: #FFFFFF;
    color: #897765;
    font-size: 11px;
    border-top: 1px solid #DDD7CF;
    padding: 2px 8px;
}
"""

"""
Mouse-Driven Drawing Canvas for Handwritten Character Recognition.

Features:
- Pure mouse input: Left-click to start, drag to draw, release to finish.
- Antialiased stroke rendering with rounded caps and joins.
- Centered empty-state instruction when canvas is clear.
- Unlimited undo and redo stack management.
- Pure in-memory conversion to PIL.Image for inference.
"""

from PIL import Image
from PySide6.QtCore import Qt, QPointF, Signal, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QImage, QFont
from PySide6.QtWidgets import QWidget


class DrawingCanvas(QWidget):
    """Clean, dedicated mouse-drawing canvas with empty-state guide."""
    stroke_drawn = Signal()
    can_undo_changed = Signal(bool)
    can_redo_changed = Signal(bool)

    def __init__(self, parent=None, canvas_size=360):
        super().__init__(parent)
        self.canvas_size = canvas_size
        self.setFixedSize(canvas_size, canvas_size)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Theme Colors
        self.pen_color = QColor("#39342F")  # Dark charcoal-warm ink
        self.bg_color = QColor("#FFFFFF")   # Crisp white canvas
        self.hint_color = QColor("#B9AA98") # Elegant taupe empty-state hint
        self.brush_size = 18.0              # Default stroke thickness

        # Canvas buffer (QImage)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(self.bg_color)

        # Undo / Redo Stacks (stores QImage copies)
        self.undo_stack = [self.image.copy()]
        self.redo_stack = []
        self.max_history = 40

        # State tracking
        self.drawing = False
        self.last_point = QPointF()
        self.has_strokes = False

    def set_brush_size(self, size):
        """Update brush thickness."""
        self.brush_size = max(4.0, float(size))

    def clear_canvas(self):
        """Resets the canvas to blank white and updates undo/redo state."""
        self.image.fill(self.bg_color)
        self.redo_stack.clear()
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        self.has_strokes = False
        self.update()
        self.can_undo_changed.emit(len(self.undo_stack) > 1)
        self.can_redo_changed.emit(False)
        self.stroke_drawn.emit()

    def undo(self):
        """Reverts the last drawn stroke."""
        if len(self.undo_stack) > 1:
            current = self.undo_stack.pop()
            self.redo_stack.append(current)
            self.image = self.undo_stack[-1].copy()
            # Check if previous image still has strokes
            self.has_strokes = len(self.undo_stack) > 1
            self.update()
            self.can_undo_changed.emit(len(self.undo_stack) > 1)
            self.can_redo_changed.emit(len(self.redo_stack) > 0)
            self.stroke_drawn.emit()

    def redo(self):
        """Restores an undone stroke."""
        if self.redo_stack:
            restored = self.redo_stack.pop()
            self.undo_stack.append(restored)
            self.image = restored.copy()
            self.has_strokes = True
            self.update()
            self.can_undo_changed.emit(len(self.undo_stack) > 1)
            self.can_redo_changed.emit(len(self.redo_stack) > 0)
            self.stroke_drawn.emit()

    # -------------------------------------------------------------
    # Drawing Primitives
    # -------------------------------------------------------------
    def _draw_stroke_segment(self, start_pt, end_pt):
        """Draws an antialiased line segment between two points on the buffer."""
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(self.pen_color, self.brush_size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(start_pt, end_pt)
        painter.end()

        self.has_strokes = True
        self.update()

    def _commit_stroke(self):
        """Saves current state to undo stack when a stroke finishes."""
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.can_undo_changed.emit(True)
        self.can_redo_changed.emit(False)
        self.stroke_drawn.emit()

    # -------------------------------------------------------------
    # Paint Event with Empty-State Hint
    # -------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

        # Draw centered empty-state instruction if canvas has no strokes
        if not self.has_strokes:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(self.hint_color)
            font = QFont("Segoe UI", 12)
            font.setStyle(QFont.Style.StyleItalic)
            painter.setFont(font)
            rect = QRect(0, 0, self.width(), self.height())
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Draw a letter or digit here")

    # -------------------------------------------------------------
    # Mouse-Only Input Handling
    # -------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position()
            self._draw_stroke_segment(self.last_point, self.last_point)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            curr_point = event.position()
            self._draw_stroke_segment(self.last_point, curr_point)
            self.last_point = curr_point

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self._commit_stroke()

    # -------------------------------------------------------------
    # Image Export
    # -------------------------------------------------------------
    def get_pil_image(self):
        """Converts internal buffer to PIL Image."""
        buffer = self.image.convertToFormat(QImage.Format.Format_RGBA8888)
        width = buffer.width()
        height = buffer.height()
        ptr = buffer.bits()
        img = Image.frombuffer('RGBA', (width, height), ptr, 'raw', 'RGBA', 0, 1)
        return img.convert('RGB')

    def save_to_file(self, file_path):
        """Exports canvas to disk as PNG."""
        return self.image.save(file_path, "PNG")

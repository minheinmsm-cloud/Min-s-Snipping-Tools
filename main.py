import math
import sys

from PySide6.QtCore import Qt, QRect, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import (QGuiApplication, QPixmap, QPainter, QColor, QPen,
                           QPainterPath, QPolygonF, QPolygon, QKeySequence,
                           QShortcut, QFont, QFontMetrics)
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QVBoxLayout,
                               QHBoxLayout, QToolButton, QMenu, QScrollArea,
                               QLabel, QFileDialog, QColorDialog, QComboBox,
                               QButtonGroup, QFrame, QMessageBox, QLineEdit)

STYLE = """
QMainWindow, QWidget#root { background: #f3f3f3; font-family: 'Segoe UI'; }
QToolButton { border: 1px solid transparent; border-radius: 5px;
              padding: 5px 9px; font-size: 13px; background: transparent; }
QToolButton:hover { background: #e5e5e5; }
QToolButton:checked { background: #dbe9f7; border: 1px solid #2a76c6; }
QToolButton:disabled { color: #aaa; }
QToolButton::menu-indicator { width: 8px; }
QComboBox { padding: 3px 6px; border: 1px solid #ccc; border-radius: 4px; background: white; }
QFrame#sep { background: #d0d0d0; }
QScrollArea { background: white; border: none; }
"""


# ---------------------------------------------------------------- capture
def grab_all():
    """Grab every monitor into one pixmap. Returns (pixmap, virtual_geometry)."""
    screens = QGuiApplication.screens()
    vgeo = screens[0].geometry()
    for s in screens[1:]:
        vgeo = vgeo.united(s.geometry())
    sc = max(s.devicePixelRatio() for s in screens)
    pm = QPixmap(round(vgeo.width() * sc), round(vgeo.height() * sc))
    pm.fill(Qt.black)
    p = QPainter(pm)
    for s in screens:
        g = s.geometry()
        target = QRect(round((g.x() - vgeo.x()) * sc), round((g.y() - vgeo.y()) * sc),
                       round(g.width() * sc), round(g.height() * sc))
        p.drawPixmap(target, s.grabWindow(0))
    p.end()
    return pm, vgeo


class SnipOverlay(QWidget):
    done = Signal(object)  # QPixmap or None

    def __init__(self, shot, vgeo, mode):
        super().__init__()
        self.shot, self.mode = shot, mode
        self.start = self.end = None
        self.points = []
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.CrossCursor)
        self.setGeometry(vgeo)
        self.show()
        self.activateWindow()
        self.setFocus()

    def ratio(self):
        return self.shot.width() / self.width()

    def sel_rect(self):
        if not self.start or not self.end:
            return QRect()
        return QRect(self.start, self.end).normalized()

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawPixmap(self.rect(), self.shot)
        dim = QColor(0, 0, 0, 110)
        if self.mode == "freeform" and len(self.points) > 1:
            path = QPainterPath()
            path.addPolygon(QPolygonF([QPointF(x) for x in self.points]))
            full = QPainterPath()
            full.addRect(QRectF(self.rect()))
            p.fillPath(full.subtracted(path), dim)
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawPath(path)
        elif self.start and self.end:
            r = self.sel_rect()
            full = QPainterPath()
            full.addRect(QRectF(self.rect()))
            hole = QPainterPath()
            hole.addRect(QRectF(r))
            p.fillPath(full.subtracted(hole), dim)
            p.setPen(QPen(QColor("#ffffff"), 2))
            p.drawRect(r)
        else:
            p.fillRect(self.rect(), dim)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.start = self.end = e.position().toPoint()
            self.points = [self.start]

    def mouseMoveEvent(self, e):
        if self.start:
            self.end = e.position().toPoint()
            if self.mode == "freeform":
                self.points.append(self.end)
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or not self.start:
            return
        k = self.ratio()
        if self.mode == "freeform" and len(self.points) > 2:
            poly = QPolygonF([QPointF(pt.x() * k, pt.y() * k) for pt in self.points])
            bounds = poly.boundingRect().toRect().intersected(self.shot.rect())
            out = QPixmap(bounds.size())
            out.fill(Qt.transparent)
            path = QPainterPath()
            path.addPolygon(poly.translated(-bounds.x(), -bounds.y()))
            p = QPainter(out)
            p.setRenderHint(QPainter.Antialiasing)
            p.setClipPath(path)
            p.drawPixmap(0, 0, self.shot, bounds.x(), bounds.y(), bounds.width(), bounds.height())
            p.end()
            self.finish(out)
        else:
            r = self.sel_rect()
            if r.width() < 3 or r.height() < 3:
                self.finish(None)
                return
            src = QRect(round(r.x() * k), round(r.y() * k), round(r.width() * k), round(r.height() * k))
            self.finish(self.shot.copy(src))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.finish(None)

    def finish(self, pm):
        self.hide()
        self.done.emit(pm)
        self.close()


# ----------------------------------------------------------------- canvas
class TextEdit(QLineEdit):
    """Inline text box. Enter / click away = place text, Esc = cancel."""
    def __init__(self, parent):
        super().__init__(parent)
        self.cancelled = False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancelled = True
            self.clearFocus()
        else:
            super().keyPressEvent(e)


class Canvas(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.pix = QPixmap()
        self.base = QPixmap()
        self.undo_stack, self.redo_stack = [], []
        self.tool = "pen"
        self.color = QColor("#e81123")
        self.width = 4
        self.a = self.b = None
        self.path = None
        self.editor = None
        self.setMouseTracking(True)

    def has_image(self):
        return not self.pix.isNull()

    def set_tool(self, tool):
        if self.editor:
            self.editor.clearFocus()  # commits pending text
        self.tool = tool

    # --- text tool
    def _font(self):
        f = QFont("Segoe UI")
        f.setPixelSize(12 + self.width * 3)
        return f

    def start_text(self, pt):
        if self.editor:
            self.editor.clearFocus()  # commit previous box first
        ed = TextEdit(self)
        font = self._font()
        ed.setFont(font)
        ed.setTextMargins(0, 0, 0, 0)
        ed.setStyleSheet(
            f"QLineEdit {{ background: transparent; color: {self.color.name()};"
            f" border: 1px dashed #2a76c6; padding: 0px; margin: 0px; }}")
        h = QFontMetrics(font).height() + 6
        ed.setGeometry(int(pt.x()), int(pt.y() - h / 2), 140, h)
        ed.textChanged.connect(lambda t, e=ed: e.setFixedWidth(
            max(140, QFontMetrics(e.font()).horizontalAdvance(t) + 24)))
        ed.editingFinished.connect(self.commit_text)
        self.editor = ed
        ed.show()
        ed.setFocus()

    def commit_text(self):
        ed, self.editor = self.editor, None
        if ed is None:
            return
        text = ed.text().strip()
        if text and not ed.cancelled:
            self._push()
            p = QPainter(self.pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.TextAntialiasing)
            p.setFont(ed.font())
            p.setPen(self.color)
            p.drawText(QRect(ed.x() + 2, ed.y(), ed.width() + 400, ed.height()),
                       Qt.AlignLeft | Qt.AlignVCenter, text)
            p.end()
            self.update()
        ed.deleteLater()

    def set_image(self, pm):
        self.pix, self.base = pm.copy(), pm.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.setFixedSize(pm.size())
        self.update()
        self.changed.emit()

    # --- undo / redo
    def _push(self):
        self.undo_stack.append((self.pix.copy(), self.base.copy()))
        self.redo_stack.clear()
        self.changed.emit()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append((self.pix.copy(), self.base.copy()))
            self.pix, self.base = self.undo_stack.pop()
            self.setFixedSize(self.pix.size())
            self.update()
            self.changed.emit()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append((self.pix.copy(), self.base.copy()))
            self.pix, self.base = self.redo_stack.pop()
            self.setFixedSize(self.pix.size())
            self.update()
            self.changed.emit()

    # --- drawing helpers
    def _pen(self):
        if self.tool == "highlighter":
            c = QColor(255, 235, 0, 110)
            return QPen(c, self.width * 3 + 6, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin)
        return QPen(self.color, self.width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _draw_shape(self, p, a, b):
        p.setPen(self._pen())
        p.setBrush(Qt.NoBrush)
        if self.tool == "rect":
            p.drawRect(QRectF(a, b).normalized())
        elif self.tool == "ellipse":
            p.drawEllipse(QRectF(a, b).normalized())
        elif self.tool == "line":
            p.drawLine(a, b)
        elif self.tool == "arrow":
            p.drawLine(a, b)
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            size = 10 + self.width * 2
            pts = [b]
            for d in (-0.5, 0.5):
                pts.append(QPointF(b.x() - math.cos(ang + d) * size, b.y() - math.sin(ang + d) * size))
            p.setBrush(self.color)
            p.drawPolygon(QPolygonF(pts))

    def paintEvent(self, _):
        p = QPainter(self)
        if self.pix.isNull():
            return
        p.drawPixmap(0, 0, self.pix)
        p.setRenderHint(QPainter.Antialiasing)
        if self.a and self.b:
            if self.tool in ("rect", "ellipse", "line", "arrow"):
                self._draw_shape(p, self.a, self.b)
            elif self.tool == "crop":
                p.setPen(QPen(Qt.white, 1, Qt.DashLine))
                p.setBrush(QColor(0, 120, 215, 50))
                p.drawRect(QRectF(self.a, self.b).normalized())
        if self.path and self.tool in ("pen", "highlighter"):
            p.setPen(self._pen())
            p.setBrush(Qt.NoBrush)
            p.drawPath(self.path)

    def mousePressEvent(self, e):
        if not self.has_image() or e.button() != Qt.LeftButton:
            return
        pt = e.position()
        if self.tool == "text":
            self.start_text(pt)
            return
        self.a = self.b = pt
        if self.tool in ("pen", "highlighter"):
            self.path = QPainterPath(pt)
            self.path.lineTo(pt + QPointF(0.01, 0))
        elif self.tool == "eraser":
            self._push()
            self._erase(pt)
        self.update()

    def mouseMoveEvent(self, e):
        if not self.a:
            return
        pt = e.position()
        self.b = pt
        if self.tool in ("pen", "highlighter") and self.path:
            self.path.lineTo(pt)
        elif self.tool == "eraser":
            self._erase(pt)
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.a or e.button() != Qt.LeftButton:
            return
        if self.tool in ("pen", "highlighter") and self.path:
            self._push()
            p = QPainter(self.pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(self._pen())
            p.drawPath(self.path)
            p.end()
        elif self.tool in ("rect", "ellipse", "line", "arrow"):
            if (self.b - self.a).manhattanLength() > 2:
                self._push()
                p = QPainter(self.pix)
                p.setRenderHint(QPainter.Antialiasing)
                self._draw_shape(p, self.a, self.b)
                p.end()
        elif self.tool == "crop":
            r = QRectF(self.a, self.b).normalized().toRect().intersected(self.pix.rect())
            if r.width() > 5 and r.height() > 5:
                self._push()
                self.pix, self.base = self.pix.copy(r), self.base.copy(r)
                self.setFixedSize(self.pix.size())
        self.a = self.b = self.path = None
        self.update()

    def _erase(self, pt):
        """Erase = paint the original screenshot back over annotations."""
        p = QPainter(self.pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self.base)
        r = 8 + self.width * 2
        p.drawEllipse(pt, r, r)
        p.end()


# ------------------------------------------------------------ main window
class SnipWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snipping Tool")
        self.resize(900, 600)
        self.mode, self.delay = "rect", 0
        self.overlay = None

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(8, 6, 8, 0)

        # ---- row 1: New / mode / delay ... undo redo save copy
        r1 = QHBoxLayout()
        self.btn_new = self._btn("＋ New", self.new_snip, "Ctrl+N")
        r1.addWidget(self.btn_new)

        self.btn_mode = QToolButton()
        self.btn_mode.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self)
        for label, key in (("Rectangle", "rect"), ("Freeform", "freeform"), ("Full screen", "full")):
            m.addAction(label, lambda k=key, l=label: self._set_mode(k, l))
        self.btn_mode.setMenu(m)
        self.btn_mode.setText("▭ Rectangle")
        r1.addWidget(self.btn_mode)

        self.btn_delay = QToolButton()
        self.btn_delay.setPopupMode(QToolButton.InstantPopup)
        dm = QMenu(self)
        for s in (0, 3, 5, 10):
            dm.addAction("No delay" if s == 0 else f"{s} seconds", lambda s=s: self._set_delay(s))
        self.btn_delay.setMenu(dm)
        self.btn_delay.setText("⏱ No delay")
        r1.addWidget(self.btn_delay)
        r1.addStretch()

        self.btn_undo = self._btn("↶", lambda: self.canvas.undo(), "Ctrl+Z")
        self.btn_redo = self._btn("↷", lambda: self.canvas.redo(), "Ctrl+Y")
        self.btn_save = self._btn("💾", self.save, "Ctrl+S")
        self.btn_copy = self._btn("⧉", self.copy, "Ctrl+C")
        self.btn_more = QToolButton()
        self.btn_more.setText("⋯")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        mm = QMenu(self)
        mm.addAction("About", lambda: QMessageBox.about(self, "About", "Snipping Tool clone\nBuilt with PySide6"))
        mm.addAction("Exit", self.close)
        self.btn_more.setMenu(mm)
        for b in (self.btn_undo, self.btn_redo, self.btn_save, self.btn_copy, self.btn_more):
            r1.addWidget(b)
        v.addLayout(r1)

        # ---- row 2: drawing tools
        r2 = QHBoxLayout()
        r2.addStretch()
        self.canvas = Canvas()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        tools = [("✎", "pen", "Ballpoint pen"), ("🖍", "highlighter", "Highlighter"),
                 ("⌫", "eraser", "Eraser"), ("▭", "rect", "Rectangle"),
                 ("◯", "ellipse", "Ellipse"), ("╱", "line", "Line"),
                 ("➚", "arrow", "Arrow"), ("T", "text", "Text"), ("✂", "crop", "Crop")]
        for txt, key, tip in tools:
            b = QToolButton()
            b.setText(txt)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.canvas.set_tool(k))
            self.group.addButton(b)
            r2.addWidget(b)
            if key == "pen":
                b.setChecked(True)
        sep = QFrame(objectName="sep")
        sep.setFixedSize(1, 22)
        r2.addWidget(sep)
        self.btn_color = QToolButton()
        self.btn_color.setText("Outline")
        self.btn_color.clicked.connect(self.pick_color)
        self._paint_color_btn()
        r2.addWidget(self.btn_color)
        self.size_box = QComboBox()
        for s in (2, 4, 6, 10, 16):
            self.size_box.addItem(f"{s} px", s)
        self.size_box.setCurrentIndex(1)
        self.size_box.currentIndexChanged.connect(lambda: setattr(self.canvas, "width", self.size_box.currentData()))
        r2.addWidget(self.size_box)
        r2.addStretch()
        v.addLayout(r2)

        # ---- canvas area
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.canvas)
        self.hint = QLabel("Press  ＋ New  (Ctrl+N)  to take a snip")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#777; font-size:14px; background:white;")
        v.addWidget(self.hint, 1)
        v.addWidget(self.scroll, 1)
        self.scroll.hide()
        self.canvas.changed.connect(self._refresh)
        self._refresh()

    def _btn(self, text, fn, shortcut=None):
        b = QToolButton()
        b.setText(text)
        b.clicked.connect(fn)
        if shortcut:
            b.setToolTip(f"{text}  ({shortcut})")
            QShortcut(QKeySequence(shortcut), self, activated=fn)
        return b

    def _paint_color_btn(self):
        self.btn_color.setStyleSheet(
            f"QToolButton {{ border-left: 14px solid {self.canvas.color.name()}; }}")

    def _refresh(self):
        has = self.canvas.has_image()
        self.scroll.setVisible(has)
        self.hint.setVisible(not has)
        self.btn_undo.setEnabled(bool(self.canvas.undo_stack))
        self.btn_redo.setEnabled(bool(self.canvas.redo_stack))
        self.btn_save.setEnabled(has)
        self.btn_copy.setEnabled(has)

    def _set_mode(self, key, label):
        self.mode = key
        self.btn_mode.setText({"rect": "▭ ", "freeform": "✐ ", "full": "🖵 "}[key] + label)

    def _set_delay(self, s):
        self.delay = s
        self.btn_delay.setText("⏱ No delay" if s == 0 else f"⏱ {s} seconds")

    def pick_color(self):
        c = QColorDialog.getColor(self.canvas.color, self, "Outline color")
        if c.isValid():
            self.canvas.color = c
            self._paint_color_btn()

    # ---- capture flow
    def new_snip(self):
        self.hide()
        QTimer.singleShot(self.delay * 1000 + 300, self._capture)

    def _capture(self):
        shot, vgeo = grab_all()
        if self.mode == "full":
            self._on_done(shot)
            return
        self.overlay = SnipOverlay(shot, vgeo, self.mode)
        self.overlay.done.connect(self._on_done)

    def _on_done(self, pm):
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if pm is not None:
            self.canvas.set_image(pm)
            QGuiApplication.clipboard().setPixmap(pm)
            self.resize(max(self.width(), min(pm.width() + 40, 1400)),
                        max(self.height(), min(pm.height() + 160, 900)))

    def save(self):
        if not self.canvas.has_image():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save as", "Screenshot.png", "PNG (*.png);;JPEG (*.jpg);;Bitmap (*.bmp)")
        if path:
            self.canvas.pix.save(path)

    def copy(self):
        if self.canvas.has_image():
            QGuiApplication.clipboard().setPixmap(self.canvas.pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = SnipWindow()
    w.show()
    sys.exit(app.exec())

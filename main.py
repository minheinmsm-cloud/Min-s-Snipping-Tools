import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QRect, QRectF, QPointF, QTimer, Signal, QSize, QSizeF, QEvent, QObject, QMimeData, QUrl, QAbstractNativeEventFilter
from PySide6.QtGui import (QGuiApplication, QPixmap, QPainter, QColor, QPen,
                           QPainterPath, QPolygonF, QPolygon, QKeySequence,
                           QShortcut, QFont, QFontMetrics, QIcon, QImage, QTextCursor, QTextDocument, QTextCharFormat)
from PySide6.QtWidgets import (QApplication, QWidget, QMainWindow, QVBoxLayout,
                               QHBoxLayout, QToolButton, QMenu, QScrollArea,
                               QLabel, QFileDialog, QColorDialog, QComboBox,
                               QButtonGroup, QFrame, QMessageBox, QLineEdit,
                               QPushButton, QGridLayout, QWidgetAction,
                               QSystemTrayIcon, QPlainTextEdit, QSizePolicy)

try:
    import cv2
    import numpy as np
    HAVE_CV = True
except Exception:
    HAVE_CV = False

try:
    import imageio_ffmpeg
    HAVE_FFMPEG = True
except Exception:
    HAVE_FFMPEG = False

STYLE = """
QMainWindow, QWidget#root { background: #f3f3f3; font-family: 'Segoe UI'; }
QToolButton { border: 1px solid transparent; border-radius: 5px;
              padding: 6px 12px; font-size: 15px; background: transparent; }
QToolButton:hover { background: #e5e5e5; }
QToolButton:checked { background: #dbe9f7; border: 1px solid #2a76c6; }
QToolButton:disabled { color: #aaa; }
QToolButton::menu-indicator { width: 8px; }
QComboBox { font-size: 14px; padding: 3px 8px; border: 1px solid #ccc; border-radius: 4px; background: white; }
QFrame#sep { background: #d0d0d0; }
QScrollArea { background: white; border: none; }
QStatusBar { background: #f3f3f3; color: #444; font-size: 12px; }
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
    region = Signal(object)  # QRect in screen coordinates (video mode)

    def __init__(self, shot, vgeo, mode, video=False):
        super().__init__()
        self.shot, self.mode, self.video = shot, mode, video
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
            if self.video:
                g = r.translated(self.geometry().topLeft())
                self.hide()
                self.region.emit(g)
                self.close()
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
class BoxEdit(QPlainTextEdit):
    """Editor shown on top of a text box / callout.
    Enter = new line.  Click outside, Esc or Ctrl+Enter = done."""
    editingFinished = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.obj, self.snap, self.was_new, self.old_text, self.zoom = None, None, False, "", 1.0
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.document().setDocumentMargin(0)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape or (e.key() in (Qt.Key_Return, Qt.Key_Enter)
                                        and (e.modifiers() & Qt.ControlModifier)):
            self.clearFocus()  # finish (the text is kept)
        else:
            super().keyPressEvent(e)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.editingFinished.emit()


def cloud_path(r, target):
    """Scalloped 'revision cloud' outline around rectangle r."""
    pts = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
    path = QPainterPath()
    path.moveTo(pts[0])
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        n = max(1, round(length / target))
        seg = length / n
        ux, uy = dx / length, dy / length
        nx, ny = uy, -ux  # outward normal (clockwise path, y down)
        h = seg * 0.5 * 4 / 3  # bezier handle that gives a ~semicircle bump
        for k in range(n):
            p0 = QPointF(a.x() + ux * seg * k, a.y() + uy * seg * k)
            p1 = QPointF(a.x() + ux * seg * (k + 1), a.y() + uy * seg * (k + 1))
            path.cubicTo(QPointF(p0.x() + nx * h, p0.y() + ny * h),
                         QPointF(p1.x() + nx * h, p1.y() + ny * h), p1)
    path.closeSubpath()
    return path


CALLOUT_FILL = QColor("#fff04d")  # default callout box colour (like the yellow note in your video)
SHAPE_TOOLS = ("rect", "ellipse", "line", "arrow", "cloud")


class Canvas(QWidget):
    changed = Signal()
    zoom_changed = Signal(float)

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
        self.objs = []               # text boxes + callouts: stay editable until you save / copy / crop
        self.sel_obj, self.obj_drag, self.editing = None, None, None
        self.fill = None             # None = no fill
        self.live = None             # (tool, a, b): last shape, still editable until you click elsewhere
        # floating snip added with "Add": drag to place it, drag its corner to resize
        self.layers, self.sel, self.fl_drag = [], -1, None
        self.zoom = 1.0   # 1.0 = real pixels (1 image pixel = 1 screen pixel, sharp)
        self._pan = None
        self.scroll = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.origin, self.view_rect = QPointF(0, 0), QRectF()
        self.co_stage, self.co_box, self.co_tip = 0, None, None  # callout drawing state
        self.setMouseTracking(True)

    def has_image(self):
        return not self.pix.isNull()

    def _scale(self):
        """Bigger screenshots (4K / HiDPI) get proportionally bigger text and line heads."""
        return max(1.0, self.pix.width() / 1200)

    # --- zoom (Ctrl + mouse wheel)
    def _z(self):
        """Image pixels -> widget (logical) pixels. zoom 1.0 = one image pixel per physical screen pixel."""
        return self.zoom / max(1.0, self.devicePixelRatioF())

    def _ip(self, e):
        """Mouse event position -> image coordinates."""
        return e.position() / self._z() + self.origin

    def set_zoom(self, z, anchor=None):
        z = min(8.0, max(0.1, z))
        if abs(z - self.zoom) < 1e-4:
            return
        if self.editor is not None:
            self._commit(self.editor)
        old_z, sa = self._z(), self.scroll
        vp = self.mapTo(sa.viewport(), anchor.toPoint()) if (sa is not None and anchor is not None) else None
        self.zoom = z
        self._relayout()
        self.update()
        self.zoom_changed.emit(self.zoom)
        if vp is not None:  # keep the point under the mouse where it was
            nz = self._z()
            sa.horizontalScrollBar().setValue(int(anchor.x() / old_z * nz - vp.x()))
            sa.verticalScrollBar().setValue(int(anchor.y() / old_z * nz - vp.y()))

    def fit_zoom(self):
        if self.scroll is None or self.view_rect.isEmpty():
            return
        vp = self.scroll.viewport()
        dpr = max(1.0, self.devicePixelRatioF())
        self.set_zoom(min(vp.width() * dpr / self.view_rect.width(),
                          vp.height() * dpr / self.view_rect.height()))

    # --- compose mode ("Add"): several pictures, each movable / resizable until you click outside
    def _lrect(self, L):
        return QRectF(L["pos"], QSizeF(L["pix"].width() * L["scale"], L["pix"].height() * L["scale"]))

    def _union(self):
        r = self._lrect(self.layers[0])
        for L in self.layers[1:]:
            r = r.united(self._lrect(L))
        return r

    def _relayout(self):
        if not self.layers:
            self.origin = QPointF(0, 0)
            self.view_rect = QRectF(0, 0, self.pix.width(), self.pix.height())
            self._apply_tool_cursor()
        else:
            u = self._union()
            self.origin = QPointF(math.floor(u.x()), math.floor(u.y()))
            self.view_rect = QRectF(self.origin, QSizeF(math.ceil(u.right() - self.origin.x()),
                                                        math.ceil(u.bottom() - self.origin.y())))
        z = self._z()
        self.setFixedSize(max(1, math.ceil(self.view_rect.width() * z)),
                          max(1, math.ceil(self.view_rect.height() * z)))

    def add_float(self, pm):
        if self.editor is not None:
            self._commit(self.editor)
        self.commit_live()
        self.flatten_objs()
        if not self.has_image():
            self.set_image(pm)
            return
        if not self.layers:  # the current picture becomes the first movable layer
            self.layers = [{"pix": self.pix.copy(), "base": self.base.copy(),
                            "pos": QPointF(0, 0), "scale": 1.0}]
        u = self._union()
        self.layers.append({"pix": pm.copy(), "base": pm.copy(),
                            "pos": QPointF(u.right() + 12 * self._scale(), u.top()), "scale": 1.0})
        self.sel, self.fl_drag = len(self.layers) - 1, None
        self._relayout()
        self.setFocus()
        self.update()
        self.changed.emit()

    def commit_float(self):
        """Apply: flatten every layer into one picture."""
        if not self.layers:
            return
        ls, self.layers, self.fl_drag = self.layers, [], None
        imgs, bases, rects = [], [], []
        for L in ls:
            r = self._lrect(L)
            if abs(L["scale"] - 1) < 1e-3:
                img, bimg = L["pix"], L["base"]
            else:
                w, h = max(1, round(r.width())), max(1, round(r.height()))
                img = L["pix"].scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                bimg = L["base"].scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            imgs.append(img)
            bases.append(bimg)
            rects.append(QRectF(r.topLeft(), QSizeF(img.width(), img.height())))
        u = rects[0]
        for r in rects[1:]:
            u = u.united(r)
        u = u.toAlignedRect()
        self._push()  # undo returns to the picture as it was before "Add"

        def compose(items):
            out = QPixmap(u.size())
            out.fill(Qt.white)
            q = QPainter(out)
            for img, r in zip(items, rects):
                q.drawPixmap(round(r.x()) - u.x(), round(r.y()) - u.y(), img)
            q.end()
            return out

        self.pix, self.base = compose(imgs), compose(bases)
        self._relayout()
        self.update()
        self.changed.emit()

    def remove_selected_layer(self):
        if len(self.layers) < 2 or not (0 <= self.sel < len(self.layers)):
            return
        self.layers.pop(self.sel)
        self.sel = len(self.layers) - 1
        if len(self.layers) == 1:
            self.commit_float()
            return
        self._relayout()
        self.update()
        self.changed.emit()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.layers:
                self.remove_selected_layer()
                return
            o = self._sel()
            if o is not None and self.editor is None:  # delete the selected text box / callout
                self._push()
                self.objs = [x for x in self.objs if x is not o]
                self.sel_obj = None
                self.update()
                return
        super().keyPressEvent(e)

    def _corner_pts(self, r):
        return {"tl": r.topLeft(), "tr": r.topRight(), "bl": r.bottomLeft(), "br": r.bottomRight()}

    def _corner_at(self, r, ip):
        hs = 15 * self._scale()
        for name, c in self._corner_pts(r).items():
            if QRectF(c.x() - hs, c.y() - hs, hs * 2, hs * 2).contains(ip):
                return name
        return None

    def _draw_handle(self, p, c, diag):
        """Round blue grip with a white two-headed diagonal arrow (diag +1 = \\ , -1 = /)."""
        sc = self._scale()
        p.setPen(QPen(QColor("white"), 2 * sc))
        p.setBrush(QColor("#2a76c6"))
        p.drawEllipse(c, 11 * sc, 11 * sc)
        d = 5.2 * sc
        a, b = QPointF(c.x() - d, c.y() - d * diag), QPointF(c.x() + d, c.y() + d * diag)
        p.setPen(QPen(QColor("white"), 1.8 * sc, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(a, b)
        for end, sg in ((b, 1), (a, -1)):
            ang = math.atan2(d * diag * sg, d * sg)
            for t in (-0.75, 0.75):
                p.drawLine(end, QPointF(end.x() - math.cos(ang + t) * 4.6 * sc,
                                        end.y() - math.sin(ang + t) * 4.6 * sc))

    def _paint_layers(self, p):
        p.fillRect(self.view_rect, Qt.white)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        for L in self.layers:
            p.drawPixmap(self._lrect(L), L["pix"], QRectF(L["pix"].rect()))
        sc = self._scale()
        for i, L in enumerate(self.layers):
            r = self._lrect(L)
            if i == self.sel:
                p.setPen(QPen(QColor("#2a76c6"), 3 * sc))
                p.setBrush(Qt.NoBrush)
                p.drawRect(r)
            else:
                p.setPen(QPen(QColor(150, 150, 150), 1, Qt.DotLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(r)
        if 0 <= self.sel < len(self.layers):
            r = self._lrect(self.layers[self.sel])
            pts = self._corner_pts(r)
            self._draw_handle(p, pts["tl"], 1)
            self._draw_handle(p, pts["br"], 1)
            self._draw_handle(p, pts["tr"], -1)
            self._draw_handle(p, pts["bl"], -1)
        if self.fl_drag and self.fl_drag[0] == "resize":  # live size badge
            L = self.fl_drag[4]
            r = self._lrect(L)
            text = f"{int(r.width())} x {int(r.height())}   ({int(L['scale'] * 100)}%)"
            f = QFont("Segoe UI")
            f.setPixelSize(int(13 * sc))
            f.setBold(True)
            p.setFont(f)
            w = QFontMetrics(f).horizontalAdvance(text) + 18 * sc
            h = QFontMetrics(f).height() + 8 * sc
            box = QRectF(r.left(), r.bottom() + 16 * sc, w, h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(30, 30, 30, 220))
            p.drawRoundedRect(box, 6 * sc, 6 * sc)
            p.setPen(QColor("white"))
            p.drawText(box, int(Qt.AlignCenter), text)

    def _snap_pos(self, idx, pos):
        """Magnetic alignment against the other pictures (edges, centre, side-by-side gap)."""
        sc = self._scale()
        tol, gap = 10 * sc, 12 * sc
        r = self._lrect(self.layers[idx])
        w, h = r.width(), r.height()
        xs, ys = [], []
        for j, O in enumerate(self.layers):
            if j == idx:
                continue
            o = self._lrect(O)
            xs += [o.right() + gap, o.left() - w - gap, o.left(), o.right() - w]
            ys += [o.top(), o.bottom() - h, o.top() + (o.height() - h) / 2, o.bottom() + gap, o.top() - h - gap]
        x, y = pos.x(), pos.y()
        bx = min(xs, key=lambda t: abs(x - t), default=None)
        by = min(ys, key=lambda t: abs(y - t), default=None)
        if bx is not None and abs(x - bx) < tol:
            x = bx
        if by is not None and abs(y - by) < tol:
            y = by
        return QPointF(x, y)

    def _float_mouse(self, e, kind):
        sc = self._scale()
        ip = self._ip(e)
        cur = self.layers[self.sel] if 0 <= self.sel < len(self.layers) else None
        corner = self._corner_at(self._lrect(cur), ip) if cur is not None else None
        hit = -1
        for i in range(len(self.layers) - 1, -1, -1):
            if self._lrect(self.layers[i]).contains(ip):
                hit = i
                break
        if kind == "press":
            if corner:
                r = self._lrect(cur)
                ax = r.left() if corner in ("tr", "br") else r.right()   # opposite corner stays fixed
                ay = r.top() if corner in ("bl", "br") else r.bottom()
                self.fl_drag = ("resize", e.globalPosition(), QPointF(ax, ay), cur["scale"], cur, corner)
            elif hit >= 0:
                L = self.layers.pop(hit)
                self.layers.append(L)  # bring to front
                self.sel = len(self.layers) - 1
                self.fl_drag = ("move", e.globalPosition(), QPointF(L["pos"]), L["scale"], L, None)
            else:
                self.commit_float()  # click outside = apply
                return
            self.setFocus()
            self.update()
        elif kind == "move":
            if self.fl_drag:
                mode, g0, ref, sc0, L, cn = self.fl_drag
                d = (e.globalPosition() - g0) / self._z()
                idx = self.layers.index(L)
                if mode == "move":
                    L["pos"] = self._snap_pos(idx, ref + d)
                else:
                    pw, ph = L["pix"].width(), L["pix"].height()
                    sx = 1 if cn in ("tr", "br") else -1
                    sy = 1 if cn in ("bl", "br") else -1
                    grow = (sx * d.x() / (pw * sc0) + sy * d.y() / (ph * sc0)) / 2
                    scale = max(0.1, sc0 * (1 + grow))
                    for j, O in enumerate(self.layers):  # match another picture's height / width
                        if j == idx:
                            continue
                        o = self._lrect(O)
                        if abs(ph * scale - o.height()) < 10 * sc:
                            scale = o.height() / ph
                            break
                        if abs(pw * scale - o.width()) < 10 * sc:
                            scale = o.width() / pw
                            break
                    L["scale"] = scale
                    w, h = pw * scale, ph * scale
                    L["pos"] = QPointF(ref.x() if sx == 1 else ref.x() - w,
                                       ref.y() if sy == 1 else ref.y() - h)
                self._relayout()
                self.update()
            else:
                if corner in ("tl", "br"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif corner in ("tr", "bl"):
                    self.setCursor(Qt.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.SizeAllCursor if hit >= 0 else Qt.ArrowCursor)
        else:
            self.fl_drag = None
            self._relayout()
            self.update()

    # --- style setters: they also restyle the selected shape / text box / callout
    def _sel(self):
        o = self.sel_obj
        return o if (o is not None and self._is_obj(o)) else None

    def set_color(self, c):
        self.color = QColor(c)
        o = self._sel()
        if o is not None:
            o["color"] = QColor(c)
        if self.editor is not None:
            self._editor_style(self.editor)
        self.update()

    def set_width(self, w):
        self.width = w
        o = self._sel()
        if o is not None:
            o["width"] = w
            o["px"] = self._obj_px(o["kind"], w)
            self._fit_obj(o)
            if self.editor is not None and self.editing is o:
                self.editor.setFont(self._font(max(6, int(o["px"] * self.editor.zoom))))
                self._place_editor(self.editor)
        self.update()

    def set_fill(self, c):
        self.fill = QColor(c) if c is not None else None
        o = self._sel()
        if o is not None:
            o["fill"] = QColor(c) if c is not None else None
        self.update()

    def commit_live(self):
        if self.live:
            tool, a, b = self.live
            self.live = None
            self._push()
            p = QPainter(self.pix)
            p.setRenderHint(QPainter.Antialiasing)
            self._draw_shape(p, a, b, tool)
            p.end()
            self.update()

    def _editor_style(self, ed):
        ed.setStyleSheet(
            f"QPlainTextEdit {{ background: transparent; color: {ed.obj['color'].name()};"
            " border: none; padding: 0px; margin: 0px; }")

    def _apply_tool_cursor(self):
        if self.tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()

    def set_tool(self, tool):
        self.commit_pending()  # commit/discard unfinished text before switching
        self.sel_obj = None
        self.co_stage, self.co_box, self.co_tip = 0, None, None
        self.a = self.b = None
        self.tool = tool
        self._apply_tool_cursor()
        self.update()

    # --- text boxes + callouts: real objects, editable until you save / copy / crop
    def _font_px(self):
        return int((16 + self.width * 3) * self._scale())

    def _obj_px(self, kind, w):
        sc = self._scale()
        return int((16 + w * 3) * sc) if kind == "text" else int((12 + w * 2.5) * sc)

    def _font(self, px=None):
        f = QFont("Segoe UI")
        f.setPixelSize(px or self._font_px())
        f.setBold(True)
        return f

    def _text_doc(self, text, px, width, color=None):
        """Same layout engine for measuring (auto-grow) and for drawing, so they always agree."""
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(self._font(px))
        doc.setPlainText(text)
        doc.setTextWidth(max(10.0, width))
        if color is not None:
            cur = QTextCursor(doc)
            cur.select(QTextCursor.Document)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cur.mergeCharFormat(fmt)
        return doc

    def _fit_obj(self, o):
        """The box grows taller as you type (never smaller than the height you gave it)."""
        h = self._text_doc(o["text"] or " ", o["px"], o["box"].width() - 16).size().height() + 8
        o["box"].setHeight(max(o["minh"], h))

    def _is_obj(self, o):
        return any(x is o for x in self.objs)

    def _copy_objs(self, objs):
        return [{**o, "box": QRectF(o["box"]),
                 "tip": QPointF(o["tip"]) if o["tip"] is not None else None,
                 "color": QColor(o["color"]),
                 "fill": QColor(o["fill"]) if o["fill"] is not None else None} for o in objs]

    def _snapshot(self):
        return (self.pix.copy(), self.base.copy(), self._copy_objs(self.objs))

    def _new_obj(self, kind, box, tip=None):
        snap = self._snapshot()  # state before the new object exists (for undo)
        fill = self.fill
        if fill is None and kind == "callout":
            fill = CALLOUT_FILL
        o = {"kind": kind, "box": QRectF(box), "tip": QPointF(tip) if tip is not None else None,
             "text": "", "color": QColor(self.color), "width": self.width,
             "fill": QColor(fill) if fill is not None else None}
        o["px"] = self._obj_px(kind, self.width)
        o["minh"] = box.height()
        if kind == "text":
            o["minh"] = max(box.height(), self._text_doc("Ag", o["px"], 100).size().height() + 8)
            if box.height() < 2:  # plain click: centre the box on the click
                o["box"].moveTop(box.top() - o["minh"] / 2)
            o["box"].setHeight(max(o["minh"], box.height()))
        self.objs.append(o)
        self._edit_obj(o, snap, True)

    def _place_editor(self, ed):
        z, r = ed.zoom, ed.obj["box"].adjusted(8, 4, -8, -4)
        ed.setGeometry(QRectF(r.x() * z, r.y() * z, r.width() * z, r.height() * z).toRect())

    def _edit_obj(self, o, snap=None, is_new=False):
        self.commit_pending()
        if not self._is_obj(o):  # it was an empty box that just got removed
            return
        self.sel_obj, self.editing = o, o
        ed = BoxEdit(self)
        ed.obj, ed.was_new, ed.old_text, ed.zoom = o, is_new, o["text"], self._z()
        ed.snap = snap if snap is not None else self._snapshot()
        ed.setFont(self._font(max(6, int(o["px"] * ed.zoom))))
        ed.setPlainText(o["text"])
        cur = ed.textCursor()
        cur.movePosition(QTextCursor.End)
        ed.setTextCursor(cur)
        self._editor_style(ed)
        self._place_editor(ed)
        ed.textChanged.connect(lambda e=ed: self._grow(e))
        ed.editingFinished.connect(lambda e=ed: self._commit(e))
        self.editor = ed
        ed.show()
        ed.setFocus()
        self.update()

    def _grow(self, ed):
        if ed is not self.editor:
            return
        o = ed.obj
        o["text"] = ed.toPlainText()
        self._fit_obj(o)
        self._place_editor(ed)
        self.update()

    def commit_pending(self):
        if self.editor is not None:
            self._commit(self.editor)
        self.commit_live()
        self.commit_float()

    def _commit(self, ed):
        if ed is not self.editor:  # stale signal from an old editor
            return
        self.editor = self.editing = None
        o = ed.obj
        text = ed.toPlainText().rstrip()
        ed.hide()
        ed.deleteLater()
        if o["kind"] == "text" and not text.strip():  # an empty text box simply disappears
            self.objs = [x for x in self.objs if x is not o]
            self.sel_obj = None
            self.update()
            return
        o["text"] = text
        self._fit_obj(o)
        if ed.was_new or text != ed.old_text:  # one undo step for "created / edited"
            self.undo_stack.append(ed.snap)
            self.redo_stack.clear()
            self.changed.emit()
        self.setFocus()
        self.update()

    def _draw_obj(self, p, o, with_text=True):
        box = o["box"]
        if o["kind"] == "callout":
            self._draw_callout(p, box, o["tip"], obj=o)
        elif o["fill"] is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(o["fill"])
            p.drawRect(box)
        if with_text and o["text"]:
            w, h = box.width() - 16, box.height() - 8
            doc = self._text_doc(o["text"], o["px"], w, o["color"])
            p.save()
            p.translate(box.left() + 8, box.top() + 4)
            doc.drawContents(p, QRectF(0, 0, w, h))
            p.restore()

    def flatten_objs(self):
        """Burn all text boxes / callouts into the picture (before crop / adding pictures)."""
        self.commit_pending()
        if not self.objs:
            return
        p = QPainter(self.pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        for o in self.objs:
            self._draw_obj(p, o)
        p.end()
        self.objs, self.sel_obj, self.editing = [], None, None

    def export_pixmap(self):
        """The picture as it should be saved / copied (objects included, still editable afterwards)."""
        self.commit_pending()
        if not self.objs:
            return self.pix
        out = self.pix.copy()
        p = QPainter(out)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        for o in self.objs:
            self._draw_obj(p, o)
        p.end()
        return out

    def set_image(self, pm):
        if self.editor is not None:
            self.editor.hide()
            self.editor.deleteLater()
            self.editor = None
        self.live = None
        self.layers, self.fl_drag = [], None
        self.objs, self.sel_obj, self.obj_drag, self.editing, self._pan = [], None, None, None, None
        self.co_stage, self.co_box, self.co_tip = 0, None, None
        self.pix, self.base = pm.copy(), pm.copy()
        self.zoom = 1.0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._relayout()
        self.zoom_changed.emit(self.zoom)
        self.update()
        self.changed.emit()

    # --- undo / redo
    def _push(self):
        self.undo_stack.append(self._snapshot())
        self.redo_stack.clear()
        self.changed.emit()

    def _restore(self, snap):
        self.pix, self.base, self.objs = snap
        self.sel_obj = self.editing = None
        self._relayout()
        self.update()
        self.changed.emit()

    def undo(self):
        if self.layers:  # composing: undo = remove the last added picture
            self.layers.pop()
            if len(self.layers) < 2:
                self.layers = []
            self.sel = len(self.layers) - 1
            self._relayout()
            self.update()
            self.changed.emit()
            return
        self.commit_pending()
        if self.undo_stack:
            self.redo_stack.append(self._snapshot())
            self._restore(self.undo_stack.pop())

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._snapshot())
            self._restore(self.redo_stack.pop())

    # --- drawing helpers
    def _pen(self):
        if self.tool == "highlighter":
            c = QColor(255, 235, 0, 110)
            return QPen(c, self.width * 3 + 6, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin)
        return QPen(self.color, self.width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _draw_shape(self, p, a, b, tool=None):
        tool = tool or self.tool
        p.setPen(self._pen())
        p.setBrush(self.fill if self.fill is not None else Qt.NoBrush)
        if tool == "rect":
            p.drawRect(QRectF(a, b).normalized())
        elif tool == "ellipse":
            p.drawEllipse(QRectF(a, b).normalized())
        elif tool == "cloud":
            r = QRectF(a, b).normalized()
            if r.width() > 6 and r.height() > 6:
                p.drawPath(cloud_path(r, (18 + self.width * 2) * self._scale()))
        elif tool == "line":
            p.setBrush(Qt.NoBrush)
            p.drawLine(a, b)
        elif tool == "arrow":
            p.setBrush(Qt.NoBrush)
            p.drawLine(a, b)
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            size = 10 + self.width * 2
            pts = [b]
            for d in (-0.5, 0.5):
                pts.append(QPointF(b.x() - math.cos(ang + d) * size, b.y() - math.sin(ang + d) * size))
            p.setBrush(self.color)
            p.drawPolygon(QPolygonF(pts))

    def _draw_callout(self, p, box, tip, preview=False, obj=None):
        """Box + leader with a bend and an open arrowhead. preview=True = thin teal 'rubber band' look."""
        sc = self._scale()
        k = 18 * sc
        if preview:
            p.setPen(QPen(QColor("#5b8794"), 1.5 * sc, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(QColor(110, 190, 200, 55))
            width = 0
        else:
            color, width, fill = obj["color"], obj["width"], obj["fill"]
            p.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(fill if fill is not None else Qt.NoBrush)
        p.drawRect(box)
        p.setBrush(Qt.NoBrush)
        cx, cy = box.center().x(), box.center().y()
        dx, dy = tip.x() - cx, tip.y() - cy
        if abs(dx) > abs(dy):  # arrow points mostly sideways: leave from the left / right edge
            if dx > 0:
                s0 = QPointF(box.right(), cy)
                knee = QPointF(s0.x() + k, s0.y())
            else:
                s0 = QPointF(box.left(), cy)
                knee = QPointF(s0.x() - k, s0.y())
        elif dy > 0:
            s0 = QPointF(cx, box.bottom())
            knee = QPointF(s0.x(), s0.y() + k)
        else:
            s0 = QPointF(cx, box.top())
            knee = QPointF(s0.x(), s0.y() - k)
        p.drawPolyline(QPolygonF([s0, knee, tip]))
        ang = math.atan2(tip.y() - knee.y(), tip.x() - knee.x())
        head = ((9 if preview else 14) + width * 3) * sc
        for d in (-0.45, 0.45):
            p.drawLine(tip, QPointF(tip.x() - math.cos(ang + d) * head, tip.y() - math.sin(ang + d) * head))

    def _pan_begin(self, e):
        if self.scroll is None:
            return False
        self._pan = (e.globalPosition(), self.scroll.horizontalScrollBar().value(),
                     self.scroll.verticalScrollBar().value())
        self.setCursor(Qt.ClosedHandCursor)
        return True

    def _pan_move(self, e):
        g0, h0, v0 = self._pan
        d = e.globalPosition() - g0
        self.scroll.horizontalScrollBar().setValue(int(h0 - d.x()))
        self.scroll.verticalScrollBar().setValue(int(v0 - d.y()))

    def _pan_end(self):
        self._pan = None
        self._apply_tool_cursor()

    def _co_handles(self, box):
        cx, cy = box.center().x(), box.center().y()
        return {"tl": box.topLeft(), "tr": box.topRight(), "bl": box.bottomLeft(), "br": box.bottomRight(),
                "t": QPointF(cx, box.top()), "b": QPointF(cx, box.bottom()),
                "l": QPointF(box.left(), cy), "r": QPointF(box.right(), cy)}

    def _obj_hit(self, o, pt):
        hs = 10 * self._scale()
        if o["tip"] is not None and (pt - o["tip"]).manhattanLength() <= hs * 1.8:
            return "tip"
        for name, c in self._co_handles(o["box"]).items():
            if abs(pt.x() - c.x()) <= hs and abs(pt.y() - c.y()) <= hs:
                return name
        return "move" if o["box"].contains(pt) else None

    def _obj_at(self, pt):
        hs = 10 * self._scale()
        for o in reversed(self.objs):
            if o["box"].contains(pt) or (o["tip"] is not None and (pt - o["tip"]).manhattanLength() <= hs * 1.8):
                return o
        return None

    def _co_cursor(self, part):
        if part in ("tl", "br"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif part in ("tr", "bl"):
            self.setCursor(Qt.SizeBDiagCursor)
        elif part in ("t", "b"):
            self.setCursor(Qt.SizeVerCursor)
        elif part in ("l", "r"):
            self.setCursor(Qt.SizeHorCursor)
        elif part == "tip":
            self.setCursor(Qt.CrossCursor)
        elif part == "move":
            self.setCursor(Qt.SizeAllCursor)
        else:
            self._apply_tool_cursor()

    def _paint_obj_handles(self, p, o):
        sc, box = self._scale(), o["box"]
        p.setPen(QPen(QColor("#2a76c6"), 1.5 * sc, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRect(box.adjusted(-3 * sc, -3 * sc, 3 * sc, 3 * sc))
        hs = 5 * sc
        p.setPen(QPen(QColor("#2a76c6"), 1.5 * sc))
        p.setBrush(QColor("white"))
        for c in self._co_handles(box).values():
            p.drawRect(QRectF(c.x() - hs, c.y() - hs, hs * 2, hs * 2))
        if o["tip"] is not None:  # arrow-tip grip
            p.setBrush(QColor("#2a76c6"))
            p.setPen(QPen(QColor("white"), 1.5 * sc))
            p.drawEllipse(o["tip"], 7 * sc, 7 * sc)

    def paintEvent(self, _):
        p = QPainter(self)
        if self.pix.isNull():
            return
        z = self._z()
        p.scale(z, z)
        p.translate(-self.origin)
        if z < 1:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.layers:
            self._paint_layers(p)
        else:
            p.drawPixmap(0, 0, self.pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        for o in self.objs:
            self._draw_obj(p, o, with_text=(o is not self.editing))
        o = self._sel()
        if o is not None:
            self._paint_obj_handles(p, o)
        if self.a and self.b:
            if self.tool in SHAPE_TOOLS:
                self._draw_shape(p, self.a, self.b)
            elif self.tool == "crop":
                p.setPen(QPen(Qt.white, 1, Qt.DashLine))
                p.setBrush(QColor(0, 120, 215, 50))
                p.drawRect(QRectF(self.a, self.b).normalized())
            elif self.tool == "text":
                p.setPen(QPen(QColor("#2a76c6"), 1.5 * self._scale(), Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(self.a, self.b).normalized())
            elif self.tool == "callout" and self.co_stage == 1:
                p.setPen(QPen(QColor("#5b8794"), 1.5 * self._scale()))
                p.setBrush(QColor(110, 190, 200, 55))
                p.drawRect(QRectF(self.a, self.b).normalized())
        if self.live:
            t, la, lb = self.live
            self._draw_shape(p, la, lb, t)
            sc = self._scale()
            r = QRectF(la, lb).normalized().adjusted(-4 * sc, -4 * sc, 4 * sc, 4 * sc)
            p.setPen(QPen(QColor("#2a76c6"), 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            p.setPen(QPen(QColor("#2a76c6"), 1))
            p.setBrush(QColor("white"))
            hs = 3.5 * sc
            for c in (r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()):
                p.drawRect(QRectF(c.x() - hs, c.y() - hs, hs * 2, hs * 2))
        if self.tool == "callout" and self.co_stage == 2 and self.co_box and self.co_tip:
            self._draw_callout(p, self.co_box, self.co_tip, preview=True)
        if self.path and self.tool in ("pen", "highlighter"):
            p.setPen(self._pen())
            p.setBrush(Qt.NoBrush)
            p.drawPath(self.path)

    def _begin_obj_drag(self, o, part, pt):
        self.sel_obj = o
        self.obj_drag = {"o": o, "part": part, "pt": pt, "moved": False, "box": QRectF(o["box"]),
                         "tip": QPointF(o["tip"]) if o["tip"] is not None else None}
        self.setFocus()
        self.update()

    def mousePressEvent(self, e):
        if not self.has_image():
            return
        if e.button() == Qt.MiddleButton or (self.tool == "pan" and e.button() == Qt.LeftButton):
            if self._pan_begin(e):  # hold the wheel button (or use the hand tool) and drag
                return
        if e.button() != Qt.LeftButton:
            return
        if self.layers:
            self._float_mouse(e, "press")
            return
        pt = self._ip(e)
        if self.tool == "callout" and self.co_stage == 2:  # second click = where the arrow points
            self.commit_pending()
            box = self.co_box
            self.co_stage, self.co_box = 0, None
            self._new_obj("callout", box, pt)
            return
        o = self._sel()
        if o is not None:  # grab a handle / the body of the selected text box or callout
            part = self._obj_hit(o, pt)
            if part:
                self.commit_pending()
                self._begin_obj_drag(o, part, pt)
                return
        if self.tool in ("text", "callout"):  # click on another one = select + move it
            o = self._obj_at(pt)
            if o is not None:
                self.commit_pending()
                self._begin_obj_drag(o, "move", pt)
                return
        self.sel_obj = None
        self.commit_pending()  # clicking elsewhere finishes (or removes, if empty) the open text box
        if self.tool == "callout":
            self.co_stage = 1
        self.a = self.b = pt
        if self.tool in ("pen", "highlighter"):
            self.path = QPainterPath(pt)
            self.path.lineTo(pt + QPointF(0.01, 0))
        elif self.tool == "eraser":
            self._push()
            self._erase(pt)
        self.update()

    def mouseMoveEvent(self, e):
        if self._pan:
            self._pan_move(e)
            return
        if self.layers:
            self._float_mouse(e, "move")
            return
        pt = self._ip(e)
        if self.obj_drag:
            d = self.obj_drag
            o, delta, part = d["o"], pt - d["pt"], d["part"]
            if not d["moved"]:
                d["moved"] = True
                self._push()  # one undo step for the whole drag
            mn = 24 * self._scale()
            if part == "move":
                o["box"] = d["box"].translated(delta)
                if o["tip"] is not None:
                    o["tip"] = d["tip"] + delta
            elif part == "tip":
                o["tip"] = d["tip"] + delta
            else:
                box0, nb = d["box"], QRectF(d["box"])
                if "l" in part:
                    nb.setLeft(min(box0.left() + delta.x(), box0.right() - mn))
                if "r" in part:
                    nb.setRight(max(box0.right() + delta.x(), box0.left() + mn))
                if "t" in part:
                    nb.setTop(min(box0.top() + delta.y(), box0.bottom() - mn))
                if "b" in part:
                    nb.setBottom(max(box0.bottom() + delta.y(), box0.top() + mn))
                o["box"] = nb
                if "t" in part or "b" in part:
                    o["minh"] = max(mn, nb.height())
                self._fit_obj(o)  # narrower box = text wraps = box grows
            self.update()
            return
        o = self._sel()
        if o is not None and not self.a and self.co_stage == 0:
            self._co_cursor(self._obj_hit(o, pt))
        if self.tool == "callout" and self.co_stage == 2:
            self.co_tip = pt
            self.update()
            return
        if not self.a:
            return
        self.b = pt
        if self.tool in ("pen", "highlighter") and self.path:
            self.path.lineTo(pt)
        elif self.tool == "eraser":
            self._erase(pt)
        self.update()

    def mouseDoubleClickEvent(self, e):
        """Double-click a text box / callout to edit its text again."""
        if e.button() == Qt.LeftButton and not self.layers:
            o = self._obj_at(self._ip(e))
            if o is not None and o is not self.editing:
                self.obj_drag = None
                self._edit_obj(o)
                return
        super().mouseDoubleClickEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan and e.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._pan_end()
            return
        if e.button() != Qt.LeftButton:
            return
        if self.obj_drag:
            self.obj_drag = None
            return
        if self.layers:
            self._float_mouse(e, "release")
            return
        if self.tool == "callout":
            if self.co_stage == 1 and self.a and self.b:
                box = QRectF(self.a, self.b).normalized()
                if box.width() > 12 and box.height() > 8:
                    self.co_box, self.co_stage, self.co_tip = box, 2, self._ip(e)
                else:
                    self.co_stage = 0
            self.a = self.b = None
            self.update()
            return
        if self.tool == "text" and self.a and self.b:
            a, b, sc = self.a, self.b, self._scale()
            self.a = self.b = None
            if (b - a).manhattanLength() > 14 * sc:  # dragged: that is the box
                self._new_obj("text", QRectF(a, b).normalized())
            else:  # plain click: default width, centred on the click
                self._new_obj("text", QRectF(a.x(), a.y(), 260 * sc, 0))
            return
        if not self.a:
            return
        if self.tool in ("pen", "highlighter") and self.path:
            self._push()
            p = QPainter(self.pix)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(self._pen())
            p.drawPath(self.path)
            p.end()
        elif self.tool in SHAPE_TOOLS:
            if (self.b - self.a).manhattanLength() > 2:
                # keep it "live": colour / fill / size can still change until you click elsewhere
                self.live = (self.tool, QPointF(self.a), QPointF(self.b))
        elif self.tool == "crop":
            r = QRectF(self.a, self.b).normalized().toRect().intersected(self.pix.rect())
            if r.width() > 5 and r.height() > 5:
                self._push()
                self.flatten_objs()
                self.pix, self.base = self.pix.copy(r), self.base.copy(r)
                self._relayout()
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


# ------------------------------------------------------------------ icons
ACCENT = {"add_img": "#0f6cbd", "text": "#0f6cbd", "rec": "#d13438", "pen": "#d13438", "highlighter": "#f2b800", "new": "#0f6cbd", "save": "#0f6cbd"}


def make_icon(name):
    """Draw a crisp vector icon (24x24 grid) so no image files are needed."""

    def render(col, acc, name):
        S = 96
        pm = QPixmap(S, S)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(S / 24, S / 24)
        if name == "redo":
            p.translate(24, 0)
            p.scale(-1, 1)
            name = "undo"
        pen = QPen(QColor(col), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        def L(x1, y1, x2, y2):
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        def poly(*pts):
            path = QPainterPath(QPointF(*pts[0]))
            for pt in pts[1:]:
                path.lineTo(QPointF(*pt))
            path.closeSubpath()
            p.drawPath(path)

        def dashed():
            d = QPen(pen)
            d.setStyle(Qt.CustomDashLine)
            d.setDashPattern([1.5, 1.3])
            d.setCapStyle(Qt.FlatCap)
            p.setPen(d)

        if name == "new":
            p.setPen(QPen(QColor(acc), 2.2, Qt.SolidLine, Qt.RoundCap))
            L(12, 4.5, 12, 19.5)
            L(4.5, 12, 19.5, 12)
        elif name == "m_rect":
            dashed()
            p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 1.5, 1.5)
        elif name == "m_free":
            dashed()
            path = QPainterPath(QPointF(4.5, 15.5))
            path.cubicTo(2.5, 7, 11, 3.5, 16, 6)
            path.cubicTo(22, 9, 18, 18, 12.5, 16.5)
            path.cubicTo(9.5, 15.5, 8, 18.5, 4.5, 15.5)
            p.drawPath(path)
        elif name == "m_full":
            p.drawRoundedRect(QRectF(3, 4.5, 18, 12), 1.5, 1.5)
            L(8, 20, 16, 20)
            L(12, 16.5, 12, 20)
        elif name == "clock":
            p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
            L(12, 7, 12, 12)
            L(12, 12, 15.5, 14)
        elif name == "undo":
            L(4.5, 12, 9.5, 7)
            L(4.5, 12, 9.5, 17)
            path = QPainterPath(QPointF(5, 12))
            path.cubicTo(10, 6.5, 19.5, 7, 20, 17.5)
            p.drawPath(path)
        elif name == "save":
            p.drawRoundedRect(QRectF(4, 4, 16, 16), 2, 2)
            p.drawRect(QRectF(8, 4, 8, 5))
            p.setPen(QPen(QColor(acc), 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawRect(QRectF(7.5, 13, 9, 7))
        elif name == "copy":
            p.drawRoundedRect(QRectF(9, 9, 11, 11), 2, 2)
            path = QPainterPath(QPointF(5, 15))
            path.lineTo(5, 6.5)
            path.quadTo(5, 5, 6.5, 5)
            path.lineTo(14.5, 5)
            p.drawPath(path)
        elif name == "cam":
            p.drawRoundedRect(QRectF(3, 7, 18, 12), 2, 2)
            path = QPainterPath(QPointF(8, 7))
            path.lineTo(9.5, 4.5)
            path.lineTo(14.5, 4.5)
            path.lineTo(16, 7)
            p.drawPath(path)
            p.drawEllipse(QPointF(12, 13), 3.4, 3.4)
        elif name == "video":
            p.drawRoundedRect(QRectF(3, 6.5, 13, 11), 2, 2)
            poly((16, 11), (21, 8), (21, 16), (16, 13))
        elif name == "rec":
            p.setPen(QPen(QColor(acc), 1.7))
            p.setBrush(QColor(acc))
            p.drawEllipse(QPointF(12, 12), 6.5, 6.5)
        elif name == "stop":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col))
            p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
        elif name == "pause":
            p.setPen(QPen(QColor(col), 3, Qt.SolidLine, Qt.RoundCap))
            L(9, 6.5, 9, 17.5)
            L(15, 6.5, 15, 17.5)
        elif name == "close":
            L(6, 6, 18, 18)
            L(18, 6, 6, 18)
        elif name == "more":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col))
            for x in (5, 12, 19):
                p.drawEllipse(QPointF(x, 12), 1.8, 1.8)
        elif name == "pen":
            poly((4, 20), (5.2, 15.2), (16, 4.5), (19.5, 8), (8.8, 18.8))
            L(13.5, 7, 17, 10.5)
            p.setBrush(QColor(acc))
            poly((4, 20), (5.2, 15.2), (8.8, 18.8))
        elif name == "highlighter":
            poly((8.5, 13.5), (15, 7), (19.5, 11.5), (13, 18))
            poly((8.5, 13.5), (13, 18), (9, 19), (6.5, 16.5))
            p.setPen(QPen(QColor(acc), 3, Qt.SolidLine, Qt.FlatCap))
            L(4, 21.5, 15, 21.5)
        elif name == "eraser":
            poly((3.5, 14.5), (12.5, 5.5), (19.5, 12.5), (12, 20), (8, 20))
            L(8, 10, 15, 17)
            L(12, 20, 21, 20)
        elif name == "t_rect":
            p.drawRoundedRect(QRectF(4, 6, 16, 12), 1.5, 1.5)
        elif name == "t_ellipse":
            p.drawEllipse(QRectF(4, 5, 16, 14))
        elif name == "t_line":
            L(5, 19, 19, 5)
        elif name == "t_arrow":
            L(5, 19, 18, 6)
            L(10, 6, 18, 6)
            L(18, 6, 18, 14)
        elif name == "text":
            p.setPen(QPen(QColor(acc), 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            L(5, 5.5, 19, 5.5)
            L(12, 5.5, 12, 19.5)
            L(8.5, 19.5, 15.5, 19.5)
        elif name == "add_img":
            p.drawRoundedRect(QRectF(3.5, 5, 12, 10), 1.5, 1.5)
            p.setPen(QPen(QColor(acc), 2.2, Qt.SolidLine, Qt.RoundCap))
            L(15, 17.5, 21, 17.5)
            L(18, 14.5, 18, 20.5)
        elif name == "callout":
            p.drawRect(QRectF(3.5, 3.5, 13, 8))
            path = QPainterPath(QPointF(10, 11.5))
            path.lineTo(10, 15.5)
            path.lineTo(19.5, 20.5)
            p.drawPath(path)
            ang = math.atan2(5, 9.5)
            for d in (-0.5, 0.5):
                L(19.5, 20.5, 19.5 - math.cos(ang + d) * 4.5, 20.5 - math.sin(ang + d) * 4.5)
        elif name == "hand":
            for x, top in ((8.5, 6.5), (11.5, 4.5), (14.5, 5), (17.5, 7.5)):
                L(x, top, x, 13)
            path = QPainterPath(QPointF(6.5, 13.5))
            path.lineTo(6.5, 15)
            path.cubicTo(6.5, 19.5, 9.5, 21, 12.5, 21)
            path.cubicTo(16, 21, 19, 19, 19, 15)
            path.lineTo(19, 11)
            p.drawPath(path)
            L(6.5, 13.5, 4.2, 10.5)
        elif name == "t_cloud":
            p.drawPath(cloud_path(QRectF(3.5, 6.5, 17, 11), 4.4))
        elif name == "crop":
            path = QPainterPath(QPointF(7, 3))
            path.lineTo(7, 17)
            path.lineTo(21, 17)
            p.drawPath(path)
            path = QPainterPath(QPointF(3, 7))
            path.lineTo(17, 7)
            path.lineTo(17, 21)
            p.drawPath(path)
        p.end()
        return pm

    ic = QIcon()
    dark = "#2b2b2b"
    ic.addPixmap(render(dark, ACCENT.get(name, dark), name), QIcon.Normal)
    ic.addPixmap(render("#b8b8b8", "#b8b8b8", name), QIcon.Disabled)
    return ic


MODE_ICON = {"rect": "m_rect", "freeform": "m_free", "full": "m_full"}


# -------------------------------------------------------------- recording
VIDEO_DIR = Path.home() / "Videos" / "SNP Editor"


def exclude_from_capture(w):
    """Hide a window from screen capture (Windows 10 2004+), so the record bar never shows in videos."""
    try:
        import ctypes
        ctypes.windll.user32.SetWindowDisplayAffinity(int(w.winId()), 0x11)
    except Exception:
        pass


class FfmpegWriter:
    """Pipes raw BGR frames into ffmpeg -> H.264 MP4 (plays in chat apps and browsers)."""

    def __init__(self, path, size, fps):
        w, h = size
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, creationflags=0x08000000)

    def isOpened(self):
        return self.p.poll() is None

    def write(self, frame):
        try:
            self.p.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass

    def release(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


class Recorder(QObject):
    tick = Signal(float)
    failed = Signal(str)

    def __init__(self, region, path, fps=15):
        super().__init__()
        self.path, self.fps = path, fps
        self.screen = QGuiApplication.screenAt(region.center()) or QGuiApplication.primaryScreen()
        sg = self.screen.geometry()
        self.local = region.intersected(sg).translated(-sg.topLeft())
        self.writer, self.size, self.frames = None, None, 0
        self.t0, self.paused, self.pause_t = 0.0, 0.0, None
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / fps))
        self.timer.timeout.connect(self._grab)

    def start(self):
        self.t0 = time.perf_counter()
        self.timer.start()
        self._grab()

    def elapsed(self):
        now = self.pause_t if self.pause_t is not None else time.perf_counter()
        return now - self.t0 - self.paused

    def pause(self):
        self.timer.stop()
        self.pause_t = time.perf_counter()

    def resume(self):
        if self.pause_t is not None:
            self.paused += time.perf_counter() - self.pause_t
            self.pause_t = None
        self.timer.start()

    def _grab(self):
        r = self.local
        pm = self.screen.grabWindow(0, r.x(), r.y(), r.width(), r.height())
        if pm.isNull():
            return
        img = pm.toImage().convertToFormat(QImage.Format_RGB32)  # memory order: B G R X
        w, h = img.width(), img.height()
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3]
        frame = np.ascontiguousarray(arr)
        if self.writer is None:
            s = min(1.0, 1920 / w, 1080 / h)
            self.size = (max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2))
            if HAVE_FFMPEG:
                try:
                    self.writer = FfmpegWriter(self.path, self.size, self.fps)
                except Exception:
                    self.writer = None
            if self.writer is None:  # fallback encoder
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                self.timer.stop()
                self.failed.emit("Could not start the video encoder.")
                return
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        t = self.elapsed()
        n = min(int(t * self.fps) + 1 - self.frames, self.fps * 2)  # keep real-time speed
        for _ in range(max(n, 0)):
            self.writer.write(frame)
            self.frames += 1
        self.tick.emit(t)

    def stop(self):
        self.timer.stop()
        if self.writer is not None:
            self.writer.release()
        return self.path, self.frames / self.fps


class RegionFrame(QWidget):
    """Red border drawn just outside the recorded area."""

    def __init__(self, rect):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setGeometry(rect.adjusted(-3, -3, 3, 3))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QPen(QColor("#e81123"), 3))
        p.drawRect(QRectF(1.5, 1.5, self.width() - 3, self.height() - 3))


class RecorderBar(QWidget):
    started = Signal()
    stop_req = Signal()
    pause_req = Signal(bool)
    cancel_req = Signal()

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setObjectName("bar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "QWidget#bar { background: #ffffff; border: 1px solid #9a9a9a; border-radius: 8px; }"
            "QLabel { font-size: 15px; font-weight: 600; color: #222; background: transparent; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(6)
        self.btn_start = self._tb("rec", "Start", self._countdown)
        self.btn_pause = self._tb("pause", "", lambda: None)
        self.btn_pause.setCheckable(True)
        self.btn_pause.setToolTip("Pause / resume")
        self.btn_pause.toggled.connect(self.pause_req)
        self.btn_stop = self._tb("stop", "Stop", self.stop_req.emit)
        self.lbl = QLabel("Ready")
        self.lbl.setMinimumWidth(70)
        self.lbl.setAlignment(Qt.AlignCenter)
        self.btn_cancel = self._tb("close", "", self.cancel_req.emit)
        self.btn_cancel.setToolTip("Cancel (Esc)")
        for w in (self.btn_start, self.btn_pause, self.btn_stop, self.lbl, self.btn_cancel):
            lay.addWidget(w)
        self.btn_pause.hide()
        self.btn_stop.hide()
        self.count = 0
        self._cd = QTimer(self)
        self._cd.timeout.connect(self._cd_tick)

    def _tb(self, icon, text, fn):
        b = QToolButton()
        b.setIcon(make_icon(icon))
        b.setIconSize(QSize(22, 22))
        b.setMinimumHeight(38)
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        else:
            b.setFixedSize(40, 38)
        b.clicked.connect(lambda _=False: fn())
        return b

    def _countdown(self):
        self.btn_start.hide()
        self.count = 3
        self.lbl.setText("Starting in 3")
        self._cd.start(1000)
        self.adjustSize()

    def _cd_tick(self):
        self.count -= 1
        if self.count <= 0:
            self._cd.stop()
            self.btn_pause.show()
            self.btn_stop.show()
            self.lbl.setText("0:00")
            self.adjustSize()
            self.started.emit()
        else:
            self.lbl.setText(f"Starting in {self.count}")

    def set_time(self, t):
        m, s = divmod(int(t), 60)
        self.lbl.setText(f"{m}:{s:02d}")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancel_req.emit()


def app_icon():
    """Blue rounded square with a dashed snip rectangle (window / tray icon)."""
    pm = QPixmap(128, 128)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#0f6cbd"))
    p.drawRoundedRect(8, 8, 112, 112, 26, 26)
    pen = QPen(QColor("white"), 9, Qt.CustomDashLine)
    pen.setDashPattern([2.0, 1.5])
    pen.setCapStyle(Qt.FlatCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawRect(34, 40, 60, 48)
    p.end()
    return QIcon(pm)


class HotkeyFilter(QAbstractNativeEventFilter):
    """Receives WM_HOTKEY messages (global shortcuts such as F12)."""

    def __init__(self, on_hotkey):
        super().__init__()
        self.on_hotkey = on_hotkey

    def nativeEventFilter(self, event_type, message):
        try:
            try:
                et = bytes(event_type)
            except Exception:
                et = str(event_type).encode()
            if et == b"windows_generic_MSG":
                import ctypes.wintypes as wt
                msg = wt.MSG.from_address(int(message))
                if msg.message == 0x0312:  # WM_HOTKEY
                    self.on_hotkey(int(msg.wParam))
                    return True, 0
        except Exception:
            pass
        return False, 0


PALETTE = ["#e81123", "#ff8c00", "#ffb900", "#107c10", "#00b7c3", "#0078d4",
           "#744da9", "#e3008c", "#000000", "#ffffff", "#767676", "#8e562e"]


def make_color_menu(parent, on_pick, allow_none=False):
    """Pop-up palette like Windows: swatches + (No fill) + More colors."""
    menu = QMenu(parent)
    if allow_none:
        act = menu.addAction("No fill")
        act.triggered.connect(lambda _=False: on_pick(None))
    w = QWidget()
    g = QGridLayout(w)
    g.setContentsMargins(8, 6, 8, 6)
    g.setSpacing(6)
    for i, c in enumerate(PALETTE):
        b = QToolButton()
        b.setFixedSize(28, 28)
        b.setStyleSheet(
            f"QToolButton {{ background: {c}; border: 1px solid #999; border-radius: 14px; padding: 0; }}"
            "QToolButton:hover { border: 2px solid #2a76c6; }")
        b.clicked.connect(lambda _=False, c=c: (on_pick(QColor(c)), menu.close()))
        g.addWidget(b, i // 6, i % 6)
    wa = QWidgetAction(menu)
    wa.setDefaultWidget(w)
    menu.addAction(wa)

    def custom():
        col = QColorDialog.getColor(QColor("#e81123"), parent, "Choose a color")
        if col.isValid():
            on_pick(col)

    menu.addAction("More colors…").triggered.connect(lambda _=False: custom())
    return menu


# ------------------------------------------------------------ main window
class SnipWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNP Editor   -   F12  =  quick snip")
        self.resize(1000, 640)
        self.mode, self.delay = "rect", 0
        self.capture, self.view = "photo", "none"
        self.overlay = self.rec = self.bar = self.frame = None
        self.adding = False
        self._quitting = self._tray_hint = False
        self.video_path = ""

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(8, 6, 8, 0)

        # ---- row 1: New / mode / delay ... undo redo save copy
        r1 = QHBoxLayout()
        r1.setSpacing(4)
        self.btn_new = self._btn("new", self.new_snip, "Ctrl+N", "New", "New snip")
        r1.addWidget(self.btn_new)
        self.btn_add = self._btn("add_img", self.add_snip, "Ctrl+Shift+N", "Add",
                                 "Add another snip next to this image")
        r1.addWidget(self.btn_add)

        self.cap_group = QButtonGroup(self)
        self.cap_group.setExclusive(True)
        for kind, icon, tip in (("photo", "cam", "Screenshot"), ("video", "video", "Screen recording")):
            cb = QToolButton()
            cb.setIcon(make_icon(icon))
            cb.setIconSize(QSize(24, 24))
            cb.setFixedSize(46, 42)
            cb.setCheckable(True)
            cb.setToolTip(tip)
            cb.clicked.connect(lambda _=False, k=kind: self._set_capture(k))
            self.cap_group.addButton(cb)
            r1.addWidget(cb)
            if kind == "photo":
                cb.setChecked(True)

        self.btn_mode = QToolButton()
        self.btn_mode.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_mode.setPopupMode(QToolButton.InstantPopup)
        self.btn_mode.setToolTip("Snipping mode")
        m = QMenu(self)
        for label, key in (("Rectangle", "rect"), ("Freeform", "freeform"), ("Full screen", "full")):
            act = m.addAction(label)
            act.setIcon(make_icon(MODE_ICON[key]))
            act.triggered.connect(lambda _=False, k=key, l=label: self._set_mode(k, l))
        self.btn_mode.setMenu(m)
        r1.addWidget(self.btn_mode)

        self.btn_delay = QToolButton()
        self.btn_delay.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_delay.setPopupMode(QToolButton.InstantPopup)
        self.btn_delay.setToolTip("Delay")
        self.btn_delay.setIcon(make_icon("clock"))
        dm = QMenu(self)
        for s in (0, 3, 5, 10):
            act = dm.addAction("No delay" if s == 0 else f"{s} seconds")
            act.triggered.connect(lambda _=False, s=s: self._set_delay(s))
        self.btn_delay.setMenu(dm)
        r1.addWidget(self.btn_delay)
        self.key_hint = QLabel("Quick snip:  press  F12")
        self.key_hint.setStyleSheet("color:#0f4c81; background:#e8f1fb; border-radius:5px; padding:4px 10px;"
                                    " font-size:13px; font-weight:600;")
        self.key_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        r1.addSpacing(10)
        r1.addWidget(self.key_hint, 1)
        r1.addStretch()

        self.btn_undo = self._btn("undo", lambda: self.canvas.undo(), "Ctrl+Z", tip="Undo")
        self.btn_redo = self._btn("redo", lambda: self.canvas.redo(), "Ctrl+Y", tip="Redo")
        self.btn_save = self._btn("save", self.save, "Ctrl+S", tip="Save")
        self.btn_copy = self._btn("copy", self.copy, "Ctrl+C", tip="Copy")
        self.btn_more = QToolButton()
        self.btn_more.setIcon(make_icon("more"))
        self.btn_more.setIconSize(QSize(24, 24))
        self.btn_more.setToolTip("See more")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        self.btn_more.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        mm = QMenu(self)
        mm.addAction("About", lambda: QMessageBox.about(self, "About", "SNP Editor\nBuilt with PySide6"))
        mm.addAction("Exit", self.quit_app)
        self.btn_more.setMenu(mm)
        for b in (self.btn_undo, self.btn_redo, self.btn_save, self.btn_copy, self.btn_more):
            r1.addWidget(b)
        v.addLayout(r1)

        # ---- row 2: drawing tools
        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addStretch()
        self.canvas = Canvas()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        tools = [("pen", "pen", "Ballpoint pen"), ("highlighter", "highlighter", "Highlighter"),
                 ("eraser", "eraser", "Eraser"), ("t_rect", "rect", "Rectangle"),
                 ("t_ellipse", "ellipse", "Ellipse"), ("t_line", "line", "Line"),
                 ("t_arrow", "arrow", "Arrow"), ("callout", "callout", "Callout"),
                 ("t_cloud", "cloud", "Cloud"), ("text", "text", "Text"), ("crop", "crop", "Crop"),
                 ("hand", "pan", "Pan - or hold the mouse wheel button and drag")]
        for icon, key, tip in tools:
            b = QToolButton()
            b.setIcon(make_icon(icon))
            b.setIconSize(QSize(26, 26))
            b.setFixedSize(46, 42)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.canvas.set_tool(k))
            self.group.addButton(b)
            r2.addWidget(b)
            if key == "pen":
                b.setChecked(True)
        sep = QFrame(objectName="sep")
        sep.setFixedSize(1, 28)
        r2.addWidget(sep)
        self.btn_fill = QToolButton()
        self.btn_fill.setText("Fill")
        self.btn_fill.setToolTip("Fill color (rectangle, ellipse, cloud, callout)")
        self.btn_fill.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_fill.setPopupMode(QToolButton.InstantPopup)
        self.btn_fill.setIconSize(QSize(22, 22))
        self.btn_fill.setMinimumHeight(42)
        self.btn_fill.setMenu(make_color_menu(self, self._pick_fill, allow_none=True))
        self._paint_fill_btn()
        r2.addWidget(self.btn_fill)

        self.btn_color = QToolButton()
        self.btn_color.setText("Outline")
        self.btn_color.setToolTip("Outline / pen color")
        self.btn_color.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_color.setPopupMode(QToolButton.InstantPopup)
        self.btn_color.setIconSize(QSize(22, 22))
        self.btn_color.setMinimumHeight(42)
        self.btn_color.setMenu(make_color_menu(self, self._pick_outline))
        self._paint_color_btn()
        r2.addWidget(self.btn_color)
        self.size_box = QComboBox()
        self.size_box.setMinimumHeight(34)
        for s in (2, 4, 6, 10, 16):
            self.size_box.addItem(f"{s} px", s)
        self.size_box.setCurrentIndex(1)
        self.size_box.currentIndexChanged.connect(lambda: self.canvas.set_width(self.size_box.currentData()))
        r2.addWidget(self.size_box)
        r2.addStretch()
        r2.setContentsMargins(0, 0, 0, 0)
        self.tools_bar = QWidget()
        self.tools_bar.setLayout(r2)
        v.addWidget(self.tools_bar)

        self._set_mode("rect", "Rectangle")
        self._set_delay(0)

        # ---- canvas area
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.canvas)
        self.canvas.scroll = self.scroll
        self.scroll.viewport().installEventFilter(self)
        self.hint = QLabel("Press  New  (Ctrl+N)  or  F12  to take a snip")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color:#777; font-size:14px; background:white;")
        self.float_hint = QLabel("Drag a picture to move it (it snaps into line)  -  drag the blue corner grips to resize  -  press Add for more  -  "
                                 "Delete removes the selected one  -  click outside to apply")
        self.float_hint.setStyleSheet("background:#e8f1fb; color:#0f4c81; padding:5px 8px; font-size:13px;")
        self.float_hint.setAlignment(Qt.AlignCenter)
        self.float_hint.hide()
        v.addWidget(self.float_hint)
        v.addWidget(self.hint, 1)
        v.addWidget(self.scroll, 1)
        self.vpanel = QWidget(objectName="vpanel")
        self.vpanel.setStyleSheet("QWidget#vpanel { background: white; }")
        vp = QVBoxLayout(self.vpanel)
        vp.addStretch()
        self.vinfo = QLabel()
        self.vinfo.setAlignment(Qt.AlignCenter)
        self.vinfo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vp.addWidget(self.vinfo)
        vb = QHBoxLayout()
        vb.addStretch()
        for text, fn in (("▶  Play", self.play_video), ("Show in folder", self.show_in_folder),
                         ("Save as…", self.save), ("Delete", self.delete_video)):
            pb = QPushButton(text)
            pb.setMinimumHeight(36)
            pb.setMinimumWidth(110)
            pb.clicked.connect(lambda _=False, f=fn: f())
            vb.addWidget(pb)
        vb.addStretch()
        vp.addLayout(vb)
        vp.addStretch()
        v.addWidget(self.vpanel, 1)
        self.vpanel.hide()
        self.scroll.hide()
        self.status_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.status_lbl)
        self.canvas.zoom_changed.connect(lambda _z: self._update_status())
        for keys, fn in (("Ctrl+0", lambda: self.canvas.set_zoom(1.0)),
                         ("Ctrl+=", lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25)),
                         ("Ctrl++", lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25)),
                         ("Ctrl+-", lambda: self.canvas.set_zoom(self.canvas.zoom / 1.25)),
                         ("Ctrl+9", lambda: self.canvas.fit_zoom())):
            QShortcut(QKeySequence(keys), self, activated=fn)
        self.canvas.changed.connect(self._refresh)
        self._refresh()
        self._init_tray()
        self._init_hotkeys()

    def _btn(self, icon, fn, shortcut=None, text="", tip=""):
        b = QToolButton()
        b.setIcon(make_icon(icon))
        b.setIconSize(QSize(24, 24))
        b.setMinimumHeight(42)
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        else:
            b.setFixedSize(46, 42)
        b.setToolTip(f"{tip or text}  ({shortcut})" if shortcut else (tip or text))
        b.clicked.connect(lambda _=False: fn())
        if shortcut:
            QShortcut(QKeySequence(shortcut), self, activated=fn)
        return b

    def _paint_color_btn(self):
        pm = QPixmap(44, 44)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(self.canvas.color, 7))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(7, 7, 30, 30)
        p.end()
        self.btn_color.setIcon(QIcon(pm))

    def _paint_fill_btn(self):
        pm = QPixmap(44, 44)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        if self.canvas.fill is None:
            p.setPen(QPen(QColor("#555"), 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(5, 5, 34, 34)
            p.drawLine(11, 33, 33, 11)
        else:
            p.setPen(QPen(QColor("#888"), 2))
            p.setBrush(self.canvas.fill)
            p.drawEllipse(5, 5, 34, 34)
        p.end()
        self.btn_fill.setIcon(QIcon(pm))

    def _pick_outline(self, c):
        self.canvas.set_color(c)
        self._paint_color_btn()

    def _pick_fill(self, c):
        self.canvas.set_fill(c)
        self._paint_fill_btn()

    def _refresh(self):
        img = self.view == "image"
        self.scroll.setVisible(img)
        self.hint.setVisible(self.view == "none")
        self.vpanel.setVisible(self.view == "video")
        self.btn_undo.setEnabled(img and bool(self.canvas.undo_stack))
        self.btn_redo.setEnabled(img and bool(self.canvas.redo_stack))
        self.btn_save.setEnabled(self.view != "none")
        self.btn_copy.setEnabled(img or self.view == "video")
        self.btn_add.setEnabled(img)
        self.float_hint.setVisible(img and bool(self.canvas.layers))
        self._update_status()

    def _update_status(self):
        if self.view == "image":
            vr = self.canvas.view_rect
            self.status_lbl.setText(
                f"{int(vr.width())} x {int(vr.height())} px      Zoom {round(self.canvas.zoom * 100)}%      "
                "Ctrl + wheel = zoom   |   hold wheel button and drag = pan   |   Ctrl+0 = 100%   |   Ctrl+9 = fit")
        else:
            self.status_lbl.setText("")

    def eventFilter(self, obj, ev):
        """Ctrl + mouse wheel over the picture area = zoom in / out."""
        if (obj is self.scroll.viewport() and ev.type() == QEvent.Wheel and self.view == "image"
                and (ev.modifiers() & Qt.ControlModifier)):
            steps = ev.angleDelta().y() / 120
            anchor = QPointF(self.canvas.mapFromGlobal(ev.globalPosition().toPoint()))
            self.canvas.set_zoom(self.canvas.zoom * (1.1 ** steps), anchor)
            return True
        return super().eventFilter(obj, ev)

    def _set_mode(self, key, label):
        self.mode = key
        self.btn_mode.setText(label)
        self.btn_mode.setIcon(make_icon(MODE_ICON[key]))
        self.btn_mode.setIconSize(QSize(24, 24))

    def _set_delay(self, s):
        self.delay = s
        self.btn_delay.setText("No delay" if s == 0 else f"{s} seconds")
        self.btn_delay.setIconSize(QSize(24, 24))

    # ---- capture flow
    def _set_capture(self, kind):
        self.capture = kind
        photo = kind == "photo"
        self.btn_mode.setVisible(photo)
        self.btn_add.setVisible(photo)
        self.btn_delay.setVisible(photo)
        self.tools_bar.setVisible(photo)

    def add_snip(self):
        """Take another snip and place it next to the current image."""
        self.new_snip(add=self.view == "image")

    def new_snip(self, add=False):
        if self.rec_active():
            return
        self.adding = add
        if self.capture == "video" and not HAVE_CV:
            QMessageBox.warning(self, "Recording unavailable",
                                "OpenCV is missing, so screen recording can't be used.")
            return
        delay = self.delay if self.capture == "photo" else 0
        self.hide()
        QTimer.singleShot(delay * 1000 + 300, self._capture)

    def _capture(self):
        shot, vgeo = grab_all()
        if self.capture == "video":
            self.overlay = SnipOverlay(shot, vgeo, "rect", video=True)
            self.overlay.region.connect(self._region_selected)
            self.overlay.done.connect(self._on_done)  # emitted only on cancel
            return
        if self.mode == "full":
            self._on_done(shot)
            return
        self.overlay = SnipOverlay(shot, vgeo, self.mode)
        self.overlay.done.connect(self._on_done)

    def _show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_done(self, pm):
        adding, self.adding = self.adding, False
        self._show_window()
        if pm is not None:
            if adding and self.view == "image":
                self.canvas.add_float(pm)
            else:
                self.view = "image"
                self.canvas.set_image(pm)
                QGuiApplication.clipboard().setPixmap(pm)
            vr = self.canvas.view_rect
            self.resize(max(self.width(), min(int(vr.width()) + 40, 1400)),
                        max(self.height(), min(int(vr.height()) + 190, 900)))
        self._refresh()

    # ---- tray + global hotkeys (F12)
    def _init_tray(self):
        self.tray = QSystemTrayIcon(app_icon(), self)
        self.tray.setToolTip("SNP Editor - press F12 to take a snip")
        tm = QMenu(self)
        tm.addAction("New snip", lambda: self.new_snip())
        tm.addAction("Open SNP Editor", self._show_window)
        self.act_hotkey = tm.addAction("F12 quick snip (on / off)")
        self.act_hotkey.setCheckable(True)
        self.act_hotkey.setChecked(True)
        self.act_hotkey.toggled.connect(self._set_hotkey_enabled)
        tm.addSeparator()
        tm.addAction("Exit", self.quit_app)
        self.tray.setContextMenu(tm)
        self.tray.activated.connect(self._tray_clicked)
        self.tray.show()

    def _tray_clicked(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self._show_window()

    def _init_hotkeys(self):
        self._hk_ids, self._hook, self._hook_proc = [], None, None
        self._hotkey_enabled, self._f12_down, self._f12_registered = True, False, False
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        NOREPEAT = 0x4000
        if user32.RegisterHotKey(None, 1, NOREPEAT, 0x2C):  # PrtSc: bonus, only if Windows isn't using it
            self._hk_ids.append(1)
        self._register_f12()
        self._hk_filter = HotkeyFilter(self._on_hotkey)
        QApplication.instance().installNativeEventFilter(self._hk_filter)
        QApplication.instance().aboutToQuit.connect(self._unregister_hotkeys)

    def _register_f12(self):
        """F12 is 'reserved for debuggers', so Windows may refuse RegisterHotKey: then use a keyboard hook."""
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        if user32.RegisterHotKey(None, 2, 0x4000, 0x7B):
            self._f12_registered = True
            return
        kernel32 = ctypes.windll.kernel32
        LRESULT = ctypes.c_ssize_t
        HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

        class KBD(ctypes.Structure):
            _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HMODULE, wintypes.DWORD]
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallNextHookEx.restype = LRESULT
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE

        def proc(n_code, w_param, l_param):
            try:
                if n_code == 0 and self._hotkey_enabled:
                    kb = KBD.from_address(l_param)
                    if kb.vkCode == 0x7B and not (kb.flags & 0x10):  # F12, not injected
                        if w_param in (0x100, 0x104):  # key down
                            if not self._f12_down:  # ignore auto-repeat
                                self._f12_down = True
                                QTimer.singleShot(0, self._hotkey_snip)
                        else:
                            self._f12_down = False
                        return 1  # swallow F12 so other programs don't react to it
            except Exception:
                pass
            return user32.CallNextHookEx(None, n_code, w_param, l_param)

        self._hook_proc = HOOKPROC(proc)  # keep a reference or it gets garbage collected
        self._hook = user32.SetWindowsHookExW(13, self._hook_proc, kernel32.GetModuleHandleW(None), 0)

    def _set_hotkey_enabled(self, on):
        """Tray switch: F12 back to normal for other programs (e.g. browser dev tools)."""
        self._hotkey_enabled = bool(on)
        if sys.platform == "win32" and self._f12_registered:
            import ctypes
            if on:
                ctypes.windll.user32.RegisterHotKey(None, 2, 0x4000, 0x7B)
            else:
                ctypes.windll.user32.UnregisterHotKey(None, 2)

    def _unregister_hotkeys(self):
        if sys.platform != "win32":
            return
        import ctypes
        for hid in self._hk_ids + ([2] if self._f12_registered else []):
            ctypes.windll.user32.UnregisterHotKey(None, hid)
        if self._hook:
            ctypes.windll.user32.UnhookWindowsHookEx(self._hook)
            self._hook = None

    def _on_hotkey(self, _hid):
        QTimer.singleShot(0, self._hotkey_snip)  # leave the native message handler first

    def _hotkey_snip(self):
        if self.rec_active() or (self.overlay is not None and self.overlay.isVisible()):
            return
        self.new_snip()

    def closeEvent(self, e):
        if self._quitting or not self.tray.isVisible():
            e.accept()
            return
        e.ignore()  # keep running in the tray so F12 still works
        self.hide()
        if not self._tray_hint:
            self._tray_hint = True
            self.tray.showMessage("SNP Editor is still running",
                                  "Press F12 to take a snip. Right-click the tray icon to exit.",
                                  QSystemTrayIcon.Information, 4000)

    def quit_app(self):
        self._quitting = True
        QApplication.quit()

    # ---- video recording
    def rec_active(self):
        return self.bar is not None

    def _region_selected(self, rect):
        self.rec_rect = rect
        self.frame = RegionFrame(rect)
        self.bar = RecorderBar()
        self.bar.started.connect(self._begin_recording)
        self.bar.stop_req.connect(self._stop_recording)
        self.bar.pause_req.connect(self._pause_recording)
        self.bar.cancel_req.connect(self._cancel_recording)
        self.frame.show()
        self.bar.show()
        screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
        sg = screen.geometry()
        self.bar.adjustSize()
        bw, bh = self.bar.width(), self.bar.height()
        x = min(max(rect.center().x() - bw // 2, sg.left() + 4), sg.right() - bw - 4)
        if rect.bottom() + 12 + bh < sg.bottom():
            y = rect.bottom() + 12
        elif rect.top() - 12 - bh > sg.top():
            y = rect.top() - 12 - bh
        else:
            y = rect.top() + 12
        self.bar.move(x, y)
        exclude_from_capture(self.frame)
        exclude_from_capture(self.bar)
        self.bar.activateWindow()
        self.bar.setFocus()

    def _begin_recording(self):
        try:
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            folder = VIDEO_DIR
        except OSError:
            folder = Path(tempfile.gettempdir())
        path = folder / f"Recording {datetime.now():%Y-%m-%d %H%M%S}.mp4"
        self.rec = Recorder(self.rec_rect, str(path))
        self.rec.tick.connect(self.bar.set_time)
        self.rec.failed.connect(self._rec_failed)
        self.rec.start()

    def _pause_recording(self, paused):
        if self.rec:
            self.rec.pause() if paused else self.rec.resume()

    def _close_rec_ui(self):
        for w in (self.bar, self.frame):
            if w is not None:
                w.close()
                w.deleteLater()
        self.bar = self.frame = None

    def _stop_recording(self):
        if not self.rec:
            return
        path, dur = self.rec.stop()
        self.rec = None
        self._close_rec_ui()
        self._show_window()
        if dur < 0.3 or not os.path.exists(path):
            QMessageBox.information(self, "Recording", "The recording was too short to save.")
            return
        self._show_video(path, dur)

    def _cancel_recording(self):
        if self.rec:
            path, _ = self.rec.stop()
            self.rec = None
            try:
                os.remove(path)
            except OSError:
                pass
        if self.bar is not None:
            self.bar._cd.stop()
        self._close_rec_ui()
        self._show_window()

    def _rec_failed(self, msg):
        self._cancel_recording()
        QMessageBox.warning(self, "Recording failed", msg)

    def _show_video(self, path, dur):
        self.video_path = path
        self.view = "video"
        m, s = divmod(int(dur), 60)
        mb = os.path.getsize(path) / 1048576
        self.vinfo.setText(
            "<div style='font-size:22px'><b>Recording saved</b></div>"
            f"<div style='color:#555;font-size:15px'>{m}:{s:02d} &nbsp;•&nbsp; {mb:.1f} MB</div>"
            f"<div style='color:#888;font-size:12px'>{path}</div>"
            "<div style='color:#0f6cbd;font-size:14px;margin-top:10px'>Copied to clipboard - press Ctrl+V in your chat to send it</div>")
        self._copy_video()
        self._refresh()

    def _copy_video(self):
        """Put the video FILE on the clipboard (like copying it in Explorer)."""
        md = QMimeData()
        md.setUrls([QUrl.fromLocalFile(self.video_path)])
        QGuiApplication.clipboard().setMimeData(md)

    def play_video(self):
        if hasattr(os, "startfile"):
            os.startfile(self.video_path)

    def show_in_folder(self):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(self.video_path)])

    def delete_video(self):
        try:
            os.remove(self.video_path)
        except OSError:
            pass
        self.view = "none"
        self._refresh()

    def save(self):
        if self.view == "video":
            name = os.path.basename(self.video_path)
            path, _ = QFileDialog.getSaveFileName(self, "Save recording as", name, "MP4 video (*.mp4)")
            if path:
                shutil.copy(self.video_path, path)
            return
        if self.view != "image":
            return
        self.canvas.commit_pending()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save as", "Screenshot.png", "PNG (*.png);;JPEG (*.jpg);;Bitmap (*.bmp)")
        if path:
            self.canvas.export_pixmap().save(path, None, 95)

    def copy(self):
        if self.view == "video":
            self._copy_video()
        elif self.canvas.has_image():
            QGuiApplication.clipboard().setPixmap(self.canvas.export_pixmap())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(app_icon())
    w = SnipWindow()
    w.show()
    sys.exit(app.exec())

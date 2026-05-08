"""
animated_buttons.py — Animated icon buttons for the Home Screen.

Each button contains a small QPainter-driven animation canvas on the left
and a text label on the right, all inside a QPushButton so existing
stylesheet theming (hover, press, disabled) works automatically.

Animations are driven by an external QTimer calling canvas.advance().
"""

import math
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QLabel, QWidget
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath


# ─── Base Canvas ────────────────────────────────────────────────────────────────

class AnimationCanvas(QWidget):
    """Base class for small animated icons drawn via QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.phase = 0.0
        self.setFixedSize(54, 54)

    def advance(self):
        self.phase += 0.05
        self.update()

    def _accent(self):
        return self.palette().highlight().color()

    def _fg(self):
        return self.palette().buttonText().color()

    def _dim(self, alpha=70):
        c = QColor(self._fg())
        c.setAlpha(alpha)
        return c


# ─── 1. EEG Signal Plotter — scrolling sine waves ──────────────────────────────

class EegWaveCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        configs = [
            (self._accent(), 2.5, 0.22, 0.0),
            (self._fg(),     3.8, 0.15, 0.8),
            (self._dim(90),  1.7, 0.13, 1.6),
        ]

        for ch, (color, freq, amp, offset) in enumerate(configs):
            p.setPen(QPen(color, 1.5))
            cy = h * (0.25 + ch * 0.25)
            path = QPainterPath()
            for x in range(w):
                t = x / w * math.pi * freq * 2 - self.phase * 3 + offset
                y = cy + math.sin(t) * h * amp
                if x == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)
        p.end()


# ─── 2. Emotion Pipeline — connected nodes with traveling pulse ─────────────────

class PipelineCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        nodes = [
            QPointF(7, h / 2),
            QPointF(w * 0.33, h * 0.25),
            QPointF(w * 0.66, h * 0.75),
            QPointF(w - 7, h / 2),
        ]

        # Connections
        p.setPen(QPen(self._dim(50), 1.5))
        for i in range(len(nodes) - 1):
            p.drawLine(nodes[i], nodes[i + 1])

        accent = self._accent()
        pulse_pos = (self.phase * 1.2) % len(nodes)
        pulse_idx = int(pulse_pos)
        pulse_frac = pulse_pos - pulse_idx

        # Static nodes
        for i, node in enumerate(nodes):
            dist = abs(i - pulse_pos)
            if dist < 1.0:
                glow = 1.0 - dist
                r = 4.5 + glow * 2.5
                c = QColor(accent)
                c.setAlpha(int(120 + glow * 135))
            else:
                r = 3.5
                c = self._fg()
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(node, r, r)

        # Traveling pulse dot
        if pulse_idx < len(nodes) - 1:
            a, b = nodes[pulse_idx], nodes[pulse_idx + 1]
            px = a.x() + (b.x() - a.x()) * pulse_frac
            py = a.y() + (b.y() - a.y()) * pulse_frac
            gc = QColor(accent)
            gc.setAlpha(200)
            p.setBrush(gc)
            p.drawEllipse(QPointF(px, py), 3, 3)
        p.end()


# ─── 3. Music Player — bouncing equalizer bars ─────────────────────────────────

class MusicBarsCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        accent = self._accent()
        n_bars = 5
        bar_w = w / (n_bars * 2 + 1)
        max_h = h * 0.72

        for i in range(n_bars):
            t = self.phase * 2.5 + i * 0.7
            bar_h = max_h * (0.25 + 0.75 * abs(math.sin(t)))
            x = bar_w + i * bar_w * 2
            y = h - 4 - bar_h

            c = QColor(accent)
            hue = (c.hue() + i * 15) % 360
            c.setHsv(hue, c.saturation(), c.value(), 200)
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 2, 2)
        p.end()


# ─── 4. Real-Time Classifier — V-A circumplex with orbiting dots ───────────────

class CircumplexCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.35

        # Crosshair axes
        dim = self._dim(40)
        p.setPen(QPen(dim, 1))
        p.drawLine(QPointF(cx - radius - 2, cy), QPointF(cx + radius + 2, cy))
        p.drawLine(QPointF(cx, cy - radius - 2), QPointF(cx, cy + radius + 2))

        # Circle outline
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Orbiting emotion dots
        dots = [
            (QColor("#a6e3a1"), 0.0, 0.75),   # Happy — green
            (QColor("#89b4fa"), 2.1, 0.60),   # Sad — blue
            (QColor("#f38ba8"), 1.2, 0.70),   # Fear — red
            (QColor("#f9e2af"), 3.3, 0.55),   # Neutral — yellow
        ]
        for color, offset, orbit_r in dots:
            angle = self.phase * 0.6 + offset
            dx = math.cos(angle) * radius * orbit_r
            dy = math.sin(angle) * radius * orbit_r
            color.setAlpha(210)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(cx + dx, cy + dy), 4, 4)
        p.end()


# ─── 5. Headset Simulator — head outline with blinking electrodes ──────────────

class HeadsetCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        fg = self._fg()
        accent = self._accent()
        head_w, head_h = w * 0.55, h * 0.65

        # Head oval
        p.setPen(QPen(fg, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy + 2), head_w / 2, head_h / 2)

        # Headband arc
        band_rect = QRectF(cx - head_w / 2 - 4, cy - head_h / 2 - 2,
                           head_w + 8, head_h * 0.6)
        p.drawArc(band_rect, 30 * 16, 120 * 16)

        # Electrode positions (relative offsets from head center)
        electrodes = [
            (-0.35, -0.20), (0.35, -0.20),   # frontal
            (-0.40,  0.15), (0.40,  0.15),    # temporal
            ( 0.00, -0.35),                    # central
            ( 0.00,  0.30),                    # parietal
        ]
        for i, (ex, ey) in enumerate(electrodes):
            t = self.phase * 2.0 + i * 1.0
            brightness = 0.3 + 0.7 * max(0, math.sin(t))
            c = QColor(accent)
            c.setAlpha(int(brightness * 255))
            r = 2.5 + brightness * 1.5
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(QPointF(cx + ex * head_w, cy + 2 + ey * head_h), r, r)
        p.end()


# ─── 6. About — pulsing info circle ────────────────────────────────────────────

class AboutCanvas(AnimationCanvas):

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        accent = self._accent()

        # Pulsing outer ring
        pulse = 0.5 + 0.5 * math.sin(self.phase * 1.5)
        ring_r = 14 + pulse * 4
        rc = QColor(accent)
        rc.setAlpha(int(60 + pulse * 80))
        p.setPen(QPen(rc, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)

        # Solid inner circle
        ic = QColor(accent)
        ic.setAlpha(180)
        p.setPen(Qt.NoPen)
        p.setBrush(ic)
        p.drawEllipse(QPointF(cx, cy), 10, 10)

        # "i" letter
        p.setPen(QPen(self.palette().highlightedText().color(), 2))
        font = p.font()
        font.setPixelSize(13)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(cx - 6, cy - 7, 12, 14), Qt.AlignCenter, "i")
        p.end()


# ─── Composite Button Widget ───────────────────────────────────────────────────

class AnimatedIconButton(QPushButton):
    """
    A QPushButton with an animated canvas on the left and a text label.
    Inherits all existing QPushButton stylesheet theming automatically.
    """

    def __init__(self, text, canvas_cls, parent=None):
        super().__init__("", parent)   # empty native text
        self.setMinimumSize(400, 80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(16)

        self.canvas = canvas_cls(self)

        self._label = QLabel(text, self)
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._label.setStyleSheet("background: transparent;")
        font = self._label.font()
        font.setPointSize(14)
        self._label.setFont(font)

        layout.addWidget(self.canvas)
        layout.addWidget(self._label, 1)

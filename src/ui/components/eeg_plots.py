import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPainterPath, QPolygonF
from src.eeg_pipeline.zscore_tracker import ZSCORE_CALIBRATION_TIME


# ──────────────────────────────────────────────────────
# Custom QPainter Plot Widgets
# ──────────────────────────────────────────────────────

class EegPlotWidget(QWidget):
    """Custom QPainter widget for rendering EEG signal waveforms."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels = []       # list of (label, np.array of amplitudes)
        self.time_axis = None    # np.array of time values in seconds
        self.title = "EEG Signal"
        
        # Appearance
        self.bg_color = QColor("#1e1e2e")
        self.grid_color = QColor("#333355")
        self.axis_color = QColor("#888899")
        self.label_color = QColor("#ccccdd")
        self.channel_colors = [
            QColor("#00FFB2"), QColor("#00AAFF"), QColor("#FF6B9D"),
            QColor("#FFD93D"), QColor("#C084FC"), QColor("#FF8C42"),
            QColor("#6EE7B7"), QColor("#67E8F9"), QColor("#FCA5A5"),
            QColor("#A3E635"), QColor("#E879F9"), QColor("#FB923C"),
        ]
        
        self.margin_left = 70
        self.margin_right = 15
        self.margin_top = 35
        self.margin_bottom = 30
        
        self.setMinimumHeight(280)
    
    def set_data(self, channels, time_axis, title="EEG Signal"):
        """channels: list of (label_str, amplitude_array)"""
        self.channels = channels
        self.time_axis = time_axis
        self.title = title
        self.update()
    
    def clear_data(self):
        self.channels = []
        self.time_axis = None
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        plot_x = self.margin_left
        plot_y = self.margin_top
        plot_w = w - self.margin_left - self.margin_right
        plot_h = h - self.margin_top - self.margin_bottom
        
        if plot_w <= 0 or plot_h <= 0:
            return
        
        # Title
        painter.setPen(QPen(self.label_color))
        title_font = QFont("Sans", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(QRectF(plot_x, 2, plot_w, self.margin_top - 4),
                         Qt.AlignCenter | Qt.AlignVCenter, self.title)
        
        # Plot border
        painter.setPen(QPen(self.grid_color, 1))
        painter.drawRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        if not self.channels or self.time_axis is None or len(self.time_axis) == 0:
            painter.setPen(QPen(self.axis_color))
            painter.setFont(QFont("Sans", 10))
            painter.drawText(QRectF(plot_x, plot_y, plot_w, plot_h),
                             Qt.AlignCenter, "No data loaded")
            return
        
        t_min = float(self.time_axis[0])
        t_max = float(self.time_axis[-1])
        t_range = t_max - t_min if t_max > t_min else 1.0
        
        # Compute global data range across all channels
        all_min = float('inf')
        all_max = float('-inf')
        for _, data in self.channels:
            all_min = min(all_min, float(np.min(data)))
            all_max = max(all_max, float(np.max(data)))
        data_range = all_max - all_min if all_max > all_min else 1.0
        padding = data_range * 0.05
        all_min -= padding
        all_max += padding
        data_range = all_max - all_min
        
        # Grid lines & axis labels
        label_font = QFont("Sans", 8)
        painter.setFont(label_font)
        
        # Horizontal grid (5 lines)
        painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        for i in range(6):
            frac = i / 5.0
            # Clinical EEG style: Negative values (all_min) plotted at the TOP
            y = plot_y + (frac * plot_h)
            painter.drawLine(QPointF(plot_x, y), QPointF(plot_x + plot_w, y))
            val = all_min + frac * data_range
            painter.setPen(QPen(self.axis_color))
            painter.drawText(QRectF(0, y - 8, self.margin_left - 5, 16),
                             Qt.AlignRight | Qt.AlignVCenter, f"{val:.0f}")
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        # Vertical grid (time ticks)
        num_ticks = min(10, max(4, int(t_range)))
        tick_step = t_range / num_ticks
        painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        for i in range(num_ticks + 1):
            t = t_min + i * tick_step
            x = plot_x + ((t - t_min) / t_range) * plot_w
            painter.drawLine(QPointF(x, plot_y), QPointF(x, plot_y + plot_h))
            painter.setPen(QPen(self.axis_color))
            painter.drawText(QRectF(x - 25, plot_y + plot_h + 2, 50, 20),
                             Qt.AlignCenter, f"{t:.1f}s")
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        # Draw waveforms
        painter.setClipRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        # Downsample for performance if there are too many points
        max_points = plot_w * 2  # 2 points per pixel max
        
        for ch_i, (label, data) in enumerate(self.channels):
            color = self.channel_colors[ch_i % len(self.channel_colors)]
            painter.setPen(QPen(color, 1.5))
            
            n = len(data)
            step = max(1, int(n / max_points))
            
            path = QPainterPath()
            first = True
            for j in range(0, n, step):
                t = float(self.time_axis[j])
                v = float(data[j])
                x = plot_x + ((t - t_min) / t_range) * plot_w
                # Clinical EEG style: Invert Y-axis (Negative UP)
                y = plot_y + ((v - all_min) / data_range) * plot_h
                if first:
                    path.moveTo(x, y)
                    first = False
                else:
                    path.lineTo(x, y)
            
            painter.drawPath(path)
        
        painter.setClipping(False)
        
        # Y-axis label
        painter.setPen(QPen(self.label_color))
        painter.setFont(QFont("Sans", 9))
        painter.save()
        painter.translate(12, plot_y + plot_h / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_h/2, -10, plot_h, 20), Qt.AlignCenter, "Amplitude (µV)")
        painter.restore()


class EmotionPlotWidget(QWidget):
    """Custom QPainter widget for rendering emotion probability curves + playhead."""
    
    EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
    EMOTION_COLORS = [
        QColor("#67E8F9"),   # Cyan - Neutral
        QColor("#818CF8"),   # Indigo - Sad
        QColor("#FCA5A5"),   # Red - Fear
        QColor("#FDE047"),   # Yellow - Happy
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.probs = None        # shape (T, 4)
        self.time_axis = None    # np.array of time in seconds
        self.view_start = 0.0
        self.view_end = 1.0
        self.playhead_time = -1.0  # negative = hidden
        
        # Appearance
        self.bg_color = QColor("#1e1e2e")
        self.grid_color = QColor("#333355")
        self.axis_color = QColor("#888899")
        self.label_color = QColor("#ccccdd")
        self.playhead_color = QColor("#ff0055")
        
        self.margin_left = 70
        self.margin_right = 15
        self.margin_top = 35
        self.margin_bottom = 30
        
        self.setMinimumHeight(280)
    
    def set_data(self, probs, time_axis, view_start, view_end):
        """probs: np.array (T, 4), time_axis: np.array (T,)"""
        self.probs = probs
        self.time_axis = time_axis
        self.view_start = view_start
        self.view_end = view_end
        self.update()
    
    def clear_data(self):
        self.probs = None
        self.time_axis = None
        self.playhead_time = -1.0
        self.update()
    
    def update_playhead(self, time_s):
        self.playhead_time = time_s
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        plot_x = self.margin_left
        plot_y = self.margin_top
        plot_w = w - self.margin_left - self.margin_right
        plot_h = h - self.margin_top - self.margin_bottom
        
        if plot_w <= 0 or plot_h <= 0:
            return
        
        # Title
        painter.setPen(QPen(self.label_color))
        title_font = QFont("Sans", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(QRectF(plot_x, 2, plot_w, self.margin_top - 4),
                         Qt.AlignCenter | Qt.AlignVCenter,
                         "Emotion Classification Probabilities")
        
        # Plot border
        painter.setPen(QPen(self.grid_color, 1))
        painter.drawRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        t_min = self.view_start
        t_max = self.view_end
        t_range = t_max - t_min if t_max > t_min else 1.0
        
        # Grid & axis labels
        label_font = QFont("Sans", 8)
        painter.setFont(label_font)
        
        # Horizontal grid (probability 0..1)
        painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        for i in range(6):
            frac = i / 5.0
            y = plot_y + plot_h - (frac * plot_h)
            painter.drawLine(QPointF(plot_x, y), QPointF(plot_x + plot_w, y))
            painter.setPen(QPen(self.axis_color))
            painter.drawText(QRectF(0, y - 8, self.margin_left - 5, 16),
                             Qt.AlignRight | Qt.AlignVCenter, f"{frac:.1f}")
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        # Vertical grid
        num_ticks = min(10, max(4, int(t_range)))
        tick_step = t_range / num_ticks
        for i in range(num_ticks + 1):
            t = t_min + i * tick_step
            x = plot_x + ((t - t_min) / t_range) * plot_w
            painter.drawLine(QPointF(x, plot_y), QPointF(x, plot_y + plot_h))
            painter.setPen(QPen(self.axis_color))
            painter.drawText(QRectF(x - 25, plot_y + plot_h + 2, 50, 20),
                             Qt.AlignCenter, f"{t:.1f}s")
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        # Draw emotion curves
        painter.setClipRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        if self.probs is not None and self.time_axis is not None and len(self.time_axis) > 0:
            for emotion_i in range(4):
                color = self.EMOTION_COLORS[emotion_i]
                painter.setPen(QPen(color, 2.0))
                
                path = QPainterPath()
                first = True
                for j in range(len(self.time_axis)):
                    t = float(self.time_axis[j])
                    if t < t_min or t > t_max:
                        continue
                    v = float(self.probs[j, emotion_i])
                    x = plot_x + ((t - t_min) / t_range) * plot_w
                    y = plot_y + plot_h - (v * plot_h)  # 0..1 mapped
                    if first:
                        path.moveTo(x, y)
                        first = False
                    else:
                        path.lineTo(x, y)
                
                painter.drawPath(path)
        else:
            painter.setPen(QPen(self.axis_color))
            painter.setFont(QFont("Sans", 10))
            painter.drawText(QRectF(plot_x, plot_y, plot_w, plot_h),
                             Qt.AlignCenter, "Run classification to see results")
        
        # Draw Playhead
        if self.playhead_time >= t_min and self.playhead_time <= t_max:
            px = plot_x + ((self.playhead_time - t_min) / t_range) * plot_w
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(QPointF(px, plot_y), QPointF(px, plot_y + plot_h))
            # Triangle marker
            triangle = QPolygonF([
                QPointF(px - 6, plot_y),
                QPointF(px + 6, plot_y),
                QPointF(px, plot_y + 10)
            ])
            painter.setBrush(self.playhead_color)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(triangle)
        
        painter.setClipping(False)
        
        # Legend
        legend_font = QFont("Sans", 9)
        painter.setFont(legend_font)
        legend_x = plot_x + plot_w - 280
        legend_y = plot_y + 8
        for i in range(4):
            lx = legend_x + i * 70
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.EMOTION_COLORS[i])
            painter.drawRoundedRect(QRectF(lx, legend_y, 10, 10), 2, 2)
            painter.setPen(QPen(self.label_color))
            painter.drawText(QRectF(lx + 13, legend_y - 2, 55, 14),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             self.EMOTION_LABELS[i])
        
        # Y-axis label
        painter.setPen(QPen(self.label_color))
        painter.setFont(QFont("Sans", 9))
        painter.save()
        painter.translate(12, plot_y + plot_h / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_h/2, -10, plot_h, 20), Qt.AlignCenter, "Probability")
        painter.restore()


class BandZScorePlotWidget(QWidget):
    """QPainter widget showing 5 EEG band Z-scores over time.
    
    Y-axis is centered at 0 (baseline) with range ±3σ.
    Each band is a colored line: delta, theta, alpha, beta, gamma.
    A horizontal dashed line at Z=0 marks the baseline.
    """
    
    BAND_LABELS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    BAND_COLORS = [
        QColor("#FF6B9D"),   # Pink - Delta
        QColor("#FDE047"),   # Yellow - Theta
        QColor("#00FFB2"),   # Green - Alpha
        QColor("#00AAFF"),   # Blue - Beta
        QColor("#C084FC"),   # Purple - Gamma
    ]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.z_history = []      # list of np.array shape (5,) — mean Z per band
        self.max_history = 300   # 5 minutes at 1Hz
        
        self.bg_color = QColor("#1e1e2e")
        self.grid_color = QColor("#333355")
        self.axis_color = QColor("#888899")
        self.label_color = QColor("#ccccdd")
        self.zero_line_color = QColor("#556677")
        
        self.margin_left = 55
        self.margin_right = 15
        self.margin_top = 35
        self.margin_bottom = 30
        
        self.setMinimumHeight(200)
        self.baseline_locked = True
        self.calibration_progress = 60
    
    def append_z_scores(self, z_scores_array, is_locked=True, progress_seconds=60):
        """Append one frame of Z-scores.
        
        Args:
            z_scores_array: np.array shape (n_channels, n_bands) — we average
                across channels to get one value per band.
            is_locked: bool — whether baseline calculation is finished.
            progress_seconds: int — calibration time progress.
        """
        self.baseline_locked = is_locked
        self.calibration_progress = progress_seconds
        
        if not is_locked:
            self.z_history.clear()
            self.update()
            return
            
        if z_scores_array is None or z_scores_array.size == 0:
            return
        # Mean across channels → shape (n_bands,)
        mean_z = z_scores_array.mean(axis=0)
        self.z_history.append(mean_z)
        if len(self.z_history) > self.max_history:
            self.z_history.pop(0)
        self.update()
    
    def clear_data(self):
        self.z_history.clear()
        self.baseline_locked = True
        self.calibration_progress = 60
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(self.rect(), self.bg_color)
        
        plot_x = self.margin_left
        plot_y = self.margin_top
        plot_w = w - self.margin_left - self.margin_right
        plot_h = h - self.margin_top - self.margin_bottom
        
        if plot_w <= 0 or plot_h <= 0:
            return
        
        # Title
        painter.setPen(QPen(self.label_color))
        title_font = QFont("Sans", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(QRectF(plot_x, 2, plot_w, self.margin_top - 4),
                         Qt.AlignCenter | Qt.AlignVCenter,
                         "Band Power Z-Scores (σ from baseline)")
        
        # Plot border
        painter.setPen(QPen(self.grid_color, 1))
        painter.drawRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        # Y range: -3 to +3
        y_min, y_max = -3.0, 3.0
        y_range = y_max - y_min
        
        label_font = QFont("Sans", 8)
        painter.setFont(label_font)
        
        # Horizontal grid lines at -3, -2, -1, 0, +1, +2, +3
        for z_val in [-3, -2, -1, 0, 1, 2, 3]:
            frac = (z_val - y_min) / y_range
            y = plot_y + plot_h - (frac * plot_h)
            
            if z_val == 0:
                painter.setPen(QPen(self.zero_line_color, 1.5, Qt.DashLine))
            else:
                painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
            painter.drawLine(QPointF(plot_x, y), QPointF(plot_x + plot_w, y))
            
            painter.setPen(QPen(self.axis_color))
            label = f"{z_val:+d}σ" if z_val != 0 else " 0"
            painter.drawText(QRectF(0, y - 8, self.margin_left - 5, 16),
                             Qt.AlignRight | Qt.AlignVCenter, label)
        
        # Time axis
        T = len(self.z_history)
        t_min = 0.0
        t_max = max(T, 30)  # show at least 30s window
        t_range = t_max - t_min
        
        num_ticks = min(10, max(4, int(t_range / 10)))
        tick_step = t_range / num_ticks
        painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        for i in range(num_ticks + 1):
            t = t_min + i * tick_step
            x = plot_x + ((t - t_min) / t_range) * plot_w
            painter.drawLine(QPointF(x, plot_y), QPointF(x, plot_y + plot_h))
            painter.setPen(QPen(self.axis_color))
            painter.drawText(QRectF(x - 25, plot_y + plot_h + 2, 50, 20),
                             Qt.AlignCenter, f"{int(t)}s")
            painter.setPen(QPen(self.grid_color, 1, Qt.DashLine))
        
        # Draw Z-score curves
        painter.setClipRect(QRectF(plot_x, plot_y, plot_w, plot_h))
        
        if T > 0:
            n_bands = min(5, self.z_history[0].shape[0]) if self.z_history else 0
            
            for band_i in range(n_bands):
                color = self.BAND_COLORS[band_i % len(self.BAND_COLORS)]
                painter.setPen(QPen(color, 2.0))
                
                path = QPainterPath()
                first = True
                for j in range(T):
                    z_val = float(np.clip(self.z_history[j][band_i], y_min, y_max))
                    t = float(j)
                    x = plot_x + ((t - t_min) / t_range) * plot_w
                    frac = (z_val - y_min) / y_range
                    y = plot_y + plot_h - (frac * plot_h)
                    if first:
                        path.moveTo(x, y)
                        first = False
                    else:
                        path.lineTo(x, y)
                
                painter.drawPath(path)
        else:
            painter.setPen(QPen(self.axis_color))
            painter.setFont(QFont("Sans", 10))
            painter.drawText(QRectF(plot_x, plot_y, plot_w, plot_h),
                             Qt.AlignCenter, "Z-scores will appear after calibration finishes")
        
        painter.setClipping(False)
        
        # Legend
        legend_font = QFont("Sans", 8)
        painter.setFont(legend_font)
        legend_x = plot_x + plot_w - 350
        legend_y = plot_y + 8
        for i in range(5):
            lx = legend_x + i * 70
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.BAND_COLORS[i])
            painter.drawRoundedRect(QRectF(lx, legend_y, 10, 10), 2, 2)
            painter.setPen(QPen(self.label_color))
            painter.drawText(QRectF(lx + 13, legend_y - 2, 55, 14),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             self.BAND_LABELS[i])
        
        # Y-axis label
        painter.setPen(QPen(self.label_color))
        painter.setFont(QFont("Sans", 9))
        painter.save()
        painter.translate(12, plot_y + plot_h / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_h/2, -10, plot_h, 20), Qt.AlignCenter, "Z-Score (σ)")
        painter.restore()
        
        # Calibration overlay
        if not self.baseline_locked:
            # Elegant semi-transparent bottom badge
            overlay_rect = QRectF(plot_x + 20, plot_y + plot_h - 45, plot_w - 40, 30)
            painter.setBrush(QColor(0, 0, 0, 160) if self.bg_color.lightness() < 128 else QColor(255, 255, 255, 180))
            painter.setPen(QPen(self.grid_color, 1))
            painter.drawRoundedRect(overlay_rect, 6, 6)
            
            painter.setPen(QPen(QColor("#89b4fa") if self.bg_color.lightness() < 128 else QColor("#4a7dff")))
            painter.setFont(QFont("Sans", 9, QFont.Bold))
            painter.drawText(overlay_rect, Qt.AlignCenter, 
                             f" Calibrating Baseline: {self.calibration_progress}/{ZSCORE_CALIBRATION_TIME}s (Resting EEG)")


class AsymmetryGaugeWidget(QWidget):
    """QPainter widget showing Frontal Alpha Asymmetry (FAA) as a balance gauge.
    
    A stable, horizontal meter centered on 0.0, indicating:
      - Left: Withdrawal (Right Dominance, FAA < 0)
      - Right: Approach (Left Dominance, FAA > 0)
    This prevents layout shifts from long status text strings.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.asymmetry = 0.0
        self.direction_text = "Balanced"
        
        self.bg_color = QColor("#1e1e2e")
        self.border_color = QColor("#333355")
        self.axis_color = QColor("#888899")
        self.label_color = QColor("#ccccdd")
        self.left_color = QColor("#FCA5A5")   # Soft red (withdrawal)
        self.right_color = QColor("#00FFB2")  # Soft green (approach)
        self.indicator_color = QColor("#C084FC") # Purple marker
        
        self.setMinimumHeight(65)
        self.setMaximumHeight(65)
        
    def set_asymmetry(self, val):
        self.asymmetry = val
        if abs(val) < 0.05:
            self.direction_text = "Balanced"
        elif val > 0:
            self.direction_text = "Approach (Left Dominance)"
        else:
            self.direction_text = "Withdrawal (Right Dominance)"
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw background container
        painter.fillRect(self.rect(), self.bg_color)
        painter.setPen(QPen(self.border_color, 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        
        # Gauge bar boundaries
        margin_x = 40
        bar_w = w - 2 * margin_x
        bar_h = 10
        bar_y = 30
        
        # Draw base bar (track)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#e0e0e0") if self.bg_color.lightness() > 128 else QColor("#11111b"))
        painter.drawRoundedRect(margin_x, bar_y, bar_w, bar_h, 5, 5)
        
        # Center line (Z = 0)
        center_x = w / 2
        painter.setPen(QPen(self.axis_color, 1.5))
        painter.drawLine(int(center_x), bar_y - 4, int(center_x), bar_y + bar_h + 4)
        
        # Map asymmetry range [-1.0, 1.0] to visual bar
        val_clipped = max(-1.0, min(1.0, self.asymmetry))
        # Invert direction: positive (Left Dominance = Approach) goes left, negative (Right Dominance = Withdrawal) goes right
        offset_x = -(val_clipped / 2.0) * bar_w
        
        # Draw active deviation bar
        if val_clipped != 0:
            if val_clipped > 0:  # Approach (Left Dominance) -> Draw to Left
                painter.setBrush(self.right_color)  # Soft green (Approach)
                painter.drawRect(QRectF(center_x + offset_x, bar_y, -offset_x, bar_h))
            else:                # Withdrawal (Right Dominance) -> Draw to Right
                painter.setBrush(self.left_color)   # Soft red (Withdrawal)
                painter.drawRect(QRectF(center_x, bar_y, offset_x, bar_h))
                
        # Draw sliding pointer marker
        pointer_x = center_x + offset_x
        painter.setBrush(self.indicator_color)
        painter.setPen(QPen(Qt.white, 1))
        painter.drawEllipse(QPointF(pointer_x, bar_y + bar_h/2), 6, 6)
        
        # Text labels
        painter.setPen(QPen(self.label_color))
        painter.setFont(QFont("Sans", 8))
        # Draw asymmetric components at fixed left and right bounds to prevent shifting
        painter.drawText(QRectF(margin_x, 5, 250, 20), Qt.AlignLeft | Qt.AlignVCenter, f"Frontal Alpha Asymmetry: {self.asymmetry:+.2f}")
        painter.drawText(QRectF(w - margin_x - 300, 5, 300, 20), Qt.AlignRight | Qt.AlignVCenter, f"State: {self.direction_text}")
        
        painter.setFont(QFont("Sans", 7))
        painter.drawText(QRectF(margin_x, bar_y + bar_h + 4, 150, 15), Qt.AlignLeft, "Approach (Engagement)")
        painter.drawText(QRectF(w - margin_x - 150, bar_y + bar_h + 4, 150, 15), Qt.AlignRight, "Withdrawal (Avoidance)")



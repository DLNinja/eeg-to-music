import numpy as np
from scipy.interpolate import Rbf
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QImage
from PyQt5.QtCore import Qt, QRectF

# SEED 62-channel approximate 2D coordinates (x, y) ranging from [-1, 1]
SEED_COORDS = [
    [-0.15, 0.9], [0, 0.9], [0.15, 0.9], [-0.35, 0.7], [0.35, 0.7], [-0.85, 0.5], 
    [-0.6, 0.5], [-0.35, 0.5], [-0.15, 0.5], [0, 0.5], [0.15, 0.5], [0.35, 0.5], 
    [0.6, 0.5], [0.85, 0.5], [-0.85, 0.25], [-0.6, 0.25], [-0.35, 0.25], [-0.15, 0.25], 
    [0, 0.25], [0.15, 0.25], [0.35, 0.25], [0.6, 0.25], [0.85, 0.25], [-0.85, 0], 
    [-0.6, 0], [-0.35, 0], [-0.15, 0], [0, 0], [0.15, 0], [0.35, 0], [0.6, 0], 
    [0.85, 0], [-0.85, -0.25], [-0.6, -0.25], [-0.35, -0.25], [-0.15, -0.25], [0, -0.25], 
    [0.15, -0.25], [0.35, -0.25], [0.6, -0.25], [0.85, -0.25], [-0.85, -0.5], [-0.6, -0.5], 
    [-0.35, -0.5], [-0.15, -0.5], [0, -0.5], [0.15, -0.5], [0.35, -0.5], [0.6, -0.5], 
    [0.85, -0.5], [-0.85, -0.7], [-0.6, -0.7], [-0.35, -0.7], [0, -0.7], [0.35, -0.7], 
    [0.6, -0.7], [0.85, -0.7], [-0.15, -1.0], [-0.15, -0.9], [0, -0.9], [0.15, -0.9], [0.15, -1.0]
]

class SingleBandTopomap(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.powers = np.zeros(62)
        self.setMinimumSize(120, 140)
        
        # Pre-compute grid for interpolation (60x60 resolution)
        res = 60
        x = np.linspace(-1.1, 1.1, res)
        # PyQt Y-axis goes down, so we map +1.1 to the top
        y = np.linspace(1.1, -1.1, res)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.points = np.array(SEED_COORDS)
        self.image = None
        
    def set_powers(self, powers):
        self.powers = powers
        self.update_heatmap()
        self.update()

    def _value_to_rgb(self, val):
        """Standard 'Jet' colormap approximation for topomaps."""
        val = max(0.0, min(1.0, val))
        r = max(0, min(255, int(255 * (1.5 - abs(val * 4 - 3)))))
        g = max(0, min(255, int(255 * (1.5 - abs(val * 4 - 2)))))
        b = max(0, min(255, int(255 * (1.5 - abs(val * 4 - 1)))))
        return r, g, b

    def update_heatmap(self):
        if len(self.powers) == 0:
            return
            
        current_min = self.powers.min()
        current_max = self.powers.max()
        
        # Initialize EMA bounds if they don't exist
        if not hasattr(self, 'v_min') or self.v_min is None:
            self.v_min = current_min
            self.v_max = current_max
        else:
            # Smooth the min/max bounds over time using Exponential Moving Average
            # This prevents the heatmap from violently flickering every second
            alpha = 0.1
            self.v_min = self.v_min * (1 - alpha) + current_min * alpha
            self.v_max = self.v_max * (1 - alpha) + current_max * alpha

        # Normalize to [0, 1] using the stable smoothed bounds
        if self.v_max - self.v_min < 1e-6:
            p_norm = np.zeros(len(self.powers))
        else:
            p_clipped = np.clip(self.powers, self.v_min, self.v_max)
            p_norm = (p_clipped - self.v_min) / (self.v_max - self.v_min)
            
        pts = self.points[:len(self.powers)]
        
        # Radial Basis Function interpolation creates a perfectly smooth continuous heatmap
        # mimicking the standard MNE/EEGLAB topomap algorithm
        rbf = Rbf(pts[:, 0], pts[:, 1], p_norm, function='thin_plate', smooth=0.0)
        grid_z = rbf(self.grid_x, self.grid_y)
        
        res = grid_z.shape[0]
        self.image = QImage(res, res, QImage.Format_ARGB32)
        
        # Map values to pixels, clipping outside the head circle
        for i in range(res):
            for j in range(res):
                # Calculate squared distance from center to make a circular mask
                x_norm = self.grid_x[i, j]
                y_norm = self.grid_y[i, j]
                dist_sq = x_norm**2 + y_norm**2
                
                if dist_sq <= 1.0:  # Exactly match the outline radius
                    r, g, b = self._value_to_rgb(grid_z[i, j])
                    # Add anti-aliased edge to the circle
                    alpha = 255 if dist_sq <= 0.95 else int(255 * (1.0 - dist_sq) / 0.05)
                    self.image.setPixelColor(j, i, QColor(r, g, b, alpha))
                else:
                    self.image.setPixelColor(j, i, QColor(0, 0, 0, 0))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw Title
        painter.setPen(QPen(Qt.black))
        painter.drawText(self.rect(), Qt.AlignTop | Qt.AlignHCenter, self.title)

        rect = self.rect()
        margin = 25
        w = rect.width() - margin * 2
        h = rect.height() - margin * 2 - 20
        size = min(w, h)
        
        cx = rect.width() / 2
        cy = 20 + margin + h / 2
        radius = size / 2

        # Draw Heatmap
        if self.image:
            # We scale the image to fit the bounding box ([-1.1, 1.1] corresponds to radius * 1.1)
            target_rect = QRectF(cx - radius * 1.1, cy - radius * 1.1, size * 1.1, size * 1.1)
            # Use smooth transformation so the 60x60 grid looks perfectly continuous
            painter.drawImage(target_rect, self.image)

        # Draw Head Outline
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(size), int(size))
        
        # Draw Nose
        painter.drawLine(int(cx - 8), int(cy - radius), int(cx), int(cy - radius - 12))
        painter.drawLine(int(cx), int(cy - radius - 12), int(cx + 8), int(cy - radius))
        
        # Draw Ears
        painter.drawArc(int(cx - radius - 6), int(cy - 10), 12, 20, 90 * 16, 180 * 16)
        painter.drawArc(int(cx + radius - 6), int(cy - 10), 12, 20, -90 * 16, 180 * 16)


class MultiBandTopomapWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        self.topomaps = {}
        
        for b in self.bands:
            tmap = SingleBandTopomap(b)
            self.topomaps[b.lower()] = tmap
            layout.addWidget(tmap)
            
    def update_bands(self, band_powers_dict):
        for band_name, tmap in self.topomaps.items():
            if band_name in band_powers_dict:
                data = band_powers_dict[band_name]
                if isinstance(data, dict) and 'channels' in data:
                    tmap.set_powers(data['channels'])

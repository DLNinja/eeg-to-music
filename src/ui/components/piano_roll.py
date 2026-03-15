from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import Qt, QRectF, QPointF

class PianoRollWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes = [] # list of (start, duration, pitch, velocity)
        self.total_time = 1.0
        self.current_time = 0.0
        self.zoom_factor = 1.0
        self.scroll_time = 0.0
        self.min_pitch = 21
        self.max_pitch = 108
        self.keyboard_width = 80
        
        # Appearance - Dark Studio Theme
        self.bg_color = QColor("#1e1e1e")
        self.grid_color = QColor("#333333")
        self.white_key_color = QColor("#ffffff")
        self.black_key_color = QColor("#111111")
        self.note_color = QColor("#00FFB2") # Ableton-style neon cyan
        self.playhead_color = QColor("#ff0055") # Neon pink red
        
        self.setMinimumHeight(200)

    def set_data(self, notes, total_time, min_pitch, max_pitch):
        self.notes = notes
        self.total_time = max(total_time, 1.0)
        # Add padding to pitch range
        self.min_pitch = max(0, min_pitch - 2)
        self.max_pitch = min(127, max_pitch + 2)
        if self.min_pitch >= self.max_pitch:
            self.min_pitch, self.max_pitch = 21, 108
        self.update()
        
    def add_note(self, start, duration, pitch, velocity):
        """Helper for real-time additions."""
        self.notes.append((start, duration, pitch, velocity))
        if pitch < self.min_pitch: self.min_pitch = max(0, pitch - 2)
        if pitch > self.max_pitch: self.max_pitch = min(127, pitch + 2)
        self.update()

    def clear_notes(self):
        self.notes = []
        self.current_time = 0.0
        self.scroll_time = 0.0
        self.total_time = 1.0
        self.update()

    def update_playhead(self, time_s):
        self.current_time = time_s
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Draw background
        painter.fillRect(rect, self.bg_color)
        
        pitch_range = self.max_pitch - self.min_pitch + 1
        note_height = height / pitch_range if pitch_range > 0 else height
        
        # Calculate time scale
        track_width = width - self.keyboard_width
        base_pixels_per_second = track_width / self.total_time if self.total_time > 0 else 100
        pixels_per_second = base_pixels_per_second * self.zoom_factor
        visible_time = track_width / pixels_per_second if pixels_per_second > 0 else self.total_time
        
        # Clip notes drawing area
        painter.setClipRect(self.keyboard_width, 0, int(track_width), height)
        
        # Draw horizontal grid lines
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)
        for i in range(pitch_range + 1):
            y = int(height - (i * note_height))
            painter.drawLine(self.keyboard_width, y, width, y)
            
        # Draw vertical grid lines (every 1 second)
        start_s = int(self.scroll_time)
        end_s = int(self.scroll_time + visible_time) + 1
        for s in range(start_s, end_s + 1):
            x = int(self.keyboard_width + ((s - self.scroll_time) * pixels_per_second))
            painter.drawLine(x, 0, x, height)
            
        # Draw Notes
        painter.setPen(Qt.NoPen)
        for start, duration, pitch, vel in self.notes:
            if start + duration < self.scroll_time or start > self.scroll_time + visible_time:
                continue
            alpha = max(60, min(255, int((vel / 127) * 255)))
            color = QColor(self.note_color)
            color.setAlpha(alpha)
            painter.setBrush(color)
            
            x = self.keyboard_width + ((start - self.scroll_time) * pixels_per_second)
            y = height - ((pitch - self.min_pitch + 1) * note_height)
            w = duration * pixels_per_second
            h = note_height
            
            h_pad = h * 0.1
            painter.drawRoundedRect(QRectF(x, y + h_pad, max(2.0, w), h - 2*h_pad), 3, 3)
            
        # Draw Playhead
        playhead_x = self.keyboard_width + ((self.current_time - self.scroll_time) * pixels_per_second)
        if self.keyboard_width <= playhead_x <= width:
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(int(playhead_x), 0, int(playhead_x), height)
            
            triangle = QPolygonF([
                QPointF(playhead_x - 6, 0),
                QPointF(playhead_x + 6, 0),
                QPointF(playhead_x, 12)
            ])
            painter.setBrush(self.playhead_color)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(triangle)
            
        painter.setClipping(False)
        
        # Draw Keyboard Ledger
        painter.fillRect(0, 0, self.keyboard_width, height, QColor("#111111"))
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            y = height - ((pitch - self.min_pitch + 1) * note_height)
            is_black = (pitch % 12) in [1, 3, 6, 8, 10]
            
            key_rect = QRectF(0, y, self.keyboard_width, note_height)
            if is_black:
                painter.fillRect(key_rect, self.black_key_color)
            else:
                painter.fillRect(key_rect, self.white_key_color)
                
            # Draw key border
            painter.setPen(QPen(QColor("#555555"), 1))
            painter.drawRect(key_rect)
            
            # C note text labels
            if pitch % 12 == 0:
                octave = (pitch // 12) - 1
                painter.setPen(QPen(QColor("#000000")))
                font = painter.font()
                font.setPointSize(max(5, int(note_height * 0.5)))
                painter.setFont(font)
                painter.drawText(key_rect.adjusted(3, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, f"C{octave}")

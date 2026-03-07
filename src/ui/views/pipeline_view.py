import os
import scipy.io
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QRadioButton, QSpinBox, 
    QGroupBox, QMessageBox, QDoubleSpinBox, QScrollBar,
    QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPainterPath

# Import Model + Feat Extraction
from src.model.signal_processing import get_de_stft, smooth_features, sf
from src.model.emotion_classifier import EEGResNet
from src.music.midi_generator import generate_midi_from_emotions
from src.ui.views.music_view import MusicView


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
            y = plot_y + plot_h - (frac * plot_h)
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
                y = plot_y + plot_h - ((v - all_min) / data_range) * plot_h
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
        painter.drawText(QRectF(-plot_h/2, -10, plot_h, 20), Qt.AlignCenter, "Amplitude")
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
            from PyQt5.QtGui import QPolygonF
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


# ──────────────────────────────────────────────────────
# Pipeline View (rewritten without matplotlib)
# ──────────────────────────────────────────────────────

class PipelineView(QWidget):
    navigate_to_home_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State variables
        self.mat_data = None
        self.trial_keys = []
        self.current_trial_data = None
        self.sf = sf  # typically 200 Hz
        
        # Pipeline Data
        self.emotion_probs = None # Will store shape (T, 4)
        self.segment_len = 1  # 1 second windows
        self.stft_n = 256
        self.loaded_file_path = ""
        
        # Load Model
        self.model = None
        self._load_model()
        
        self._setup_ui()

    def _load_model(self):
        try:
            self.model = EEGResNet(num_classes=4) 
            checkpoint = torch.load("models/best_model_stft_smooth.pt", map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        except Exception as e:
            print(f"Warning: Failed to load model. Ensure 'models/best_model_stft_smooth.pt' exists. Error: {e}")

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        root_layout.addWidget(self.scroll_area)
        
        self.content_widget = QWidget()
        self.scroll_area.setWidget(self.content_widget)
        
        main_layout = QVBoxLayout(self.content_widget)
        
        # Top toolbar: Back button & File selection & Run Model
        top_bar = QHBoxLayout()
        
        self.back_btn = QPushButton("← Back to Menu")
        self.back_btn.clicked.connect(self.navigate_to_home_signal.emit)
        top_bar.addWidget(self.back_btn)
        
        self.open_file_btn = QPushButton("Open .mat File")
        self.open_file_btn.clicked.connect(self.open_file)
        top_bar.addWidget(self.open_file_btn)
        
        top_bar.addWidget(QLabel("Trial:"))
        self.trial_combo = QComboBox()
        self.trial_combo.currentIndexChanged.connect(self.on_trial_selected)
        self.trial_combo.setEnabled(False)
        top_bar.addWidget(self.trial_combo)
        
        # New Button: Run classification
        self.run_pipeline_btn = QPushButton("Run Emotion Classification")
        self.run_pipeline_btn.clicked.connect(self.run_classification)
        self.run_pipeline_btn.setEnabled(False)
        self.run_pipeline_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        top_bar.addWidget(self.run_pipeline_btn)
        
        self.generate_music_btn = QPushButton("Generate Music")
        self.generate_music_btn.clicked.connect(self.generate_music)
        self.generate_music_btn.setEnabled(False)
        self.generate_music_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        top_bar.addWidget(self.generate_music_btn)
        
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # --- Controls Area ---
        controls_layout = QHBoxLayout()
        
        # 1. Channel Selection UI
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QHBoxLayout()
        
        self.radio_single = QRadioButton("Single Channel")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self.plot_data)
        
        self.spin_single = QSpinBox()
        self.spin_single.setMinimum(1)
        self.spin_single.setValue(1)
        self.spin_single.valueChanged.connect(self.plot_data)
        
        self.radio_range = QRadioButton("Channel Range")
        self.radio_range.toggled.connect(self.plot_data)
        
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(1)
        self.spin_start.setValue(1)
        self.spin_start.valueChanged.connect(self.plot_data)
        
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(1)
        self.spin_end.setValue(5)
        self.spin_end.valueChanged.connect(self.plot_data)
        
        channel_layout.addWidget(self.radio_single)
        channel_layout.addWidget(QLabel("Ch:"))
        channel_layout.addWidget(self.spin_single)
        channel_layout.addSpacing(20)
        channel_layout.addWidget(self.radio_range)
        channel_layout.addWidget(QLabel("Start:"))
        channel_layout.addWidget(self.spin_start)
        channel_layout.addWidget(QLabel("End:"))
        channel_layout.addWidget(self.spin_end)
        channel_layout.addStretch()
        channel_group.setLayout(channel_layout)
        controls_layout.addWidget(channel_group)
        
        # 2. Zoom / View Mode UI
        zoom_group = QGroupBox("Zoom & View")
        zoom_layout = QHBoxLayout()
        
        self.radio_full_view = QRadioButton("Full Signal")
        self.radio_full_view.setChecked(True)
        self.radio_full_view.toggled.connect(self.on_view_mode_changed)
        
        self.radio_window_view = QRadioButton("Windowed View")
        self.radio_window_view.toggled.connect(self.on_view_mode_changed)
        
        self.label_window_size = QLabel("Sensitivity (s):")
        self.spin_window_size = QDoubleSpinBox()
        self.spin_window_size.setRange(0.5, 60.0)
        self.spin_window_size.setValue(5.0)
        self.spin_window_size.setSingleStep(0.5)
        self.spin_window_size.valueChanged.connect(self.on_window_size_changed)
        
        zoom_layout.addWidget(self.radio_full_view)
        zoom_layout.addSpacing(10)
        zoom_layout.addWidget(self.radio_window_view)
        zoom_layout.addSpacing(10)
        zoom_layout.addWidget(self.label_window_size)
        zoom_layout.addWidget(self.spin_window_size)
        zoom_layout.addStretch()
        zoom_group.setLayout(zoom_layout)
        controls_layout.addWidget(zoom_group)
        
        main_layout.addLayout(controls_layout)
        
        # Custom Plot Widgets (replacing matplotlib)
        self.eeg_plot = EegPlotWidget()
        main_layout.addWidget(self.eeg_plot)
        
        self.emotion_plot = EmotionPlotWidget()
        main_layout.addWidget(self.emotion_plot)
        
        # Horizontal Scrollbar for Time navigation
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.valueChanged.connect(self.plot_data)
        main_layout.addWidget(self.time_scrollbar)
        
        # Add visual separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # Add embedded MusicView
        self.music_view = MusicView(embedded_mode=True)
        self.music_view.playback_progress_signal.connect(self.update_music_playhead)
        main_layout.addWidget(self.music_view)
        
        # Init view mode states
        self.on_view_mode_changed()
        
    def open_file(self):
        start_dir = "data/raw/eeg_seed" if os.path.exists("data/raw/eeg_seed") else "."
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open EEG .mat File", 
            start_dir, 
            "MAT Files (*.mat);;All Files (*)"
        )
        
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        try:
            self.loaded_file_path = file_path
            self.mat_data = scipy.io.loadmat(file_path)
            
            # Extract keys that don't start with '__'
            self.trial_keys = [k for k in self.mat_data.keys() if not k.startswith('__')]
            self.trial_keys.sort()
            
            if not self.trial_keys:
                QMessageBox.warning(self, "No Data", "No trial arrays found in the selected file.")
                return

            # Update UI
            self.trial_combo.clear()
            self.trial_combo.addItems(self.trial_keys)
            self.trial_combo.setEnabled(True)
            self.run_pipeline_btn.setEnabled(True)
            self.generate_music_btn.setEnabled(False)
            
            # Select first trial automatically
            self.trial_combo.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Failed to load user data:\n{str(e)}")
            self.mat_data = None
            self.trial_keys = []
            self.trial_combo.clear()
            self.trial_combo.setEnabled(False)
            self.run_pipeline_btn.setEnabled(False)
            self.generate_music_btn.setEnabled(False)
            self.emotion_probs = None
            self.eeg_plot.clear_data()
            self.emotion_plot.clear_data()

    def on_trial_selected(self, index):
        if index < 0 or not self.trial_keys or not self.mat_data:
            return
            
        key = self.trial_keys[index]
        self.current_trial_data = self.mat_data[key]
        self.emotion_probs = None # Clear previous probabilities
        self.generate_music_btn.setEnabled(False)
        
        # Update spinbox maximums based on data shape (channels)
        num_channels = self.current_trial_data.shape[0]
        self.spin_single.setMaximum(num_channels)
        self.spin_start.setMaximum(num_channels)
        self.spin_end.setMaximum(num_channels)
        
        if self.spin_end.value() > num_channels:
            self.spin_end.setValue(num_channels)
            
        self.update_scrollbar_range()
        self.plot_data()

    def run_classification(self):
        if self.current_trial_data is None:
            return
            
        if self.model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Classification model could not be loaded. Please ensure models/best_model_stft_smooth.pt exists.")
            return

        try:
            self.run_pipeline_btn.setText("Processing...")
            self.run_pipeline_btn.setEnabled(False)
            # 1. Extract Features
            features = get_de_stft(self.current_trial_data, self.segment_len, self.stft_n, self.sf)
            
            # 2. Smooth
            smoothed_features = smooth_features(features)
            
            # 3. Predict
            # shape required: (Batch, Channels=1, H=62, W=5)
            input_tensor = torch.tensor(smoothed_features, dtype=torch.float32).unsqueeze(1)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                
            self.emotion_probs = torch.nn.functional.softmax(output, dim=1).numpy()
            
            self.run_pipeline_btn.setText("Run Emotion Classification")
            self.run_pipeline_btn.setEnabled(True)
            self.generate_music_btn.setEnabled(True)
            self.plot_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Classification Error", f"An error occurred during classification:\n{str(e)}")
            self.run_pipeline_btn.setText("Run Emotion Classification")
            self.run_pipeline_btn.setEnabled(True)

    def generate_music(self):
        if self.emotion_probs is None:
            return
            
        try:
            # Parse file path: e.g. data/raw/eeg_seed/3/9_20140620.mat
            dataset = "unknown"
            session = "unknown"
            subject = "unknown"
            
            path_parts = os.path.normpath(self.loaded_file_path).split(os.sep)
            if len(path_parts) >= 3:
                dataset = path_parts[-3]
                session = path_parts[-2]
                filename = path_parts[-1]
                subject = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                
            trial = self.trial_combo.currentText()
            
            output_dir = "music"
            os.makedirs(output_dir, exist_ok=True)
            
            out_filename = f"{dataset}_sub{subject}_sess{session}_trial{trial}.mid"
            out_filepath = os.path.join(output_dir, out_filename)
            
            generate_midi_from_emotions(self.emotion_probs, filename=out_filepath)
            
            self.music_view.load_data(out_filepath)
            
            QMessageBox.information(self, "Music Generated", f"Successfully generated music!\nLoaded into Music Player below.")
            
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"Failed to generate music:\n{str(e)}")

    def on_view_mode_changed(self):
        is_windowed = self.radio_window_view.isChecked()
        
        self.label_window_size.setEnabled(is_windowed)
        self.spin_window_size.setEnabled(is_windowed)
        self.time_scrollbar.setEnabled(is_windowed)
        self.time_scrollbar.setVisible(is_windowed)
        
        if self.current_trial_data is not None:
            self.update_scrollbar_range()
            self.plot_data()
            
    def on_window_size_changed(self):
        if self.current_trial_data is not None:
            self.update_scrollbar_range()
            self.plot_data()

    def update_scrollbar_range(self):
        if self.current_trial_data is None:
            return
            
        num_samples = self.current_trial_data.shape[1]
        total_seconds = num_samples / self.sf
        
        if self.radio_window_view.isChecked():
            window_size = self.spin_window_size.value()
            max_scroll = max(0, int((total_seconds - window_size) * self.sf))
            
            self.time_scrollbar.setMaximum(max_scroll)
            self.time_scrollbar.setSingleStep(int(self.sf * 0.1)) 
            self.time_scrollbar.setPageStep(int(self.sf * window_size * 0.5)) 
        else:
            self.time_scrollbar.setMaximum(0)
            self.time_scrollbar.setValue(0)

    def plot_data(self):
        if self.current_trial_data is None:
            return
            
        data = self.current_trial_data
        num_channels = data.shape[0]
        num_samples = data.shape[1]
        
        # Determine time window to plot
        if self.radio_full_view.isChecked():
            start_sample = 0
            end_sample = num_samples
        else:
            window_size_samples = int(self.spin_window_size.value() * self.sf)
            start_sample = self.time_scrollbar.value()
            end_sample = min(num_samples, start_sample + window_size_samples)
            
        if start_sample >= end_sample:
            return
            
        plot_data = data[:, start_sample:end_sample]
        time_axis = np.arange(start_sample, end_sample) / self.sf
        
        # 1. Build EEG channels
        channels = []
        title = "EEG Signal"
        
        if self.radio_single.isChecked():
            ch_idx = self.spin_single.value() - 1
            if 0 <= ch_idx < num_channels:
                channels.append((f"Ch {ch_idx+1}", plot_data[ch_idx, :]))
                title = f"EEG Signal (Trial: {self.trial_combo.currentText()}, Ch: {ch_idx+1})"
        else:
            start_idx = self.spin_start.value() - 1
            end_idx = self.spin_end.value() - 1
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx 
            start_idx = max(0, start_idx)
            end_idx = min(num_channels - 1, end_idx)
            
            for i, ch_idx in enumerate(range(start_idx, end_idx + 1)):
                offset = i * 150 
                channels.append((f"Ch {ch_idx+1}", plot_data[ch_idx, :] + offset))
                
            title = f"EEG Signal (Trial: {self.trial_combo.currentText()}, Chs: {start_idx+1}-{end_idx+1})"

        self.eeg_plot.set_data(channels, time_axis, title)
        
        # 2. Build Emotion data
        start_time = start_sample / self.sf
        end_time = end_sample / self.sf
        
        if self.emotion_probs is not None:
            num_segments = self.emotion_probs.shape[0]
            emotion_time = np.arange(num_segments) * self.segment_len
            
            e_start = max(0, int(start_time / self.segment_len))
            e_end = min(num_segments, int(np.ceil(end_time / self.segment_len)))
            
            if e_start < e_end:
                self.emotion_plot.set_data(
                    self.emotion_probs[e_start:e_end, :],
                    emotion_time[e_start:e_end],
                    start_time, end_time
                )
            else:
                self.emotion_plot.set_data(None, None, start_time, end_time)
        else:
            self.emotion_plot.set_data(None, None, start_time, end_time)
        
    def update_music_playhead(self, time_s):
        self.emotion_plot.update_playhead(time_s)

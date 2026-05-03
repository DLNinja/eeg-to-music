from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QGridLayout
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont

from src.ui.components.animated_buttons import (
    AnimatedIconButton,
    EegWaveCanvas,
    PipelineCanvas,
    MusicBarsCanvas,
    CircumplexCanvas,
    HeadsetCanvas,
)

class HomeView(QWidget):
    navigate_to_plot_signal = pyqtSignal()
    navigate_to_pipeline_signal = pyqtSignal()
    navigate_to_music_signal = pyqtSignal()
    navigate_to_realtime_signal = pyqtSignal()
    navigate_to_simulator_signal = pyqtSignal()
    theme_changed_signal = pyqtSignal(str)  # "dark" or "light"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_animation_timer()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title_label = QLabel("Welcome to EEG to Music App")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(24)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        subtitle_label = QLabel("Select a tool below to begin.")
        subtitle_label.setAlignment(Qt.AlignCenter)
        font = subtitle_label.font()
        font.setPointSize(14)
        subtitle_label.setFont(font)
        layout.addWidget(subtitle_label)

        layout.addSpacing(40)

        # Animated icon buttons in a 2-column grid
        button_layout = QGridLayout()
        button_layout.setSpacing(20)
        
        self.plotter_btn = AnimatedIconButton("EEG Signal Plotter", EegWaveCanvas)
        self.plotter_btn.clicked.connect(self.navigate_to_plot_signal.emit)
        button_layout.addWidget(self.plotter_btn, 0, 0)

        self.pipeline_btn = AnimatedIconButton("Emotion Pipeline", PipelineCanvas)
        self.pipeline_btn.clicked.connect(self.navigate_to_pipeline_signal.emit)
        button_layout.addWidget(self.pipeline_btn, 0, 1)

        self.music_btn = AnimatedIconButton("Music Player + Visualizer", MusicBarsCanvas)
        self.music_btn.clicked.connect(self.navigate_to_music_signal.emit)
        button_layout.addWidget(self.music_btn, 1, 0)

        self.realtime_btn = AnimatedIconButton("Real-Time Emotion Classifier", CircumplexCanvas)
        self.realtime_btn.clicked.connect(self.navigate_to_realtime_signal.emit)
        button_layout.addWidget(self.realtime_btn, 1, 1)

        self.simulator_btn = AnimatedIconButton("Headset Simulator", HeadsetCanvas)
        self.simulator_btn.clicked.connect(self.navigate_to_simulator_signal.emit)
        # Span the 5th button across both columns, centered
        button_layout.addWidget(self.simulator_btn, 2, 0, 1, 2, alignment=Qt.AlignCenter)

        layout.addLayout(button_layout)
        layout.addSpacing(40)

        # Theme selector
        theme_bar = QHBoxLayout()
        theme_bar.setAlignment(Qt.AlignCenter)

        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont("Sans", 11))
        theme_bar.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setMinimumWidth(120)
        self.theme_combo.setFont(QFont("Sans", 11))
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        theme_bar.addWidget(self.theme_combo)

        layout.addLayout(theme_bar)

    def _setup_animation_timer(self):
        """Single 30 fps timer driving all animation canvases."""
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)  # ~30 fps

        # Collect all animated button canvases
        self._canvases = [
            self.plotter_btn.canvas,
            self.pipeline_btn.canvas,
            self.music_btn.canvas,
            self.realtime_btn.canvas,
            self.simulator_btn.canvas,
        ]
        self._anim_timer.timeout.connect(self._tick_animations)
        self._anim_timer.start()

    def _tick_animations(self):
        for canvas in self._canvases:
            canvas.advance()

    def _on_theme_changed(self, text):
        self.theme_changed_signal.emit(text.lower())
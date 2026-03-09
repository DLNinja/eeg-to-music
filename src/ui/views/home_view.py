from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

class HomeView(QWidget):
    navigate_to_plot_signal = pyqtSignal()
    navigate_to_pipeline_signal = pyqtSignal()
    navigate_to_music_signal = pyqtSignal()
    navigate_to_realtime_signal = pyqtSignal()
    theme_changed_signal = pyqtSignal(str)  # "dark" or "light"
    # Emit signal to request navigation to 'about' view
    navigate_to_about_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

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

        btn_font = QFont()
        btn_font.setPointSize(12)

        # Plotter button
        self.plotter_btn = QPushButton("Go to EEG Signal Plotter")
        self.plotter_btn.setMinimumSize(300, 60)
        self.plotter_btn.setFont(btn_font)
        self.plotter_btn.clicked.connect(self.navigate_to_plot_signal.emit)
        layout.addWidget(self.plotter_btn, alignment=Qt.AlignCenter)
        
        layout.addSpacing(20)
        
        # Pipeline button
        self.pipeline_btn = QPushButton("Go to Emotion Pipeline")
        self.pipeline_btn.setMinimumSize(300, 60)
        self.pipeline_btn.setFont(btn_font)
        self.pipeline_btn.clicked.connect(self.navigate_to_pipeline_signal.emit)
        layout.addWidget(self.pipeline_btn, alignment=Qt.AlignCenter)
        
        layout.addSpacing(20)
        
        # Music button
        self.music_btn = QPushButton("Music Player & Visualizer")
        self.music_btn.setMinimumSize(300, 60)
        self.music_btn.setFont(btn_font)
        self.music_btn.clicked.connect(self.navigate_to_music_signal.emit)
        layout.addWidget(self.music_btn, alignment=Qt.AlignCenter)
        
        layout.addSpacing(20)
        
        # Real-Time Classifier button
        self.realtime_btn = QPushButton("Real-Time Emotion Classifier")
        self.realtime_btn.setMinimumSize(300, 60)
        self.realtime_btn.setFont(btn_font)
        self.realtime_btn.clicked.connect(self.navigate_to_realtime_signal.emit)
        layout.addWidget(self.realtime_btn, alignment=Qt.AlignCenter)
        
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

    def _on_theme_changed(self, text):
        self.theme_changed_signal.emit(text.lower())
        # About button
        self.about_btn = QPushButton("About Music Generation")
        self.about_btn.setMinimumSize(300, 60)
        self.about_btn.setFont(btn_font)
        self.about_btn.clicked.connect(self.navigate_to_about_signal.emit)
        
        layout.addWidget(self.about_btn, alignment=Qt.AlignCenter)

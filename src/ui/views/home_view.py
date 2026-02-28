from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal, Qt

class HomeView(QWidget):
    # Emit signal to request navigation to 'plot' view
    navigate_to_plot_signal = pyqtSignal()
    # Emit signal to request navigation to 'pipeline' view
    navigate_to_pipeline_signal = pyqtSignal()
    # Emit signal to request navigation to 'music' view
    navigate_to_music_signal = pyqtSignal()

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

        # Plotter button
        self.plotter_btn = QPushButton("Go to EEG Signal Plotter")
        self.plotter_btn.setMinimumSize(300, 60)
        btn_font = self.plotter_btn.font()
        btn_font.setPointSize(12)
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

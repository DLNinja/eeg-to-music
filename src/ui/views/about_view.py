import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap

class AboutView(QWidget):
    navigate_to_home_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Top toolbar: Back button
        top_bar = QHBoxLayout()
        self.back_btn = QPushButton("← Back to Menu")
        self.back_btn.clicked.connect(self.navigate_to_home_signal.emit)
        top_bar.addWidget(self.back_btn)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # Image display area with scrolling in case the image is large
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Path to modes.jpg
        # Since this file is in src/ui/views/about_view.py
        # The project root is three directories up
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        image_path = os.path.join(base_dir, "modes.jpg")
        
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            # Scale to a middle ground (450px) to balance size and clarity
            scaled_pixmap = pixmap.scaledToWidth(450, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText(f"Image not found at:\n{image_path}")
            
        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area)

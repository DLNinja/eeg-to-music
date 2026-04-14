import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QApplication
from src.ui.views.home_view import HomeView
from src.ui.views.plot_view import PlotView
from src.ui.views.pipeline_view import PipelineView, EegPlotWidget, EmotionPlotWidget
from src.ui.views.realtime_view import RealTimeView
from src.ui.views.simulator_view import SimulatorView
from src.ui.views.music_view import MusicView
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt

# ──────────────────────────────────────────────────────
# Theme Stylesheets
# ──────────────────────────────────────────────────────

DARK_STYLESHEET = """
    QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    }
    QPushButton {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 6px;
        padding: 6px 14px;
    }
    QPushButton:hover {
        background-color: #45475a;
        border-color: #89b4fa;
    }
    QPushButton:pressed {
        background-color: #585b70;
    }
    QPushButton:disabled {
        background-color: #1e1e2e;
        color: #585b70;
        border-color: #313244;
    }
    QComboBox {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 4px 8px;
    }
    QComboBox:hover {
        border-color: #89b4fa;
    }
    QComboBox QAbstractItemView {
        background-color: #313244;
        color: #cdd6f4;
        selection-background-color: #45475a;
    }
    QGroupBox {
        border: 1px solid #45475a;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: bold;
        color: #89b4fa;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }
    QRadioButton {
        color: #cdd6f4;
        spacing: 5px;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
    }
    QLabel {
        color: #cdd6f4;
    }
    QScrollBar:horizontal {
        background: #181825;
        height: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal {
        background: #585b70;
        border-radius: 6px;
        min-width: 30px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #89b4fa;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    QScrollBar:vertical {
        background: #181825;
        width: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background: #585b70;
        border-radius: 6px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background: #89b4fa;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QSlider::groove:horizontal {
        background: #313244;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #89b4fa;
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QMessageBox {
        background-color: #1e1e2e;
    }
    QFrame[frameShape="4"] {
        color: #45475a;
    }
"""

LIGHT_STYLESHEET = """
    QWidget {
        background-color: #f5f5f5;
        color: #1e1e2e;
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    }
    QPushButton {
        background-color: #ffffff;
        color: #1e1e2e;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 6px 14px;
    }
    QPushButton:hover {
        background-color: #e8e8e8;
        border-color: #4a7dff;
    }
    QPushButton:pressed {
        background-color: #d5d5d5;
    }
    QPushButton:disabled {
        background-color: #f0f0f0;
        color: #b0b0b0;
        border-color: #e0e0e0;
    }
    QComboBox {
        background-color: #ffffff;
        color: #1e1e2e;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 4px 8px;
    }
    QComboBox:hover {
        border-color: #4a7dff;
    }
    QComboBox QAbstractItemView {
        background-color: #ffffff;
        color: #1e1e2e;
        selection-background-color: #e0e7ff;
    }
    QGroupBox {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: bold;
        color: #4a7dff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }
    QRadioButton {
        color: #1e1e2e;
        spacing: 5px;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
    }
    QLabel {
        color: #1e1e2e;
    }
    QScrollBar:horizontal {
        background: #e8e8e8;
        height: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:horizontal {
        background: #b0b0b0;
        border-radius: 6px;
        min-width: 30px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #4a7dff;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }
    QScrollBar:vertical {
        background: #e8e8e8;
        width: 12px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background: #b0b0b0;
        border-radius: 6px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background: #4a7dff;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QSlider::groove:horizontal {
        background: #d0d0d0;
        height: 6px;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #4a7dff;
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QMessageBox {
        background-color: #f5f5f5;
    }
    QFrame[frameShape="4"] {
        color: #d0d0d0;
    }
"""

# QPainter color palettes for plot widgets
DARK_PLOT_COLORS = {
    "bg": QColor("#1e1e2e"),
    "grid": QColor("#333355"),
    "axis": QColor("#888899"),
    "label": QColor("#ccccdd"),
}

LIGHT_PLOT_COLORS = {
    "bg": QColor("#ffffff"),
    "grid": QColor("#e0e0e0"),
    "axis": QColor("#666666"),
    "label": QColor("#1e1e2e"),
}

from src.ui.views.about_view import AboutView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG to Music App")
        self.resize(1000, 600)
        self.current_theme = "dark"

        self._setup_stacked_widget()
        self._apply_theme("dark")

    def _setup_stacked_widget(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.home_view = HomeView()
        self.plot_view = PlotView()
        self.pipeline_view = PipelineView()
        self.music_view = MusicView()
        self.realtime_view = RealTimeView()
        self.simulator_view = SimulatorView()
        self.about_view = AboutView()
        
        # Add views to stack
        self.stacked_widget.addWidget(self.home_view)
        self.stacked_widget.addWidget(self.plot_view)
        self.stacked_widget.addWidget(self.pipeline_view)
        self.stacked_widget.addWidget(self.music_view)
        self.stacked_widget.addWidget(self.realtime_view)
        self.stacked_widget.addWidget(self.simulator_view)
        self.stacked_widget.addWidget(self.about_view)
        
        # Connect navigation signals
        self.home_view.navigate_to_plot_signal.connect(self.show_plot_view)
        self.home_view.navigate_to_pipeline_signal.connect(self.show_pipeline_view)
        self.home_view.navigate_to_music_signal.connect(self.show_music_view)
        self.home_view.navigate_to_realtime_signal.connect(self.show_realtime_view)
        self.home_view.navigate_to_simulator_signal.connect(self.show_simulator_view)
        self.home_view.theme_changed_signal.connect(self._apply_theme)
        self.plot_view.navigate_to_home_signal.connect(self.show_home_view)
        self.pipeline_view.navigate_to_home_signal.connect(self.show_home_view)
        self.music_view.navigate_to_home_signal.connect(self.show_home_view)
        self.realtime_view.navigate_to_home_signal.connect(self.show_home_view)
        self.simulator_view.navigate_to_home_signal.connect(self.show_home_view)
        self.home_view.navigate_to_about_signal.connect(self.show_about_view)
        self.plot_view.navigate_to_home_signal.connect(self.show_home_view)
        self.pipeline_view.navigate_to_home_signal.connect(self.show_home_view)
        self.music_view.navigate_to_home_signal.connect(self.show_home_view)
        self.about_view.navigate_to_home_signal.connect(self.show_home_view)
        
        # Show HomeView initially
        self.stacked_widget.setCurrentWidget(self.home_view)

    def _apply_theme(self, theme_name):
        self.current_theme = theme_name
        app = QApplication.instance()
        
        palette = QPalette()
        
        if theme_name == "dark":
            self.setStyleSheet(DARK_STYLESHEET)
            plot_colors = DARK_PLOT_COLORS
            
            # Dark palette for Fusion-drawn elements (arrows, indicators)
            palette.setColor(QPalette.Window, QColor("#1e1e2e"))
            palette.setColor(QPalette.WindowText, QColor("#cdd6f4"))
            palette.setColor(QPalette.Base, QColor("#313244"))
            palette.setColor(QPalette.AlternateBase, QColor("#45475a"))
            palette.setColor(QPalette.Text, QColor("#cdd6f4"))
            palette.setColor(QPalette.Button, QColor("#45475a"))
            palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
            palette.setColor(QPalette.Highlight, QColor("#89b4fa"))
            palette.setColor(QPalette.HighlightedText, QColor("#1e1e2e"))
            palette.setColor(QPalette.Light, QColor("#cdd6f4"))
            palette.setColor(QPalette.Mid, QColor("#585b70"))
            palette.setColor(QPalette.Dark, QColor("#181825"))
            
            self.music_view.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
            if hasattr(self.pipeline_view, 'music_view'):
                self.pipeline_view.music_view.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        else:
            self.setStyleSheet(LIGHT_STYLESHEET)
            plot_colors = LIGHT_PLOT_COLORS
            
            # Light palette
            palette.setColor(QPalette.Window, QColor("#f5f5f5"))
            palette.setColor(QPalette.WindowText, QColor("#1e1e2e"))
            palette.setColor(QPalette.Base, QColor("#ffffff"))
            palette.setColor(QPalette.AlternateBase, QColor("#e8e8e8"))
            palette.setColor(QPalette.Text, QColor("#1e1e2e"))
            palette.setColor(QPalette.Button, QColor("#e0e0e0"))
            palette.setColor(QPalette.ButtonText, QColor("#000000"))
            palette.setColor(QPalette.Highlight, QColor("#4a7dff"))
            palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
            palette.setColor(QPalette.Light, QColor("#ffffff"))
            palette.setColor(QPalette.Mid, QColor("#b0b0b0"))
            palette.setColor(QPalette.Dark, QColor("#808080"))
            
            self.music_view.setStyleSheet("background-color: #ffffff; color: #1e1e2e;")
            if hasattr(self.pipeline_view, 'music_view'):
                self.pipeline_view.music_view.setStyleSheet("background-color: #ffffff; color: #1e1e2e;")
        
        app.setPalette(palette)
        
        # Update QPainter plot widgets
        self._apply_plot_colors(self.plot_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.pipeline_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.pipeline_view.emotion_plot, plot_colors)
        self._apply_plot_colors(self.realtime_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.realtime_view.emotion_plot, plot_colors)
        self._apply_plot_colors(self.simulator_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.simulator_view.emotion_plot, plot_colors)

    def _apply_plot_colors(self, widget, colors):
        widget.bg_color = colors["bg"]
        widget.grid_color = colors["grid"]
        widget.axis_color = colors["axis"]
        widget.label_color = colors["label"]
        widget.update()

    def show_plot_view(self):
        self.stacked_widget.setCurrentWidget(self.plot_view)
        
    def show_pipeline_view(self):
        self.stacked_widget.setCurrentWidget(self.pipeline_view)
        
    def show_music_view(self):
        self.stacked_widget.setCurrentWidget(self.music_view)
    
    def show_realtime_view(self):
        self.stacked_widget.setCurrentWidget(self.realtime_view)
        
    def show_simulator_view(self):
        self.stacked_widget.setCurrentWidget(self.simulator_view)
        
    def show_about_view(self):
        self.stacked_widget.setCurrentWidget(self.about_view)
        
    def show_home_view(self):
        self.stacked_widget.setCurrentWidget(self.home_view)

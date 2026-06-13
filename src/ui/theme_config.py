from PyQt5.QtGui import QColor, QPalette

# Theme Stylesheets

DARK_STYLESHEET = """
    QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    }
    QWidget#topSettingsBar {
        background-color: #11111b;
        border-bottom: 1px solid #313244;
    }
    QLabel#settingsBarTitle {
        color: #a6adc8;
        font-weight: bold;
        font-size: 11px;
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
    QWidget#dashboardCard {
        background-color: #22223b;
        border: 1px solid #3b3b5c;
        border-radius: 6px;
    }
    QLabel#cardTitle {
        color: #888899;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        background: transparent;
    }
    QLabel#cardValue {
        color: #ffffff;
        font-size: 13px;
        font-weight: bold;
        background: transparent;
    }
    QLabel#subLabel {
        font-style: italic;
        color: #aaaaaa;
    }
    QPushButton#playBtn {
        background-color: #1a1a1a;
        color: #00FFB2;
        border: 1px solid #00FFB2;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#playBtn:hover {
        background-color: #00FFB2;
        color: #1a1a1a;
    }
    QPushButton#playBtn:disabled {
        background-color: #121212;
        color: #555555;
        border-color: #333333;
    }
    QPushButton#pauseBtn {
        background-color: #1a1a1a;
        color: #FF9800;
        border: 1px solid #FF9800;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#pauseBtn:hover {
        background-color: #FF9800;
        color: #1a1a1a;
    }
    QPushButton#pauseBtn:disabled {
        background-color: #121212;
        color: #555555;
        border-color: #333333;
    }
    QPushButton#stopBtn {
        background-color: #1a1a1a;
        color: #F44336;
        border: 1px solid #F44336;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#stopBtn:hover {
        background-color: #F44336;
        color: #1a1a1a;
    }
    QPushButton#stopBtn:disabled {
        background-color: #121212;
        color: #555555;
        border-color: #333333;
    }
    QPushButton#skipBtn {
        background-color: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#skipBtn:hover {
        background-color: #45475a;
        border-color: #89b4fa;
    }
    QPushButton#skipBtn:disabled {
        background-color: #1e1e2e;
        color: #585b70;
        border-color: #313244;
    }
    QPushButton#loadModelBtn {
        background-color: #313244;
        color: #89b4fa;
        border: 1px solid #89b4fa;
        border-radius: 4px;
        font-weight: bold;
        padding: 5px 12px;
    }
    QPushButton#loadModelBtn:hover {
        background-color: #89b4fa;
        color: #11111b;
    }
"""

LIGHT_STYLESHEET = """
    QWidget {
        background-color: #f5f5f5;
        color: #1e1e2e;
        font-family: 'Segoe UI', 'Ubuntu', sans-serif;
    }
    QWidget#topSettingsBar {
        background-color: #e8e8e8;
        border-bottom: 1px solid #d0d0d0;
    }
    QLabel#settingsBarTitle {
        color: #555555;
        font-weight: bold;
        font-size: 11px;
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
    QWidget#dashboardCard {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
    }
    QLabel#cardTitle {
        color: #666666;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        background: transparent;
    }
    QLabel#cardValue {
        color: #1e1e2e;
        font-size: 13px;
        font-weight: bold;
        background: transparent;
    }
    QLabel#subLabel {
        font-style: italic;
        color: #666666;
    }
    QPushButton#playBtn {
        background-color: #ffffff;
        color: #00a877;
        border: 1px solid #00a877;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#playBtn:hover {
        background-color: #e6f7f2;
    }
    QPushButton#playBtn:disabled {
        background-color: #f5f5f5;
        color: #b0b0b0;
        border-color: #e0e0e0;
    }
    QPushButton#pauseBtn {
        background-color: #ffffff;
        color: #e67e22;
        border: 1px solid #e67e22;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#pauseBtn:hover {
        background-color: #fdf5e6;
    }
    QPushButton#pauseBtn:disabled {
        background-color: #f5f5f5;
        color: #b0b0b0;
        border-color: #e0e0e0;
    }
    QPushButton#stopBtn {
        background-color: #ffffff;
        color: #d35400;
        border: 1px solid #d35400;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#stopBtn:hover {
        background-color: #fbeee6;
    }
    QPushButton#stopBtn:disabled {
        background-color: #f5f5f5;
        color: #b0b0b0;
        border-color: #e0e0e0;
    }
    QPushButton#skipBtn {
        background-color: #ffffff;
        color: #1e1e2e;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 15px;
    }
    QPushButton#skipBtn:hover {
        background-color: #e8e8e8;
        border-color: #4a7dff;
    }
    QPushButton#skipBtn:disabled {
        background-color: #f0f0f0;
        color: #b0b0b0;
        border-color: #e0e0e0;
    }
    QPushButton#loadModelBtn {
        background-color: #ffffff;
        color: #4a7dff;
        border: 1px solid #4a7dff;
        border-radius: 4px;
        font-weight: bold;
        padding: 5px 12px;
    }
    QPushButton#loadModelBtn:hover {
        background-color: #4a7dff;
        color: #ffffff;
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

THEME_CONFIGS = {
    "dark": {
        "stylesheet": DARK_STYLESHEET,
        "plot_colors": DARK_PLOT_COLORS,
        "music_style": "background-color: #1e1e2e; color: #cdd6f4;",
        "palette": {
            QPalette.Window: QColor("#1e1e2e"),
            QPalette.WindowText: QColor("#cdd6f4"),
            QPalette.Base: QColor("#313244"),
            QPalette.AlternateBase: QColor("#45475a"),
            QPalette.Text: QColor("#cdd6f4"),
            QPalette.Button: QColor("#45475a"),
            QPalette.ButtonText: QColor("#ffffff"),
            QPalette.Highlight: QColor("#89b4fa"),
            QPalette.HighlightedText: QColor("#1e1e2e"),
            QPalette.Light: QColor("#cdd6f4"),
            QPalette.Mid: QColor("#585b70"),
            QPalette.Dark: QColor("#181825"),
        }
    },
    "light": {
        "stylesheet": LIGHT_STYLESHEET,
        "plot_colors": LIGHT_PLOT_COLORS,
        "music_style": "background-color: #ffffff; color: #1e1e2e;",
        "palette": {
            QPalette.Window: QColor("#f5f5f5"),
            QPalette.WindowText: QColor("#1e1e2e"),
            QPalette.Base: QColor("#ffffff"),
            QPalette.AlternateBase: QColor("#e8e8e8"),
            QPalette.Text: QColor("#1e1e2e"),
            QPalette.Button: QColor("#e0e0e0"),
            QPalette.ButtonText: QColor("#000000"),
            QPalette.Highlight: QColor("#4a7dff"),
            QPalette.HighlightedText: QColor("#ffffff"),
            QPalette.Light: QColor("#ffffff"),
            QPalette.Mid: QColor("#b0b0b0"),
            QPalette.Dark: QColor("#808080"),
        }
    }
}
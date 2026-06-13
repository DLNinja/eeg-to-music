import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from src.ui.views.home_view import HomeView
from src.ui.views.plot_view import PlotView
from src.ui.views.pipeline_view import PipelineView
from src.ui.components.eeg_plots import EegPlotWidget, EmotionPlotWidget
from src.ui.views.realtime_view import RealTimeView
from src.ui.views.simulator_view import SimulatorView
from src.ui.views.music_view import MusicView
from src.ui.components.model_selector import ModelSelectorWidget
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
from src.ui.theme_config import THEME_CONFIGS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG to Music App")
        self.resize(1000, 650)
        self.current_theme = "dark"

        self._setup_ui_layout()
        self._apply_theme("dark")
        self._load_initial_model()

    def _setup_ui_layout(self):
        self.central_area = QWidget()
        self.setCentralWidget(self.central_area)
        main_layout = QVBoxLayout(self.central_area)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create settings bar header
        self.settings_header = QWidget()
        self.settings_header.setObjectName("topSettingsBar")
        header_layout = QHBoxLayout(self.settings_header)
        header_layout.setContentsMargins(15, 6, 15, 6)
        header_layout.setSpacing(15)

        title_lbl = QLabel("ACTIVE MODEL CONFIGURATION")
        title_lbl.setObjectName("settingsBarTitle")
        header_layout.addWidget(title_lbl)

        self.model_selector = ModelSelectorWidget()
        self.model_selector.model_loaded.connect(self._on_model_loaded)
        header_layout.addWidget(self.model_selector)
        header_layout.addStretch()

        main_layout.addWidget(self.settings_header)

        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        self.home_view = HomeView()
        self.plot_view = PlotView()
        self.pipeline_view = PipelineView()
        self.music_view = MusicView()
        self.realtime_view = RealTimeView()
        self.simulator_view = SimulatorView()
        
        # Add views to stack
        self.stacked_widget.addWidget(self.home_view)
        self.stacked_widget.addWidget(self.plot_view)
        self.stacked_widget.addWidget(self.pipeline_view)
        self.stacked_widget.addWidget(self.music_view)
        self.stacked_widget.addWidget(self.realtime_view)
        self.stacked_widget.addWidget(self.simulator_view)
        
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
        
        # Listen to page change to show/hide settings bar
        self.stacked_widget.currentChanged.connect(self._on_page_changed)
        
        # Show HomeView initially
        self.stacked_widget.setCurrentWidget(self.home_view)
        self._on_page_changed(0)

    def _load_initial_model(self):
        from src.eeg_pipeline.emotion_classifier import load_emotion_model
        arch = self.model_selector.arch_combo.currentText()
        chk = self.model_selector.checkpoint_combo.currentText()
        if arch and chk:
            chk_path = os.path.join("models/classifiers", arch, chk)
            model = load_emotion_model(model_path=chk_path, arch_name=arch)
            if model is not None:
                self._on_model_loaded(model, arch, chk_path)

    def _on_model_loaded(self, model, arch_name, checkpoint_path):
        """Propagate loaded model to all classification views."""
        self.pipeline_view.set_model(model)
        self.realtime_view.set_model(model)
        self.simulator_view.set_model(model)

    def _on_page_changed(self, index):
        current_widget = self.stacked_widget.currentWidget()
        show_selector = isinstance(current_widget, (PipelineView, RealTimeView, SimulatorView))
        self.settings_header.setVisible(show_selector)

    def _apply_theme(self, theme_name):
        self.current_theme = theme_name
        cfg = THEME_CONFIGS.get(theme_name, THEME_CONFIGS["dark"])
        
        self.setStyleSheet(cfg["stylesheet"])
        self.music_view.setStyleSheet(cfg["music_style"])
        if hasattr(self.pipeline_view, 'music_view'):
            self.pipeline_view.music_view.setStyleSheet(cfg["music_style"])
            
        palette = QPalette()
        for role, color in cfg["palette"].items():
            palette.setColor(role, color)
        QApplication.instance().setPalette(palette)
        
        # Update QPainter plot and custom widgets
        plot_colors = cfg["plot_colors"]
        self._apply_plot_colors(self.plot_view.eeg_plot, plot_colors)
        
        self._apply_plot_colors(self.pipeline_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.pipeline_view.emotion_plot, plot_colors)
        if hasattr(self.pipeline_view, 'music_view') and hasattr(self.pipeline_view.music_view, 'piano_roll'):
            self._apply_plot_colors(self.pipeline_view.music_view.piano_roll, plot_colors)
        
        self._apply_plot_colors(self.realtime_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.realtime_view.emotion_plot, plot_colors)
        if hasattr(self.realtime_view, 'zscore_plot'):
            self._apply_plot_colors(self.realtime_view.zscore_plot, plot_colors)
        if hasattr(self.realtime_view, 'asymmetry_gauge'):
            self._apply_plot_colors(self.realtime_view.asymmetry_gauge, plot_colors)
        if hasattr(self.realtime_view, 'piano_roll'):
            self._apply_plot_colors(self.realtime_view.piano_roll, plot_colors)
            
        self._apply_plot_colors(self.simulator_view.eeg_plot, plot_colors)
        self._apply_plot_colors(self.simulator_view.emotion_plot, plot_colors)
        if hasattr(self.simulator_view, 'zscore_plot'):
            self._apply_plot_colors(self.simulator_view.zscore_plot, plot_colors)
        if hasattr(self.simulator_view, 'asymmetry_gauge'):
            self._apply_plot_colors(self.simulator_view.asymmetry_gauge, plot_colors)
        if hasattr(self.simulator_view, 'piano_roll'):
            self._apply_plot_colors(self.simulator_view.piano_roll, plot_colors)
            
        if hasattr(self.music_view, 'piano_roll'):
            self._apply_plot_colors(self.music_view.piano_roll, plot_colors)

    def _apply_plot_colors(self, widget, colors):
        if widget is None:
            return
        widget.bg_color = colors["bg"]
        if hasattr(widget, "grid_color"):
            widget.grid_color = colors["grid"]
        if hasattr(widget, "axis_color"):
            widget.axis_color = colors["axis"]
        if hasattr(widget, "label_color"):
            widget.label_color = colors["label"]
        if hasattr(widget, "border_color"):
            widget.border_color = colors["grid"]
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
        
    
    def show_home_view(self):
        self.stacked_widget.setCurrentWidget(self.home_view)

    def closeEvent(self, event):
        # Stop simulator view worker thread and synthesizer
        if hasattr(self, 'simulator_view') and self.simulator_view is not None:
            try:
                if hasattr(self.simulator_view, 'worker_thread') and self.simulator_view.worker_thread.isRunning():
                    if hasattr(self.simulator_view, 'worker') and self.simulator_view.worker is not None:
                        self.simulator_view.worker.stop()
                    self.simulator_view.worker_thread.quit()
                    self.simulator_view.worker_thread.wait(1000)
                if hasattr(self.simulator_view, 'synth') and self.simulator_view.synth is not None:
                    self.simulator_view.synth.stop()
            except Exception as e:
                print(f"Error cleaning up simulator view threads: {e}")
                
        # Stop realtime view worker thread and synthesizer
        if hasattr(self, 'realtime_view') and self.realtime_view is not None:
            try:
                if hasattr(self.realtime_view, 'worker_thread') and self.realtime_view.worker_thread.isRunning():
                    if hasattr(self.realtime_view, 'worker') and self.realtime_view.worker is not None:
                        self.realtime_view.worker.stop()
                    self.realtime_view.worker_thread.quit()
                    self.realtime_view.worker_thread.wait(1000)
                if hasattr(self.realtime_view, 'synth') and self.realtime_view.synth is not None:
                    self.realtime_view.synth.stop()
            except Exception as e:
                print(f"Error cleaning up realtime view threads: {e}")
                
        event.accept()

import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from src.ui.views.home_view import HomeView
from src.ui.views.plot_view import PlotView
from src.ui.views.pipeline_view import PipelineView
from src.ui.views.music_view import MusicView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG to Music App")
        self.resize(1000, 600)

        self._setup_stacked_widget()

    def _setup_stacked_widget(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize views
        self.home_view = HomeView()
        self.plot_view = PlotView()
        self.pipeline_view = PipelineView()
        self.music_view = MusicView()
        
        # Add views to stack
        self.stacked_widget.addWidget(self.home_view)
        self.stacked_widget.addWidget(self.plot_view)
        self.stacked_widget.addWidget(self.pipeline_view)
        self.stacked_widget.addWidget(self.music_view)
        
        # Connect navigation signals
        self.home_view.navigate_to_plot_signal.connect(self.show_plot_view)
        self.home_view.navigate_to_pipeline_signal.connect(self.show_pipeline_view)
        self.home_view.navigate_to_music_signal.connect(self.show_music_view)
        self.plot_view.navigate_to_home_signal.connect(self.show_home_view)
        self.pipeline_view.navigate_to_home_signal.connect(self.show_home_view)
        self.music_view.navigate_to_home_signal.connect(self.show_home_view)
        
        # Show HomeView initially
        self.stacked_widget.setCurrentWidget(self.home_view)

    def show_plot_view(self):
        self.stacked_widget.setCurrentWidget(self.plot_view)
        
    def show_pipeline_view(self):
        self.stacked_widget.setCurrentWidget(self.pipeline_view)
        
    def show_music_view(self):
        self.stacked_widget.setCurrentWidget(self.music_view)
        
    def show_home_view(self):
        self.stacked_widget.setCurrentWidget(self.home_view)

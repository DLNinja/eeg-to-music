import os
import scipy.io
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QRadioButton, QSpinBox, 
    QGroupBox, QMessageBox, QDoubleSpinBox, QScrollBar
)
from PyQt5.QtCore import pyqtSignal, Qt

# Reuse the custom QPainter EEG plot widget from pipeline_view
from src.ui.views.pipeline_view import EegPlotWidget

class PlotView(QWidget):
    navigate_to_home_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State variables
        self.mat_data = None
        self.trial_keys = []
        self.current_trial_data = None
        self.sf = 200  # Sampling frequency in Hz
        
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Top toolbar: Back button & File selection
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
        
        top_bar.addStretch()
        main_layout.addLayout(top_bar)
        
        # --- Controls Area ---
        controls_layout = QHBoxLayout()
        
        # 1. Channel Selection UI
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QHBoxLayout()
        
        self.radio_single = QRadioButton("Single Channel")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self.plot_trial)
        
        self.spin_single = QSpinBox()
        self.spin_single.setMinimum(1)
        self.spin_single.setValue(1)
        self.spin_single.valueChanged.connect(self.plot_trial)
        
        self.radio_range = QRadioButton("Channel Range")
        self.radio_range.toggled.connect(self.plot_trial)
        
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(1)
        self.spin_start.setValue(1)
        self.spin_start.valueChanged.connect(self.plot_trial)
        
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(1)
        self.spin_end.setValue(5)
        self.spin_end.valueChanged.connect(self.plot_trial)
        
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
        
        # Custom QPainter Plot Widget (replaces matplotlib)
        self.eeg_plot = EegPlotWidget()
        self.eeg_plot.setMinimumHeight(400)
        main_layout.addWidget(self.eeg_plot, 1)  # stretch factor
        
        # Horizontal Scrollbar for Time navigation
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.valueChanged.connect(self.plot_trial)
        main_layout.addWidget(self.time_scrollbar)
        
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
            self.mat_data = scipy.io.loadmat(file_path)
            
            self.trial_keys = [k for k in self.mat_data.keys() if not k.startswith('__')]
            self.trial_keys.sort()
            
            if not self.trial_keys:
                QMessageBox.warning(self, "No Data", "No trial arrays found in the selected file.")
                return

            # Update UI
            self.trial_combo.clear()
            self.trial_combo.addItems(self.trial_keys)
            self.trial_combo.setEnabled(True)
            
            # Select first trial automatically
            self.trial_combo.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Failed to load user data:\n{str(e)}")
            self.mat_data = None
            self.trial_keys = []
            self.trial_combo.clear()
            self.trial_combo.setEnabled(False)
            self.eeg_plot.clear_data()

    def on_trial_selected(self, index):
        if index < 0 or not self.trial_keys or not self.mat_data:
            return
            
        key = self.trial_keys[index]
        self.current_trial_data = self.mat_data[key]
        
        num_channels = self.current_trial_data.shape[0]
        self.spin_single.setMaximum(num_channels)
        self.spin_start.setMaximum(num_channels)
        self.spin_end.setMaximum(num_channels)
        
        if self.spin_end.value() > num_channels:
            self.spin_end.setValue(num_channels)
            
        self.update_scrollbar_range()
        self.plot_trial()

    def on_view_mode_changed(self):
        is_windowed = self.radio_window_view.isChecked()
        
        self.label_window_size.setEnabled(is_windowed)
        self.spin_window_size.setEnabled(is_windowed)
        self.time_scrollbar.setEnabled(is_windowed)
        self.time_scrollbar.setVisible(is_windowed)
        
        if self.current_trial_data is not None:
            self.update_scrollbar_range()
            self.plot_trial()
            
    def on_window_size_changed(self):
        if self.current_trial_data is not None:
            self.update_scrollbar_range()
            self.plot_trial()

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

    def plot_trial(self):
        if self.current_trial_data is None:
            return
            
        data = self.current_trial_data
        num_channels = data.shape[0]
        num_samples = data.shape[1]
        
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

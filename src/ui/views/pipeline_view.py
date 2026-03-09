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
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

# Import Model + Feat Extraction
from src.model.signal_processing import get_de_stft, smooth_features, sf
from src.model.emotion_classifier import EEGResNet
from src.music.midi_generator import generate_midi_from_emotions
from src.ui.views.music_view import MusicView

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
        self.spin_window_size.setRange(0.5, 60.0) # 0.5s up to 60s windows
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
        
        # Plot area
        # 2 subplots, share x-axis
        self.figure = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(600)
        
        self.ax_eeg = self.figure.add_subplot(211)
        self.ax_eeg.set_title("EEG Signal")
        self.ax_eeg.set_ylabel("Amplitude")
        
        self.ax_emotion = self.figure.add_subplot(212, sharex=self.ax_eeg)
        self.ax_emotion.set_title("Emotion Classification Probabilities")
        self.ax_emotion.set_ylabel("Probability")
        self.ax_emotion.set_xlabel("Time (s)")
        
        self.figure.tight_layout()
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
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
            self.ax_eeg.clear()
            self.ax_emotion.clear()
            self.canvas.draw()

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
        
        # 1. Plot EEG
        self.ax_eeg.clear()
        
        if self.radio_single.isChecked():
            ch_idx = self.spin_single.value() - 1
            if 0 <= ch_idx < num_channels:
                self.ax_eeg.plot(time_axis, plot_data[ch_idx, :], label=f"Ch {ch_idx+1}")
                self.ax_eeg.set_title(f"EEG Signal (Trial: {self.trial_combo.currentText()}, Ch: {ch_idx+1})")
                self.ax_eeg.set_ylabel("Amplitude")
        else:
            start_idx = self.spin_start.value() - 1
            end_idx = self.spin_end.value() - 1
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx 
            start_idx = max(0, start_idx)
            end_idx = min(num_channels - 1, end_idx)
            
            for i, ch_idx in enumerate(range(start_idx, end_idx + 1)):
                offset = i * 150 
                self.ax_eeg.plot(time_axis, plot_data[ch_idx, :] + offset, label=f"Ch {ch_idx+1}")
                
            self.ax_eeg.set_title(f"EEG Signal (Trial: {self.trial_combo.currentText()}, Chs: {start_idx+1}-{end_idx+1})")
            self.ax_eeg.set_ylabel("Amplitude + Offset")

        self.ax_eeg.set_xlim(time_axis[0], time_axis[-1])
        
        # 2. Plot Emotion Probabilities
        self.ax_emotion.clear()
        
        if self.emotion_probs is not None:
            # self.emotion_probs has shape (T, 4), where T is num_segments (e.g. 1 second per segment)
            emotion_labels = ["Neutral", "Sad", "Fear", "Happy"]
            
            # Reconstruct the time_axis for emotions
            num_segments = self.emotion_probs.shape[0]
            # Since segment length is 1s, the emotion times are [0, 1, 2, ... num_segments-1]
            emotion_time = np.arange(num_segments) * self.segment_len
            
            # Slice the probabilities to match current view
            start_time = start_sample / self.sf
            end_time = end_sample / self.sf
            
            # Find indices
            e_start = max(0, int(start_time / self.segment_len))
            e_end = min(num_segments, int(np.ceil(end_time / self.segment_len)))
            
            if e_start < e_end:
                plot_probs = self.emotion_probs[e_start:e_end, :]
                plot_etimes = emotion_time[e_start:e_end]
                
                for i in range(4):
                    self.ax_emotion.plot(plot_etimes, plot_probs[:, i], label=emotion_labels[i])
                    
            self.ax_emotion.set_ylim(0, 1.05)
            self.ax_emotion.legend(loc='lower right', ncol=4, fontsize='small')
            
        self.ax_emotion.set_title("Emotion Probabilities Over Time")
        self.ax_emotion.set_ylabel("Probability")
        self.ax_emotion.set_xlabel("Time (s)")
        self.ax_emotion.grid(True, alpha=0.3)
        self.ax_emotion.set_xlim(time_axis[0], time_axis[-1]) # ensures synced plotting width
        
        self.music_playhead_line = self.ax_emotion.axvline(x=0, color='red', linewidth=2, linestyle='--', zorder=10)
        self.music_playhead_line.set_visible(False)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_music_playhead(self, time_s):
        if hasattr(self, 'music_playhead_line') and self.music_playhead_line is not None:
            # Scale the music playback time to match the EEG timeline length
            # (Because MIDI generator changes BPM dynamically, music duration != EEG duration)
            if hasattr(self.music_view, 'total_time_s') and self.music_view.total_time_s > 0 and self.emotion_probs is not None:
                total_eeg_s = len(self.emotion_probs) * self.segment_len
                mapped_time_s = time_s * (total_eeg_s / self.music_view.total_time_s)
            else:
                mapped_time_s = time_s
                
            self.music_playhead_line.set_xdata([mapped_time_s, mapped_time_s])
            self.music_playhead_line.set_visible(True)
            self.canvas.draw_idle()

import os
import sys
import numpy as np
import torch
import socket
import struct
from queue import Queue
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QGroupBox, QMessageBox, QFrame,
    QRadioButton, QDoubleSpinBox, QScrollBar, QSpinBox, QLineEdit, QCheckBox,
    QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QThread, QObject
from PyQt5.QtGui import QIntValidator

from src.model.signal_processing import (
    sf, n_channels, bands,
    create_bandpass_filter, filter_segment,
    extract_single_window_features, smooth_features, moving_average
)
from src.model.emotion_classifier import EEGResNet
from src.ui.views.pipeline_view import EegPlotWidget, EmotionPlotWidget
from src.ui.components.piano_roll import PianoRollWidget
from src.music.realtime_generator import RealTimeMusicSynthesizer

# Default values (used as UI defaults, can be changed by user)
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 65432
DEFAULT_CHANNELS = 62
DEFAULT_SAMPLE_RATE = 200

# BioSemi protocol defaults
# BioSemi ADC resolution: 1 bit = 31.25 nV = 0.00003125 µV
# So 1 µV = 1 / 0.00003125 = 32000 bits
DEFAULT_UV_TO_BITS = 32000
DEFAULT_BYTES_PER_SAMPLE = 3  # 24-bit samples

# ──────────────────────────────────────────────────────
# Data Stream Thread
# ──────────────────────────────────────────────────────

class DataStreamThread(QThread):
    """
    Background thread to listen to the TCP socket continuously
    without blocking the GUI main loop.
    Decodes BioSemi ActiveView 24-bit LE signed integer format.
    """
    new_data_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    disconnected_signal = pyqtSignal()

    def __init__(self, host, port, channels, bytes_per_sample, uv_to_bits):
        super().__init__()
        self.host = host
        self.port = port
        self.channels = channels
        self.bytes_per_sample = bytes_per_sample
        self.uv_to_bits = uv_to_bits
        self.packet_size = channels * bytes_per_sample
        self.running = False
        self.sock = None

    def recvall(self, n):
        data = bytearray()
        while len(data) < n and self.running:
            try:
                packet = self.sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except Exception:
                return None
        return data

    def run(self):
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        
        try:
            self.sock.connect((self.host, self.port))
        except Exception as e:
            self.error_signal.emit(f"Failed to connect: {e}")
            self.running = False
            return

        while self.running:
            try:
                raw_data = self.recvall(self.packet_size)
                if not self.running:
                    break
                    
                if not raw_data:
                    self.disconnected_signal.emit()
                    self.running = False
                    break
                    
                # Unpack N-bit little-endian signed integers (BioSemi format)
                values = []
                for ch in range(self.channels):
                    offset = ch * self.bytes_per_sample
                    sample_bytes = raw_data[offset:offset + self.bytes_per_sample]
                    raw_int = int.from_bytes(sample_bytes, byteorder='little', signed=True)
                    uv_value = raw_int / self.uv_to_bits
                    values.append(uv_value)
                self.new_data_signal.emit(values)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.error_signal.emit(f"Connection error: {e}")
                self.running = False
                break
                
        self.sock.close()

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass

# ──────────────────────────────────────────────────────
# Background classification worker
# ──────────────────────────────────────────────────────

class ClassificationWorker(QObject):
    """Runs filter → features → smooth → classify in a background thread."""
    result_ready = pyqtSignal(object, object)  # (features_62x5, probs_4)
    
    def __init__(self, sos, zi_template, model, stft_n, sample_rate):
        super().__init__()
        self.sos = sos
        self._zi_template = zi_template
        self.model = model
        self.stft_n = stft_n
        self.sf = sample_rate
        
        self._queue = Queue()
        self._running = False
        
        self.filter_zi = None
        self.raw_features = []
    
    def reset(self):
        n_sections = self._zi_template.shape[0]
        self.filter_zi = np.zeros((n_channels, n_sections, 2))
        for ch in range(n_channels):
            self.filter_zi[ch] = self._zi_template.copy()
        self.raw_features = []
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except: break
    
    def enqueue(self, segment):
        self._queue.put(("segment", segment.copy()))
    
    def stop(self):
        self._running = False
        self._queue.put(("stop", None))
        
    def is_empty(self):
        return self._queue.empty()
    
    def run(self):
        self._running = True
        while self._running:
            item = self._queue.get()
            tag, data = item
            if tag == "stop":
                break
            elif tag == "segment":
                self._process(data)
    
    def _process(self, segment):
        filtered, self.filter_zi = filter_segment(segment, self.sos, self.filter_zi)
        features = extract_single_window_features(filtered, self.stft_n, self.sf)
        self.raw_features.append(features)
        
        features_arr = np.array(self.raw_features)
        T = features_arr.shape[0]
        window = min(5, T)
        smoothed = features_arr[max(0, T-window):T].mean(axis=0)
        
        if self.model is not None:
            input_tensor = torch.tensor(smoothed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        else:
            probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        self.result_ready.emit(features, probs)


# ──────────────────────────────────────────────────────
# Real-Time View
# ──────────────────────────────────────────────────────

class SimulatorView(QWidget):
    navigate_to_home_signal = pyqtSignal()
    
    EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.sf = sf
        self.stft_n = 256
        self.window_samples = self.sf
        
        self.is_connected = False
        self.playhead_samples = 0
        
        # Buffers for UI
        self.display_buffer_len = self.sf * 10 # 10 seconds history
        self.display_data = np.zeros((n_channels, self.display_buffer_len))
        self.emotion_probs = []
        
        # Full history buffer for post-disconnect review
        self.full_history_capacity = self.sf * 60 * 5  # Pre-allocate 5 minutes
        self.full_history = np.zeros((n_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.review_mode = False
        self.awaiting_worker_finish = False
        
        # Buffer for Worker Model
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        
        self.pending_samples = []
        
        # Update plotting at ~30Hz
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_gui_plot)
        
        self.model = None
        self._load_model()
        
        self.sos, self._zi_template = create_bandpass_filter(self.sf, 0.1, 75.0)
        
        self.worker_thread = QThread()
        self.worker = ClassificationWorker(
            self.sos, self._zi_template, self.model, self.stft_n, self.sf
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self._on_classification_result)
        
        self.stream_thread = None
        
        # Music synthesizer
        self.synth = RealTimeMusicSynthesizer()
        self.synth.note_played.connect(self._on_note_played)
        self.synth.state_update.connect(self._on_synth_state_update)
        self.synth.start()
        
        self._setup_ui()
    
    def _load_model(self):
        try:
            self.model = EEGResNet(num_classes=4)
            checkpoint = torch.load("models/best_model_stft_smooth.pt", map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
    
    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        root_layout.addWidget(self.scroll_area)

        content_widget = QWidget()
        self.scroll_area.setWidget(content_widget)
        main_layout = QVBoxLayout(content_widget)
        
        # ── Top bar ──
        top_bar = QHBoxLayout()
        self.back_btn = QPushButton("← Back to Menu")
        self.back_btn.clicked.connect(self._on_back_clicked)
        top_bar.addWidget(self.back_btn)
        
        top_bar.addStretch()
        
        title_lbl = QLabel("Headset Simulator Monitor")
        font = title_lbl.font()
        font.setPointSize(14)
        font.setBold(True)
        title_lbl.setFont(font)
        top_bar.addWidget(title_lbl)
        
        top_bar.addStretch()
        
        self.status_lbl = QLabel("Status: Disconnected")
        self.status_lbl.setFont(font)
        top_bar.addWidget(self.status_lbl)
        
        main_layout.addLayout(top_bar)
        
        # ── Connection Settings ──
        self.settings_group = QGroupBox("Connection Settings")
        settings_layout = QVBoxLayout()
        
        # Row 1: Host + Port
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Host:"))
        self.input_host = QLineEdit(DEFAULT_HOST)
        self.input_host.setMaximumWidth(160)
        row1.addWidget(self.input_host)
        
        row1.addSpacing(15)
        row1.addWidget(QLabel("Port:"))
        self.input_port = QLineEdit(str(DEFAULT_PORT))
        self.input_port.setValidator(QIntValidator(1, 65535))
        self.input_port.setMaximumWidth(80)
        row1.addWidget(self.input_port)
        
        row1.addStretch()
        settings_layout.addLayout(row1)
        
        # Row 2: Channels + Sample Rate
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Channels:"))
        self.input_channels = QLineEdit(str(DEFAULT_CHANNELS))
        self.input_channels.setValidator(QIntValidator(1, 256))
        self.input_channels.setMaximumWidth(60)
        row2.addWidget(self.input_channels)
        
        row2.addSpacing(15)
        row2.addWidget(QLabel("Sample Rate (Hz):"))
        self.input_sample_rate = QLineEdit(str(DEFAULT_SAMPLE_RATE))
        self.input_sample_rate.setValidator(QIntValidator(1, 10000))
        self.input_sample_rate.setMaximumWidth(80)
        row2.addWidget(self.input_sample_rate)
        
        row2.addStretch()
        settings_layout.addLayout(row2)
        
        # Row 3: Bytes/Sample + µV/Bit
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Bytes per Sample:"))
        self.input_bytes_per_sample = QLineEdit(str(DEFAULT_BYTES_PER_SAMPLE))
        self.input_bytes_per_sample.setValidator(QIntValidator(1, 8))
        self.input_bytes_per_sample.setMaximumWidth(40)
        row3.addWidget(self.input_bytes_per_sample)
        
        row3.addSpacing(15)
        row3.addWidget(QLabel("µV to Bits (ADC res.):"))
        self.input_uv_to_bits = QLineEdit(str(DEFAULT_UV_TO_BITS))
        self.input_uv_to_bits.setValidator(QIntValidator(1, 1000000))
        self.input_uv_to_bits.setMaximumWidth(100)
        row3.addWidget(self.input_uv_to_bits)
        
        row3.addStretch()
        settings_layout.addLayout(row3)
        
        # Packet info label
        self.packet_info_lbl = QLabel()
        self._update_packet_info()
        settings_layout.addWidget(self.packet_info_lbl)
        
        self.input_channels.textChanged.connect(self._update_packet_info)
        self.input_bytes_per_sample.textChanged.connect(self._update_packet_info)
        
        self.settings_group.setLayout(settings_layout)
        main_layout.addWidget(self.settings_group)
        
        # ── Network controls ──
        controls_bar = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Listening")
        self.start_btn.clicked.connect(self.toggle_listening)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(150)
        controls_bar.addWidget(self.start_btn)
        
        controls_bar.addSpacing(20)
        
        self.detection_checkbox = QCheckBox("Enable Emotion Detection")
        self.detection_checkbox.setChecked(True)
        self.detection_checkbox.toggled.connect(self._on_detection_toggled)
        controls_bar.addWidget(self.detection_checkbox)
        
        controls_bar.addSpacing(20)
        
        self.music_checkbox = QCheckBox("Enable Music Generation")
        self.music_checkbox.setChecked(False)
        self.music_checkbox.toggled.connect(self._on_music_toggled)
        controls_bar.addWidget(self.music_checkbox)
        
        controls_bar.addSpacing(20)
        
        self.clear_btn = QPushButton("Clear Signal")
        self.clear_btn.clicked.connect(self._clear_signal)
        controls_bar.addWidget(self.clear_btn)
        
        controls_bar.addStretch()
        main_layout.addLayout(controls_bar)
        
        # ── Music controls ──
        music_bar = QHBoxLayout()
        music_bar.addWidget(QLabel("🔈"))
        self.vol_slider = QScrollBar(Qt.Horizontal)
        self.vol_slider.setRange(0, 127)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        music_bar.addWidget(self.vol_slider)

        music_bar.addSpacing(20)
        self.time_label = QLabel("00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        music_bar.addWidget(self.time_label)
        music_bar.addStretch()
        main_layout.addLayout(music_bar)
        
        # ── Channel selection ──
        ch_bar = QHBoxLayout()
        ch_bar.addWidget(QLabel("Channels:"))
        
        self.channel_mode_combo = QComboBox()
        self.channel_mode_combo.addItems(["Single Channel", "Channel Range", "All Channels"])
        self.channel_mode_combo.currentIndexChanged.connect(self._on_channel_mode_changed)
        ch_bar.addWidget(self.channel_mode_combo)
        
        self.label_ch_from = QLabel("Ch:")
        ch_bar.addWidget(self.label_ch_from)
        self.spin_ch_from = QSpinBox()
        self.spin_ch_from.setRange(1, n_channels)
        self.spin_ch_from.setValue(1)
        self.spin_ch_from.valueChanged.connect(self._on_channel_changed)
        ch_bar.addWidget(self.spin_ch_from)
        
        self.label_ch_to = QLabel("to:")
        ch_bar.addWidget(self.label_ch_to)
        self.spin_ch_to = QSpinBox()
        self.spin_ch_to.setRange(1, n_channels)
        self.spin_ch_to.setValue(5)
        self.spin_ch_to.valueChanged.connect(self._on_channel_changed)
        ch_bar.addWidget(self.spin_ch_to)
        
        # Init visibility for single channel mode
        self.label_ch_to.setVisible(False)
        self.spin_ch_to.setVisible(False)
        
        ch_bar.addStretch()
        main_layout.addLayout(ch_bar)
        
        # ── Review controls (hidden during streaming) ──
        self.review_widget = QWidget()
        review_layout = QHBoxLayout(self.review_widget)
        review_layout.setContentsMargins(0, 0, 0, 0)
        
        review_layout.addWidget(QLabel("Review Mode:"))
        
        self.radio_full_view = QRadioButton("Full Signal")
        self.radio_full_view.setChecked(True)
        self.radio_full_view.toggled.connect(self._on_review_mode_changed)
        review_layout.addWidget(self.radio_full_view)
        
        self.radio_window_view = QRadioButton("Windowed View")
        self.radio_window_view.toggled.connect(self._on_review_mode_changed)
        review_layout.addWidget(self.radio_window_view)
        
        review_layout.addSpacing(10)
        self.label_window_size = QLabel("Window (s):")
        review_layout.addWidget(self.label_window_size)
        
        self.spin_window_size = QDoubleSpinBox()
        self.spin_window_size.setRange(0.5, 60.0)
        self.spin_window_size.setValue(5.0)
        self.spin_window_size.setSingleStep(0.5)
        self.spin_window_size.valueChanged.connect(self._on_review_window_changed)
        review_layout.addWidget(self.spin_window_size)
        
        review_layout.addStretch()
        self.review_widget.setVisible(False)
        main_layout.addWidget(self.review_widget)
        
        # ── Plots (fixed heights — page scrolls vertically) ──
        self.eeg_plot = EegPlotWidget()
        self.eeg_plot.setFixedHeight(350)
        main_layout.addWidget(self.eeg_plot)
        
        self.emotion_plot = EmotionPlotWidget()
        self.emotion_plot.setFixedHeight(280)
        main_layout.addWidget(self.emotion_plot)
        
        # ── Piano Roll ──
        self.piano_roll = PianoRollWidget()
        self.piano_roll.setFixedHeight(320)
        main_layout.addWidget(self.piano_roll)
        
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.valueChanged.connect(self._update_review_plots)
        self.time_scrollbar.setVisible(False)
        main_layout.addWidget(self.time_scrollbar)
        
    def _on_back_clicked(self):
        self.stop_listening()
        self.synth.pause()
        self.synth.reset_state()
        self.navigate_to_home_signal.emit()

    def _on_detection_toggled(self, checked):
        """When emotion detection is off, music must also be off."""
        if not checked:
            self.music_checkbox.setChecked(False)
            self.music_checkbox.setEnabled(False)
        else:
            self.music_checkbox.setEnabled(True)

    def _on_music_toggled(self, checked):
        if checked:
            self.synth.play()
        else:
            self.synth.pause()
            self.synth.reset_state()
            self.piano_roll.clear_notes()

    def _on_volume_changed(self, value):
        self.synth.set_volume(value)

    def _on_note_played(self, channel, pitch, velocity, start_time, duration):
        total_s = max(60.0, self.total_samples_received / self.sf) if self.total_samples_received > 0 else 60.0
        self.piano_roll.total_time = total_s
        self.piano_roll.add_note(start_time, duration, pitch, velocity)
        self.piano_roll.update_playhead(start_time)
        # Update time label
        mins = int(start_time // 60)
        secs = int(start_time % 60)
        self.time_label.setText(f"{mins:02d}:{secs:02d}")

    def _on_synth_state_update(self, mode, chord_type, bpm):
        pass

    # ── Channel helpers ──────────────────────────────────
    
    def _get_selected_channels(self):
        """Return list of (label, data_row_index) based on current channel selection."""
        mode = self.channel_mode_combo.currentIndex()
        if mode == 0:  # Single
            ch = self.spin_ch_from.value() - 1
            return [(f"Ch {ch+1}", ch)]
        elif mode == 1:  # Range
            ch_from = self.spin_ch_from.value() - 1
            ch_to = self.spin_ch_to.value() - 1
            if ch_from > ch_to:
                ch_from, ch_to = ch_to, ch_from
            return [(f"Ch {i+1}", i) for i in range(ch_from, ch_to + 1)]
        else:  # All
            return [(f"Ch {i+1}", i) for i in range(n_channels)]
    
    def _on_channel_mode_changed(self, index):
        is_range = (index == 1)
        is_all = (index == 2)
        
        self.label_ch_from.setVisible(not is_all)
        self.spin_ch_from.setVisible(not is_all)
        self.label_ch_to.setVisible(is_range)
        self.spin_ch_to.setVisible(is_range)
        self.label_ch_from.setText("Ch:" if not is_range else "From:")
        
        if self.review_mode:
            self._update_review_plots()
            
    def _on_channel_changed(self):
        if self.review_mode:
            self._update_review_plots()

    def _update_packet_info(self):
        try:
            ch = int(self.input_channels.text())
            bps = int(self.input_bytes_per_sample.text())
        except (ValueError, AttributeError):
            ch, bps = DEFAULT_CHANNELS, DEFAULT_BYTES_PER_SAMPLE
        packet = ch * bps
        self.packet_info_lbl.setText(
            f"Packet size: {ch} ch × {bps} bytes = {packet} bytes/sample"
        )

    def _clear_signal(self):
        """Reset all buffers and plots while optionally staying connected."""
        self.playhead_samples = 0
        self.display_data = np.zeros((n_channels, self.display_buffer_len))
        self.emotion_probs = []
        self.full_history_capacity = self.sf * 60 * 5
        self.full_history = np.zeros((n_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.pending_samples = []
        
        # Clear the plots
        self.eeg_plot.set_data([], np.array([]), "EEG Signal")
        self.emotion_plot.set_data(
            np.zeros((1, 4)), np.array([0]), 0, 1
        )
        self.piano_roll.clear_notes()
        self.time_label.setText("00:00")
        
        # Reset music state
        self.synth.pause()
        self.synth.reset_state()
        
        # Hide review mode if active
        self.review_mode = False
        self.review_widget.setVisible(False)
        self.time_scrollbar.setVisible(False)

    # ── Streaming state ──────────────────────────────────

    def toggle_listening(self):
        if not self.is_connected:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        self.start_btn.setText("Connecting...")
        self.start_btn.setEnabled(False)
        
        # Read user-configured parameters
        host = self.input_host.text().strip() or DEFAULT_HOST
        port = int(self.input_port.text() or DEFAULT_PORT)
        channels = int(self.input_channels.text() or DEFAULT_CHANNELS)
        sample_rate = int(self.input_sample_rate.text() or DEFAULT_SAMPLE_RATE)
        bytes_per_sample = int(self.input_bytes_per_sample.text() or DEFAULT_BYTES_PER_SAMPLE)
        uv_to_bits = int(self.input_uv_to_bits.text() or DEFAULT_UV_TO_BITS)
        
        # Reset state
        self.review_mode = False
        self.awaiting_worker_finish = False
        self.review_widget.setVisible(False)
        self.time_scrollbar.setVisible(False)
        self.playhead_samples = 0
        self.display_data = np.zeros((n_channels, self.display_buffer_len))
        self.emotion_probs = []
        self.full_history_capacity = self.sf * 60 * 5
        self.full_history = np.zeros((n_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.pending_samples = []

        self.worker.reset()
        if not self.worker_thread.isRunning():
            self.worker_thread.start()

        self.stream_thread = DataStreamThread(host, port, channels, bytes_per_sample, uv_to_bits)
        self.stream_thread.new_data_signal.connect(self.on_new_data)
        self.stream_thread.error_signal.connect(self.on_connection_error)
        self.stream_thread.disconnected_signal.connect(self.on_disconnected)
        
        # Disable settings while connected
        self.settings_group.setEnabled(False)
        
        # Change state
        self.is_connected = True
        self.status_lbl.setText(f"Status: Connected to {host}:{port}")
        self.start_btn.setText("Stop Listening")
        self.start_btn.setEnabled(True)
        
        self.stream_thread.start()
        self.ui_timer.start(33) # ~30 fps
        
        # Start music if enabled
        if self.music_checkbox.isChecked():
            self.synth.play()

    def stop_listening(self, wait_worker=False):
        self.is_connected = False
        self.ui_timer.stop()
        
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread.wait()
            self.stream_thread = None
            
        if not wait_worker:
            if self.worker_thread.isRunning():
                self.worker.stop()
                self.worker_thread.quit()
                self.worker_thread.wait(2000)
            
        self.status_lbl.setText("Status: Disconnected")
        self.start_btn.setText("Start Listening")
        self.settings_group.setEnabled(True)
        
        # Pause music on disconnect (but keep piano roll visible for review)
        self.synth.pause()

    def on_new_data(self, data):
        # Thread-safely push data into pending list
        self.pending_samples.append(data)

    def on_connection_error(self, err_msg):
        self.stop_listening()
        QMessageBox.critical(self, "Connection Error", err_msg)

    def on_disconnected(self):
        # We don't shut the worker down yet, just the stream.
        self.stop_listening(wait_worker=True)
        self.status_lbl.setText("Status: Stream ended. Processing remaining data...")
        self.awaiting_worker_finish = True
        
        # We handle entering review mode in `_on_classification_result`
        # once the worker queue is empty, or immediately if already empty.
        if self.worker.is_empty():
            self._enter_review_mode()
            
    def _enter_review_mode(self):
        self.status_lbl.setText("Status: Reviewing Completed Session")
        self.review_mode = True
        self.awaiting_worker_finish = False
        self.review_widget.setVisible(True)
        self.radio_full_view.setChecked(True)
        self._on_review_mode_changed()
        
        # Stop the worker thread now that we are done
        if self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(2000)

    # ── Review controls ──────────────────────────────────

    def _on_review_mode_changed(self):
        if not self.review_mode:
            return
            
        is_windowed = self.radio_window_view.isChecked()
        self.label_window_size.setEnabled(is_windowed)
        self.spin_window_size.setEnabled(is_windowed)
        self.time_scrollbar.setVisible(is_windowed)
        self.time_scrollbar.setEnabled(is_windowed)
        
        if is_windowed:
            self._update_review_scrollbar()
            
        self._update_review_plots()
        
    def _on_review_window_changed(self):
        if self.review_mode and self.radio_window_view.isChecked():
            self._update_review_scrollbar()
            self._update_review_plots()
            
    def _update_review_scrollbar(self):
        if self.total_samples_received == 0:
            return
            
        total_seconds = self.total_samples_received / self.sf
        window_size = self.spin_window_size.value()
        max_scroll = max(0, int((total_seconds - window_size) * self.sf))
        
        self.time_scrollbar.setMaximum(max_scroll)
        self.time_scrollbar.setSingleStep(int(self.sf * 0.1))
        self.time_scrollbar.setPageStep(int(self.sf * window_size * 0.5))

    def _update_review_plots(self):
        if not self.review_mode or self.total_samples_received == 0:
            return
            
        if self.radio_full_view.isChecked():
            start_sample = 0
            end_sample = self.total_samples_received
        else:
            window_size_samples = int(self.spin_window_size.value() * self.sf)
            start_sample = self.time_scrollbar.value()
            end_sample = min(self.total_samples_received, start_sample + window_size_samples)
            
        if start_sample >= end_sample:
            return
            
        # Plot EEG
        time_axis = np.arange(start_sample, end_sample) / self.sf
        channels = self._get_selected_channels()
        ch_data = [(label, self.full_history[idx, start_sample:end_sample]) for label, idx in channels]
        self.eeg_plot.set_data(ch_data, time_axis, "EEG Signal — Review Mode")
        
        # Plot Emotions
        if len(self.emotion_probs) > 0:
            probs_arr = np.array(self.emotion_probs)
            emotion_time = np.arange(len(self.emotion_probs)) * 1.0 # 1 second segments
            
            start_time = start_sample / self.sf
            end_time = end_sample / self.sf
            
            e_start = max(0, int(start_time))
            e_end = min(len(self.emotion_probs), int(np.ceil(end_time)))
            
            if e_start < e_end:
                self.emotion_plot.set_data(
                    probs_arr[e_start:e_end],
                    emotion_time[e_start:e_end],
                    start_time, end_time
                )

    def _update_gui_plot(self):
        """Timer callback (30 FPS) to flush collected network samples to the rolling display"""
        if not self.is_connected or not self.pending_samples:
            return
            
        # Flush all pending samples
        samples_to_flush = self.pending_samples[:]
        self.pending_samples.clear()
        
        num_new = len(samples_to_flush)
        if num_new == 0:
            return
            
        # Convert to numpy block (num_new, 62) -> transpose to (62, num_new)
        new_block = np.array(samples_to_flush).T
        
        # Append to full history, resizing if necessary
        if self.total_samples_received + num_new > self.full_history_capacity:
            self.full_history_capacity = max(self.full_history_capacity * 2, self.total_samples_received + num_new + self.sf * 60)
            new_history = np.zeros((n_channels, self.full_history_capacity))
            new_history[:, :self.total_samples_received] = self.full_history[:, :self.total_samples_received]
            self.full_history = new_history
            
        self.full_history[:, self.total_samples_received:self.total_samples_received + num_new] = new_block
        self.total_samples_received += num_new
        
        # Shift display buffer left by num_new and insert new data
        shift = min(num_new, self.display_buffer_len)
        if shift < self.display_buffer_len:
            self.display_data[:, :-shift] = self.display_data[:, shift:]
        self.display_data[:, -shift:] = new_block[:, -shift:]
        
        # Build classification buffer
        remaining = num_new
        src_offset = 0
        while remaining > 0:
            space = self.window_samples - self.buffer_pos
            take = min(remaining, space)
            
            self.classification_buffer[:, self.buffer_pos:self.buffer_pos + take] = new_block[:, src_offset:src_offset + take]
            self.buffer_pos += take
            src_offset += take
            remaining -= take
            
            if self.buffer_pos >= self.window_samples:
                if self.detection_checkbox.isChecked():
                    self.worker.enqueue(self.classification_buffer)
                self.buffer_pos = 0

        self.playhead_samples += num_new

        # Update EEG widget (showing full 10 seconds rolling)
        cur_t = self.playhead_samples / self.sf
        t_start = max(0, cur_t - (self.display_buffer_len / self.sf))
        time_axis = np.linspace(t_start, cur_t, self.display_buffer_len)
        
        channels = self._get_selected_channels()
        ch_data = [(label, self.display_data[idx, :]) for label, idx in channels]
        
        self.eeg_plot.set_data(
            ch_data, time_axis,
            f"EEG Signal — Live stream ({cur_t:.1f}s)"
        )

    # ── Worker callbacks (main thread) ───────────────────
    
    def _on_classification_result(self, features, probs):
        self.emotion_probs.append(probs)
        
        # Send to music synthesizer if enabled
        if self.music_checkbox.isChecked():
            timestamp = len(self.emotion_probs) - 1  # 1-second segments
            self.synth.update_emotion(probs, timestamp)
        
        n_segs = len(self.emotion_probs)
        if n_segs > 0:
            probs_arr = np.array(self.emotion_probs)
            e_time = np.arange(n_segs)
            
            # Show full timeline from the beginning
            view_start = 0
            view_end = e_time[-1] + 1
            
            self.emotion_plot.set_data(
                probs_arr, e_time,
                view_start,
                view_end
            )
            
            dominant = self.EMOTION_LABELS[np.argmax(probs)]
            # Only update status if we aren't waiting for the finish
            if not self.awaiting_worker_finish:
                self.status_lbl.setText(f"Status: Connected | Dominant Emotion: {dominant}")
        
        # Check if stream ended and we just drained the last item from the queue
        if self.awaiting_worker_finish and self.worker.is_empty():
            self._enter_review_mode()

# This view is for the live-acquisiton headset case, where the user is prompted to
# connect to te TCP/IP stream of the headset and employs the pipeline live

import os
import sys
import math
import numpy as np
import torch
import scipy.signal
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QGroupBox, QMessageBox, QFrame,
    QRadioButton, QDoubleSpinBox, QScrollBar, QSpinBox, QLineEdit, QCheckBox,
    QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QThread
from PyQt5.QtGui import QIntValidator

from src.eeg_pipeline.signal_processing import sf, n_channels
from src.eeg_pipeline.emotion_classifier import load_emotion_model
from src.eeg_pipeline.classification_worker import ClassificationWorker
from src.eeg_pipeline.segment_processor import SegmentProcessor
from src.ui.components.eeg_plots import EegPlotWidget, EmotionPlotWidget, BandZScorePlotWidget, AsymmetryGaugeWidget
from src.ui.components.piano_roll import PianoRollWidget
from src.ui.components.channel_selector import ChannelSelectorWidget
from src.ui.components.data_stream import (
    DataStreamThread,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_UV_TO_BITS,
    DEFAULT_BYTES_PER_SAMPLE,
    DEFAULT_SAMPLES_PER_PACKET
)
from src.music.orchestrators.realtime_generator import RealTimeMusicSynthesizer

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
        
        self.display_channels = n_channels
        self.display_buffer_len = self.sf * 10
        self.display_data = np.zeros((self.display_channels, self.display_buffer_len))
        self.emotion_probs = []
        
        self.full_history_capacity = self.sf * 60 * 5
        self.full_history = np.zeros((self.display_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.review_mode = False
        self.awaiting_worker_finish = False
        
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.pending_samples = []
        
        self.headset_sr = sf
        
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_gui_plot)
        
        self.model = load_emotion_model()
        self._pending_band_powers: dict = {}  # keyed by timestamp, matches probs to band_powers

        # Separate threads for SegmentProcessor and ClassificationWorker
        self.seg_proc_thread = QThread()
        self.worker_thread   = QThread()
        self.seg_proc        = SegmentProcessor(self.stft_n, self.sf)
        self.worker          = ClassificationWorker(self.model)
        self.seg_proc.moveToThread(self.seg_proc_thread)
        self.worker.moveToThread(self.worker_thread)
        self.seg_proc_thread.started.connect(self.seg_proc.run)
        self.worker_thread.started.connect(self.worker.run)
        # SegmentProcessor - cache band_powers + forward de to ClassificationWorker
        self.seg_proc.segment_processed.connect(self._on_segment_processed)
        # ClassificationWorker - combine with cached band_powers - emit result
        self.worker.classification_done.connect(self._on_classification_done)
        
        self.stream_thread = None
        
        # Music synthesizer
        self.synth = RealTimeMusicSynthesizer()
        self.synth.note_played.connect(self._on_note_played)
        self.synth.state_update.connect(self._on_synth_state_update)
        self.synth.start()
        
        self._setup_ui()
    

    
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
        
        main_layout.addLayout(top_bar)
        
        self.settings_group = QGroupBox("Connection Settings")
        settings_layout = QVBoxLayout()
        
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
        
        row2.addSpacing(15)
        row2.addWidget(QLabel("Samples/Packet:"))
        self.input_samples_per_packet = QLineEdit(str(DEFAULT_SAMPLES_PER_PACKET))
        self.input_samples_per_packet.setValidator(QIntValidator(1, 1024))
        self.input_samples_per_packet.setMaximumWidth(60)
        row2.addWidget(self.input_samples_per_packet)
        
        row2.addStretch()
        settings_layout.addLayout(row2)
        
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
        
        self.packet_info_lbl = QLabel()
        self._update_packet_info()
        settings_layout.addWidget(self.packet_info_lbl)
        
        self.input_channels.textChanged.connect(self._update_packet_info)
        self.input_bytes_per_sample.textChanged.connect(self._update_packet_info)
        self.input_samples_per_packet.textChanged.connect(self._update_packet_info)
        
        self.settings_group.setLayout(settings_layout)
        main_layout.addWidget(self.settings_group)
        
        controls_bar = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Listening")
        self.start_btn.clicked.connect(self.toggle_listening)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(150)
        controls_bar.addWidget(self.start_btn)
        
        controls_bar.addSpacing(20)
        
        self.detection_checkbox = QCheckBox("Enable Emotion Detection")
        self.detection_checkbox.setChecked(False)
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
        
        dashboard_layout = QHBoxLayout()
        
        self.status_card = QWidget()
        self.status_card.setObjectName("dashboardCard")
        status_card_layout = QVBoxLayout(self.status_card)
        status_card_layout.setContentsMargins(8, 4, 8, 4)
        status_title = QLabel("System Status")
        status_title.setObjectName("cardTitle")
        self.status_val = QLabel("Disconnected")
        self.status_val.setObjectName("cardValue")
        self.status_val.setWordWrap(True)
        status_card_layout.addWidget(status_title)
        status_card_layout.addWidget(self.status_val)
        dashboard_layout.addWidget(self.status_card, stretch=2)
        
        self.segments_card = QWidget()
        self.segments_card.setFixedWidth(140)
        self.segments_card.setObjectName("dashboardCard")
        segments_card_layout = QVBoxLayout(self.segments_card)
        segments_card_layout.setContentsMargins(8, 4, 8, 4)
        segments_title = QLabel("EEG Segments")
        segments_title.setObjectName("cardTitle")
        self.segments_val = QLabel("0 classified")
        self.segments_val.setObjectName("cardValue")
        self.segments_val.setAlignment(Qt.AlignCenter)
        segments_card_layout.addWidget(segments_title)
        segments_card_layout.addWidget(self.segments_val)
        dashboard_layout.addWidget(self.segments_card)
        
        self.emotion_card = QWidget()
        self.emotion_card.setFixedWidth(160)
        self.emotion_card.setObjectName("dashboardCard")
        emotion_card_layout = QVBoxLayout(self.emotion_card)
        emotion_card_layout.setContentsMargins(8, 4, 8, 4)
        emotion_title = QLabel("Dominant Emotion")
        emotion_title.setObjectName("cardTitle")
        self.emotion_val = QLabel("—")
        self.emotion_val.setObjectName("cardValue")
        self.emotion_val.setAlignment(Qt.AlignCenter)
        emotion_card_layout.addWidget(emotion_title)
        emotion_card_layout.addWidget(self.emotion_val)
        dashboard_layout.addWidget(self.emotion_card)
        
        dashboard_layout.setSpacing(10)
        main_layout.addLayout(dashboard_layout)
        
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
        
        self.channel_selector = ChannelSelectorWidget(max_channels=n_channels)
        self.channel_selector.selection_changed.connect(self._on_channel_changed)
        main_layout.addWidget(self.channel_selector)
        
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
        
        self.eeg_plot = EegPlotWidget()
        self.eeg_plot.setFixedHeight(250)
        main_layout.addWidget(self.eeg_plot)
        
        self.emotion_plot = EmotionPlotWidget()
        self.emotion_plot.setFixedHeight(280)
        main_layout.addWidget(self.emotion_plot)
        
        self.asymmetry_gauge = AsymmetryGaugeWidget()
        main_layout.addWidget(self.asymmetry_gauge)
        
        self.zscore_plot = BandZScorePlotWidget()
        self.zscore_plot.setFixedHeight(250)
        main_layout.addWidget(self.zscore_plot)
        
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
        # music generation is off when classification is off
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
    
    def _get_selected_channels(self):
        return self.channel_selector.get_selected_channels(self.display_channels)
    
    def _on_channel_changed(self):
        if self.review_mode:
            self._update_review_plots()

    def _update_packet_info(self):
        try:
            ch = int(self.input_channels.text())
            bps = int(self.input_bytes_per_sample.text())
            spp = int(self.input_samples_per_packet.text())
        except (ValueError, AttributeError):
            ch, bps, spp = DEFAULT_CHANNELS, DEFAULT_BYTES_PER_SAMPLE, DEFAULT_SAMPLES_PER_PACKET
        packet = ch * bps * spp
        self.packet_info_lbl.setText(
            f"Packet size: {ch} ch × {bps} bytes × {spp} samples = {packet} bytes/packet"
            + (f"  —  first {n_channels} channels used by model" if ch > n_channels else "")
        )

    def _clear_signal(self):
        # Reset all buffers and plots
        self.playhead_samples = 0
        self.display_data = np.zeros((self.display_channels, self.display_buffer_len))
        self.emotion_probs = []
        self.full_history_capacity = self.headset_sr * 60 * 5
        self.full_history = np.zeros((self.display_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.window_samples = self.headset_sr if hasattr(self, 'headset_sr') else sf
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.pending_samples = []
        
        # Clear the plots
        self.eeg_plot.set_data([], np.array([]), "EEG Signal")
        self.zscore_plot.clear_data()
        self.asymmetry_gauge.set_asymmetry(0.0)
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
        samples_per_packet = int(self.input_samples_per_packet.text() or DEFAULT_SAMPLES_PER_PACKET)
        
        # Store headset SR for downsampling in _update_gui_plot
        self.headset_sr = max(1, sample_rate)
        self.display_channels = channels
        self.channel_selector.set_max_channels(channels)
        
        # Reset state
        self.review_mode = False
        self.awaiting_worker_finish = False
        self.review_widget.setVisible(False)
        self.time_scrollbar.setVisible(False)
        self.playhead_samples = 0
        
        self.display_buffer_len = self.headset_sr * 10
        self.display_data = np.zeros((self.display_channels, self.display_buffer_len))
        
        self.emotion_probs = []
        self.full_history_capacity = self.headset_sr * 60 * 5
        self.full_history = np.zeros((self.display_channels, self.full_history_capacity))
        self.total_samples_received = 0
        self.window_samples = self.headset_sr
        self.classification_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.pending_samples = []

        self.seg_proc.reset()
        self.worker.reset()
        self._pending_band_powers.clear()
        if not self.seg_proc_thread.isRunning():
            self.seg_proc_thread.start()
        if not self.worker_thread.isRunning():
            self.worker_thread.start()

        self.stream_thread = DataStreamThread(host, port, channels, bytes_per_sample, uv_to_bits, samples_per_packet)
        self.stream_thread.new_data_signal.connect(self.on_new_data)
        self.stream_thread.error_signal.connect(self.on_connection_error)
        self.stream_thread.disconnected_signal.connect(self.on_disconnected)
        
        self.settings_group.setEnabled(False)
        self.is_connected = True
        self.status_val.setText(f"Connected to {host}:{port}")
        self.emotion_val.setText("—")
        self.segments_val.setText("0 classified")
        self.start_btn.setText("Stop Listening")
        self.start_btn.setEnabled(True)
        
        self.stream_thread.start()
        self.ui_timer.start(33)
        
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
            if self.seg_proc_thread.isRunning():
                self.seg_proc.stop()
                self.seg_proc_thread.quit()
                self.seg_proc_thread.wait(2000)
            if self.worker_thread.isRunning():
                self.worker.stop()
                self.worker_thread.quit()
                self.worker_thread.wait(2000)
            
        self.status_val.setText("Disconnected")
        self.emotion_val.setText("—")
        self.start_btn.setText("Start Listening")
        self.settings_group.setEnabled(True)
        
        self.synth.pause()

    def on_new_data(self, all_samples):
        # all_samples is a list of per-sample lists (one list per sample in the packet)
        for sample in all_samples:
            self.pending_samples.append(sample[:self.display_channels])

    def on_connection_error(self, err_msg):
        self.stop_listening()
        QMessageBox.critical(self, "Connection Error", err_msg)

    def on_disconnected(self):
        # We don't shut the worker down yet, just the stream.
        self.stop_listening(wait_worker=True)
        self.status_val.setText("Stream ended. Processing remaining data...")
        self.awaiting_worker_finish = True
        
        # We handle entering review mode in _on_classification_result
        # once the worker queue is empty
        if self.seg_proc.is_empty() and self.worker.is_empty():
            self._enter_review_mode()
            
    def _enter_review_mode(self):
        self.status_val.setText("Reviewing Completed Session")
        self.review_mode = True
        self.awaiting_worker_finish = False
        self.review_widget.setVisible(True)
        self.radio_full_view.setChecked(True)
        self._on_review_mode_changed()
        
        # Stop the worker thread now that we are done
        if self.seg_proc_thread.isRunning():
            self.seg_proc.stop()
            self.seg_proc_thread.quit()
            self.seg_proc_thread.wait(2000)
        if self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(2000)

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
            
        total_seconds = self.total_samples_received / self.headset_sr
        window_size = self.spin_window_size.value()
        max_scroll = max(0, int((total_seconds - window_size) * self.headset_sr))
        
        self.time_scrollbar.setMaximum(max_scroll)
        self.time_scrollbar.setSingleStep(int(self.headset_sr * 0.1))
        self.time_scrollbar.setPageStep(int(self.headset_sr * window_size * 0.5))

    def _update_review_plots(self):
        if not self.review_mode or self.total_samples_received == 0:
            return
            
        if self.radio_full_view.isChecked():
            start_sample = 0
            end_sample = self.total_samples_received
        else:
            window_size_samples = int(self.spin_window_size.value() * self.headset_sr)
            start_sample = self.time_scrollbar.value()
            end_sample = min(self.total_samples_received, start_sample + window_size_samples)
            
        if start_sample >= end_sample:
            return
            
        time_axis = np.arange(start_sample, end_sample) / self.headset_sr
        channels = self._get_selected_channels()
        ch_data = []
        for i, (label, idx) in enumerate(channels):
            data = self.full_history[idx, start_sample:end_sample].copy()
            if len(data) > 0:
                mean_val = np.mean(data)
                centered = data - mean_val
            else:
                centered = data
            offset = i * 150.0
            ch_data.append((label, centered + offset))
            
        self.eeg_plot.set_data(ch_data, time_axis, "EEG Signal — Review Mode")
        
        if len(self.emotion_probs) > 0:
            probs_arr = np.array(self.emotion_probs)
            emotion_time = np.arange(len(self.emotion_probs)) * 1.0
            
            start_time = start_sample / self.headset_sr
            end_time = end_sample / self.headset_sr
            
            e_start = max(0, int(start_time))
            e_end = min(len(self.emotion_probs), int(np.ceil(end_time)))
            
            if e_start < e_end:
                self.emotion_plot.set_data(
                    probs_arr[e_start:e_end],
                    emotion_time[e_start:e_end],
                    start_time, end_time
                )

    def _update_gui_plot(self):
        if not self.is_connected:
            return
        new_block = self._flush_pending_samples()
        if new_block is None:
            return
        self._update_history_buffer(new_block)
        self._update_display_buffer(new_block)
        if self.detection_checkbox.isChecked():
            self._feed_classification_buffer(new_block)
        self.playhead_samples += new_block.shape[1]
        self._refresh_eeg_plot()

    def _flush_pending_samples(self):
        # Move pending_samples list into a (channels, n) numpy array
        if not self.pending_samples:
            return None
        samples = self.pending_samples[:]
        self.pending_samples.clear()
        return np.array(samples, dtype=np.float32).T

    def _update_history_buffer(self, new_block: np.ndarray):
        # Append new samples to full_history
        # Array is extended if needed
        num_new = new_block.shape[1]
        if self.total_samples_received + num_new > self.full_history_capacity:
            self.full_history_capacity = max(
                self.full_history_capacity * 2,
                self.total_samples_received + num_new + self.headset_sr * 60
            )
            new_hist = np.zeros((self.display_channels, self.full_history_capacity))
            new_hist[:, :self.total_samples_received] = self.full_history[:, :self.total_samples_received]
            self.full_history = new_hist
        self.full_history[:, self.total_samples_received:self.total_samples_received + num_new] = new_block
        self.total_samples_received += num_new

    def _update_display_buffer(self, new_block: np.ndarray):
        num_new = new_block.shape[1]
        shift = min(num_new, self.display_buffer_len)
        if shift < self.display_buffer_len:
            self.display_data[:, :-shift] = self.display_data[:, shift:]
        self.display_data[:, -shift:] = new_block[:, -shift:]

    def _feed_classification_buffer(self, new_block: np.ndarray):
        # Fill 1-second window from new_block, downsample if needed and enqueue to worker
        model_block = new_block[:min(new_block.shape[0], n_channels), :]
        if model_block.shape[0] < n_channels:
            pad = np.zeros((n_channels - model_block.shape[0], model_block.shape[1]), dtype=np.float32)
            model_block = np.vstack([model_block, pad])

        remaining  = model_block.shape[1]
        src_offset = 0
        while remaining > 0:
            space = self.window_samples - self.buffer_pos
            take  = min(remaining, space)
            self.classification_buffer[:, self.buffer_pos:self.buffer_pos + take] = \
                model_block[:, src_offset:src_offset + take]
            self.buffer_pos += take
            src_offset      += take
            remaining       -= take
            if self.buffer_pos >= self.window_samples:
                ds_block = self._downsample_block(self.classification_buffer)
                seg_timestamp = (self.playhead_samples + src_offset) / self.headset_sr
                self.seg_proc.enqueue(ds_block, seg_timestamp)
                self.buffer_pos = 0

    def _downsample_block(self, block: np.ndarray) -> np.ndarray:
        if self.headset_sr == sf:
            return block.copy()
        g    = math.gcd(sf, self.headset_sr)
        up   = sf // g
        down = self.headset_sr // g
        return scipy.signal.resample_poly(block, up, down, axis=1).astype(np.float32)

    def _refresh_eeg_plot(self):
        cur_t   = self.playhead_samples / self.headset_sr
        t_start = max(0, cur_t - self.display_buffer_len / self.headset_sr)
        time_axis = np.linspace(t_start, cur_t, self.display_buffer_len)
        channels  = self._get_selected_channels()
        ch_data   = []
        for i, (label, idx) in enumerate(channels):
            data        = self.display_data[idx, :].copy()
            valid_start = max(0, self.display_buffer_len - self.playhead_samples)
            if self.playhead_samples > 0:
                valid_data = data[valid_start:]
                mean_val   = np.mean(valid_data) if len(valid_data) > 0 else 0.0
                if valid_start > 0:
                    data[:valid_start] = mean_val
                centered = data - mean_val
            else:
                centered = data
            ch_data.append((label, centered + i * 150.0))
        self.eeg_plot.set_data(
            ch_data, time_axis,
            f"EEG Signal (Real-time) — {self.headset_sr} Hz"
        )
        


    # ── Worker callbacks (main thread) ───────────────────

    def _on_segment_processed(self, de, band_powers, timestamp):
        # Cache band_powers so they can be matched when probs arrive
        self._pending_band_powers[timestamp] = band_powers
        self.worker.enqueue(de, timestamp)

    def _on_classification_done(self, probs, timestamp):
        band_powers = self._pending_band_powers.pop(timestamp, {})
        self.emotion_probs.append(probs)

        # Always process band powers through EEGTexturingEngine so the Z-score plot
        # has data even when music is disabled.
        te = self.synth.eeg_texturing_engine
        te.process(band_powers)

        if self.music_checkbox.isChecked():
            ts = timestamp if timestamp is not None else (len(self.emotion_probs) - 1)
            self.synth.update_emotion(probs, ts, band_powers)

        # Update Z-score plot & Asymmetry Gauge (sourced from EEGTexturingEngine)
        self.zscore_plot.append_z_scores(te.last_z_scores, te.is_calibrated, te.calibration_progress)
        self.asymmetry_gauge.set_asymmetry(band_powers.get('asymmetry', 0.0))

        n_segs = len(self.emotion_probs)
        self.segments_val.setText(f"{n_segs} classified")
        if n_segs > 0:
            probs_arr  = np.array(self.emotion_probs)
            e_time     = np.arange(n_segs)
            self.emotion_plot.set_data(probs_arr, e_time, 0, e_time[-1] + 1)

            dominant_idx = int(np.argmax(probs))
            dominant = self.EMOTION_LABELS[dominant_idx]
            self.emotion_val.setText(dominant)

        # Check if stream ended and the last queue item was just processed
        if self.awaiting_worker_finish and self.seg_proc.is_empty() and self.worker.is_empty():
            self._enter_review_mode()

    def set_model(self, model):
        self.model = model
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.set_model(model)

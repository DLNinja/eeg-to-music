import os
import scipy.io
import numpy as np
import torch
from queue import Queue
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QGroupBox, QMessageBox, QFrame,
    QRadioButton, QDoubleSpinBox, QScrollBar, QSpinBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QThread, QObject

from src.model.signal_processing import (
    sf, n_channels, bands,
    create_bandpass_filter, filter_segment,
    extract_single_window_features, smooth_features, moving_average
)
from src.model.emotion_classifier import EEGResNet
from src.ui.views.pipeline_view import EegPlotWidget, EmotionPlotWidget
from src.ui.components.piano_roll import PianoRollWidget
from src.music.realtime_generator import RealTimeMusicSynthesizer


# ──────────────────────────────────────────────────────
# Background classification worker
# ──────────────────────────────────────────────────────

class ClassificationWorker(QObject):
    """Runs filter → features → smooth → classify in a background thread."""
    result_ready = pyqtSignal(object, object)  # (features_62x5, probs_4)
    all_done = pyqtSignal()  # emitted when queue is drained after finish signal
    
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
    
    def finish(self):
        """Signal that no more segments will be added. Worker will drain then emit all_done."""
        self._queue.put(("finish", None))
    
    def stop(self):
        self._running = False
        self._queue.put(("stop", None))
    
    def run(self):
        self._running = True
        while self._running:
            item = self._queue.get()
            tag, data = item
            if tag == "stop":
                break
            elif tag == "finish":
                self.all_done.emit()
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

class RealTimeView(QWidget):
    navigate_to_home_signal = pyqtSignal()
    
    EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.mat_data = None
        self.trial_keys = []
        self.current_trial_data = None
        self.sf = sf
        self.stft_n = 256
        self.window_samples = self.sf
        
        self.is_playing = False
        self.waiting_for_worker = False  # True when EEG done but worker still processing
        self.review_mode = False
        self.playhead_idx = 0
        self.total_segments_expected = 0
        self.sample_buffer = None
        self.buffer_pos = 0
        self.emotion_probs = []
        
        self.speed_multiplier = 1
        self.timer_interval_ms = 50
        self.samples_per_tick = int(self.sf * self.timer_interval_ms / 1000)
        
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._on_timer_tick)
        
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
        self.worker.all_done.connect(self._on_worker_finished)
        
        self.synth = RealTimeMusicSynthesizer()
        self.synth.note_played.connect(self._on_note_played)
        self.synth.state_update.connect(self._on_synth_state_update)
        self.synth.start() # Start the background thread loop
        
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
        main_layout = QVBoxLayout(self)
        
        # ── Top bar ──
        top_bar = QHBoxLayout()
        self.back_btn = QPushButton("← Back to Menu")
        self.back_btn.clicked.connect(self.stop_playback)
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
        
        # ── Playback controls ──
        controls_bar = QHBoxLayout()
        controls_bar.addStretch()
        
        # Style helpers
        btn_style = "border-radius: 4px; font-weight: bold; padding: 6px 15px;"

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.start_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(f"background-color: #1a1a1a; color: #00FFB2; border: 1px solid #00FFB2; {btn_style}")
        controls_bar.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_playback)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(f"background-color: #1a1a1a; color: #FF9800; border: 1px solid #FF9800; {btn_style}")
        controls_bar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background-color: #1a1a1a; color: #F44336; border: 1px solid #F44336; {btn_style}")
        controls_bar.addWidget(self.stop_btn)

        controls_bar.addSpacing(20)
        controls_bar.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "5x", "10x"])
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        controls_bar.addWidget(self.speed_combo)
        
        controls_bar.addStretch()
        
        self.status_label = QLabel("Load a .mat file and select a trial to begin.")
        self.status_label.setStyleSheet("font-size: 12px; padding: 4px;")
        controls_bar.addWidget(self.status_label)
        
        main_layout.addLayout(controls_bar)

        # ── Music Controls ──
        music_bar = QHBoxLayout()
        music_bar.addWidget(QLabel("🔈"))
        self.vol_slider = QScrollBar(Qt.Horizontal)
        self.vol_slider.setRange(0, 127)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        music_bar.addWidget(self.vol_slider)

        music_bar.addSpacing(20)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        music_bar.addWidget(self.time_label)

        self.music_time_slider = QScrollBar(Qt.Horizontal)
        self.music_time_slider.setEnabled(False)
        music_bar.addWidget(self.music_time_slider)
        
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
        
        review_layout.addWidget(QLabel("Review:"))
        
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
        
        # ── Plots ──
        self.eeg_plot = EegPlotWidget()
        self.eeg_plot.setMinimumHeight(280)
        main_layout.addWidget(self.eeg_plot, 1)
        
        self.emotion_plot = EmotionPlotWidget()
        self.emotion_plot.setMinimumHeight(200)
        main_layout.addWidget(self.emotion_plot, 1)
        
        # ── Piano Roll ──
        self.piano_roll = PianoRollWidget()
        self.piano_roll.setMinimumHeight(200)
        main_layout.addWidget(self.piano_roll, 2)
        
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.valueChanged.connect(self._update_review_plots)
        self.time_scrollbar.setVisible(False)
        main_layout.addWidget(self.time_scrollbar)
    
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
        
        self._on_channel_changed()
    
    def _on_channel_changed(self):
        """Refresh plots with new channel selection (during review or live)."""
        if self.review_mode:
            self._update_review_plots()
    
    # ── File / Trial ─────────────────────────────────────
    
    def open_file(self):
        start_dir = "data/raw/eeg_seed" if os.path.exists("data/raw/eeg_seed") else "."
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open EEG .mat File", start_dir,
            "MAT Files (*.mat);;All Files (*)"
        )
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path):
        try:
            self.stop_playback()
            self.mat_data = scipy.io.loadmat(file_path)
            self.trial_keys = sorted([k for k in self.mat_data.keys() if not k.startswith('__')])
            
            if not self.trial_keys:
                QMessageBox.warning(self, "No Data", "No trial arrays found.")
                return
            
            self.trial_combo.clear()
            self.trial_combo.addItems(self.trial_keys)
            self.trial_combo.setEnabled(True)
            self.trial_combo.setCurrentIndex(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
    
    def on_trial_selected(self, index):
        if index < 0 or not self.trial_keys or not self.mat_data:
            return
        
        self.stop_playback()
        self._exit_review_mode()
        key = self.trial_keys[index]
        self.current_trial_data = self.mat_data[key]
        
        self.play_btn.setEnabled(True)
        total_s = self.current_trial_data.shape[1] / self.sf
        self.status_label.setText(f"Trial loaded: {self.current_trial_data.shape[1]} samples ({total_s:.1f}s). Press Play.")
        
        self.eeg_plot.clear_data()
        self.emotion_plot.clear_data()
    
    # ── Streaming state ──────────────────────────────────
    
    def _reset_streaming_state(self):
        self.playhead_idx = 0
        self.sample_buffer = np.zeros((n_channels, self.window_samples))
        self.buffer_pos = 0
        self.emotion_probs = []
        self.total_segments_expected = 0
        self.waiting_for_worker = False
        self.worker.reset()
    
    # ── Playback controls ────────────────────────────────
    
    def start_playback(self):
        if self.current_trial_data is None:
            return
        
        if not self.is_playing:
            if self.playhead_idx == 0:
                self._reset_streaming_state()
                self._exit_review_mode()
            
            self.is_playing = True
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            if not self.worker_thread.isRunning():
                self.worker_thread.start()
                
            self.synth.play()
            
            self.stream_timer.start(self.timer_interval_ms)
    
    def pause_playback(self):
        self.is_playing = False
        self.stream_timer.stop()
        self.synth.pause()
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ Resume")
        self.pause_btn.setEnabled(False)
    
    def stop_playback(self):
        was_playing = self.is_playing or self.playhead_idx > 0
        self.is_playing = False
        self.waiting_for_worker = False
        self.stream_timer.stop()
        self.synth.pause() # Stop playing notes, keep thread alive for next play
        
        
        if self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        
        self.piano_roll.clear_notes()
        if was_playing and len(self.emotion_probs) > 0:
            self._enter_review_mode()
        
        self.playhead_idx = 0
        self.play_btn.setEnabled(self.current_trial_data is not None)
        self.play_btn.setText("▶ Play")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
    
    def _on_speed_changed(self, text):
        self.speed_multiplier = int(text.replace("x", ""))
        self.samples_per_tick = int(self.sf * self.timer_interval_ms / 1000 * self.speed_multiplier)
    
    # ── Worker callbacks (main thread) ───────────────────
    
    def _on_classification_result(self, features, probs):
        self.emotion_probs.append(probs)
        
        # Send to synthesizer
        self.synth.update_emotion(probs, features)
        
        n_segs = len(self.emotion_probs)
        
        # Update emotion plot
        if (self.is_playing or self.waiting_for_worker) and n_segs > 0:
            probs_arr = np.array(self.emotion_probs)
            e_time = np.arange(n_segs)
            self.emotion_plot.set_data(
                probs_arr, e_time,
                max(0, e_time[-1] - 30),
                e_time[-1] + 1
            )
        
        # Update status while waiting
        if self.waiting_for_worker:
            remaining = self.total_segments_expected - n_segs
            self.status_label.setText(
                f"⏳ Classifying remaining segments... {n_segs}/{self.total_segments_expected} ({remaining} left)"
            )
    
    def _on_worker_finished(self):
        """Called when worker has drained its queue after we sent finish()."""
        self.waiting_for_worker = False
        
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        
        self._enter_review_mode()
        self.playhead_idx = 0
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ Replay")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ── Review mode ──────────────────────────────────────
    
    def _enter_review_mode(self):
        self.review_mode = True
        self.review_widget.setVisible(True)
        self.radio_full_view.setChecked(True)
        self._on_review_mode_changed()
        
        n_segs = len(self.emotion_probs)
        self.status_label.setText(
            f"✓ Review mode — {n_segs} segments classified. Use controls to browse."
        )
    
    def _exit_review_mode(self):
        self.review_mode = False
        self.review_widget.setVisible(False)
        self.time_scrollbar.setVisible(False)
    
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
        if self.current_trial_data is None:
            return
        
        total_samples = self.current_trial_data.shape[1]
        total_seconds = total_samples / self.sf
        window_size = self.spin_window_size.value()
        max_scroll = max(0, int((total_seconds - window_size) * self.sf))
        
        self.time_scrollbar.setMaximum(max_scroll)
        self.time_scrollbar.setSingleStep(int(self.sf * 0.1))
        self.time_scrollbar.setPageStep(int(self.sf * window_size * 0.5))
    
    def _update_review_plots(self):
        if self.current_trial_data is None or not self.review_mode:
            return
        
        played_samples = self.current_trial_data.shape[1]
        
        if self.radio_full_view.isChecked():
            start_sample = 0
            end_sample = played_samples
        else:
            window_size_samples = int(self.spin_window_size.value() * self.sf)
            start_sample = self.time_scrollbar.value()
            end_sample = min(played_samples, start_sample + window_size_samples)
        
        if start_sample >= end_sample:
            return
        
        # EEG plot with channel selection
        time_axis = np.arange(start_sample, end_sample) / self.sf
        channels = self._get_selected_channels()
        ch_data = [(label, self.current_trial_data[idx, start_sample:end_sample]) for label, idx in channels]
        self.eeg_plot.set_data(ch_data, time_axis, f"EEG Signal — Review")
        
        # Emotion plot
        if len(self.emotion_probs) > 0:
            probs_arr = np.array(self.emotion_probs)
            emotion_time = np.arange(len(self.emotion_probs)) * 1.0
            
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
    
    # ── Timer / Streaming (main thread — EEG only) ───────
    
    def _on_timer_tick(self):
        if self.current_trial_data is None or not self.is_playing:
            return
        
        total_samples = self.current_trial_data.shape[1]
        
        n_new = min(self.samples_per_tick, total_samples - self.playhead_idx)
        if n_new <= 0:
            # EEG playback done — tell worker to drain, then wait
            self.is_playing = False
            self.stream_timer.stop()
            self.waiting_for_worker = True
            
            self.total_segments_expected = total_samples // self.window_samples
            n_classified = len(self.emotion_probs)
            remaining = self.total_segments_expected - n_classified
            
            if remaining > 0:
                self.status_label.setText(
                    f"⏳ EEG done. Classifying remaining {remaining} segments..."
                )
                self.play_btn.setEnabled(False)
                self.pause_btn.setEnabled(False)
                # Send finish sentinel — worker will drain queue then emit all_done
                self.worker.finish()
            else:
                # All segments already classified
                self._on_worker_finished()
            return
        
        new_data = self.current_trial_data[:, self.playhead_idx:self.playhead_idx + n_new]
        self.playhead_idx += n_new
        
        # Feed into 1-second buffer
        remaining = n_new
        src_offset = 0
        
        while remaining > 0:
            space = self.window_samples - self.buffer_pos
            take = min(remaining, space)
            
            self.sample_buffer[:, self.buffer_pos:self.buffer_pos + take] = new_data[:, src_offset:src_offset + take]
            self.buffer_pos += take
            src_offset += take
            remaining -= take
            
            if self.buffer_pos >= self.window_samples:
                self.worker.enqueue(self.sample_buffer)
                self.buffer_pos = 0
        
        # Update EEG plot with selected channels — show last 5 seconds
        display_window = 5 * self.sf
        start = max(0, self.playhead_idx - display_window)
        end = self.playhead_idx
        
        time_axis = np.arange(start, end) / self.sf
        channels = self._get_selected_channels()
        ch_data = [(label, self.current_trial_data[idx, start:end]) for label, idx in channels]
        
        self.eeg_plot.set_data(
            ch_data, time_axis,
            f"EEG Signal — Real-Time ({self.playhead_idx / self.sf:.1f}s)"
        )
        
        # Update status
        elapsed = self.playhead_idx / self.sf
        total = total_samples / self.sf
        n_segs = len(self.emotion_probs)
        dominant = ""
        if n_segs > 0:
            latest = self.emotion_probs[-1]
            dominant = f" | Dominant: {self.EMOTION_LABELS[np.argmax(latest)]}"
        
        self.status_label.setText(
            f"⏱ {elapsed:.1f}s / {total:.1f}s | Segments: {n_segs}{dominant}"
        )

    def _on_note_played(self, channel, pitch, velocity, start_time, duration):
        # We only really care about the total time for the piano roll to scale
        # If the total time isn't set yet, we might need a dynamic one
        total_s = self.current_trial_data.shape[1] / self.sf if self.current_trial_data is not None else 60.0
        self.piano_roll.total_time = total_s
        self.piano_roll.add_note(start_time, duration, pitch, velocity)
        self.piano_roll.update_playhead(start_time)
        
        # Update time UI
        self._update_time_ui_realtime(start_time, total_s)

    def _on_synth_state_update(self, mode, chord_type, bpm):
        # Could show this in status or a specialized label
        pass

    def _on_volume_changed(self, value):
        self.synth.set_volume(value)

    def _update_time_ui_realtime(self, pos_s, total_s):
        pos_s = min(pos_s, total_s)
        self.music_time_slider.setRange(0, int(total_s * 1000))
        self.music_time_slider.setValue(int(pos_s * 1000))
        
        cur_mins = int(pos_s // 60)
        cur_secs = int(pos_s % 60)
        total_mins = int(total_s // 60)
        total_secs = int(total_s % 60)
        self.time_label.setText(f"{cur_mins:02d}:{cur_secs:02d} / {total_mins:02d}:{total_secs:02d}")

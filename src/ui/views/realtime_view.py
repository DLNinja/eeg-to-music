import os
import time
import scipy.io
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QFileDialog, QPushButton, QGroupBox, QMessageBox, QFrame,
    QRadioButton, QDoubleSpinBox, QScrollBar, QSpinBox, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QThread

from src.eeg_pipeline.signal_processing import sf, n_channels
from src.eeg_pipeline.emotion_classifier import load_emotion_model
from src.eeg_pipeline.classification_worker import ClassificationWorker
from src.eeg_pipeline.emotion_result import EmotionResult
from src.ui.components.eeg_plots import EegPlotWidget, EmotionPlotWidget, BandZScorePlotWidget, AsymmetryGaugeWidget
from src.ui.components.piano_roll import PianoRollWidget
from src.ui.components.channel_selector import ChannelSelectorWidget
from src.music.realtime_generator import RealTimeMusicSynthesizer


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
        self.playback_start_wall_time = 0.0
        self.total_segments_expected = 0
        self.sample_buffer = None
        self.buffer_pos = 0
        self.emotion_probs = []
        
        self.speed_multiplier = 1
        self.timer_interval_ms = 50
        self.samples_per_tick = int(self.sf * self.timer_interval_ms / 1000)
        
        self.tick_count = 0 # For plot throttling
        
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._on_timer_tick)
        
        self.model = load_emotion_model()
        
        self.sos, self._zi_template = None, None  # managed by ClassificationWorker internally

        self.worker_thread = QThread()
        self.worker = ClassificationWorker(self.model, self.stft_n, self.sf)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.thread_result.connect(self._on_classification_result)
        self.worker.all_done.connect(self._on_worker_finished)
        
        self.synth = RealTimeMusicSynthesizer()
        self.synth.note_played.connect(self._on_note_played)
        self.synth.state_update.connect(self._on_synth_state_update)
        self.synth.start() # Start the background thread loop
        
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
        
        # ── Playback controls (Left-aligned & static to prevent shifting) ──
        controls_bar = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("playBtn")
        self.play_btn.clicked.connect(self.start_playback)
        self.play_btn.setEnabled(False)
        controls_bar.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.clicked.connect(self.pause_playback)
        self.pause_btn.setEnabled(False)
        controls_bar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        controls_bar.addWidget(self.stop_btn)

        controls_bar.addSpacing(20)
        controls_bar.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "5x", "10x"])
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        controls_bar.addWidget(self.speed_combo)
        
        controls_bar.addStretch()
        main_layout.addLayout(controls_bar)
        
        # ── Dashboard Cards Row (Fixed-width boxes to prevent any UI movement) ──
        dashboard_layout = QHBoxLayout()
        
        # 1. Pipeline Status Card (Stretches)
        self.status_card = QWidget()
        self.status_card.setObjectName("dashboardCard")
        status_card_layout = QVBoxLayout(self.status_card)
        status_card_layout.setContentsMargins(8, 4, 8, 4)
        status_title = QLabel("System Status")
        status_title.setObjectName("cardTitle")
        self.status_val = QLabel("Load a .mat file and select a trial to begin.")
        self.status_val.setObjectName("cardValue")
        self.status_val.setWordWrap(True)
        status_card_layout.addWidget(status_title)
        status_card_layout.addWidget(self.status_val)
        dashboard_layout.addWidget(self.status_card, stretch=2)
        
        # 2. Playback Time Card (Fixed Width)
        self.time_card = QWidget()
        self.time_card.setFixedWidth(160)
        self.time_card.setObjectName("dashboardCard")
        time_card_layout = QVBoxLayout(self.time_card)
        time_card_layout.setContentsMargins(8, 4, 8, 4)
        time_title = QLabel("Playback Time")
        time_title.setObjectName("cardTitle")
        self.time_val = QLabel("0.0s / 0.0s")
        self.time_val.setObjectName("cardValue")
        self.time_val.setAlignment(Qt.AlignCenter)
        time_card_layout.addWidget(time_title)
        time_card_layout.addWidget(self.time_val)
        dashboard_layout.addWidget(self.time_card)
        
        # 3. Segments Classified Card (Fixed Width)
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
        
        # 4. Emotion Card (Fixed Width)
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
 
        # ── Music Controls: Volume Row (Left-aligned) ──
        music_vol_bar = QHBoxLayout()
        music_vol_bar.addWidget(QLabel("🔈"))
        self.vol_slider = QScrollBar(Qt.Horizontal)
        self.vol_slider.setRange(0, 127)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        music_vol_bar.addWidget(self.vol_slider)
        music_vol_bar.addStretch()
        main_layout.addLayout(music_vol_bar)
        
        # ── Music Controls: Playback Progress Row (Stretching) ──
        music_progress_bar = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        self.time_label.setFixedWidth(110)
        music_progress_bar.addWidget(self.time_label)
 
        self.music_time_slider = QScrollBar(Qt.Horizontal)
        self.music_time_slider.setEnabled(False)
        music_progress_bar.addWidget(self.music_time_slider)
        
        main_layout.addLayout(music_progress_bar)
        
        # ── Channel selection ──
        self.channel_selector = ChannelSelectorWidget(max_channels=n_channels)
        self.channel_selector.selection_changed.connect(self._on_channel_changed)
        main_layout.addWidget(self.channel_selector)
        
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
        
        # ── Plots (fixed heights — page scrolls vertically) ──
        self.eeg_plot = EegPlotWidget()
        self.eeg_plot.setFixedHeight(350)
        main_layout.addWidget(self.eeg_plot)

        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.valueChanged.connect(self._update_review_plots)
        self.time_scrollbar.setVisible(False)
        main_layout.addWidget(self.time_scrollbar)

        self.emotion_plot = EmotionPlotWidget()
        self.emotion_plot.setFixedHeight(280)
        main_layout.addWidget(self.emotion_plot)
        
        # ── Frontal Alpha Asymmetry Gauge ──
        self.asymmetry_gauge = AsymmetryGaugeWidget()
        main_layout.addWidget(self.asymmetry_gauge)
        
        # ── Band Z-Score Plot ──
        self.zscore_plot = BandZScorePlotWidget()
        self.zscore_plot.setFixedHeight(220)
        main_layout.addWidget(self.zscore_plot)
        
        # ── Piano Roll ──
        self.piano_roll = PianoRollWidget()
        self.piano_roll.setFixedHeight(320)
        main_layout.addWidget(self.piano_roll)
    def _get_selected_channels(self):
        return self.channel_selector.get_selected_channels()
    
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
        self.status_val.setText("Trial loaded successfully. Press Play.")
        self.time_val.setText(f"0.0s / {total_s:.1f}s")
        self.segments_val.setText("0 classified")
        self.emotion_val.setText("—")
        
        self.eeg_plot.clear_data()
        self.emotion_plot.clear_data()
    
    # ── Streaming state ──────────────────────────────────
    
    def _reset_streaming_state(self):
        self.playhead_idx = 0
        self.playback_start_wall_time = time.time()
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
            if self.playhead_idx == 0: # Only reset if starting from beginning
                self._reset_streaming_state()
                self._exit_review_mode()
            
            self.is_playing = True
            # Adjust start time if we are resuming
            self.playback_start_wall_time = time.time() - (self.playhead_idx / self.sf / self.speed_multiplier)
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
        self.synth.pause() # Stop playing notes
        self.synth.reset_state() # Reset internal clock and musical state
        
        self.piano_roll.clear_notes()
        self._update_time_ui_realtime(0, 0)
        
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
    
    def _on_classification_result(self, result: EmotionResult):
        self.emotion_probs.append(result.probs)

        # Send to synthesizer — EEGTexturingEngine.process() runs inside update_emotion
        self.synth.update_emotion(result.probs, result.timestamp, result.band_powers)

        # Update Z-score plot & Asymmetry Gauge (sourced from EEGTexturingEngine)
        if self.is_playing or self.waiting_for_worker:
            te = self.synth.eeg_texturing_engine
            self.zscore_plot.append_z_scores(te.last_z_scores, te.is_calibrated, te.calibration_progress)
            self.asymmetry_gauge.set_asymmetry(result.asymmetry)

        n_segs = len(self.emotion_probs)

        # Update emotion plot
        if (self.is_playing or self.waiting_for_worker) and n_segs > 0:
            probs_arr = np.array(self.emotion_probs)
            e_time = np.arange(n_segs)
            self.emotion_plot.set_data(probs_arr, e_time, 0, e_time[-1] + 1)

        # Update status while waiting
        if self.waiting_for_worker:
            self.status_val.setText("Classifying remaining segments...")
            self.segments_val.setText(f"{n_segs} / {self.total_segments_expected}")
    
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
        self.status_val.setText("Review Mode — use timeline to browse.")
        self.segments_val.setText(f"{n_segs} classified")
    
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
        if self._check_playback_complete(total_samples):
            return
        n_new = self._calculate_samples_to_process()
        if n_new <= 0:
            return
        new_data = self._advance_playhead(n_new, total_samples)
        self._feed_worker(new_data)
        self._refresh_eeg_plot(total_samples)
        self._refresh_dashboard(total_samples)

    def _calculate_samples_to_process(self) -> int:
        """Return wall-clock-corrected number of new samples to consume this tick."""
        elapsed = time.time() - self.playback_start_wall_time
        target  = int(elapsed * self.sf * self.speed_multiplier)
        n_new   = target - self.playhead_idx
        return max(0, min(n_new, 20 * self.samples_per_tick))

    def _check_playback_complete(self, total_samples: int) -> bool:
        """Handle end-of-trial and worker drain. Returns True when playback is done."""
        if self.playhead_idx < total_samples:
            return False
        self.is_playing = False
        self.stream_timer.stop()
        self.waiting_for_worker = True
        self.total_segments_expected = total_samples // self.window_samples
        n_classified = len(self.emotion_probs)
        remaining = self.total_segments_expected - n_classified
        if remaining > 0:
            self.status_val.setText("Finishing remaining classification...")
            self.segments_val.setText(f"{n_classified} / {self.total_segments_expected}")
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.worker.finish()
        else:
            self._on_worker_finished()
        return True

    def _advance_playhead(self, n_new: int, total_samples: int) -> np.ndarray:
        """Slice n_new samples from the trial data and advance the playhead index."""
        take = min(n_new, total_samples - self.playhead_idx)
        data = self.current_trial_data[:, self.playhead_idx:self.playhead_idx + take]
        self.playhead_idx += take
        return data

    def _feed_worker(self, new_data: np.ndarray):
        """Fill the 1-second ring buffer and enqueue completed segments to the worker."""
        remaining  = new_data.shape[1]
        src_offset = 0
        while remaining > 0:
            space = self.window_samples - self.buffer_pos
            take  = min(remaining, space)
            self.sample_buffer[:, self.buffer_pos:self.buffer_pos + take] = \
                new_data[:, src_offset:src_offset + take]
            self.buffer_pos += take
            src_offset      += take
            remaining       -= take
            if self.buffer_pos >= self.window_samples:
                seg_timestamp = (self.playhead_idx - self.window_samples) / self.sf
                self.worker.enqueue(self.sample_buffer, seg_timestamp)
                self.buffer_pos = 0

    def _refresh_eeg_plot(self, total_samples: int):
        """Throttled EEG waveform update — fires every 3 ticks (~150 ms)."""
        self.tick_count += 1
        if self.tick_count % 3 != 0:
            return
        display_window = 5 * self.sf
        start = max(0, self.playhead_idx - display_window)
        time_axis = np.arange(start, self.playhead_idx) / self.sf
        channels  = self._get_selected_channels()
        ch_data   = [(lbl, self.current_trial_data[idx, start:self.playhead_idx])
                     for lbl, idx in channels]
        self.eeg_plot.set_data(
            ch_data, time_axis,
            f"EEG Signal — Real-Time ({self.playhead_idx / self.sf:.1f}s)"
        )

    def _refresh_dashboard(self, total_samples: int):
        """Update the time, segment-count, and dominant emotion dashboard labels."""
        elapsed = self.playhead_idx / self.sf
        total   = total_samples / self.sf
        n_segs  = len(self.emotion_probs)
        self.status_val.setText("Streaming live EEG data...")
        self.time_val.setText(f"{elapsed:.1f}s / {total:.1f}s")
        self.segments_val.setText(f"{n_segs} classified")
        if n_segs > 0:
            dom_name = self.EMOTION_LABELS[np.argmax(self.emotion_probs[-1])]
            self.emotion_val.setText(dom_name)

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

    def set_model(self, model):
        """Update the classification model."""
        self.model = model
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.set_model(model)

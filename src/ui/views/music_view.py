import os
import mido
import time
import fluidsynth
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFileDialog, QPushButton, QMessageBox, QSlider, QScrollBar
)
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QMutex, QMutexLocker, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF

class SuppressStderr:
    """Context manager to suppress C-level stderr (ALSA/Jack warnings)."""
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

class PianoRollWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes = [] # list of (start, duration, pitch, velocity)
        self.total_time = 1.0
        self.current_time = 0.0
        self.zoom_factor = 1.0
        self.scroll_time = 0.0
        self.min_pitch = 21
        self.max_pitch = 108
        self.keyboard_width = 80
        
        # Appearance - Dark Studio Theme
        self.bg_color = QColor("#1e1e1e")
        self.grid_color = QColor("#333333")
        self.white_key_color = QColor("#ffffff")
        self.black_key_color = QColor("#111111")
        self.note_color = QColor("#00FFB2") # Ableton-style neon cyan
        self.playhead_color = QColor("#ff0055") # Neon pink red
        
        self.setMinimumHeight(200)

    def set_data(self, notes, total_time, min_pitch, max_pitch):
        self.notes = notes
        self.total_time = max(total_time, 1.0)
        # Add padding to pitch range
        self.min_pitch = max(0, min_pitch - 2)
        self.max_pitch = min(127, max_pitch + 2)
        if self.min_pitch >= self.max_pitch:
            self.min_pitch, self.max_pitch = 21, 108
        self.update()
        
    def update_playhead(self, time_s):
        self.current_time = time_s
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # Draw background
        painter.fillRect(rect, self.bg_color)
        
        pitch_range = self.max_pitch - self.min_pitch + 1
        note_height = height / pitch_range if pitch_range > 0 else height
        
        # Calculate time scale
        track_width = width - self.keyboard_width
        base_pixels_per_second = track_width / self.total_time if self.total_time > 0 else 100
        pixels_per_second = base_pixels_per_second * self.zoom_factor
        visible_time = track_width / pixels_per_second if pixels_per_second > 0 else self.total_time
        
        # Clip notes drawing area
        painter.setClipRect(self.keyboard_width, 0, int(track_width), height)
        
        # Draw horizontal grid lines
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)
        for i in range(pitch_range + 1):
            y = int(height - (i * note_height))
            painter.drawLine(self.keyboard_width, y, width, y)
            
        # Draw vertical grid lines (every 1 second)
        start_s = int(self.scroll_time)
        end_s = int(self.scroll_time + visible_time) + 1
        for s in range(start_s, end_s + 1):
            x = int(self.keyboard_width + ((s - self.scroll_time) * pixels_per_second))
            painter.drawLine(x, 0, x, height)
            
        # Draw Notes
        painter.setPen(Qt.NoPen)
        for start, duration, pitch, vel in self.notes:
            if start + duration < self.scroll_time or start > self.scroll_time + visible_time:
                continue
            alpha = max(60, min(255, int((vel / 127) * 255)))
            color = QColor(self.note_color)
            color.setAlpha(alpha)
            painter.setBrush(color)
            
            x = self.keyboard_width + ((start - self.scroll_time) * pixels_per_second)
            y = height - ((pitch - self.min_pitch + 1) * note_height)
            w = duration * pixels_per_second
            h = note_height
            
            h_pad = h * 0.1
            painter.drawRoundedRect(QRectF(x, y + h_pad, max(2.0, w), h - 2*h_pad), 3, 3)
            
        # Draw Playhead
        playhead_x = self.keyboard_width + ((self.current_time - self.scroll_time) * pixels_per_second)
        if self.keyboard_width <= playhead_x <= width:
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(int(playhead_x), 0, int(playhead_x), height)
            
            triangle = QPolygonF([
                QPointF(playhead_x - 6, 0),
                QPointF(playhead_x + 6, 0),
                QPointF(playhead_x, 12)
            ])
            painter.setBrush(self.playhead_color)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(triangle)
            
        painter.setClipping(False)
        
        # Draw Keyboard Ledger
        painter.fillRect(0, 0, self.keyboard_width, height, QColor("#111111"))
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            y = height - ((pitch - self.min_pitch + 1) * note_height)
            is_black = (pitch % 12) in [1, 3, 6, 8, 10]
            
            key_rect = QRectF(0, y, self.keyboard_width, note_height)
            if is_black:
                painter.fillRect(key_rect, self.black_key_color)
            else:
                painter.fillRect(key_rect, self.white_key_color)
                
            # Draw key border
            painter.setPen(QPen(QColor("#555555"), 1))
            painter.drawRect(key_rect)
            
            # C note text labels
            if pitch % 12 == 0:
                octave = (pitch // 12) - 1
                painter.setPen(QPen(QColor("#000000")))
                font = painter.font()
                font.setPointSize(max(5, int(note_height * 0.5)))
                painter.setFont(font)
                painter.drawText(key_rect.adjusted(3, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, f"C{octave}")


class MidiPlaybackThread(QThread):
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()
    
    def __init__(self, synth, events, total_time):
        super().__init__()
        self.synth = synth
        self.events = events # list of (absolute_time, type, pitch, velocity)
        self.total_time = total_time
        
        self.is_playing = False
        self.is_paused = False
        self.cursor_time = 0.0
        self.mutex = QMutex()
        
    def run(self):
        self.is_playing = True
        last_time = time.time()
        
        while self.is_playing and self.cursor_time <= self.total_time:
            with QMutexLocker(self.mutex):
                if not self.is_paused:
                    now = time.time()
                    dt = now - last_time
                    last_time = now
                    
                    next_time = self.cursor_time + dt
                    
                    # Dispatch events between self.cursor_time and next_time
                    for ev_time, ev_type, pitch, vel in self.events:
                        if self.cursor_time <= ev_time <= next_time:
                            if ev_type == 'note_on':
                                self.synth.noteon(0, pitch, vel)
                            elif ev_type == 'note_off':
                                self.synth.noteoff(0, pitch)
                                
                    self.cursor_time = next_time
                    self.progress_signal.emit(self.cursor_time)
                else:
                    last_time = time.time()
            time.sleep(0.01) # 10ms resolution
            
        if self.cursor_time > self.total_time:
            self.finished_signal.emit()
            
    def stop(self):
        with QMutexLocker(self.mutex):
            self.is_playing = False
            for p in range(128):
                self.synth.noteoff(0, p)
            
    def set_pause(self, paused):
        with QMutexLocker(self.mutex):
            self.is_paused = paused
            if paused:
                for p in range(128):
                    self.synth.noteoff(0, p)
            
    def seek(self, new_time):
        with QMutexLocker(self.mutex):
            for p in range(128):
                self.synth.noteoff(0, p)
            self.cursor_time = max(0.0, min(new_time, self.total_time))
            self.progress_signal.emit(self.cursor_time)

class MusicView(QWidget):
    navigate_to_home_signal = pyqtSignal()
    playback_progress_signal = pyqtSignal(float)
    
    def __init__(self, parent=None, embedded_mode=False):
        super().__init__(parent)
        self.embedded_mode = embedded_mode
        
        self.loaded_file_path = ""
        self.total_time_s = 0.0
        self.midi_events = []
        
        self.playback_thread = None
        
        # Initialize fluidsynth
        try:
            with SuppressStderr():
                self.synth = fluidsynth.Synth()
                driver_name = "waveout" if os.name == "nt" else "pulseaudio"
                self.synth.start(driver=driver_name, midi_driver="winmidi" if os.name == "nt" else "alsa_seq")
                
                # self.synth.start(driver="pulseaudio") # linux only
                
                soundfonts = [
                    "MuseScore_General.sf3", # Local downloaded SoundFont
                    # System Path Fallbacks
                    "/usr/share/soundfonts/freepats-general-midi.sf2", 
                    "/usr/share/sounds/sf2/FluidR3_GM.sf2", 
                    "/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"
                ]
                self.sfid = -1
                for sf in soundfonts:
                    if os.path.exists(sf):
                        self.sfid = self.synth.sfload(sf)
                        break
                        
                if self.sfid != -1:
                    self.synth.program_select(0, self.sfid, 0, 0)
                else:
                    print("Warning: No soundfont found. Playback will be silent.")        
        except Exception as e:
            print(f"Warning: Failed to initialize FluidSynth: {e}")
            self.synth = None
        
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        main_layout = QVBoxLayout(self)
        
        top_bar = QHBoxLayout()
        
        self.back_btn = QPushButton("← Back to Menu")
        self.back_btn.setStyleSheet("background-color: #444444; border-radius: 4px; padding: 6px 12px;")
        self.back_btn.clicked.connect(self.stop_music)
        self.back_btn.clicked.connect(self.navigate_to_home_signal.emit)
        top_bar.addWidget(self.back_btn)
        
        self.open_file_btn = QPushButton("Open MIDI File")
        self.open_file_btn.setStyleSheet("background-color: #444444; border-radius: 4px; padding: 6px 12px;")
        self.open_file_btn.clicked.connect(self.open_file)
        top_bar.addWidget(self.open_file_btn)
        
        self.current_file_label = QLabel("No file loaded")
        self.current_file_label.setStyleSheet("font-style: italic; color: #aaaaaa;")
        top_bar.addWidget(self.current_file_label)
        
        if self.embedded_mode:
            self.back_btn.hide()
            self.open_file_btn.hide()
            self.current_file_label.hide()
            
        top_bar.addStretch()
        
        # Style helpers
        btn_style = "border-radius: 4px; font-weight: bold; padding: 6px 15px;"
        
        self.skip_back_btn = QPushButton("⏪ -5s")
        self.skip_back_btn.clicked.connect(lambda: self.seek_relative(-5.0))
        self.skip_back_btn.setEnabled(False)
        self.skip_back_btn.setStyleSheet(f"background-color: #555555; {btn_style}")
        top_bar.addWidget(self.skip_back_btn)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_music)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(f"background-color: #1a1a1a; color: #00FFB2; border: 1px solid #00FFB2; {btn_style}")
        top_bar.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(f"background-color: #1a1a1a; color: #FF9800; border: 1px solid #FF9800; {btn_style}")
        top_bar.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.clicked.connect(self.stop_music)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background-color: #1a1a1a; color: #F44336; border: 1px solid #F44336; {btn_style}")
        top_bar.addWidget(self.stop_btn)
        
        self.skip_fwd_btn = QPushButton("⏩ +5s")
        self.skip_fwd_btn.clicked.connect(lambda: self.seek_relative(5.0))
        self.skip_fwd_btn.setEnabled(False)
        self.skip_fwd_btn.setStyleSheet(f"background-color: #555555; {btn_style}")
        top_bar.addWidget(self.skip_fwd_btn)
        
        main_layout.addLayout(top_bar)
        
        mid_bar = QHBoxLayout()
        
        mid_bar.addWidget(QLabel("🔈"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setMinimum(0)
        self.vol_slider.setMaximum(127)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.valueChanged.connect(self.on_volume_changed)
        mid_bar.addWidget(self.vol_slider)
        
        mid_bar.addSpacing(20)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        mid_bar.addWidget(self.time_label)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setAttribute(Qt.WA_TransparentForMouseEvents) 
        mid_bar.addWidget(self.time_slider)
        
        main_layout.addLayout(mid_bar)
        
        # Custom Piano Roll Widget
        self.piano_roll = PianoRollWidget()
        main_layout.addWidget(self.piano_roll, 1) # stretch factor 1
        
        # Zoom and Scroll controls below Piano Roll Widget
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(QLabel("🔍 Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(100) # 1x
        self.zoom_slider.setMaximum(1000) # 10x
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self.update_scroll_bounds)
        bottom_bar.addWidget(self.zoom_slider)
        
        self.track_scrollbar = QScrollBar(Qt.Horizontal)
        self.track_scrollbar.valueChanged.connect(self.on_track_scroll)
        self.track_scrollbar.setEnabled(False)
        bottom_bar.addWidget(self.track_scrollbar, 1) # Stretch to fill
        
        main_layout.addLayout(bottom_bar)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scroll_bounds()
        
    def update_scroll_bounds(self):
        width = self.piano_roll.width()
        track_width = width - self.piano_roll.keyboard_width
        if track_width <= 0: return
        
        base_pps = track_width / self.total_time_s if self.total_time_s > 0 else 100
        zoom = getattr(self, 'zoom_slider', None)
        zoom_val = zoom.value() / 100.0 if zoom else 1.0
        pps = base_pps * zoom_val
        
        visible_time = track_width / pps if pps > 0 else self.total_time_s
        max_scroll = max(0.0, self.total_time_s - visible_time)
        
        scrollbar = getattr(self, 'track_scrollbar', None)
        if scrollbar:
            if max_scroll > 0:
                scrollbar.setEnabled(True)
                scrollbar.setMaximum(int(max_scroll * 1000))
                scrollbar.setPageStep(int(visible_time * 1000))
                scrollbar.setSingleStep(int(visible_time * 100)) # 10% step
            else:
                scrollbar.setEnabled(False)
                scrollbar.setValue(0)
                scrollbar.setMaximum(0)
            
        self.piano_roll.zoom_factor = zoom_val
        self.piano_roll.update()

    def on_track_scroll(self, value):
        self.piano_roll.scroll_time = value / 1000.0
        self.piano_roll.update()

    def open_file(self):
        start_dir = "music" if os.path.exists("music") else "."
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Generated MIDI File", 
            start_dir, 
            "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        try:
            self.stop_music()
            mid = mido.MidiFile(file_path)
            
            self.loaded_file_path = file_path
            self.current_file_label.setText(f"Loaded: {os.path.basename(file_path)}")
            
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.skip_back_btn.setEnabled(True)
            self.skip_fwd_btn.setEnabled(True)
            
            self.parse_and_plot(mid)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Failed to load MIDI file:\n{str(e)}")
            self.loaded_file_path = ""
            self.current_file_label.setText("No file loaded")
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.skip_back_btn.setEnabled(False)
            self.skip_fwd_btn.setEnabled(False)
            self.piano_roll.set_data([], 1.0, 21, 108)
            
    def parse_and_plot(self, mid: mido.MidiFile):
        notes = [] 
        active_notes = {}
        current_time = 0.0
        
        self.midi_events = []
        
        for msg in mid:
            current_time += msg.time
            if not msg.is_meta:
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity)
                    self.midi_events.append((current_time, 'note_on', msg.note, msg.velocity))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, vel = active_notes.pop(msg.note)
                        duration = current_time - start_time
                        if duration > 0:
                            notes.append((start_time, duration, msg.note, vel))
                    self.midi_events.append((current_time, 'note_off', msg.note, 0))
                            
        self.total_time_s = current_time
        
        if not notes:
            self.piano_roll.set_data([], 1.0, 21, 108)
            return

        min_pitch = min(p[2] for p in notes)
        max_pitch = max(p[2] for p in notes)
        
        self.piano_roll.set_data(notes, self.total_time_s, min_pitch, max_pitch)
        self.update_scroll_bounds()
        self._update_time_ui(0.0)

    def _update_time_ui(self, pos_s):
        pos_s = min(pos_s, self.total_time_s)
        
        pos_ms = int(pos_s * 1000)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(int(self.total_time_s * 1000))
        self.time_slider.setValue(pos_ms)
        
        cur_mins = int(pos_s // 60)
        cur_secs = int(pos_s % 60)
        total_mins = int(self.total_time_s // 60)
        total_secs = int(self.total_time_s % 60)
        self.time_label.setText(f"{cur_mins:02d}:{cur_secs:02d} / {total_mins:02d}:{total_secs:02d}")
        
        self.piano_roll.update_playhead(pos_s)
        
        # Auto-scroll logic if zoomed strictly when playing
        if self.playback_thread is not None and self.playback_thread.isRunning() and not self.playback_thread.is_paused:
            current_scroll = self.track_scrollbar.value() / 1000.0
            width = self.piano_roll.width()
            track_width = width - self.piano_roll.keyboard_width
            base_pps = track_width / self.total_time_s if self.total_time_s > 0 else 100
            zoom = self.zoom_slider.value() / 100.0
            pps = base_pps * zoom
            visible_time = track_width / pps if pps > 0 else self.total_time_s
            
            # If playhead flows out of view to the right, page it forward
            if pos_s > current_scroll + visible_time - 0.2:
                new_scroll = min(self.total_time_s - visible_time, pos_s - visible_time * 0.1) # scroll so playhead is at 10%
                if new_scroll > 0:
                    self.track_scrollbar.setValue(int(new_scroll * 1000))
            elif pos_s < current_scroll:
                self.track_scrollbar.setValue(int(pos_s * 1000))

    def on_playback_progress(self, current_time):
        self._update_time_ui(current_time)
        self.playback_progress_signal.emit(current_time)

    def on_playback_finished(self):
        self.stop_music()

    def play_music(self):
        if not self.loaded_file_path or not self.synth:
            return
            
        if self.playback_thread and self.playback_thread.isRunning():
            self.stop_music()
            
        self.on_volume_changed(self.vol_slider.value())
            
        self.playback_thread = MidiPlaybackThread(self.synth, self.midi_events, self.total_time_s)
        self.playback_thread.progress_signal.connect(self.on_playback_progress)
        self.playback_thread.finished_signal.connect(self.on_playback_finished)
        
        self.playback_thread.start()
        
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ Pause")

    def toggle_pause(self):
        if not self.playback_thread or not self.playback_thread.isRunning():
            return
            
        if self.playback_thread.is_paused:
            self.playback_thread.set_pause(False)
            self.pause_btn.setText("⏸ Pause")
        else:
            self.playback_thread.set_pause(True)
            self.pause_btn.setText("▶ Resume")

    def seek_relative(self, dt):
        if self.playback_thread and self.playback_thread.isRunning():
            new_time = self.playback_thread.cursor_time + dt
            self.playback_thread.seek(new_time)

    def on_volume_changed(self, value):
        if self.synth:
            self.synth.cc(0, 7, value)

    def stop_music(self):
        if self.playback_thread and self.playback_thread.isRunning():
            self.playback_thread.stop()
            self.playback_thread.wait()
            self.playback_thread = None
            
        self._update_time_ui(0.0)
            
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Pause")

import os
import mido
import time
import fluidsynth
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFileDialog, QPushButton, QMessageBox, QSlider, QScrollBar
)
from PyQt5.QtGui import QPainter, QColor, QPen, QPolygonF
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QMutex, QMutexLocker, QRectF, QPointF, QObject
from src.ui.components.piano_roll import PianoRollWidget

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
        self.synth = None
        self.sfid = -1
        
        self.init_thread = QThread()
        self.init_worker = SynthInitWorker()
        self.init_worker.moveToThread(self.init_thread)
        self.init_thread.started.connect(self.init_worker.run)
        self.init_worker.finished.connect(self._on_synth_init_finished)
        # We don't start the thread here anymore; we'll start it in load_data
        
        self._setup_ui()


    def _on_synth_init_finished(self, synth, sfid):
        self.synth = synth
        self.sfid = sfid
        if self.synth and self.sfid != -1:
            self.synth.program_select(0, self.sfid, 0, 0)
        self.init_thread.quit()
        self.init_thread.wait()

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
            # Initialize synth lazily if not already done
            if self.synth is None and not self.init_thread.isRunning():
                self.init_thread.start()
            
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
        self.playback_progress_signal.emit(pos_s) # Ensure parent views are updated
        
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
            
        self.total_time_s = 0.0
        self.midi_events = []
        self._update_time_ui(0.0)
        self.track_scrollbar.setValue(0)
        self.piano_roll.clear_notes()
            
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Pause")

class SynthInitWorker(QObject):
    finished = pyqtSignal(object, int)
    
    def run(self):
        synth = None
        sfid = -1
        try:
            with SuppressStderr():
                synth = fluidsynth.Synth()
                if os.name == "nt":
                    try:
                        synth.start(driver="waveout")
                    except:
                        synth.start()
                else:
                    try:
                        synth.start(driver="pulseaudio")
                    except:
                        synth.start()
                
                soundfonts = [
                    "models/soundfont.sf2",
                    "soundfont.sf2",
                    "MuseScore_General.sf3",
                    "/usr/share/soundfonts/freepats-general-midi.sf2", 
                    "/usr/share/sounds/sf2/FluidR3_GM.sf2", 
                    "/Library/Audio/Sounds/Banks/FluidR3_GM.sf2"
                ]
                for sf in soundfonts:
                    if os.path.exists(sf):
                        sfid = synth.sfload(sf)
                        break
        except Exception as e:
            print(f"Warning: SynthInitWorker - Failed: {e}")
            
        self.finished.emit(synth, sfid)


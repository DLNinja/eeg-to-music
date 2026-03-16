import os
import time
import mido
import fluidsynth
import random
import numpy as np
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from src.music.midi_generator import get_mode_pool, get_chord

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

class RealTimeMusicSynthesizer(QThread):
    note_played = pyqtSignal(int, int, int, float, float) # channel, pitch, velocity, start_time, duration
    state_update = pyqtSignal(str, str, float) # mode, chord_type, bpm

    def __init__(self, base_key_offset=0):
        super().__init__()
        
        self.base_key_offset = base_key_offset
        self.mutex = QMutex()
        self.is_running = False
        self.is_playing = False
        self.playback_start_time = 0.0
        
        # State Queue (to receive 1-second updates from EEG worker)
        self.update_queue = []
        # Note Queue: list of (timestamp, channel, pitch, velocity, duration, is_note_on)
        self.note_queue = []
        
        # Internal Music State
        self.current_bpm = 100
        self.prev_dominant_idx = -1
        self.emotion_streak = 0
        self.melody_idx = 21
        self.fundamental_bass_root = 36 + self.base_key_offset
        
        # Active notes to turn off cleanly
        self.active_chord_notes = []
        self.active_melody_note = None
        self.current_chord_degree = 0      # Track which scale degree is sounding
        self.current_dominant_idx = -1     # Track which emotion type is active for the chord

        # Timing — rhythm lists match MIDI generator (proportions of a 2-beat step)
        self.rhythms = {
            'slow': [[1.0], [0.5, 0.5], [0.75, 0.25]],
            'med':  [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0.333, 0.333, 0.333]],
            'fast': [[0.25, 0.25, 0.25, 0.25], [0.125, 0.125, 0.25, 0.5], [0.25, 0.125, 0.125, 0.5]]
        }
        
        self.synth = None

    def _init_synth(self):
        try:
            with SuppressStderr():
                self.synth = fluidsynth.Synth()
                if os.name == "nt":
                    try:
                        self.synth.start(driver="waveout")
                    except:
                        self.synth.start()
                else:
                    try:
                        self.synth.start(driver="pulseaudio")
                    except:
                        self.synth.start()
                
                soundfonts = ["models/soundfont.sf2", "soundfont.sf2", "MuseScore_General.sf3"]
                sfid = -1
                for sf in soundfonts:
                    if os.path.exists(sf):
                        sfid = self.synth.sfload(sf)
                        break
                        
                if sfid != -1:
                    self.synth.program_select(0, sfid, 0, 0)
                    self.synth.program_select(1, sfid, 0, 0)
                else:
                    print("Warning: RealTimeSynth - No soundfont found.")
        except Exception as e:
            print(f"Warning: RealTimeSynth - Failed to initialize FluidSynth: {e}")
            self.synth = None

    def update_emotion(self, probs, features):
        with QMutexLocker(self.mutex):
            self.update_queue.append((probs, features))

    def play(self):
        self.is_playing = True
        if self.playback_start_time == 0.0:
            self.playback_start_time = time.time()

    def pause(self):
        self.is_playing = False
        self._all_notes_off()

    def clear_queue(self):
        with QMutexLocker(self.mutex):
            self.update_queue.clear()
            self.note_queue.clear()

    def reset_state(self):
        with QMutexLocker(self.mutex):
            self.playback_start_time = 0.0
            self.update_queue.clear()
            self.note_queue.clear()
            self.emotion_streak = 0
            self.prev_dominant_idx = -1
            self.current_chord_degree = 0
            self.current_dominant_idx = -1
            self._all_notes_off()

    def stop(self):
        self.is_running = False
        self.is_playing = False
        self.playback_start_time = 0.0
        self._all_notes_off()
        self.wait()

    def _all_notes_off(self):
        if not self.synth: return
        for channel in [0, 1]:
            for pitch in range(128):
                self.synth.noteoff(channel, pitch)
        self.active_chord_notes.clear()
        self.active_melody_note = None

    def run(self):
        self.is_running = True
        # Track when the NEXT 1-second chunk should start playing
        # Initialize with a 1.0s delay to create a stable playback buffer
        # This prevents jitter and ensures we always have notes ready to play.
        playback_clock = time.time() + 1.0
        
        while self.is_running:
            if not self.is_playing:
                time.sleep(0.01)
                playback_clock = time.time() + 1.0
                continue
            
            if self.synth is None:
                self._init_synth()

            now = time.time()
            
            # 1. Pull new 1-second chunks from update_queue
            # We look ahead up to 0.5 seconds to ensure we always have notes ready
            if now >= (playback_clock - 0.5):
                state = None
                with QMutexLocker(self.mutex):
                    if len(self.update_queue) > 0:
                        state = self.update_queue.pop(0)

                if state:
                    p, f = state
                    # Schedule this chunk relative to our internal playback clock
                    self._generate_and_schedule_1s_chunk(p, f, playback_clock)
                    playback_clock += 1.0 # Advance clock by exactly 1s
                else:
                    # If queue is empty, we must wait. 
                    # Do NOT advance playback_clock blindly, otherwise we create a gap.
                    # If we are falling too far behind (e.g. queue empty for > 5s), 
                    # we might eventually need to resync, but for now, just wait.
                    if now > playback_clock + 5.0:
                        playback_clock = now
                
            # 2. Process scheduled notes in note_queue
            with QMutexLocker(self.mutex):
                remaining_notes = []
                # Process notes that are due
                for note_event in self.note_queue:
                    sched_time, ch, pitch, vel, duration, is_on = note_event
                    if now >= (sched_time - 0.002): # 2ms early start for smoother playback
                        if self.synth:
                            if is_on:
                                self.synth.noteon(ch, pitch, vel)
                                self.note_played.emit(ch, pitch, vel, sched_time - self.playback_start_time, duration)
                            else:
                                self.synth.noteoff(ch, pitch)
                    else:
                        remaining_notes.append(note_event)
                self.note_queue = remaining_notes

            time.sleep(0.002) # Higher frequency for tighter scheduling

    def _generate_and_schedule_1s_chunk(self, p, features, chunk_start_time):
        if not self.synth: return

        # 1. State Update
        dominant_idx = int(np.argmax(p))
        intensity = float(p[dominant_idx])

        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
        self.prev_dominant_idx = dominant_idx

        # 2. Mode Selection
        if dominant_idx == 3:   # Happy
            current_mode = 'lydian' if intensity > 0.65 else 'ionian'
            chord_type = "triad"
        elif dominant_idx == 0: # Neutral
            current_mode = 'mixolydian' if p[3] > p[1] else 'dorian'
            chord_type = "sus2"
        elif dominant_idx == 1: # Sad
            current_mode = 'phrygian' if intensity > 0.65 else 'aeolian'
            chord_type = "triad"
        elif dominant_idx == 2: # Fear
            current_mode = 'locrian' if intensity > 0.65 else 'phrygian'
            chord_type = "dim"
        else:
            current_mode = 'ionian'
            chord_type = "triad"

        self.state_update.emit(current_mode, chord_type, float(self.current_bpm))
        pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)

        # 3. Dynamics & Tempo
        arousal = float(p[3]) + float(p[2])
        target_bpm = 80 + arousal * 50
        self.current_bpm = 0.7 * self.current_bpm + 0.3 * target_bpm
        velocity = max(30, min(110, int(40 + (arousal * 70))))

        sec_per_beat = 60.0 / self.current_bpm
        
        # Rhythmic Selection (streak-based) - Matches MIDI generator logic
        if self.emotion_streak > 4 and random.random() < 0.3:
            chosen_ratios = [0.5, 0.5] if arousal < 0.5 else [0.25, 0.25, 0.5]
        else:
            if arousal > 0.7:
                chosen_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
            elif arousal > 0.4:
                chosen_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
            else:
                chosen_ratios = [1.0] if random.random() > 0.5 else [0.5, 0.5]

        # 4. chord selection logic matching offline generator
        base_chord_pool_idx = 14
        progression_degrees = {
            'lydian': [0, 3, 4, 5], 'ionian': [0, 3, 4, 5], 'mixolydian': [0, 3, 6, 4],
            'dorian': [0, 2, 3, 4], 'aeolian': [0, 5, 2, 3], 'phrygian': [0, 1, 5, 3], 'locrian': [0, 4, 5, 2]
        }
        
        change_chord_prob = 0.2 if arousal < 0.6 else 0.4
        # Persistent chord degree selection
        if self.current_dominant_idx == -1 or self.emotion_streak == 0 or (self.emotion_streak % 4 == 0 and random.random() < change_chord_prob):
            prog_pattern = progression_degrees.get(current_mode, [0, 3, 4, 5])
            self.current_chord_degree = random.choice(prog_pattern)
            self.current_dominant_idx = dominant_idx

        chord_root_idx = base_chord_pool_idx + self.current_chord_degree
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # 5. Schedule Accompaniment
        with QMutexLocker(self.mutex):
            if dominant_idx == 3: # HAPPY
                chord_vel = max(20, velocity - 10)
                for n in chord_notes:
                    self.note_queue.append((chunk_start_time, 0, int(n), chord_vel, 1.0, True))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0, 0, False))

            elif dominant_idx == 1: # SAD (Arpeggio)
                chord_vel = max(20, velocity - 10)
                low_idx = max(0, chord_root_idx - 7)
                arpeggio = [pool[low_idx], pool[low_idx + 2], pool[low_idx + 4], pool[low_idx + 2]]
                arp_step = 1.0 / len(arpeggio)
                for i, n in enumerate(arpeggio):
                    t_on = chunk_start_time + (i * arp_step)
                    t_off = t_on + (arp_step * 0.99)
                    self.note_queue.append((t_on, 0, int(n), chord_vel, arp_step, True))
                    self.note_queue.append((t_off, 0, int(n), 0, 0, False))

            elif dominant_idx == 2: # FEAR
                chord_vel = max(10, velocity - 30)
                eerie_idx = random.choice([chord_root_idx, chord_root_idx + 2, chord_root_idx + 4]) - 14
                eerie_note = pool[max(0, eerie_idx)]
                self.note_queue.append((chunk_start_time, 0, int(eerie_note), chord_vel, 1.0, True))
                self.note_queue.append((chunk_start_time + 0.99, 0, int(eerie_note), 0, 0, False))

            else: # NEUTRAL
                chord_vel = max(20, velocity - 25)
                for n in chord_notes:
                    self.note_queue.append((chunk_start_time, 0, int(n), chord_vel, 1.0, True))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0, 0, False))

            # 6. Schedule Melody
            time_offset = 0.0
            for r_frac in chosen_ratios:
                dur_sec = float(r_frac) * sec_per_beat * 2
                # Clamp duration to 1s window
                if time_offset + dur_sec > 1.0:
                    dur_sec = 1.0 - time_offset
                if dur_sec < 0.01: break

                chord_adherence_prob = 0.95 if dominant_idx == 3 else 0.70
                if random.random() < chord_adherence_prob:
                    # Sync melody notes with chord notes + octave shift
                    target_note = random.choice(chord_notes) + random.choice([12, 24])
                    note = int(target_note)
                    try: self.melody_idx = min(range(len(pool)), key=lambda k: abs(pool[k] - note))
                    except: pass
                else:
                    step = random.choices([-1, 0, 1], weights=[30, 40, 30])[0]
                    self.melody_idx += step
                    self.melody_idx = max(21, min(35, self.melody_idx))
                    note = int(pool[self.melody_idx])

                if dominant_idx == 3 and self.emotion_streak > 6:
                    note += 12 if random.random() < 0.5 else 0
                if dominant_idx == 2 and random.random() < 0.3:
                    note += random.choice([-1, 1])

                is_rest = (dominant_idx == 0 and random.random() < 0.3)
                if not is_rest:
                    t_on = chunk_start_time + time_offset
                    t_off = t_on + (dur_sec * 0.99)
                    self.note_queue.append((t_on, 1, int(note), int(velocity), dur_sec, True))
                    self.note_queue.append((t_off, 1, int(note), 0, 0, False))

                time_offset += dur_sec

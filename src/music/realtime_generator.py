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
        # In MIDI generator ticks_per_step=960. We map proportionally to seconds.
        # slow: [960] → [1.0], med: proportional, fast: proportional
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
                
                # Load soundfont...
                
                soundfonts = [
                    "models/soundfont.sf2",
                    "soundfont.sf2",
                    "MuseScore_General.sf3"
                ]
                sfid = -1
                for sf in soundfonts:
                    if os.path.exists(sf):
                        sfid = self.synth.sfload(sf)
                        break
                        
                if sfid != -1:
                    # Program 0 is Acoustic Grand Piano
                    self.synth.program_select(0, sfid, 0, 0)
                    self.synth.program_select(1, sfid, 0, 0) # Melody channel
                else:
                    print("Warning: RealTimeSynth - No soundfont found. Playback will be silent.")
        except Exception as e:
            print(f"Warning: RealTimeSynth - Failed to initialize FluidSynth: {e}")
            self.synth = None

    def update_emotion(self, probs, features):
        """Called by the main thread when a new 1-second segment is classified."""
        with QMutexLocker(self.mutex):
            self.update_queue.append((probs, features))

    def play(self):
        self.is_playing = True
        if self.playback_start_time == 0.0:
            self.playback_start_time = time.time()

    def pause(self):
        self.is_playing = False
        self._all_notes_off()

    def stop(self):
        self.is_running = False
        self.is_playing = False
        self.playback_start_time = 0.0
        self._all_notes_off()
        self.wait()

    def set_volume(self, value):
        if self.synth:
            self.synth.cc(0, 7, value)
            self.synth.cc(1, 7, value)

    def _all_notes_off(self):
        if not self.synth: return
        for channel in [0, 1]: # Chord and Melody channels
            for pitch in range(128):
                self.synth.noteoff(channel, pitch)
        self.active_chord_notes.clear()
        self.active_melody_note = None

    def run(self):
        self.is_running = True
        
        # FluidSynth will be initialized lazily when Play is first called
        # to avoid startup crashes if drivers are busy.
        
        next_chunk_time = time.time()
        
        while self.is_running:
            if not self.is_playing:
                time.sleep(0.05)
                next_chunk_time = time.time()
                continue
            
            # Lazy init on first play
            if self.synth is None:
                self._init_synth()

            current_time = time.time()
            if current_time >= next_chunk_time:
                # We reached a 1-second boundary, pull next update if available
                state = None
                with QMutexLocker(self.mutex):
                    if len(self.update_queue) > 0:
                        state = self.update_queue.pop(0)

                if state:
                    p, f = state
                    self._generate_and_play_1s_chunk(p, f)
                
                # Advance boundary by 1 second. 
                # Doing it this way prevents drift compared to simply time.sleep(1.0)
                next_chunk_time += 1.0 
                
                # If we fell way behind (e.g. system suspended), catch up
                if current_time > next_chunk_time + 1.0:
                    next_chunk_time = current_time + 1.0

    def _generate_and_play_1s_chunk(self, p, features):
        if not self.synth: return

        # Get relative time for the start of this chunk
        chunk_rel_time = time.time() - self.playback_start_time

        # 1. State Update
        dominant_idx = int(np.argmax(p))
        intensity = float(p[dominant_idx])

        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
        self.prev_dominant_idx = dominant_idx

        # 2. Mode Selection (Same rules as midi_generator)
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

        # 3. Timbre/Textures via CC (from features)
        if features is not None:
            # Features shape is (62, 5) normally. Average across channels.
            if len(features.shape) == 2:
                bands_mean = np.mean(features, axis=0)
            else:
                bands_mean = features # fallback

            scaled_bands = np.clip((bands_mean + 5) * 6, 0, 127).astype(int) 
            delta, theta, alpha, beta, gamma = int(scaled_bands[0]), int(scaled_bands[1]), int(scaled_bands[2]), int(scaled_bands[3]), int(scaled_bands[4])

            # We apply CC to both channel 0 (chords) and 1 (melody)
            # NOTE: Commented out to match the clean "spot on" sound of the pipeline view.
            # When enabling, use CC 11 (Expression) instead of CC 1 to avoid piano vibrato.
            # for ch in [0, 1]:
            #     self.synth.cc(ch, 93, delta)              # Chorus
            #     self.synth.cc(ch, 11, theta)              # Expression (NOT CC 1)
            #     self.synth.cc(ch, 91, alpha)              # Reverb
            #     self.synth.cc(ch, 73, min(90, 127 - beta)) # Attack
            #     self.synth.cc(ch, 74, gamma)              # Brightness

        # 4. Dynamics & Tempo
        arousal = float(p[3]) + float(p[2])
        
        target_bpm = 80 + arousal * 50
        self.current_bpm = 0.7 * self.current_bpm + 0.3 * target_bpm
        
        velocity = max(30, min(110, int(40 + (arousal * 70))))

        # Rhythmic Selection (mirrors MIDI generator streak-based logic)
        if self.emotion_streak > 4 and random.random() < 0.3:
            # Rhythmic breakdown — a sparse measure
            chosen_rhythm = [0.5, 0.5] if arousal < 0.5 else [0.25, 0.25, 0.5]
        else:
            if arousal > 0.7:
                chosen_rhythm = random.choice(self.rhythms['fast'])
            elif arousal > 0.4:
                chosen_rhythm = random.choice(self.rhythms['med'])
            else:
                chosen_rhythm = random.choice(self.rhythms['slow'])

        # 5. Chord Progression (matches midi_generator exactly)
        base_chord_pool_idx = 14  # C3 in an 8-octave diatonic pool rooted at C1

        progression_degrees = {
            'lydian':     [0, 3, 4, 5],
            'ionian':     [0, 3, 4, 5],
            'mixolydian': [0, 3, 6, 4],
            'dorian':     [0, 2, 3, 4],
            'aeolian':    [0, 5, 2, 3],
            'phrygian':   [0, 1, 5, 3],
            'locrian':    [0, 4, 5, 2]
        }
        
        change_chord_prob = 0.2 if arousal < 0.6 else 0.4
        
        # Emotion change OR periodic chord progression change
        emotion_changed = (dominant_idx != self.current_dominant_idx)
        should_change_chord = (
            emotion_changed
            or self.emotion_streak == 0
            or (self.emotion_streak % 4 == 0 and random.random() < change_chord_prob)
        )

        if should_change_chord:
            prog_pattern = progression_degrees.get(current_mode, [0, 3, 4, 5])
            self.current_chord_degree = random.choice(prog_pattern)
            self.current_dominant_idx = dominant_idx

            # Turn off any previously held chord notes now that we're changing
            for n in self.active_chord_notes:
                self.synth.noteoff(0, n)
            self.active_chord_notes.clear()

        chord_root_idx = base_chord_pool_idx + self.current_chord_degree

        # Build chord note list from diatonic pool (same logic as midi_generator)
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # 6. Accompaniment — mirrors midi_generator's per-emotion style

        sec_per_beat = 60.0 / self.current_bpm
        # Each 1-second chunk maps to roughly (current_bpm/60) beats.
        # ticks_per_step in MIDI = 960 = 2 beats. Here 1 chunk ≈ 1 second ≈ bpm/60 beats.
        step_duration = 1.0   # The full 1-second chunk for sustain purposes.

        if dominant_idx == 3:  # HAPPY — majestic sustained block chord
            chord_vel = max(20, velocity - 10)
            # Only re-attack if chord changed (new block chord sound)
            if not self.active_chord_notes:
                for n in chord_notes:
                    self.synth.noteon(0, n, chord_vel)
                    self.active_chord_notes.append(n)
                    self.note_played.emit(0, n, chord_vel, chunk_rel_time, step_duration)

        elif dominant_idx == 1:  # SAD — cascading broken arpeggio (downward, one octave lower)
            # Turn off any held chord before playing arpeggio
            for n in self.active_chord_notes:
                self.synth.noteoff(0, n)
            self.active_chord_notes.clear()

            chord_vel = max(20, velocity - 10)
            # Mirror MIDI generator: arpeggio uses pool[chord_root_idx - 7] (one diatonic octave down)
            low_idx = max(0, chord_root_idx - 7)
            arpeggio = [
                pool[max(0, low_idx)],
                pool[max(0, low_idx + 2)],
                pool[max(0, low_idx + 4)],
                pool[max(0, low_idx + 2)],  # repeated 3rd (same as MIDI generator)
            ]
            arp_step = step_duration / len(arpeggio)
            for n in arpeggio:
                self.synth.noteon(0, int(n), chord_vel)
                self.note_played.emit(0, int(n), chord_vel, chunk_rel_time, arp_step)
                time.sleep(arp_step * 0.9)
                self.synth.noteoff(0, int(n))
                time.sleep(arp_step * 0.1)
            # After arpeggio completes, active_chord_notes remains empty (arpeggio notes were turn-off'd)

        elif dominant_idx == 2:  # FEAR — one subtle eerie low note
            # Turn off any held chord
            for n in self.active_chord_notes:
                self.synth.noteoff(0, n)
            self.active_chord_notes.clear()

            chord_vel = max(10, velocity - 30)
            eerie_idx = random.choice([chord_root_idx, chord_root_idx + 2, chord_root_idx + 4]) - 14
            eerie_note = pool[max(0, eerie_idx)]
            self.synth.noteon(0, int(eerie_note), chord_vel)
            self.active_chord_notes.append(int(eerie_note))
            self.note_played.emit(0, int(eerie_note), chord_vel, chunk_rel_time, step_duration)

        else:  # NEUTRAL — long floating sustained block chord (softer)
            chord_vel = max(20, velocity - 25)
            if not self.active_chord_notes:
                for n in chord_notes:
                    self.synth.noteon(0, n, chord_vel)
                    self.active_chord_notes.append(n)
                    self.note_played.emit(0, n, chord_vel, chunk_rel_time, step_duration)

        # 7. Melody Generation — mirrors midi_generator's harmonic adherence logic
        # Turn off previous melody note
        if self.active_melody_note is not None:
            self.synth.noteoff(1, self.active_melody_note)
            self.active_melody_note = None

        # Build note list for this chunk (same structure as midi_generator)
        melody_notes_and_durations = []

        for dur_frac in chosen_rhythm:
            dur_sec = float(dur_frac) * sec_per_beat * 2  # ×2 because MIDI step = 2 beats

            chord_adherence_prob = 0.95 if dominant_idx == 3 else 0.70
            if random.random() < chord_adherence_prob:
                # Snap to a chord tone shifted into melody range
                target_note = random.choice(chord_notes) + random.choice([12, 24])
                note = int(target_note)
                try:
                    self.melody_idx = min(range(len(pool)), key=lambda k: abs(pool[k] - note))
                except Exception:
                    pass
            else:
                # Diatonic random walk
                step = random.choices([-1, 0, 1], weights=[30, 40, 30])[0]
                self.melody_idx += step
                self.melody_idx = max(21, min(35, self.melody_idx))
                note = int(pool[self.melody_idx])

            # Phrasing: sustained happy streak → occasional octave jump
            if self.emotion_streak > 6 and dominant_idx == 3:
                note += 12 if random.random() < 0.5 else 0

            # Chromaticism for Fear (intentional dissonance)
            if dominant_idx == 2 and random.random() < 0.3:
                note += random.choice([-1, 1])

            # Rests for Neutral sparseness
            if dominant_idx == 0 and random.random() < 0.3:
                melody_notes_and_durations.append((None, dur_sec))
            else:
                melody_notes_and_durations.append((int(note), dur_sec))

        # Play melody notes sequentially (blocking within this 1-second window)
        time_offset = 0.0
        for note, dur_sec in melody_notes_and_durations:
            # Clamp so we never exceed the 1-second budget
            remaining = 1.0 - time_offset
            if remaining < 0.05:
                break
            dur_sec = min(dur_sec, remaining)

            if note is None:
                time.sleep(dur_sec)
            else:
                self.synth.noteon(1, int(note), int(velocity))
                self.active_melody_note = int(note)
                self.note_played.emit(1, int(note), int(velocity), chunk_rel_time + time_offset, dur_sec)
                time.sleep(dur_sec * 0.9)
                self.synth.noteoff(1, int(note))
                self.active_melody_note = None
                time.sleep(dur_sec * 0.1)

            time_offset += dur_sec

import os
import time
import mido
import fluidsynth
import random
import numpy as np
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from src.music.midi_generator import get_mode_pool, get_chord
from src.music.emotion_tracker import EmotionTracker
from src.music.markov_engine import MarkovEngine

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
        # Melody state
        self.melody_idx = 35
        self.prev_melody_intervals = [0, 0, 0]
        self.force_snap = False
        self.dynamic_key_set = (base_key_offset != 0)
        self.fundamental_bass_root = 36 + self.base_key_offset
        
        # Active notes to turn off cleanly
        self.active_chord_notes = []
        self.active_melody_note = None
        self.current_chord_degree = 0      # Track which scale degree is sounding
        self.current_dominant_idx = -1     # Track which emotion type is active for the chord
        self.progression_step = 0          # Step within the current emotion's progression

        # Timing — rhythm lists match MIDI generator (proportions of a 2-beat step)
        self.rhythms = {
            'slow': [[1.0], [0.5, 0.5], [0.75, 0.25]],
            'med':  [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0.333, 0.333, 0.333]],
            'fast': [[0.25, 0.25, 0.25, 0.25], [0.125, 0.125, 0.25, 0.5], [0.25, 0.125, 0.125, 0.5]]
        }
        
        self.tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
        self.markov_engine = MarkovEngine()
        self.prev_melody_interval = 0
        self.synth = None

    def _init_synth(self):
        try:
            with SuppressStderr():
                # On Windows, 'dsound' or 'waveout' are most reliable. 
                # On Linux, 'pulseaudio' or 'alsa'.
                self.synth = fluidsynth.Synth()
                
                # FluidSynth settings to prevent trying to open MIDI INPUT devices
                # which causes the "Expected:1 found:0" error on many systems.
                self.synth.setting('midi.driver', 'none') 
                
                if os.name == "nt":
                    # For Windows, try dsound then waveout
                    drivers = ["dsound", "waveout", "winmidi"]
                else:
                    drivers = ["pulseaudio", "alsa", "jack"]

                success = False
                for driver in drivers:
                    try:
                        self.synth.start(driver=driver)
                        success = True
                        break
                    except:
                        continue
                
                if not success:
                    # Final attempt with default start
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
                    print("[RealTimeSynth] Warning: No soundfont found.")
        except Exception as e:
            print(f"[RealTimeSynth] Error: Failed to initialize FluidSynth: {e}")
            self.synth = None

    def update_emotion(self, probs, features, timestamp):
        with QMutexLocker(self.mutex):
            # Update the continuous V-A tracker
            dominant_idx = int(np.argmax(probs))
            confidence = float(probs[dominant_idx])
            self.tracker.update_from_discrete(dominant_idx, confidence)
            
            self.update_queue.append((probs, features, timestamp))

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

    def set_volume(self, value):
        """Sets the volume (CC 7) for all active channels (0 and 1)."""
        if self.synth:
            self.synth.cc(0, 7, value)
            self.synth.cc(1, 7, value)

    def run(self):
        self.is_running = True
        # Track when the NEXT 1-second chunk should start playing
        # We use a 2.2s initial delay to provide:
        # 1s for the first EEG window, 1s for user-requested lag, 0.2s for processing headroom.
        playback_clock = time.time() + 2.2
        
        while self.is_running:
            if not self.is_playing:
                time.sleep(0.01)
                playback_clock = time.time() + 2.2
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
                    p, f, ts = state
                    # Schedule this chunk relative to our internal playback clock
                    self._generate_and_schedule_1s_chunk(p, f, playback_clock, ts)
                    playback_clock += 1.0 # Advance clock by exactly 1s
                else:
                    # If queue is empty, we must wait. 
                    if now > playback_clock + 5.0:
                        playback_clock = now
                
            # 2. Process scheduled notes in note_queue
            with QMutexLocker(self.mutex):
                remaining_notes = []
                # Process notes that are due
                for note_event in self.note_queue:
                    # Structure: (sched_time, ch, pitch, vel, duration, is_on, emit_ts)
                    if len(note_event) == 7:
                        sched_time, ch, pitch, vel, duration, is_on, emit_ts = note_event
                    else:
                        # Fallback for old 6-element format if any remain
                        sched_time, ch, pitch, vel, duration, is_on = note_event
                        emit_ts = sched_time - self.playback_start_time

                    if now >= (sched_time - 0.002): # 2ms early start for smoother playback
                        if self.synth:
                            if is_on:
                                self.synth.noteon(ch, pitch, vel)
                                # Emit the EEG-aligned timestamp for the piano roll
                                self.note_played.emit(ch, pitch, vel, emit_ts, duration)
                            else:
                                self.synth.noteoff(ch, pitch)
                    else:
                        remaining_notes.append(note_event)
                self.note_queue = remaining_notes

            time.sleep(0.002) # Higher frequency for tighter scheduling

    def _generate_and_schedule_1s_chunk(self, p, features, chunk_start_time, eeg_timestamp):
        if not self.synth: return

        # ── 1. Pull both Macro (slow mood) and Micro (instantaneous delta) states ──
        state = self.tracker.get_state()
        macro_v = state['macro_v']   
        macro_a = state['macro_a']   
        micro_v = state['micro_v']   
        micro_a = state['micro_a']   
        is_spike = state['is_spike']

        # ── Dynamic Key Selection (Schubert) ──
        if not self.dynamic_key_set:
            self.dynamic_key_set = True
            dom_idx = int(np.argmax(p))
            # 0=Neutral, 1=Sad, 2=Fear, 3=Happy
            EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
            if dom_idx == 3: # Happy -> C Maj (0) or G Maj (7)
                self.base_key_offset = random.choice([0, 7])
            elif dom_idx == 1: # Sad -> D Min (2) or F Min (5)
                self.base_key_offset = random.choice([2, 5])
            elif dom_idx == 2: # Fear -> C# Min (1) or Eb Min (3)
                self.base_key_offset = random.choice([1, 3])
            else: # Neutral -> F Maj (5) or A Min (9)
                self.base_key_offset = random.choice([5, 9])
            
            self.fundamental_bass_root = 36 + self.base_key_offset
            print(f"[Synthesizer] Dynamic Key Set! Emotion: {EMOTION_LABELS[dom_idx]}, Offset: +{self.base_key_offset}")

        # Track emotion streak for chord-change logic
        dominant_idx = int(np.argmax(p))
        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
            # NEW LOGIC: Clear interval buffer and force a melodic anchor on emotion shift
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        if is_spike:
            # Spikes also reset momentum to prevent wildly dissonant leaps 
            # from mismatched state matrices.
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        self.prev_dominant_idx = dominant_idx

        # ── 2. Mode & Chord Quality (driven entirely by MACRO valence) ──
        if macro_v > 0.5:
            current_mode = 'lydian' if macro_v > 0.75 else 'ionian'
            chord_type = "triad"
        elif macro_v > 0.1:
            current_mode = 'mixolydian'
            chord_type = "sus2"
        elif macro_v > -0.2:
            current_mode = 'dorian'
            chord_type = "sus2"
        elif macro_v > -0.6:
            current_mode = 'aeolian'
            chord_type = "triad"
        else:
            current_mode = 'phrygian' if macro_v > -0.8 else 'locrian'
            chord_type = "dim"

        self.state_update.emit(current_mode, chord_type, float(self.current_bpm))
        pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)

        # ── 3. Tempo (MACRO target, MICRO smoothing speed) ──
        # Macro arousal sets the target BPM (slow drift: 60–140 BPM).
        target_bpm = 100 + (macro_a * 40)
        # When there's a micro spike the smoothing is snappier so the tempo
        # reacts quickly; during calm periods it glides gradually.
        alpha = 0.3 + 0.5 * min(1.0, abs(micro_a))  # 0.3 (smooth) → 0.8 (snappy)
        self.current_bpm = (1.0 - alpha) * self.current_bpm + alpha * target_bpm
        sec_per_beat = 60.0 / self.current_bpm

        # ── 4. Velocity (MACRO base + MICRO intensity boost) ──
        # Base from macro arousal (30–110), boosted by micro arousal magnitude.
        velocity = int(70 + (macro_a * 30) + (abs(micro_a) * 20))
        velocity = max(30, min(110, velocity))

        # ── 5. Rhythmic Density (MACRO arousal category, MICRO bias toward dense) ──
        # micro_density_bias shifts probability toward the denser variant within
        # the macro-selected density category.
        micro_density_bias = min(1.0, abs(micro_a))  # 0 → 1
        if macro_a > 0.6:
            # Fast category — micro bias further prefers shortest sub-divisions
            chosen_ratios = (
                [0.125, 0.125, 0.25, 0.5] if random.random() < (0.4 + 0.4 * micro_density_bias)
                else [0.25, 0.25, 0.25, 0.25]
            )
        elif macro_a > 0.0:
            # Medium category — micro bias toward triplet feel
            chosen_ratios = (
                [0.333, 0.333, 0.333] if random.random() < (0.3 + 0.4 * micro_density_bias)
                else [0.5, 0.25, 0.25]
            )
        else:
            # Slow category — micro bias allows a split even at low arousal
            chosen_ratios = (
                [0.5, 0.5] if random.random() < (0.2 + 0.5 * micro_density_bias)
                else [1.0]
            )

        # ── 6. Harmonic Progression (Emotion-Specific Categories) ──
        if macro_v > 0.4 and macro_a > 0.0:
            emotion_cat = 'happy'
        elif macro_v < -0.3 and macro_a < 0.0:
            emotion_cat = 'sad'
        elif macro_v < -0.3 and macro_a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        PROGRESSIONS = {
            'happy':   [0, 4, 5, 3],
            'sad':     [0, 6, 5, 4],
            'fear':    [0, 0, 1, 0],
            'neutral': [1, 0, 3, 4],
        }

        if macro_a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif macro_a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        if self.current_dominant_idx == -1 or self.emotion_streak == 0:
            self.progression_step = 0
            self.current_chord_degree = PROGRESSIONS[emotion_cat][self.progression_step]
            self.current_dominant_idx = dominant_idx
        elif self.emotion_streak % harmonic_rhythm == 0:
            self.progression_step = (self.progression_step + 1) % 4
            self.current_chord_degree = PROGRESSIONS[emotion_cat][self.progression_step]

        chord_root_idx = 14 + self.current_chord_degree
        chord_notes = get_chord(current_mode, pool[chord_root_idx], chord_type)

        # ── 7. Schedule Notes ──
        with QMutexLocker(self.mutex):

            # ── Accompaniment style (MACRO category + MICRO spike) ──
            # Spike surge: purely dynamic (velocity boost), no fast arpeggio trills
            micro_chord_boost = 0
            if is_spike:
                micro_intensity = min(1.0, abs(micro_a))
                micro_chord_boost = int(micro_intensity * 40)

            chord_vel = min(110, max(20, velocity - 10 + micro_chord_boost))

            if emotion_cat == 'happy':
                # sustained_block
                for n in chord_notes:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'sad':
                # open_wide
                root  = int(chord_notes[0])
                fifth = max(0, root - 5)
                third = min(127, int(chord_notes[1]) + 12)
                for n in [root, fifth, third]:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'fear':
                # drone_pedal
                root = max(0, int(chord_notes[0]) - 12)
                vel = min(110, max(10, chord_vel - 10))
                
                self.note_queue.append((chunk_start_time,        0, root, vel, 1.0, True,  eeg_timestamp))
                self.note_queue.append((chunk_start_time + 0.99, 0, root, 0,   0,   False, eeg_timestamp + 0.99))
                
                if is_spike:
                    # Occasional tense 5th added very softly during a surge to thicken
                    fifth = min(127, root + 7)
                    self.note_queue.append((chunk_start_time,        0, fifth, max(10, vel - 15), 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, fifth, 0,                 0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'neutral':
                # quartal_float
                root   = int(chord_notes[0])
                fourth = min(127, root + 5)
                flat7  = min(127, fourth + 5)
                for n in [root, fourth, flat7]:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            # ── Melody (MACRO for harmonic adherence, MICRO for expression) ──
            time_offset = 0.0
            for r_frac in chosen_ratios:
                dur_sec = float(r_frac) * sec_per_beat * 2
                if time_offset + dur_sec > 1.0:
                    dur_sec = 1.0 - time_offset
                if dur_sec < 0.01:
                    break

                # Harmonic adherence: driven by macro valence (stable pull toward key)
                # plus a small corrective nudge from micro valence (momentary brightness/darkness)
                chord_adherence_prob = 0.7 + (macro_v * 0.2) + (micro_v * 0.1)
                chord_adherence_prob = max(0.4, min(0.95, chord_adherence_prob))

                if self.force_snap or random.random() < chord_adherence_prob:
                    # Ground the melody back to the current safe chord tones
                    target_note = random.choice(chord_notes) + random.choice([12, 24])
                    note = int(target_note)
                    try:
                        self.melody_idx = min(range(len(pool)), key=lambda k: abs(pool[k] - note))
                    except Exception:
                        pass
                    
                    if self.force_snap:
                        self.force_snap = False
                else:
                    step = self.markov_engine.query_next_interval(emotion_cat, self.prev_melody_intervals)
                    self.prev_melody_intervals.pop(0)
                    self.prev_melody_intervals.append(step)
                    self.melody_idx += step
                    self.melody_idx = max(21, min(len(pool)-1, self.melody_idx))
                    note = int(pool[self.melody_idx])

                # Octave jumps / chromaticism: driven by MICRO valence so they feel
                # like an immediate emotional reaction, not a slow stylistic choice.
                if micro_v > 0.4 and random.random() < 0.35:
                    note += 12  # Sudden brightness / elation
                if micro_v < -0.4 and random.random() < 0.35:
                    note += random.choice([-1, 1])  # Sudden tension / unease


                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

                is_rest = (abs(macro_v) < 0.2 and abs(macro_a) < 0.2
                           and random.random() < 0.3)
                if not is_rest:
                    t_on  = chunk_start_time + time_offset
                    t_off = t_on + (dur_sec * 0.99)
                    e_ts  = eeg_timestamp + time_offset
                    self.note_queue.append((t_on,  1, int(note), int(velocity), dur_sec, True,  e_ts))
                    self.note_queue.append((t_off, 1, int(note), 0,             0,       False, e_ts + dur_sec * 0.99))

                time_offset += dur_sec

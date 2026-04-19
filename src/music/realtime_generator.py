import os
import time
import mido
import fluidsynth
import random
import numpy as np
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from src.music.midi_generator import (
    get_mode_pool, get_mode_intervals, get_chord, CHORD_TRANSITIONS,
    SPIKE_PROFILES, apply_spike_chord_color, detect_trill, resolve_dissonance
)
from src.music.emotion_tracker import EmotionTracker
from src.music.markov_engine import MarkovEngine

class SuppressStderr:
    """Context manager to suppress C-level audio driver warnings"""
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
        
        # STATE QUEUE (to receive 1-second updates from EEG worker)
        self.update_queue = []
        # NOTES QUEUE: list of (timestamp, channel, pitch, velocity, duration, is_note_on)
        self.note_queue = []
        
        # INTERNAL MUSIC PARAMETERS
        self.current_bpm = 100
        self.prev_dominant_idx = -1
        self.emotion_streak = 0

        # MELODY STATE
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

        # RHYTHM DICTIONARY:
        # this matches MIDI generator (proportions of a 2-beat step)
        self.rhythms = {
            'slow': [[1.0], [0.5, 0.5], [0.75, 0.25]],
            'med':  [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5], [0.333, 0.333, 0.333]],
            'fast': [[0.25, 0.25, 0.25, 0.25], [0.125, 0.125, 0.25, 0.5], [0.25, 0.125, 0.125, 0.5]]
        }
        
        self.tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
        self.markov_engine = MarkovEngine()
        self.prev_melody_interval = 0
        self.synth = None

        # Phase 3A: Anti-trill tracker
        self.consecutive_trill_count = 0
        # Phase 4: Spike profile tracking
        self.spike_duration_counter = 0
        self.active_spike_profile = None
        # Phase 2: Fear sustain pedal state
        self.fear_sustain_active = False
        # Phase 7: Fear State Machine
        self.fear_submode = 'ambiguity'
        self.fear_tick_counter = 0
        self.fear_ramp_velocity = 35

    def _init_synth(self):
        try:
            with SuppressStderr():
                # On WINDOWS, 'dsound' or 'waveout' are most reliable. 
                # On LINUX, 'pulseaudio' or 'alsa'.
                self.synth = fluidsynth.Synth()
                
                # FluidSynth settings to prevent trying to open MIDI INPUT devices
                # which causes the "Expected:1 found:0" error on many systems.
                self.synth.setting('midi.driver', 'none') 
                
                if os.name == "nt":
                    # For WINDOWS, try dsound then waveout
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

    def update_emotion(self, probs, timestamp):
        with QMutexLocker(self.mutex):
            # Update the continuous V-A tracker
            dominant_idx = int(np.argmax(probs))
            confidence = float(probs[dominant_idx])
            self.tracker.update_from_discrete(dominant_idx, confidence)
            
            self.update_queue.append((probs, timestamp))

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
            self.consecutive_trill_count = 0
            self.spike_duration_counter = 0
            self.active_spike_profile = None
            self.fear_sustain_active = False
            self.fear_submode = 'ambiguity'
            self.fear_tick_counter = 0
            self.fear_ramp_velocity = 35
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
            self.synth.cc(channel, 64, 0)  # Release sustain pedal
            for pitch in range(128):
                self.synth.noteoff(channel, pitch)
        self.active_chord_notes.clear()
        self.active_melody_note = None
        self.fear_sustain_active = False

    def set_volume(self, value):
        """Sets the volume (CC 7) for all active channels (0 and 1)."""
        if self.synth:
            self.synth.cc(0, 7, value)
            self.synth.cc(1, 7, value)

    def run(self):
        self.is_running = True
        # Track when the NEXT 1-second chunk should start playing
        # We use a 2.2s initial delay to provide:
        # -> 1s for the first EEG window
        # -> 1s for user-requested lag
        # -> 0.2s for processing headroom.
        playback_clock = time.time() + 2.2
        
        while self.is_running:
            if not self.is_playing:
                time.sleep(0.01)
                playback_clock = time.time() + 2.2
                continue
            
            if self.synth is None:
                self._init_synth()

            now = time.time()
            
            # 1. PULL NEW 1-SECOND CHUNKS FROM UPDATE_QUEUE
            # We dont just wait for all current chunks to be processed before pulling new ones
            # We look ahead up to 0.5 seconds to ensure we dont run out of notes to play
            if now >= (playback_clock - 0.5):
                state = None
                with QMutexLocker(self.mutex):
                    if len(self.update_queue) > 0:
                        state = self.update_queue.pop(0)

                if state:
                    p, ts = state
                    # Schedule current chunk relative to our internal playback clock
                    self._generate_and_schedule_1s_chunk(p, playback_clock, ts)
                    playback_clock += 1.0 # clock advances by 1s
                else:
                    # In the unlikely event the update queue is empty, 
                    # we must wait for the next chunk to arrive.
                    if now > playback_clock + 5.0:
                        playback_clock = now
                
            # 2. PROCESS SCHEDULED NOTES IN NOTE_QUEUE
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

    def _generate_and_schedule_1s_chunk(self, p, chunk_start_time, eeg_timestamp):
        if not self.synth: return

        # ── 1. PULL BOTH MACRO (SLOW MOOD) AND MICRO (INSTANTANEOUS DELTA) STATES ──
        state = self.tracker.get_state()
        macro_v = state['macro_v']   
        macro_a = state['macro_a']   
        micro_v = state['micro_v']   
        micro_a = state['micro_a']   
        is_spike = state['is_spike']
        spike_label = state['spike_label']
        macro_label = state['macro_label']

        # ── DYNAMIC KEY SELECTION (Schubert) ──
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

        # Track emotion streak for chord-change logic (adds variety for long sustained emotions)
        dominant_idx = int(np.argmax(p))
        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
            # Clear interval buffer and force a melodic anchor on emotion shift
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        if is_spike:
            # Spikes also reset momentum to prevent wildly dissonant leaps 
            # from mismatched state matrices.
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        self.prev_dominant_idx = dominant_idx

        # Secondary emotion for neutral chameleon behavior
        sorted_emotions = np.argsort(p)
        secondary_idx = sorted_emotions[-2]
        LABEL_NAMES = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}
        secondary_label = LABEL_NAMES.get(secondary_idx, 'neutral')

        # Map MACRO V/A to discrete emotional categories early
        if macro_v > 0.0 and macro_a > 0.0:
            emotion_cat = 'happy'
        elif macro_v < 0.0 and macro_a < 0.0:
            emotion_cat = 'sad'
        elif macro_v < 0.0 and macro_a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        # ── SPIKE PROFILE MANAGEMENT ──
        if is_spike and macro_label != spike_label:
            spike_key = (macro_label, spike_label)
            self.active_spike_profile = SPIKE_PROFILES.get(spike_key)
            self.spike_duration_counter += 1
        else:
            self.spike_duration_counter = 0
            self.active_spike_profile = None

        # --------------------------------------------------------------------------
        # 1. MODE SELECTION
        # VALENCE determines the "color" of the music (scale)
        is_neutral = (emotion_cat == 'neutral')

        if emotion_cat == 'neutral':
            current_mode = 'mixolydian'
            chord_type = 'sus2'
        elif emotion_cat == 'happy':
            current_mode = 'lydian' if macro_v > 0.75 else 'ionian'
            chord_type = "triad"
        elif emotion_cat == 'sad':
            current_mode = 'aeolian'
            chord_type = "triad"
        else: # fear
            if macro_v > -0.2:
                current_mode = 'dorian'
                chord_type = "sus2"
            elif macro_v > -0.6:
                current_mode = 'aeolian'
                chord_type = "triad"
            else:
                # Use locrian extremely rarely, stick to phrygian mostly
                current_mode = 'phrygian' if macro_v > -0.95 else 'locrian'
                chord_type = "dim"

        self.state_update.emit(current_mode, chord_type, float(self.current_bpm))
        pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)

        # ── Neutral pentatonic restriction ──
        if is_neutral:
            pentatonic_degrees = [0, 1, 3, 4, 5]
            pentatonic_pool = []
            intervals = get_mode_intervals(current_mode)
            root = 24 + self.base_key_offset
            for oct in range(8):
                for deg in pentatonic_degrees:
                    note = root + (oct * 12) + intervals[deg]
                    if note <= 127:
                        pentatonic_pool.append(note)
            melody_pool = pentatonic_pool
        else:
            melody_pool = pool

        # ── 3. TEMPO (MACRO TARGET, MICRO SMOOTHING SPEED) ──
        # MACRO arousal sets the target BPM (slow drift: 60–140 BPM).
        target_bpm = 100 + (macro_a * 40)

        # Apply spike tempo modifier
        if self.active_spike_profile:
            if self.spike_duration_counter <= 2:
                # Short spike: subtle tempo nudge (5-8%)
                nudge = 0.06 if micro_a > 0 else -0.06
                target_bpm *= (1.0 + nudge)
            else:
                # Full spike: apply profile tempo multiplier
                target_bpm *= self.active_spike_profile['tempo_mult']

        # When there's a MICRO spike the smoothing is more abrupt so the tempo
        # reacts quickly; during calm periods it glides gradually.
        alpha = 0.3 + 0.5 * min(1.0, abs(micro_a))  # 0.3 (smooth) → 0.8 (abrupt)
        self.current_bpm = (1.0 - alpha) * self.current_bpm + alpha * target_bpm
        sec_per_beat = 60.0 / self.current_bpm

        # ── 4. VELOCITY (MACRO BASE + MICRO INTENSITY BOOST) ──
        # Base from MACRO arousal (30–110), boosted by MICRO arousal magnitude.
        velocity = int(70 + (macro_a * 30) + (abs(micro_a) * 20))
        velocity = max(30, min(110, velocity))

        # Sad needs more presence (higher velocity)
        if macro_label == 'sad':
            velocity = max(50, min(110, velocity + 15))

        # Apply spike velocity modifier
        if self.active_spike_profile:
            if self.spike_duration_counter <= 2:
                # Short spike: subtle velocity boost (12%)
                velocity = int(velocity * 1.12)
            else:
                velocity = velocity + self.active_spike_profile['vel_shift']
            velocity = max(30, min(110, velocity))

        # ── 5. RHYTHMIC DENSITY (MACRO AROUSAL CATEGORY, MICRO BIAS TOWARD DENSE) ──
        # MICRO density bias shifts probability toward the denser variant within
        # the MACRO-selected density category.
        micro_density_bias = min(1.0, abs(micro_a))  # 0 → 1
        if macro_a > 0.6:
            # Fast category — MICRO bias further prefers shortest sub-divisions
            chosen_ratios = (
                [0.125, 0.125, 0.25, 0.5] if random.random() < (0.4 + 0.4 * micro_density_bias)
                else [0.25, 0.25, 0.25, 0.25]
            )
        elif macro_a > 0.0:
            # Medium category — MICRO bias toward triplet feel
            chosen_ratios = (
                [0.333, 0.333, 0.333] if random.random() < (0.3 + 0.4 * micro_density_bias)
                else [0.5, 0.25, 0.25]
            )
        else:
            # Slow category — MICRO bias allows a split even at low arousal
            chosen_ratios = (
                [0.5, 0.5] if random.random() < (0.2 + 0.5 * micro_density_bias)
                else [1.0]
            )

        # ── 6. HARMONIC PROGRESSION (EMOTION-SPECIFIC CATEGORIES) ──

        # --- CHORD PROGRESSION (Markov Transition Matrix) ---

        if macro_a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif macro_a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        # Sad & Neutral chord variety: prevent boring repetition at slow tempos
        if emotion_cat in ('sad', 'neutral') and self.emotion_streak > 2:
            harmonic_rhythm = min(harmonic_rhythm, 2)
        # Even more aggressive for neutral — never hold more than 3 steps
        if emotion_cat == 'neutral':
            harmonic_rhythm = min(harmonic_rhythm, 3)

        # Advance chord via Markov dice roll
        if emotion_cat == 'fear':
            # Fear State Machine & chord override (eerie submodes)
            if self.current_dominant_idx == -1 or self.emotion_streak == 0 or self.prev_dominant_idx != 2:
                self.fear_submode = random.choices(['ambiguity', 'eerie_melodic', 'climax'], weights=[65, 30, 5])[0]
                self.fear_tick_counter = 0
                self.fear_ramp_velocity = 35
                self.current_chord_degree = 0
                self.current_dominant_idx = dominant_idx

            self.fear_tick_counter += 1
            if self.fear_submode == 'eerie_melodic':
                # Descending harmony (e.g. 0 -> 6 -> 5 -> 4)
                if self.emotion_streak % harmonic_rhythm == 0:
                    self.current_chord_degree = (self.current_chord_degree - 1) % 7
            else:
                # Ambiguity/Climax stays mostly on tonic or steps slightly
                if self.emotion_streak % harmonic_rhythm == 0:
                    matrix = CHORD_TRANSITIONS.get(emotion_cat, {})
                    if self.current_chord_degree in matrix:
                        options = matrix[self.current_chord_degree]['options']
                        weights = matrix[self.current_chord_degree]['weights']
                        self.current_chord_degree = random.choices(options, weights=weights, k=1)[0]
                    else:
                        self.current_chord_degree = 0

        elif self.current_dominant_idx == -1 or self.emotion_streak == 0:
            # On start or emotion shift, force a safe root
            self.current_chord_degree = 0
            self.current_dominant_idx = dominant_idx
        elif self.emotion_streak % harmonic_rhythm == 0:
            # Time to change the chord — look up the transition matrix
            matrix = CHORD_TRANSITIONS.get(emotion_cat, {})
            if self.current_chord_degree in matrix:
                options = matrix[self.current_chord_degree]['options']
                weights = matrix[self.current_chord_degree]['weights']
                self.current_chord_degree = random.choices(options, weights=weights, k=1)[0]
            else:
                self.current_chord_degree = 0  # Fallback

        chord_root_idx = 14 + self.current_chord_degree
        
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "sus4":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 3], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # Harmonic Minor Alteration: Tension chord (Major V) for Sad emotion
        if emotion_cat == 'sad' and self.current_chord_degree == 4 and chord_type == "triad":
            chord_notes[1] += 1  # Raise minor 3rd to Major 3rd

        # Apply spike chord coloring (only for sustained spikes, 3+ samples)
        if self.active_spike_profile and self.spike_duration_counter > 2:
            chord_notes = apply_spike_chord_color(chord_notes, self.active_spike_profile['chord_color'])

        # Spike rest probability override for melody
        spike_rest_prob = self.active_spike_profile['rest_prob'] if self.active_spike_profile and self.spike_duration_counter > 2 else None

        # ── 7. SCHEDULE NOTES ──
        with QMutexLocker(self.mutex):

            # ── Fear sustain pedal management ──
            if emotion_cat == 'fear' and not self.fear_sustain_active:
                self.synth.cc(0, 64, 127)  # Sustain ON
                self.fear_sustain_active = True
            elif emotion_cat != 'fear' and self.fear_sustain_active:
                self.synth.cc(0, 64, 0)    # Sustain OFF
                self.fear_sustain_active = False

            # ── Accompaniment style (MACRO category + MICRO spike) ──
            # Spike surge: purely dynamic (velocity boost), no fast arpeggio trills
            micro_chord_boost = 0
            if is_spike:
                micro_intensity = min(1.0, abs(micro_a))
                micro_chord_boost = int(micro_intensity * 40)

            # Emotion-specific chord velocity
            if emotion_cat == 'sad':
                chord_vel = min(110, max(40, velocity + 5 + micro_chord_boost))
            else:
                chord_vel = min(110, max(20, velocity - 10 + micro_chord_boost))

            if emotion_cat == 'happy':
                # sustained_block
                # Register cap: keep happy chords warm, not tense (max C4 = 72)
                chord_notes = [n - 12 if n > 72 else n for n in chord_notes]
                for n in chord_notes:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'sad':
                # open_wide (Diatonic): Root + 3rd (octave up) + 5th (octave down if possible)
                root  = int(chord_notes[0])
                third = int(chord_notes[1]) + 12
                fifth = int(chord_notes[2]) - 12
                
                fifth = fifth if fifth >= 0 else fifth + 12
                third = third if third <= 127 else third - 12
                
                for n in [root, fifth, third]:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'fear':
                # Brings chords higher up to bridge the gap with the melody
                root = max(0, int(chord_notes[0]) - 12)
                fifth_down = max(0, int(chord_notes[2]) - 12)

                # Route submode logic
                if self.fear_submode == 'climax':
                    vel = 110 if random.random() < 0.2 else 0 # 20% chance of sudden stab
                    if vel > 0:
                        minor_2nd = min(127, root + 1)
                        self.note_queue.append((chunk_start_time,        0, root, vel, 1.0, True,  eeg_timestamp))
                        self.note_queue.append((chunk_start_time,        0, fifth_down, vel - 10, 1.0, True,  eeg_timestamp))
                        self.note_queue.append((chunk_start_time,        0, minor_2nd, vel - 10, 1.0, True,  eeg_timestamp))
                        
                        self.note_queue.append((chunk_start_time + 0.25, 0, root, 0,   0,   False, eeg_timestamp + 0.25))
                        self.note_queue.append((chunk_start_time + 0.99, 0, fifth_down, 0,   0,   False, eeg_timestamp + 0.99))
                        self.note_queue.append((chunk_start_time + 0.25, 0, minor_2nd, 0,   0,   False, eeg_timestamp + 0.25))
                    else:
                        self.note_queue.append((chunk_start_time + 0.99, 0, root, 0,   0,   False, eeg_timestamp + 0.99))
                else:
                    if self.fear_submode == 'ambiguity':
                        if self.fear_tick_counter % 15 == 0:
                            self.fear_ramp_velocity = 45 # Reset, increased base intensity
                        else:
                            self.fear_ramp_velocity += 4
                        vel = min(100, self.fear_ramp_velocity)
                    else:
                        vel = random.randint(55, 75) # Constant brooding, higher base

                    minor_2nd = min(127, root + 1)
                    self.note_queue.append((chunk_start_time,        0, root, vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, root, 0,   0,   False, eeg_timestamp + 0.99))
                    
                    if random.random() < 0.15:
                        self.note_queue.append((chunk_start_time,        0, minor_2nd, max(10, vel // 3), 1.0, True,  eeg_timestamp))
                        self.note_queue.append((chunk_start_time + 0.99, 0, minor_2nd, 0,                 0,   False, eeg_timestamp + 0.99))
                        
                    self.note_queue.append((chunk_start_time,        0, fifth_down, max(10, vel - 15), 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, fifth_down, 0,                 0,   False, eeg_timestamp + 0.99))

            elif emotion_cat == 'neutral':
                # quartal_float (Diatonic): Stack 3 notes using pool indices to guarantee diatonic 4ths/3rds
                idx = chord_root_idx
                q1 = pool[idx]
                q2 = pool[idx + 3] if (idx + 3) < len(pool) else pool[-1]
                q3 = pool[idx + 6] if (idx + 6) < len(pool) else pool[-1]
                
                for n in [q1, q2, q3]:
                    self.note_queue.append((chunk_start_time,        0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0,         0,   False, eeg_timestamp + 0.99))

            # ── Melody (MACRO for harmonic adherence, MICRO for expression) ──
            melody_notes_for_trill = []  # Track for anti-trill detection
            time_offset = 0.0
            
            # Create a mutable copy of the pool for this chunk
            active_pool = list(melody_pool)
            
            # Harmonic Minor Alteration: Align melody with the tension V chord
            if emotion_cat == 'sad' and self.current_chord_degree == 4:
                for idx in range(len(active_pool)):
                    if (idx % 7) == 6:
                        active_pool[idx] += 1
            for r_frac in chosen_ratios:
                dur_sec = float(r_frac) * sec_per_beat * 2
                if time_offset + dur_sec > 1.0:
                    dur_sec = 1.0 - time_offset
                if dur_sec < 0.01:
                    break

                if emotion_cat == 'fear':
                    # Sparse eerie right hand: mostly rests with occasional notes
                    if random.random() < 0.55:
                        time_offset += dur_sec
                        continue  # REST — silence builds tension
                    # Bring melody register down to approach the chords (C4-C5 range)
                    self.melody_idx = random.randint(28, min(len(active_pool) - 1, 36))
                    note = int(active_pool[self.melody_idx])
                    # Rare chromatic coloring for eeriness (sparing for musicality)
                    if random.random() < 0.15:
                        note += random.choice([-1, 1, 6, -6])
                else:
                    # Harmonic adherence: driven by macro valence (stable pull toward key)
                    # plus a small corrective nudge from micro valence (momentary brightness/darkness)
                    chord_adherence_prob = 0.7 + (macro_v * 0.2) + (micro_v * 0.1)
                    chord_adherence_prob = max(0.15, min(0.95, chord_adherence_prob))
                    # Happy/Sad: higher floor to prevent dissonant clashes
                    if emotion_cat in ('happy', 'sad'):
                        chord_adherence_prob = max(0.80, chord_adherence_prob)
                    # Neutral: lower adherence to reduce tonic repetition, but prefer non-root tones
                    if emotion_cat == 'neutral':
                        chord_adherence_prob = min(0.55, chord_adherence_prob)

                    if self.force_snap or random.random() < chord_adherence_prob:
                        # Ground the melody back to the current safe chord tones
                        # Snap to safe triad tones only (Root, 3rd, 5th)
                        # to prevent dissonant clashes with any 7ths or extensions
                        safe_snap_notes = chord_notes[:3]
                        # Neutral: prefer non-root tones to reduce tonic repetition
                        if emotion_cat == 'neutral' and len(safe_snap_notes) > 1:
                            safe_snap_notes = safe_snap_notes[1:]  # skip root
                        target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                        note = int(target_note)
                        try:
                            self.melody_idx = min(range(len(active_pool)), key=lambda k: abs(active_pool[k] - note))
                        except Exception:
                            pass
                        
                        if self.force_snap:
                            self.force_snap = False
                    else:
                        step = self.markov_engine.query_next_interval(emotion_cat, self.prev_melody_intervals)
                        self.prev_melody_intervals.pop(0)
                        self.prev_melody_intervals.append(step)
                        self.melody_idx += step
                        self.melody_idx = max(21, min(len(active_pool)-1, self.melody_idx))
                        note = int(active_pool[self.melody_idx])

                    # Micro valence expression — ONLY for fear/neutral transitions
                    if micro_v > 0.4 and random.random() < 0.35:
                        note += 12  # Octave jump is always safe
                    if emotion_cat == 'fear' and micro_v < -0.4 and random.random() < 0.25:
                        note += random.choice([-1, 1])  # Chromatic only for fear

                # Apply spike melody register shift
                if self.active_spike_profile and self.spike_duration_counter > 2:
                    note += self.active_spike_profile['melody_register']

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

                note = max(0, min(127, note))
                
                # Universal Dissonance Guard
                if emotion_cat in ('happy', 'sad', 'neutral'):
                    note = resolve_dissonance(note, chord_notes)

                # Determine rest probability
                if spike_rest_prob is not None:
                    is_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.3:
                    is_rest = True
                else:
                    is_rest = False

                if not is_rest:
                    # Fear melody: separate velocity for eerie dynamics
                    if emotion_cat == 'fear':
                        mel_vel = random.randint(90, 110) if random.random() < 0.12 else random.randint(55, 75)
                    else:
                        mel_vel = int(velocity)

                    t_on  = chunk_start_time + time_offset
                    t_off = t_on + (dur_sec * 0.99)
                    e_ts  = eeg_timestamp + time_offset
                    self.note_queue.append((t_on,  1, int(note), mel_vel, dur_sec, True,  e_ts))
                    self.note_queue.append((t_off, 1, int(note), 0,             0,       False, e_ts + dur_sec * 0.99))
                    melody_notes_for_trill.append((int(note), dur_sec))

                time_offset += dur_sec

            # ── Anti-trill guard (applies to happy, sad, neutral) ──
            # Rule: max 1 trill per 5 seconds — if ANY trill detected, force snap next chunk
            if emotion_cat in ('happy', 'sad', 'neutral') and len(melody_notes_for_trill) >= 4:
                if detect_trill(melody_notes_for_trill):
                    self.consecutive_trill_count += 1
                else:
                    self.consecutive_trill_count = 0
                # Trigger on FIRST trill detection
                if self.consecutive_trill_count >= 1:
                    self.force_snap = True
                    self.consecutive_trill_count = 0

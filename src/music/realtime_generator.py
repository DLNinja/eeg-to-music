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
        # Neutral: lock Dorian/Mixolydian per passage
        self.neutral_locked_mode = None

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
            self.neutral_locked_mode = None
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
        spike_intensity = state['spike_intensity']
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
            # Lock mode at start of each neutral passage based on sad vs happy probability
            if self.emotion_streak == 0 or self.neutral_locked_mode is None:
                if p[1] >= p[3]:  # sad probability >= happy probability
                    self.neutral_locked_mode = 'dorian'   # Melancholic but dignified
                else:
                    self.neutral_locked_mode = 'mixolydian'  # Wonder and discovery
            current_mode = self.neutral_locked_mode
            # Pure triads only — modal character comes from scale + progression
            chord_type = 'triad'
        elif emotion_cat == 'happy':
            current_mode = 'lydian' if macro_v > 0.75 else 'ionian'
            chord_type = "triad"
        elif emotion_cat == 'sad':
            current_mode = 'aeolian'
            chord_type = "triad"
        else: # fear
            # All fear uses non-tertian voicings (no 3rd = no major/minor identity)
            if macro_v > -0.4:
                current_mode = 'phrygian'        # ♭2 = brooding, oppressive, claustrophobic
            elif macro_v > -0.7:
                current_mode = 'harmonic_minor'   # Augmented 2nd = gothic, eerie, Dracula
            else:
                current_mode = 'phrygian_dominant' # ♭2 + M3 = alien, extreme, snake-charmer
            chord_type = "fear_open"  # Root + P5 + 6th (non-tertian, avoids happy/sad territory)

        self.state_update.emit(current_mode, chord_type, float(self.current_bpm))
        pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)

        melody_pool = pool

        # ── 3. TEMPO (MACRO TARGET, MICRO SMOOTHING SPEED) ──
        # MACRO arousal sets the target BPM (slow drift: 60–140 BPM).
        target_bpm = 100 + (macro_a * 40)

        # Apply spike tempo modifier (graduated by spike_intensity)
        if self.active_spike_profile and spike_intensity > 0:
            # Graduated: scale tempo multiplier by spike intensity
            tempo_mult = self.active_spike_profile['tempo_mult']
            # Blend toward profile tempo based on intensity (subtle at low, full at high)
            blended_mult = 1.0 + (tempo_mult - 1.0) * min(1.0, spike_intensity)
            target_bpm *= blended_mult

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
        # Neutral needs enough presence to feel emotionally engaged (matched closer to sad)
        if emotion_cat == 'neutral':
            velocity = max(60, velocity)

        # Apply spike velocity modifier (graduated by spike_intensity)
        if self.active_spike_profile and spike_intensity > 0:
            # Graduated scaling: 0-30% → 30% effect, 30-60% → 60% effect, 60-100% → full effect
            if spike_intensity < 0.3:
                effect_scale = 0.3
            elif spike_intensity < 0.6:
                effect_scale = 0.6
            else:
                effect_scale = 1.0
            vel_shift = int(self.active_spike_profile['vel_shift'] * effect_scale)
            velocity = velocity + vel_shift
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

        # ANXIETY spike (happy→fear): rhythmic density doubling for nervous energy
        if self.active_spike_profile and self.active_spike_profile.get('name') == 'ANXIETY' and spike_intensity > 0.3:
            # Jump to the next-faster rhythm category
            if chosen_ratios == [1.0] or chosen_ratios == [0.5, 0.5]:
                chosen_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
            elif len(chosen_ratios) == 3:
                chosen_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]

        # Sad & Neutral micro-timing humanization: ±5-12% timing jitter on note durations
        # Breaks the mechanical grid feel at slower tempos
        if emotion_cat in ('sad', 'neutral') and len(chosen_ratios) > 1:
            jittered = []
            for i, r in enumerate(chosen_ratios):
                jitter = random.uniform(-0.12, 0.12) * r
                jittered.append(r + jitter)
            # Normalize to preserve total duration
            total = sum(jittered)
            chosen_ratios = [r / total for r in jittered]

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
        # Neutral: AGGRESSIVE chord movement — never hold more than 2 steps
        if emotion_cat == 'neutral':
            harmonic_rhythm = min(harmonic_rhythm, 2)
        # Fear: keep chords moving at a steady moderate pace
        if emotion_cat == 'fear':
            harmonic_rhythm = min(harmonic_rhythm, 2)

        # Advance chord via Markov dice roll
        if self.current_dominant_idx == -1 or self.emotion_streak == 0:
            # On start or emotion shift, force a safe root
            self.current_chord_degree = 0
            self.current_dominant_idx = dominant_idx
        elif self.emotion_streak % harmonic_rhythm == 0:
            # Time to change the chord — look up the transition matrix
            # Neutral uses mode-specific matrices for distinct character
            if emotion_cat == 'neutral':
                matrix_key = 'neutral_dorian' if current_mode == 'dorian' else 'neutral_mixolydian'
            else:
                matrix_key = emotion_cat
            matrix = CHORD_TRANSITIONS.get(matrix_key, {})
            if self.current_chord_degree in matrix:
                options = matrix[self.current_chord_degree]['options']
                weights = matrix[self.current_chord_degree]['weights']
                self.current_chord_degree = random.choices(options, weights=weights, k=1)[0]
            else:
                self.current_chord_degree = 0  # Fallback

        # Fear near 100%: enhanced chord movement to prevent stagnation
        # ~30% of the time when intensity > 0.90, force a move to iv, bVI, or bvii
        confidence = float(p[int(np.argmax(p))])
        if emotion_cat == 'fear' and confidence > 0.90:
            if random.random() < 0.30:
                # Break the i↔bII oscillation with doom descent or tension pull
                self.current_chord_degree = random.choice([3, 5, 6])  # iv, bVI, bvii

        chord_root_idx = 14 + self.current_chord_degree
        
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "sus4":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 3], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        elif chord_type == "fear_open":
            # Non-tertian: Root + P5 + 6th degree (skips 3rd)
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
        elif chord_type == "cinematic_open":
            # Cinematic modal: triad + mode signature note for distinct character
            # Dorian: minor add6 (m3 + M6) — spiritual, noble (Force Theme)
            # Mixolydian: major add♭7 (M3 + ♭7) — prehistoric power (Jurassic Park)
            if current_mode == 'dorian':
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
            else:  # mixolydian
                # ♭7 as color note — exposes the Mixolydian signature directly
                flat7 = pool[chord_root_idx + 6] if (chord_root_idx + 6) < len(pool) else min(127, pool[chord_root_idx] + 10)
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], flat7]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # Harmonic Minor Alteration: Tension chord (Major V) for Sad emotion
        if emotion_cat == 'sad' and self.current_chord_degree == 4 and chord_type == "triad":
            chord_notes[1] += 1  # Raise minor 3rd to Major 3rd

        # Apply spike chord coloring (graduated by spike_intensity)
        if self.active_spike_profile and spike_intensity > 0:
            # EPIC MODAL override: force heroic i→bVI→bVII cycle before coloring
            if self.active_spike_profile.get('chord_color') == 'epic_modal' and spike_intensity > 0.6:
                epic_sequence = [0, 5, 6]  # i → bVI → bVII heroic loop
                self.current_chord_degree = epic_sequence[self.emotion_streak % 3]
                chord_root_idx = 14 + self.current_chord_degree
                # Rebuild chord from the forced epic degree
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
            # Apply chord coloring at any spike intensity (subtle at low, full at high)
            chord_notes = apply_spike_chord_color(chord_notes, self.active_spike_profile['chord_color'])

        # Spike rest probability override for melody (graduated)
        if self.active_spike_profile and spike_intensity > 0.3:
            spike_rest_prob = self.active_spike_profile['rest_prob'] * min(1.0, spike_intensity / 0.6)
        else:
            spike_rest_prob = None

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
            elif emotion_cat == 'neutral':
                chord_vel = min(70, max(35, velocity - 10 + micro_chord_boost))  # C418 delicate touch — soft, contemplative
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
                
                # Simple subtle arpeggio (~10%): bass note first, then each voice enters in quick succession
                if random.random() < 0.10:
                    roll_voices = [root, fifth, third]
                    roll_gap_sec = 0.03  # ~30ms between each note entry
                    for vi, n in enumerate(roll_voices):
                        vel_taper = max(30, chord_vel - (vi * 3))
                        t_on = chunk_start_time + (vi * roll_gap_sec)
                        self.note_queue.append((t_on, 0, int(n), vel_taper, 1.0, True, eeg_timestamp + (vi * roll_gap_sec)))
                    
                    # All notes off together at end
                    for n in roll_voices:
                        self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0, 0, False, eeg_timestamp + 0.99))
                else:
                    # Standard full block chord
                    for n in [root, fifth, third]:
                        self.note_queue.append((chunk_start_time, 0, n, chord_vel, 1.0, True, eeg_timestamp))
                        self.note_queue.append((chunk_start_time + 0.99, 0, n, 0, 0, False, eeg_timestamp + 0.99))

            elif emotion_cat == 'neutral':
                # C418 Minecraft voicing (Sweden / Danny):
                # 1. Soft bass note in low register (C2-C3 range), played gently
                # 2. Upper voices (3rd, 5th) arpeggiated gently in mid register
                # 3. Open spacing — big gap between bass and upper voices
                # 4. All notes sustain together until end of step
                
                root  = int(chord_notes[0])
                third = int(chord_notes[1])
                fifth = int(chord_notes[2])
                
                # Bass: push root down to C2-C3 range (36-48)
                bass = root
                while bass > 48:
                    bass -= 12
                while bass < 36:
                    bass += 12
                
                # Upper voices: keep in warm mid register (C4-C5 range, 60-72)
                while third < 60:
                    third += 12
                while third > 72:
                    third -= 12
                while fifth < 60:
                    fifth += 12
                while fifth > 72:
                    fifth -= 12
                
                # Bass velocity is softer than upper voices (C418 signature)
                bass_vel = max(25, chord_vel - 12)
                upper_vel = chord_vel
                
                if random.random() < 0.10:
                    # Arpeggio gaps (rare spice ~10%): bass first, then 3rd, then 5th
                    arp_gap_sec = 0.08  # ~80ms gap
                    
                    # Bass note (first, soft)
                    self.note_queue.append((chunk_start_time, 0, int(bass), bass_vel, 1.0, True, eeg_timestamp))
                    # Third (after gap)
                    self.note_queue.append((chunk_start_time + arp_gap_sec, 0, int(third), upper_vel, 1.0 - arp_gap_sec, True, eeg_timestamp + arp_gap_sec))
                    # Fifth (after another gap)
                    self.note_queue.append((chunk_start_time + arp_gap_sec * 2, 0, int(fifth), max(25, upper_vel - 3), 1.0 - arp_gap_sec * 2, True, eeg_timestamp + arp_gap_sec * 2))
                    
                    # All notes off together at end of step
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(bass), 0, 0, False, eeg_timestamp + 0.99))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(third), 0, 0, False, eeg_timestamp + 0.99))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(fifth), 0, 0, False, eeg_timestamp + 0.99))
                else:
                    # Full block chord (standard ~90%)
                    self.note_queue.append((chunk_start_time, 0, int(bass), bass_vel, 1.0, True, eeg_timestamp))
                    self.note_queue.append((chunk_start_time, 0, int(third), upper_vel, 1.0, True, eeg_timestamp))
                    self.note_queue.append((chunk_start_time, 0, int(fifth), max(25, upper_vel - 3), 1.0, True, eeg_timestamp))
                    
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(bass), 0, 0, False, eeg_timestamp + 0.99))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(third), 0, 0, False, eeg_timestamp + 0.99))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(fifth), 0, 0, False, eeg_timestamp + 0.99))

            elif emotion_cat == 'fear':
                # Fear: heavy non-tertian voicing (Root + P5 + 6th)
                fear_chord = list(chord_notes)
                while fear_chord[0] > 48:  # Root in deep bass (below C3)
                    fear_chord[0] -= 12
                fear_chord[1] = fear_chord[1] - 12 if fear_chord[1] > 69 else fear_chord[1]
                fear_chord[2] = fear_chord[2] - 12 if fear_chord[2] > 69 else fear_chord[2]
                vel = min(110, max(50, velocity + micro_chord_boost))

                for n in fear_chord:
                    self.note_queue.append((chunk_start_time, 0, int(n), vel, 1.0, True, eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99, 0, int(n), 0, 0, False, eeg_timestamp + 0.99))

                # RARE dissonance: ~5% chance of a quiet tritone or minor 2nd
                if random.random() < 0.05:
                    dissonant_note = min(127, int(fear_chord[0]) + random.choice([1, 6]))
                    self.note_queue.append((chunk_start_time, 0, dissonant_note, max(10, vel // 4), 0.5, True, eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.5, 0, dissonant_note, 0, 0, False, eeg_timestamp + 0.5))


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

                # Harmonic adherence
                chord_adherence_prob = 0.7 + (macro_v * 0.2) + (micro_v * 0.1)
                chord_adherence_prob = max(0.15, min(0.95, chord_adherence_prob))
                if emotion_cat in ('happy', 'sad'):
                    chord_adherence_prob = max(0.80, chord_adherence_prob)
                if emotion_cat == 'neutral':
                    chord_adherence_prob = max(0.80, chord_adherence_prob)
                if emotion_cat == 'fear':
                    chord_adherence_prob = max(0.75, chord_adherence_prob)

                if self.force_snap or random.random() < chord_adherence_prob:
                    # Ground the melody to chord tones
                    safe_snap_notes = chord_notes[:3]
                    if emotion_cat == 'neutral' and len(safe_snap_notes) > 1 and random.random() < 0.4:
                        safe_snap_notes = safe_snap_notes[1:]  # skip root occasionally
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    # Fear: snap to lower octave for darker register
                    if emotion_cat == 'fear':
                        target_note = random.choice(safe_snap_notes) + random.choice([0, 12])
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

                # Fear: lower register with descending bias
                if emotion_cat == 'fear':
                    self.melody_idx = max(14, min(len(active_pool) - 1, self.melody_idx))
                    if random.random() < 0.30 and self.melody_idx > 16:
                        self.melody_idx -= 1
                    note = int(active_pool[self.melody_idx])

                # Micro valence expression
                if micro_v > 0.4 and random.random() < 0.35:
                    note += 12

                # Apply spike melody register shift (graduated)
                if self.active_spike_profile and spike_intensity > 0.3:
                    note += int(self.active_spike_profile['melody_register'] * min(1.0, spike_intensity))

                # Cap super high notes
                while note > 84:
                    note -= 12
                note = max(0, min(127, note))

                # Universal Dissonance Guard
                if emotion_cat in ('happy', 'sad', 'neutral', 'fear'):
                    note = resolve_dissonance(note, chord_notes)

                # Determine rest probability
                if spike_rest_prob is not None:
                    is_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.10:
                    is_rest = True  # Occasional breaths, not constant gaps
                elif emotion_cat == 'fear' and random.random() < 0.15:
                    is_rest = True
                else:
                    is_rest = False

                if not is_rest:
                    # Fear melody: slightly lower velocity for brooding feel
                    if emotion_cat == 'fear':
                        mel_vel = max(45, min(85, velocity - 5))
                    else:
                        mel_vel = int(velocity)

                    t_on  = chunk_start_time + time_offset
                    t_off = t_on + (dur_sec * 0.99)
                    e_ts  = eeg_timestamp + time_offset
                    self.note_queue.append((t_on,  1, int(note), mel_vel, dur_sec, True,  e_ts))
                    self.note_queue.append((t_off, 1, int(note), 0,             0,       False, e_ts + dur_sec * 0.99))
                    melody_notes_for_trill.append((int(note), dur_sec))

                time_offset += dur_sec

            # ── Anti-trill guard (applies to all emotions) ──
            if len(melody_notes_for_trill) >= 4:
                if detect_trill(melody_notes_for_trill):
                    self.consecutive_trill_count += 1
                else:
                    self.consecutive_trill_count = 0
                if self.consecutive_trill_count >= 1:
                    self.force_snap = True
                    self.consecutive_trill_count = 0

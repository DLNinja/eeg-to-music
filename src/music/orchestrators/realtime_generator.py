from src.ui import theme_config
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
from src.music.rule_based import music_theory
import os
import time
import mido
import fluidsynth
import random
import numpy as np
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from ..rule_based.music_theory import (
    CHORD_TRANSITIONS, SPIKE_PROFILES,
    get_mode_pool, select_key_offset
)
from ..rule_based.spike_state import SpikeState
from ..rule_based.harmony import (
    select_mode,
    compute_harmonic_rhythm, advance_chord_degree, build_chord_notes,
    apply_harmonic_minor_chord, apply_chord_spike_overrides
)
from ..utility.dynamics import apply_anxiety_rhythm_doubling, apply_humanization_jitter
from ..utility.melody import (
    detect_trill, resolve_dissonance, pool_idx_nearest,
    apply_harmonic_minor_to_pool, compute_chord_adherence
)
from ..emotion_tracker.emotion_tracker import EmotionTracker
from ..markov.markov_engine import MarkovEngine
from src.eeg_pipeline.eeg_texturing_engine import EEGTexturingEngine


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
        self._initial_base_key_offset = base_key_offset
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
        self.motif_buffer = []  # last generated 3-4 note pattern (matches MIDI generator)
        
        # Active notes to turn off cleanly
        self.active_chord_notes = []
        self.active_melody_note = None
        self.current_chord_degree = 0      # Track which scale degree is sounding
        self.current_dominant_idx = -1     # Track which emotion type is active for the chord

        
        self.tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
        self.spike_state = SpikeState()
        self.markov_engine = MarkovEngine()
        self.eeg_texturing_engine = EEGTexturingEngine()
        self.prev_melody_interval = 0
        self.synth = None

        # Latest band powers (updated each second, used by EEG texturing)
        self.current_band_powers = {}

        # Phase 3A: Anti-trill tracker
        self.consecutive_trill_count = 0
        # Phase 4: Spike profile tracking
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

    def update_emotion(self, probs, timestamp, band_powers=None):
        """Receive a new 1-second classification result.

        Drives EEGTexturingEngine.process() so Z-score baseline, per-band scalars,
        and trend history are all updated in one call on the main thread.
        """
        with QMutexLocker(self.mutex):
            # Update the continuous V-A tracker
            dominant_idx = int(np.argmax(probs))
            confidence = float(probs[dominant_idx])
            self.tracker.update_from_discrete(dominant_idx, confidence)

            if band_powers is not None:
                self.current_band_powers = band_powers
                self.eeg_texturing_engine.process(band_powers)

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
            self.base_key_offset = self._initial_base_key_offset
            self.dynamic_key_set = (self.base_key_offset != 0)
            self.update_queue.clear()
            self.note_queue.clear()
            self.emotion_streak = 0
            self.prev_dominant_idx = -1
            self.current_chord_degree = 0
            self.current_dominant_idx = -1
            self.consecutive_trill_count = 0
            self.spike_state = SpikeState()
            self.fear_sustain_active = False
            self.fear_submode = 'ambiguity'
            self.fear_tick_counter = 0
            self.fear_ramp_velocity = 35
            self.neutral_locked_mode = None
            self.motif_buffer = []
            self._all_notes_off()

    def stop(self):
        self.is_running = False
        self.is_playing = False
        self.playback_start_time = 0.0
        self._all_notes_off()
        self.wait()

    def _apply_eeg_texturing(self, emotion_label):
        # Apply CC mapping to EEGTexturingEngine.
        # Copies band_z_scalars, band_trends and asymmetry under the mutex 
        # then calls apply_cc() outside the lock.
        
        if not self.synth:
            return
        with QMutexLocker(self.mutex):
            band_z_scalars = dict(self.eeg_texturing_engine.band_z_scalars)
            band_trends    = dict(self.eeg_texturing_engine.band_trends)
            asymmetry      = float(self.current_band_powers.get('asymmetry', 0.0))

        self.eeg_texturing_engine.apply_cc(
            emotion_label  = emotion_label,
            band_z_scalars = band_z_scalars,
            band_trends    = band_trends,
            asymmetry      = asymmetry,
            synth          = self.synth,
        )

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
        # Sets the volume (CC 7) for all active channels (0 and 1)
        if self.synth:
            self.synth.cc(0, 7, value)
            self.synth.cc(1, 7, value)

    def run(self):
        self.is_running = True
        # Track when the NEXT 1-second of music should start playing
        # 2.2s initial delay to provide:
        # -> 1s for the first EEG window
        # -> 1s safety buffer
        # -> 0.2s for processing delay
        playback_clock = time.time() + 1.5
        
        while self.is_running:
            if not self.is_playing:
                time.sleep(0.01)
                playback_clock = time.time() + 1.5
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
                    # Schedule current chunk relative to internal playback clock
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
                for note_event in self.note_queue:
                    if len(note_event) == 7:
                        sched_time, ch, pitch, vel, duration, is_on, emit_ts = note_event
                    else:
                        sched_time, ch, pitch, vel, duration, is_on = note_event
                        emit_ts = sched_time - self.playback_start_time

                    if now >= (sched_time - 0.002):
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

            time.sleep(0.002)


    def _process_emotion_state(self, p):
        state = self.tracker.get_state()
        macro_v = state['macro_v']   
        macro_a = state['macro_a']   
        micro_v = state['micro_v']   
        micro_a = state['micro_a']   
        is_spike = state['is_spike']
        spike_intensity = state['spike_intensity']
        spike_label = state['spike_label']
        macro_label = state['macro_label']

        if not self.dynamic_key_set:
            self.dynamic_key_set = True
            dom_idx = int(np.argmax(p))
            EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
            self.base_key_offset = select_key_offset(dom_idx)
            self.fundamental_bass_root = 36 + self.base_key_offset
            print(f"[Synthesizer] Dynamic Key Set! Emotion: {EMOTION_LABELS[dom_idx]}, Offset: +{self.base_key_offset}")

        dominant_idx = int(np.argmax(p))
        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
            self.motif_buffer = []
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        if is_spike:
            self.motif_buffer = []
            self.prev_melody_intervals = [0, 0, 0]
            self.force_snap = True
            
        self.prev_dominant_idx = dominant_idx

        sorted_emotions = np.argsort(p)
        secondary_idx = sorted_emotions[-2]
        LABEL_NAMES = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}
        secondary_label = LABEL_NAMES.get(secondary_idx, 'neutral')

        if macro_v > 0.0 and macro_a > 0.0:
            emotion_cat = 'happy'
        elif macro_v < 0.0 and macro_a < 0.0:
            emotion_cat = 'sad'
        elif macro_v < 0.0 and macro_a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        self.spike_state.update(is_spike, macro_label, spike_label, spike_intensity)
        
        return emotion_cat, state, spike_intensity, macro_v, macro_a, micro_v, micro_a, dominant_idx, macro_label, is_spike

    def _compute_dynamics(self, emotion_cat, macro_a, micro_a, spike_intensity, macro_label):
        target_bpm = 100 + (macro_a * 40)

        if self.spike_state.is_active:
            tempo_mult = self.spike_state.active_profile['tempo_mult']
            blended_mult = 1.0 + (tempo_mult - 1.0) * min(1.0, spike_intensity)
            target_bpm *= blended_mult

        alpha = 0.3 + 0.5 * min(1.0, abs(micro_a))  
        self.current_bpm = (1.0 - alpha) * self.current_bpm + alpha * target_bpm
        sec_per_beat = 60.0 / self.current_bpm

        velocity = int(70 + (macro_a * 30) + (abs(micro_a) * 20))
        velocity = max(30, min(110, velocity))

        if macro_label == 'sad':
            velocity = max(50, min(110, velocity + 15))
        if emotion_cat == 'neutral':
            velocity = max(60, velocity)

        if self.spike_state.is_active:
            if spike_intensity < 0.3:
                effect_scale = 0.3
            elif spike_intensity < 0.6:
                effect_scale = 0.6
            else:
                effect_scale = 1.0
            vel_shift = int(self.spike_state.active_profile['vel_shift'] * effect_scale)
            velocity = velocity + vel_shift
            velocity = max(30, min(110, velocity))

        micro_density_bias = min(1.0, abs(micro_a))
        if macro_a > 0.6:
            chosen_ratios = (
                [0.125, 0.125, 0.25, 0.5] if random.random() < (0.4 + 0.4 * micro_density_bias)
                else [0.25, 0.25, 0.25, 0.25]
            )
        elif macro_a > 0.0:
            chosen_ratios = (
                [0.333, 0.333, 0.333] if random.random() < (0.3 + 0.4 * micro_density_bias)
                else [0.5, 0.25, 0.25]
            )
        else:
            chosen_ratios = (
                [0.5, 0.5] if random.random() < (0.2 + 0.5 * micro_density_bias)
                else [1.0]
            )

        chosen_ratios = apply_anxiety_rhythm_doubling(
            chosen_ratios, self.spike_state.name, spike_intensity
        )
        chosen_ratios = apply_humanization_jitter(chosen_ratios, emotion_cat)
        
        return velocity, chosen_ratios, sec_per_beat

    def _generate_piano_accompaniment(self, chunk_start_time, eeg_timestamp, emotion_cat, chord_notes, chord_vel, velocity, micro_chord_boost, spike_intensity, LEGATO_EARLY, LEGATO_LATE):
        if emotion_cat == 'happy':
            chord_notes = [n - 12 if n > 72 else n for n in chord_notes]
            for n in chord_notes:
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(n), chord_vel, 1.0, True,  eeg_timestamp))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(n), 0, 0, False, eeg_timestamp + 0.99))

        elif emotion_cat == 'sad':
            root  = int(chord_notes[0])
            third = int(chord_notes[1]) + 12
            fifth = int(chord_notes[2]) - 12
            
            fifth = fifth if fifth >= 0 else fifth + 12
            third = third if third <= 127 else third - 12
            
            if random.random() < 0.10:
                roll_voices = [root, fifth, third]
                roll_gap_sec = 0.03  
                for vi, n in enumerate(roll_voices):
                    vel_taper = max(30, chord_vel - (vi * 3))
                    t_on = chunk_start_time - LEGATO_EARLY + (vi * roll_gap_sec)
                    self.note_queue.append((t_on, 0, int(n), vel_taper, 1.0, True, eeg_timestamp + (vi * roll_gap_sec)))
                
                for n in roll_voices:
                    self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(n), 0, 0, False, eeg_timestamp + 0.99))
            else:
                for n in [root, fifth, third]:
                    self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, n, chord_vel, 1.0, True, eeg_timestamp))
                    self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, n, 0, 0, False, eeg_timestamp + 0.99))

        elif emotion_cat == 'neutral':
            root  = int(chord_notes[0])
            third = int(chord_notes[1])
            fifth = int(chord_notes[2])
            
            bass = root
            while bass > 48:
                bass -= 12
            while bass < 36:
                bass += 12
            
            while third < 60:
                third += 12
            while third > 72:
                third -= 12
            while fifth < 60:
                fifth += 12
            while fifth > 72:
                fifth -= 12
            
            bass_vel = max(25, chord_vel - 12)
            upper_vel = chord_vel
            
            if random.random() < 0.10:
                arp_gap_sec = 0.08  
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(bass), bass_vel, 1.0, True, eeg_timestamp))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY + arp_gap_sec, 0, int(third), upper_vel, 1.0 - arp_gap_sec, True, eeg_timestamp + arp_gap_sec))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY + arp_gap_sec * 2, 0, int(fifth), max(25, upper_vel - 3), 1.0 - arp_gap_sec * 2, True, eeg_timestamp + arp_gap_sec * 2))
                
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(bass),  0, 0, False, eeg_timestamp + 0.99))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(third), 0, 0, False, eeg_timestamp + 0.99))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(fifth), 0, 0, False, eeg_timestamp + 0.99))
            else:
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(bass),  bass_vel, 1.0, True, eeg_timestamp))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(third), upper_vel, 1.0, True, eeg_timestamp))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(fifth), max(25, upper_vel - 3), 1.0, True, eeg_timestamp))
                
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(bass),  0, 0, False, eeg_timestamp + 0.99))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(third), 0, 0, False, eeg_timestamp + 0.99))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(fifth), 0, 0, False, eeg_timestamp + 0.99))

        elif emotion_cat == 'fear':
            fear_chord = list(chord_notes)
            while fear_chord[0] > 48:
                fear_chord[0] -= 12
            fear_chord[1] = fear_chord[1] - 12 if fear_chord[1] > 69 else fear_chord[1]
            fear_chord[2] = fear_chord[2] - 12 if fear_chord[2] > 69 else fear_chord[2]
            vel = min(110, max(50, velocity + micro_chord_boost))

            for n in fear_chord:
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(n), vel, 1.0, True, eeg_timestamp))
                self.note_queue.append((chunk_start_time + 0.99 + LEGATO_LATE, 0, int(n), 0, 0, False, eeg_timestamp + 0.99))

            if random.random() < 0.05:
                dissonant_note = min(127, int(fear_chord[0]) + random.choice([1, 6]))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, dissonant_note, max(10, vel // 4), 0.5, True, eeg_timestamp))
                self.note_queue.append((chunk_start_time - LEGATO_EARLY + 0.5, 0, dissonant_note, 0, 0, False, eeg_timestamp + 0.5))

        if self.spike_state.name == 'UNEASE' and spike_intensity > 0.5:
            if chord_notes[0] > 36:
                chord_notes[0] = chord_notes[0] - 12
            ghost_note = min(127, chord_notes[0] + 1)
            ghost_vel = max(15, int(chord_vel * 0.25))
            self.note_queue.append((chunk_start_time - LEGATO_EARLY, 0, int(ghost_note), ghost_vel, 0.5, True, eeg_timestamp))
            self.note_queue.append((chunk_start_time - LEGATO_EARLY + 0.5, 0, int(ghost_note), 0, 0, False, eeg_timestamp + 0.5))

    def _generate_melody_phrase(self, active_pool, emotion_cat, macro_v, micro_v, spike_intensity, chosen_ratios, chord_notes, sec_per_beat, spike_rest_prob, is_neutral):
        melody_notes_and_durations_sec = []
        melody_notes_for_trill = []  
        time_offset = 0.0
        
        use_motif = len(self.motif_buffer) > 0 and random.random() < 0.40

        if use_motif:
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, r_frac in self.motif_buffer:
                dur_sec = float(r_frac) * sec_per_beat * 2
                if time_offset + dur_sec > 1.0:
                    dur_sec = 1.0 - time_offset
                if dur_sec < 0.01:
                    break

                new_deg = self.melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note = int(active_pool[new_deg])

                if self.spike_state.is_active and spike_intensity > 0.3:
                    note += self.spike_state.get_melody_register_shift()
                while note > 84:
                    note -= 12
                note = max(0, min(127, note))

                if emotion_cat in ('happy', 'sad', 'neutral'):
                    note = resolve_dissonance(note, chord_notes)

                melody_notes_and_durations_sec.append((note, dur_sec, time_offset))
                melody_notes_for_trill.append((int(note), dur_sec))
                time_offset += dur_sec
        else:
            chosen_contour = []
            prev_intervals = [0, 0, 0]
            for _ in range(len(chosen_ratios)):
                next_interval = self.markov_engine.query_next_interval(emotion_cat, prev_intervals)
                chosen_contour.append(next_interval)
                prev_intervals.pop(0)
                prev_intervals.append(next_interval)

            new_motif = []
            chord_adherence_prob, force_snap_consumed = compute_chord_adherence(
                emotion_cat, macro_v, micro_v,
                self.spike_state.active_profile, spike_intensity, self.force_snap
            )
            if force_snap_consumed:
                self.force_snap = False

            for i, r_frac in enumerate(chosen_ratios):
                dur_sec = float(r_frac) * sec_per_beat * 2
                if time_offset + dur_sec > 1.0:
                    dur_sec = 1.0 - time_offset
                if dur_sec < 0.01:
                    break

                if i < len(chosen_contour):
                    self.melody_idx += chosen_contour[i]
                    self.melody_idx = max(21, min(len(active_pool) - 1, self.melody_idx))

                if emotion_cat == 'fear':
                    self.melody_idx = max(14, min(len(active_pool) - 1, self.melody_idx))
                    if random.random() < 0.30 and self.melody_idx > 16:
                        self.melody_idx -= 1

                if random.random() < chord_adherence_prob:
                    safe_snap_notes = chord_notes[:3]
                    if emotion_cat == 'neutral' and len(safe_snap_notes) > 1 and random.random() < 0.4:
                        safe_snap_notes = safe_snap_notes[1:]
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    if emotion_cat == 'fear':
                        target_note = random.choice(safe_snap_notes) + random.choice([0, 12])
                    self.melody_idx = pool_idx_nearest(target_note, active_pool)
                    self.melody_idx = max(21, min(len(active_pool) - 1, self.melody_idx))

                note = int(active_pool[self.melody_idx])

                if micro_v > 0.4 and random.random() < 0.35:
                    note += 12  

                if self.spike_state.is_active and spike_intensity > 0.3:
                    note += self.spike_state.get_melody_register_shift()

                while note > 84:
                    note -= 12
                note = max(0, min(127, note))

                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, r_frac))

                if spike_rest_prob is not None:
                    is_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.10:
                    is_rest = True
                elif emotion_cat == 'fear' and random.random() < 0.15:
                    is_rest = True
                else:
                    is_rest = False

                if not is_rest:
                    # Universal Dissonance Guard
                    if emotion_cat in ('happy', 'sad', 'neutral', 'fear'):
                        note = resolve_dissonance(note, chord_notes)
                    melody_notes_and_durations_sec.append((note, dur_sec, time_offset))
                    melody_notes_for_trill.append((int(note), dur_sec))

                time_offset += dur_sec

            self.motif_buffer = new_motif

        if len(melody_notes_for_trill) >= 4:
            if detect_trill(melody_notes_for_trill):
                self.consecutive_trill_count += 1
            else:
                self.consecutive_trill_count = 0
            if self.consecutive_trill_count >= 1:
                self.force_snap = True
                self.consecutive_trill_count = 0

        return melody_notes_and_durations_sec

    def _append_melody_events(self, chunk_start_time, eeg_timestamp, melody_notes_and_durations_sec, emotion_cat, velocity):
        for note, dur_sec, time_offset in melody_notes_and_durations_sec:
            mel_vel = max(45, min(85, velocity - 5)) if emotion_cat == 'fear' else int(velocity)
            t_on  = chunk_start_time + time_offset
            t_off = t_on + (dur_sec * 0.99)
            e_ts  = eeg_timestamp + time_offset
            self.note_queue.append((t_on,  1, int(note), mel_vel, dur_sec, True,  e_ts))
            self.note_queue.append((t_off, 1, int(note), 0,       0,       False, e_ts + dur_sec * 0.99))

    def _generate_and_schedule_1s_chunk(self, p, chunk_start_time, eeg_timestamp):
        if not self.synth: return

        emotion_cat, state, spike_intensity, macro_v, macro_a, micro_v, micro_a, dominant_idx, macro_label, is_spike = self._process_emotion_state(p)
        is_neutral = (emotion_cat == 'neutral')

        current_mode, chord_type, self.neutral_locked_mode = select_mode(
            emotion_cat, macro_v, p, self.emotion_streak, self.neutral_locked_mode
        )

        self.state_update.emit(current_mode, chord_type, float(self.current_bpm))
        pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)

        velocity, chosen_ratios, sec_per_beat = self._compute_dynamics(
            emotion_cat, macro_a, micro_a, spike_intensity, macro_label
        )

        # EEG Texturing: map band power Z-scores to MIDI CCs
        self._apply_eeg_texturing(emotion_cat)

        harmonic_rhythm = compute_harmonic_rhythm(macro_a, emotion_cat, self.emotion_streak)

        self.current_chord_degree = advance_chord_degree(
            emotion_cat, current_mode, self.emotion_streak,
            harmonic_rhythm, self.current_chord_degree,
            is_first_step=(self.current_dominant_idx == -1)
        )
        if self.current_dominant_idx == -1:
            self.current_dominant_idx = dominant_idx

        confidence = float(p[int(np.argmax(p))])
        if emotion_cat == 'fear' and confidence > 0.90:
            if random.random() < 0.30:
                self.current_chord_degree = random.choice([3, 5, 6])  

        chord_root_idx = 14 + self.current_chord_degree
        chord_notes = build_chord_notes(pool, chord_root_idx, chord_type, current_mode)
        chord_notes = apply_harmonic_minor_chord(chord_notes, emotion_cat, self.current_chord_degree, chord_type)

        chord_notes, self.current_chord_degree = apply_chord_spike_overrides(
            pool, 14, chord_notes,
            self.spike_state.get_chord_color(), self.spike_state.name, spike_intensity,
            self.emotion_streak, self.current_chord_degree
        )

        spike_rest_prob = self.spike_state.get_rest_probability()

        LEGATO_EARLY = 0.05
        LEGATO_LATE  = -0.06  

        with QMutexLocker(self.mutex):
            if emotion_cat == 'fear' and not self.fear_sustain_active:
                self.synth.cc(0, 64, 127)  
                self.fear_sustain_active = True
            elif emotion_cat != 'fear' and self.fear_sustain_active:
                self.synth.cc(0, 64, 0)    
                self.fear_sustain_active = False

            micro_chord_boost = 0
            if is_spike:
                micro_intensity = min(1.0, abs(micro_a))
                micro_chord_boost = int(micro_intensity * 40)

            if emotion_cat == 'sad':
                chord_vel = min(110, max(40, velocity + 5 + micro_chord_boost))
            elif emotion_cat == 'neutral':
                chord_vel = min(70, max(35, velocity - 10 + micro_chord_boost))  
            else:
                chord_vel = min(110, max(20, velocity - 10 + micro_chord_boost))

            self._generate_piano_accompaniment(
                chunk_start_time, eeg_timestamp, emotion_cat, chord_notes, chord_vel, 
                velocity, micro_chord_boost, spike_intensity, LEGATO_EARLY, LEGATO_LATE
            )

            active_pool = list(pool)
            active_pool = apply_harmonic_minor_to_pool(active_pool, emotion_cat, self.current_chord_degree)

            melody_notes_and_durations_sec = self._generate_melody_phrase(
                active_pool, emotion_cat, macro_v, micro_v, spike_intensity, 
                chosen_ratios, chord_notes, sec_per_beat, spike_rest_prob, is_neutral
            )

            self._append_melody_events(
                chunk_start_time, eeg_timestamp, melody_notes_and_durations_sec, 
                emotion_cat, velocity
            )

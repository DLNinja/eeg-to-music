import json
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np
from ..emotion_tracker.emotion_tracker import EmotionTracker
from ..markov.markov_engine import MarkovEngine

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


class OfflineMidiGenerator:
    def __init__(self, base_key_offset=0):
        self.base_key_offset = base_key_offset
        self.fundamental_bass_root = 36 + base_key_offset
        
        self.tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
        self.spike_state = SpikeState()
        self.markov_engine = MarkovEngine()
        
        self.current_bpm = None
        self.prev_dominant_idx = -1
        self.emotion_streak = 0
        self.melody_idx = 21 
        
        self.motif_buffer = []
        self.current_chord_degree = 0
        self.consecutive_trill_count = 0
        self.neutral_locked_mode = None

    def _process_emotion_state(self, p):
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx]
        self.tracker.update_from_discrete(dominant_idx, intensity)
        
        state = self.tracker.get_state()
        v = state['macro_v']
        a = state['macro_a']
        is_spike = state['is_spike']
        spike_intensity = state['spike_intensity']
        spike_label = state['spike_label']
        macro_label = state['macro_label']

        # secondary emotion for neutral mode selection
        sorted_emotions = np.argsort(p)
        secondary_idx = sorted_emotions[-2]
        LABEL_NAMES = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}
        secondary_label = LABEL_NAMES.get(secondary_idx, 'neutral')

        # MACRO MOOD streak 
        if dominant_idx == self.prev_dominant_idx:
            self.emotion_streak += 1
        else:
            self.emotion_streak = 0
        self.prev_dominant_idx = dominant_idx

        # SPIKE PROFILE MANAGEMENT
        self.spike_state.update(is_spike, macro_label, spike_label, spike_intensity)

        # MACRO V/A mapped to discrete emotional categories 
        if v > 0.0 and a > 0.0:
            emotion_cat = 'happy'
        elif v < 0.0 and a < 0.0:
            emotion_cat = 'sad'
        elif v < 0.0 and a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'
            
        return emotion_cat, v, a, intensity, state, spike_intensity, macro_label

    def _generate_piano_accompaniment(self, emotion_cat, chord_notes, chord_vel, velocity, ticks_per_step, chord_track):
        if emotion_cat == 'happy':
            chord_notes = [n - 12 if n > 72 else n for n in chord_notes]
            for n in chord_notes:
                chord_track.append(Message('note_on',  note=int(n), velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=int(chord_notes[0]), velocity=0, time=int(ticks_per_step)))
            for n in chord_notes[1:]:
                chord_track.append(Message('note_off', note=int(n), velocity=0, time=0))

        elif emotion_cat == 'sad':
            root  = int(chord_notes[0])
            third = int(chord_notes[1]) + 12
            fifth = int(chord_notes[2]) - 12
            
            fifth = fifth if fifth >= 0 else fifth + 12
            third = third if third <= 127 else third - 12

            if random.random() < 0.10:
                roll_voices = [root, fifth, third]
                roll_gap = int(ticks_per_step * 0.03)
                for vi, n in enumerate(roll_voices):
                    vel_taper = max(30, chord_vel - (vi * 3))
                    chord_track.append(Message('note_on', note=n, velocity=vel_taper, time=(roll_gap if vi > 0 else 0)))
                
                sustain_remaining = int(ticks_per_step - roll_gap * (len(roll_voices) - 1))
                chord_track.append(Message('note_off', note=roll_voices[0], velocity=0, time=sustain_remaining))
                for n in roll_voices[1:]:
                    chord_track.append(Message('note_off', note=n, velocity=0, time=0))
            else:
                for n in [root, fifth, third]:
                    chord_track.append(Message('note_on', note=n, velocity=chord_vel, time=0))
                chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))

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
                arp_gap = int(ticks_per_step * 0.08)
                
                chord_track.append(Message('note_on', note=bass, velocity=bass_vel, time=0))
                chord_track.append(Message('note_on', note=third, velocity=upper_vel, time=arp_gap))
                chord_track.append(Message('note_on', note=fifth, velocity=max(25, upper_vel - 3), time=arp_gap))
                
                sustain = int(ticks_per_step - arp_gap * 2)
                chord_track.append(Message('note_off', note=bass,  velocity=0, time=sustain))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
            else:
                chord_track.append(Message('note_on', note=bass, velocity=bass_vel, time=0))
                chord_track.append(Message('note_on', note=third, velocity=upper_vel, time=0))
                chord_track.append(Message('note_on', note=fifth, velocity=max(25, upper_vel - 3), time=0))
                
                chord_track.append(Message('note_off', note=bass,  velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))

        elif emotion_cat == 'fear':
            chord_track.append(Message('control_change', control=64, value=127, time=0))

            fear_chord = list(chord_notes)
            while fear_chord[0] > 48:
                fear_chord[0] -= 12
            fear_chord[1] = fear_chord[1] - 12 if fear_chord[1] > 69 else fear_chord[1]
            fear_chord[2] = fear_chord[2] - 12 if fear_chord[2] > 69 else fear_chord[2]
            vel = max(50, velocity)

            for n in fear_chord:
                chord_track.append(Message('note_on', note=int(n), velocity=vel, time=0))

            if random.random() < 0.05:
                dissonant_note = min(127, int(fear_chord[0]) + random.choice([1, 6]))
                chord_track.append(Message('note_on', note=dissonant_note, velocity=max(10, vel // 4), time=0))
                chord_track.append(Message('note_off', note=dissonant_note, velocity=0, time=int(ticks_per_step // 2)))
                chord_track.append(Message('note_off', note=int(fear_chord[0]), velocity=0, time=int(ticks_per_step - (ticks_per_step // 2))))
            else:
                chord_track.append(Message('note_off', note=int(fear_chord[0]), velocity=0, time=int(ticks_per_step)))
            for n in fear_chord[1:]:
                chord_track.append(Message('note_off', note=int(n), velocity=0, time=0))

            chord_track.append(Message('control_change', control=64, value=0, time=0))

    def _generate_melody_phrase(self, melody_pool, emotion_cat, v, state, spike_intensity, chosen_rhythm, chord_notes):
        melody_notes_and_durations = []
        active_pool = list(melody_pool)  
        
        # SAD humanization: apply the harmonic minor chord to the note pool
        active_pool = apply_harmonic_minor_to_pool(active_pool, emotion_cat, self.current_chord_degree)

        use_motif = len(self.motif_buffer) > 0 and random.random() < 0.40
        is_neutral = (emotion_cat == 'neutral')
        spike_rest_prob = self.spike_state.get_rest_probability()
        register_shift = 0

        if use_motif:
            # Use saved motif transposed 
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, duration in self.motif_buffer:
                new_deg = self.melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note    = int(active_pool[new_deg]) + register_shift

                register_offset = self.spike_state.get_melody_register_shift()
                if register_offset != 0:
                    note += register_offset

                # Cap super high notes (C6 is 84)
                while note > 84:
                    note -= 12
                    
                note = max(0, min(127, note))
                
                # Dissonance Guard (applies to all emotions but FEAR)
                if emotion_cat in ('happy', 'sad', 'neutral'):
                    note = resolve_dissonance(note, chord_notes)
                    
                melody_notes_and_durations.append((note, duration))
        else:
            # New musical phrase using VGMIDI Markov Chain
            chosen_contour = []
            prev_intervals = [0, 0, 0] 
            
            for _ in range(len(chosen_rhythm)):
                next_interval = self.markov_engine.query_next_interval(emotion_cat, prev_intervals)
                chosen_contour.append(next_interval)
                prev_intervals.pop(0)
                prev_intervals.append(next_interval)
                
            new_motif = []

            # MACRO VALENCE anchors to chord, MICRO VALENCE deviates
            chord_adherence_prob, _ = compute_chord_adherence(
                emotion_cat, v, state['micro_v'],
                self.spike_state.is_active, spike_intensity, force_snap=False
            )

            for i, duration in enumerate(chosen_rhythm):
                if i < len(chosen_contour):
                    self.melody_idx += chosen_contour[i]
                    self.melody_idx  = max(21, min(len(active_pool) - 1, self.melody_idx))

                # FEAR: lower register (C3-C5 range) with descending bias
                if emotion_cat == 'fear':
                    self.melody_idx = max(14, min(len(active_pool) - 1, self.melody_idx))
                    if random.random() < 0.30 and self.melody_idx > 16:
                        self.melody_idx -= 1

                # Snap to safe triad tones only
                if random.random() < chord_adherence_prob:
                    safe_snap_notes = chord_notes[:3]
                    if emotion_cat == 'neutral' and len(safe_snap_notes) > 1 and random.random() < 0.4:
                        safe_snap_notes = safe_snap_notes[1:]
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    if emotion_cat == 'fear':
                        target_note = random.choice(safe_snap_notes) + random.choice([0, 12])
                    self.melody_idx  = pool_idx_nearest(target_note, active_pool)
                    self.melody_idx  = max(21, min(len(active_pool) - 1, self.melody_idx))

                note = int(active_pool[self.melody_idx]) + register_shift

                # Micro valence expression
                if state['micro_v'] > 0.4 and random.random() < 0.35:
                    note += 12  

                # Spike melody register shift
                register_offset = self.spike_state.get_melody_register_shift()
                if register_offset != 0:
                    note += register_offset

                # Cap super high notes
                while note > 84:
                    note -= 12

                note = max(0, min(127, note))

                # Record motif offsets
                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, duration))

                if spike_rest_prob is not None:
                    should_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.10:
                    should_rest = True  
                elif emotion_cat == 'fear' and random.random() < 0.15:
                    should_rest = True  
                else:
                    should_rest = False

                if should_rest:
                    melody_notes_and_durations.append((None, duration))
                else:
                    # Universal Dissonance Guard
                    if emotion_cat in ('happy', 'sad', 'neutral', 'fear'):
                        note = resolve_dissonance(note, chord_notes)
                    melody_notes_and_durations.append((note, duration))

            # Save the new phrase to the motif buffer
            self.motif_buffer[:] = new_motif

        # Anti-trill guard
        if detect_trill(melody_notes_and_durations):
            self.consecutive_trill_count += 1
        else:
            self.consecutive_trill_count = 0

        if self.consecutive_trill_count >= 1:
            melody_notes_and_durations = []
            for i_r, duration in enumerate(chosen_rhythm):
                ct_idx = i_r % len(chord_notes)
                note = int(chord_notes[ct_idx]) + 12
                while note > 84:
                    note -= 12
                note = max(0, min(127, note))
                melody_notes_and_durations.append((note, duration))
            self.consecutive_trill_count = 0
            
        return melody_notes_and_durations

    def _append_melody_events(self, melody_notes_and_durations, emotion_cat, velocity, melody_track):
        for note, duration in melody_notes_and_durations:
            if note is None:
                melody_track.append(Message('note_off', note=0, velocity=0, time=int(duration)))
            else:
                # Fear melody: slightly lower velocity for building tension
                if emotion_cat == 'fear':
                    mel_vel = max(45, min(85, velocity - 5))
                else:
                    mel_vel = velocity
                melody_track.append(Message('note_on',  note=int(note), velocity=int(mel_vel), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0,              time=int(duration)))

    def generate(self, emotions_array, filename="eeg_music.mid"):
        mid = MidiFile()

        # Left hand CHORDS + right hand MELODY
        chord_track = MidiTrack()
        melody_track = MidiTrack()
        mid.tracks.extend([chord_track, melody_track])
        
        # Acoustic Grand Piano (Program 0)
        chord_track.append(Message('program_change', program=0, time=0))
        melody_track.append(Message('program_change', program=0, time=0))

        ticks_per_beat = 480
        ticks_per_step = ticks_per_beat * 2 
        
        for t, p in enumerate(emotions_array):
            emotion_cat, v, a, intensity, state, spike_intensity, macro_label = self._process_emotion_state(p)

            is_neutral = (emotion_cat == 'neutral')

            current_mode, chord_type, self.neutral_locked_mode = select_mode(
                emotion_cat, v, p, self.emotion_streak, self.neutral_locked_mode
            )

            # DIATONIC MODE POOL
            pool = get_mode_pool(current_mode, root_midi=(24 + self.base_key_offset), octaves=8)
            melody_pool = pool

            #____________________________________________________________
            # TEMPO
            target_bpm = 100 + (a * 40)
            target_bpm *= self.spike_state.get_tempo_multiplier()

            if self.current_bpm is None:
                self.current_bpm = target_bpm
            else:
                self.current_bpm = 0.7 * self.current_bpm + 0.3 * target_bpm
                
            tempo = bpm2tempo(int(self.current_bpm))
            chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

            # ticks_per_step ensures 1 iteration = exactly 1 second
            ticks_per_step = int((self.current_bpm / 60.0) * ticks_per_beat)

            #____________________________________________________________
            # VELOCITY
            velocity = int(70 + (a * 40))
            velocity = max(30, min(110, velocity))

            # Sad/Neutral velocity boost
            if macro_label == 'sad':
                velocity = max(50, min(110, velocity + 15))
            if emotion_cat == 'neutral':
                velocity = max(60, velocity)

            vel_shift = self.spike_state.get_velocity_shift()
            if vel_shift != 0:
                velocity = max(20, min(127, velocity + vel_shift))
                velocity = max(30, min(110, velocity))

            #________________________________________________________________________________
            # RHYTHM RATIOS
            if a > 0.6:
                chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
            elif a > 0.0:
                chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
            else:
                chosen_rhythm_ratios = [1.0] if random.random() > 0.5 else [0.5, 0.5]

            # Anxiety spike (happy->fear)
            chosen_rhythm_ratios = apply_anxiety_rhythm_doubling(
                chosen_rhythm_ratios, self.spike_state.name, spike_intensity
            )

            # Sad/Neutral humanization
            chosen_rhythm_ratios = apply_humanization_jitter(chosen_rhythm_ratios, emotion_cat)

            # Ratios to ticks conversion
            chosen_rhythm = [int(r * ticks_per_step) for r in chosen_rhythm_ratios]
            if sum(chosen_rhythm) != ticks_per_step:
                chosen_rhythm[-1] += (ticks_per_step - sum(chosen_rhythm))

            #__________________________________________________________________________________
            # 3. CHORDS & ACCOMPANIMENT
            harmonic_rhythm = compute_harmonic_rhythm(a, emotion_cat, self.emotion_streak)

            self.current_chord_degree = advance_chord_degree(
                emotion_cat, current_mode, self.emotion_streak,
                harmonic_rhythm, self.current_chord_degree, is_first_step=(t == 0)
            )

            # FEAR chord movement humanization
            if emotion_cat == 'fear' and intensity > 0.90:
                if random.random() < 0.30:
                    self.current_chord_degree = random.choice([3, 5, 6])

            base_chord_pool_idx = 14
            chord_root_idx = base_chord_pool_idx + self.current_chord_degree

            chord_notes = build_chord_notes(pool, chord_root_idx, chord_type, current_mode)

            # SAD humanization: added harmonic minor chord
            chord_notes = apply_harmonic_minor_chord(chord_notes, emotion_cat, self.current_chord_degree, chord_type)

            if emotion_cat == 'sad':
                chord_vel = max(40, velocity + 5)
            elif emotion_cat == 'neutral':
                chord_vel = max(35, min(70, velocity - 10))
            else:
                chord_vel = max(20, velocity - 10)

            chord_notes, self.current_chord_degree = apply_chord_spike_overrides(
                pool, base_chord_pool_idx, chord_notes,
                self.spike_state.get_chord_color(), self.spike_state.name, spike_intensity,
                self.emotion_streak, self.current_chord_degree
            )

            # UNEASE spike humanization
            if self.spike_state.name in ('UNEASE',) and spike_intensity > 0.5:
                    if chord_notes[0] > 36:
                        chord_notes[0] = chord_notes[0] - 12
                    ghost_note = min(127, chord_notes[0] + 1)
                    ghost_vel = max(15, int(chord_vel * 0.25))  
                    chord_track.append(Message('note_on', note=int(ghost_note), velocity=ghost_vel, time=0))
                    chord_track.append(Message('note_off', note=int(ghost_note), velocity=0, time=int(ticks_per_step // 2)))

            self._generate_piano_accompaniment(emotion_cat, chord_notes, chord_vel, velocity, ticks_per_step, chord_track)

            #____________________________________________________________________________________
            # 4. MELODY GENERATION
            melody_notes_and_durations = self._generate_melody_phrase(
                melody_pool, emotion_cat, v, state, spike_intensity, chosen_rhythm, chord_notes
            )

            self._append_melody_events(melody_notes_and_durations, emotion_cat, velocity, melody_track)

        mid.save(filename)
        print(f"Saved Final Cohesive MIDI: {filename}")


def generate_midi_from_emotions(emotions_array, base_key_offset=0, filename="eeg_music.mid"):
    """
    Wrapper function to maintain compatibility with existing pipeline scripts.
    Instantiates the OfflineMidiGenerator and runs the generation.
    """
    if base_key_offset == 0 and len(emotions_array) > 0:
        dom_idx = int(np.argmax(emotions_array[0]))
        EMOTION_LABELS = ["NEUTRAL", "SAD", "FEAR", "HAPPY"]
        base_key_offset = select_key_offset(dom_idx)
        print(f"[OfflineGenerator] Dynamic Key Set! Emotion: {EMOTION_LABELS[dom_idx]}, Offset: +{base_key_offset}")

    generator = OfflineMidiGenerator(base_key_offset=base_key_offset)
    generator.generate(emotions_array, filename)

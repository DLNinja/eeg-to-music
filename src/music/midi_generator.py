import json
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np
from .emotion_tracker import EmotionTracker
from .markov_engine import MarkovEngine

def get_mode_intervals(mode_name):
    # Greek Mode interval integer scales relative to the Root
    modes = {
        'lydian':     [0, 2, 4, 6, 7, 9, 11],
        'ionian':     [0, 2, 4, 5, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'dorian':     [0, 2, 3, 5, 7, 9, 10],
        'aeolian':    [0, 2, 3, 5, 7, 8, 10],
        'phrygian':   [0, 1, 3, 5, 7, 8, 10],
        'locrian':    [0, 1, 3, 5, 6, 8, 10]
    }
    return modes.get(mode_name, modes['ionian'])

def get_mode_pool(mode_name, root_midi=24, octaves=8):
    # generates parallel mode notes across octaves grounded to a shared root fundamental
    intervals = get_mode_intervals(mode_name)
    pool = []
    for oct in range(octaves):
        for interval in intervals:
            note = root_midi + (oct * 12) + interval
            if note <= 127:
                pool.append(note)
    return pool

def get_chord(mode_name, root_midi, chord_type="triad"):
    intervals = get_mode_intervals(mode_name)
    if chord_type == "triad":
        return [root_midi, root_midi + intervals[2], root_midi + intervals[4]]
    elif chord_type == "sus2":
        return [root_midi, root_midi + intervals[1], root_midi + intervals[4]]
    elif chord_type == "dim":
        # Force a flat 5 for horror/fear
        return [root_midi, root_midi + intervals[2], root_midi + 6]
    return [root_midi, root_midi + intervals[2], root_midi + intervals[4]]

def generate_midi_from_emotions(emotions_array, eeg_features=None, base_key_offset=0, filename="eeg_music.mid"):
    mid = MidiFile()

    # left hand chords + right hand melody
    chord_track = MidiTrack()
    melody_track = MidiTrack()
    mid.tracks.extend([chord_track, melody_track])
    
    # Explicitly set the instrument to Acoustic Grand Piano (Program 0) for both tracks
    chord_track.append(Message('program_change', program=0, time=0))
    melody_track.append(Message('program_change', program=0, time=0))

    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat * 2 # each time step holds for 2 beats

    # Rhythm dicts: lists of note durations in ticks adding up to ticks_per_step (960)
    rhythms = {
        'slow': [[960], [480, 480], [720, 240]],
        'med': [[480, 240, 240], [240, 240, 480], [320, 320, 320]], # triplets included
        'fast': [[240, 240, 240, 240], [120, 120, 240, 480], [240, 120, 120, 480]]
    }

    # Tracking variables for Cohesion and Phrasing
    current_bpm = None
    prev_dominant_idx = -1
    emotion_streak = 0
    
    # Start the melody index around middle C.
    # index 21 in the 8-octave pool is exactly the Root interval of the 3rd octave
    melody_idx = 21 

    # Shared Root note across all modes for parallel cohesion (24 is C1, 36 is C2, 48 is C3)
    fundamental_bass_root = 36 + base_key_offset
    
    tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
    markov_engine = MarkovEngine()

    # Step 3: Motif memory — stores the last generated 3-4 note contour phrase
    motif_buffer = []  # list of scale-degree offsets relative to melody_idx

    progression_step = 0

    for t, p in enumerate(emotions_array):
        # Update continuous tracker
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx]
        tracker.update_from_discrete(dominant_idx, intensity)
        
        state = tracker.get_state()
        v = state['macro_v']
        a = state['macro_a']
        
        # Track streak to implement variations within identical sustained emotion periods
        if dominant_idx == prev_dominant_idx:
            emotion_streak += 1
        else:
            emotion_streak = 0
        prev_dominant_idx = dominant_idx

        # --------------------------------------------------------------------------
        # 1. MODE SELECTION
        # Valence determines the "color" (Scale)
        is_neutral = abs(v) < 0.2 and abs(a) < 0.2

        if is_neutral:
            # ── Step 1: Chameleon Neutral ─────────────────────────────────────
            # Use the nearest expressive mode so neutral doesn't pick a random
            # colour each bar — it "leans" toward whichever quadrant is closest.
            if v >= 0:
                current_mode = 'mixolydian'   # Slightly bright but ambiguous
            else:
                current_mode = 'dorian'        # Slightly dark but smooth
            # Suspended chords = tonally ambiguous, works for any mood
            chord_type = random.choice(['sus2', 'sus4'])
        elif v > 0.5:
            current_mode = 'lydian' if v > 0.75 else 'ionian'
            chord_type = "triad"
        elif v > 0.1:
            current_mode = 'mixolydian'
            chord_type = "sus2"
        elif v > -0.2:
            current_mode = 'dorian'
            chord_type = "sus2"
        elif v > -0.6:
            current_mode = 'aeolian'
            chord_type = "triad"
        else:
            current_mode = 'phrygian' if v > -0.8 else 'locrian'
            chord_type = "dim"

        # Generate the pool of valid diatonic notes for this exact parallel mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)

        # ── Step 1 (cont.): Neutral pentatonic restriction & register shift ──
        if is_neutral:
            # Build a pentatonic subset: degrees 0, 1, 3, 4, 5 of the diatonic pool
            # (skips the 3rd and 7th — removes modal "colour" notes for ambiguity)
            pentatonic_degrees = [0, 1, 3, 4, 5]
            pentatonic_pool = []
            intervals = get_mode_intervals(current_mode)
            root = 24 + base_key_offset
            for oct in range(8):
                for deg in pentatonic_degrees:
                    note = root + (oct * 12) + intervals[deg]
                    if note <= 127:
                        pentatonic_pool.append(note)
            melody_pool = pentatonic_pool  # melody uses this restricted pool
            # Register shift: slightly negative V → drop one octave (dark ambient)
            #                 slightly positive V → raise one octave (serene)
            register_shift = 12 if v > 0.05 else (-12 if v < -0.05 else 0)
        else:
            melody_pool = pool
            register_shift = 0
        
        # 1.5. TEXTURING: Map raw EEG bands to MIDI Control Change (CC) messages
        if eeg_features is not None and t < len(eeg_features):
            # eeg_features shape: (T, 62, 5) -> 5 bands: Delta, Theta, Alpha, Beta, Gamma
            # Average across all channels for this time step to get global brain state
            bands = np.mean(eeg_features[t], axis=0) # shape (5,)
            
            # Normalize bands roughly to 0-127 MIDI range. 
            # DE features are typically between -10 and 20 (log energy). 
            # We'll use a simple min-max scaling assumption or clipping for safety.
            # You can tweak these multiplier/offset values depending on the actual data range!
            scaled_bands = np.clip((bands + 5) * 6, 0, 127).astype(int) 
            
            delta_val = scaled_bands[0]
            theta_val = scaled_bands[1]
            alpha_val = scaled_bands[2]
            beta_val  = scaled_bands[3]
            gamma_val = scaled_bands[4]
            
            # Texture 1: Delta -> Chorus/Sub-bass (CC 93)
            chord_track.append(Message('control_change', control=93, value=delta_val, time=0))
            melody_track.append(Message('control_change', control=93, value=delta_val, time=0))
            
            # Texture 2: Theta -> Modulation Wheel (CC 1)
            chord_track.append(Message('control_change', control=1, value=theta_val, time=0))
            melody_track.append(Message('control_change', control=1, value=theta_val, time=0))
            
            # Texture 3: Alpha -> Reverb (CC 91)
            chord_track.append(Message('control_change', control=91, value=alpha_val, time=0))
            melody_track.append(Message('control_change', control=91, value=alpha_val, time=0))
            
            # Texture 4: Beta -> Envelope Attack (CC 73)
            # Higher Beta (alert) = faster attack (lower CC value). Low Beta = slow attack (higher CC value)
            attack_val = 127 - beta_val
            chord_track.append(Message('control_change', control=73, value=attack_val, time=0))
            melody_track.append(Message('control_change', control=73, value=attack_val, time=0))
            
            # Texture 5: Gamma -> Brightness/Filter Cutoff (CC 74)
            chord_track.append(Message('control_change', control=74, value=gamma_val, time=0))
            melody_track.append(Message('control_change', control=74, value=gamma_val, time=0))
            
        # 2. DYNAMICS & TEMPO (Mapped to Arousal)
        # BPM scale: -1.0 (60) to 1.0 (140)
        target_bpm = 100 + (a * 40)
        
        # Exponential Moving Average for smooth cinematic tempo transitions
        if current_bpm is None:
            current_bpm = target_bpm
        else:
            current_bpm = 0.7 * current_bpm + 0.3 * target_bpm
            
        tempo = bpm2tempo(int(current_bpm))
        chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # Dynamic ticks_per_step to ensure 1 iteration = exactly 1 second
        # 1 second = (current_bpm / 60) beats
        # ticks_per_step = (current_bpm / 60) * ticks_per_beat
        ticks_per_step = int((current_bpm / 60.0) * ticks_per_beat)

        # Velocity mapping: -1.0 (30) to 1.0 (110)
        velocity = int(70 + (a * 40))
        velocity = max(30, min(110, velocity))
        
        # Rhythmic Density based on Arousal
        if a > 0.6:
            chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
        elif a > 0.0:
            chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
        else:
            chosen_rhythm_ratios = [1.0] if random.random() > 0.5 else [0.5, 0.5]

        # Convert ratios to actual ticks for this specific step
        chosen_rhythm = [int(r * ticks_per_step) for r in chosen_rhythm_ratios]
        # Adjust last note to fix rounding errors and ensure sum is exactly ticks_per_step
        if sum(chosen_rhythm) != ticks_per_step:
            chosen_rhythm[-1] += (ticks_per_step - sum(chosen_rhythm))

        # -------------------------------------------------------------------------
        # 3. CHORD & ACCOMPANIMENT GENERATION (Block styles only)

        # Map macro V/A to discrete emotional categories for the progression
        if v > 0.4 and a > 0.0:
            emotion_cat = 'happy'
        elif v < -0.3 and a < 0.0:
            emotion_cat = 'sad'
        elif v < -0.3 and a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        # --- EMOTION SPECIFIC CHORD PROGRESSIONS ---
        PROGRESSIONS = {
            # Happy: classic pop I-V-vi-IV (Lydian/Ionian degrees)
            'happy':   [0, 4, 5, 3],
            # Sad: Andalusian descending cadence i-bVII-VI-V (Aeolian degrees)
            'sad':     [0, 6, 5, 4],
            # Fear: mostly drones on the root, occasionally touching bII (Phrygian)
            'fear':    [0, 0, 1, 0],
            # Neutral: Dorian/Mixolydian shifts (II-I-IV-V)
            'neutral': [1, 0, 3, 4],
        }

        # Harmonic Rhythm: How many steps to hold a chord based on Macro Arousal
        if a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        # Advance the progression sequence
        if t == 0 or emotion_streak == 0:
            progression_step = 0
            # On state change, force chord change immediately
            current_chord_degree = PROGRESSIONS[emotion_cat][progression_step]
        elif emotion_streak % harmonic_rhythm == 0:
            progression_step = (progression_step + 1) % 4
            current_chord_degree = PROGRESSIONS[emotion_cat][progression_step]
        else:
            # Hold chord
            pass

        base_chord_pool_idx = 14
        chord_root_idx = base_chord_pool_idx + current_chord_degree

        # Diatonic mapping (Fear / Neutral get some suspended/dim flavours naturally)
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

        chord_vel = max(20, velocity - 10)

        # --- ACCOMPANIMENT STYLES (NO ARPEGGIOS) ---
        if emotion_cat == 'happy':
            # sustained_block: Classic block chord right on the beat
            for n in chord_notes:
                chord_track.append(Message('note_on',  note=int(n), velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=int(chord_notes[0]), velocity=0, time=int(ticks_per_step)))
            for n in chord_notes[1:]:
                chord_track.append(Message('note_off', note=int(n), velocity=0, time=0))

        elif emotion_cat == 'sad':
            # open_wide: Root + 5th below + 3rd octave up (Tragic, spacious)
            root  = int(chord_notes[0])
            fifth = max(0, root - 5)
            third = min(127, int(chord_notes[1]) + 12)
            for n in [root, fifth, third]:
                chord_track.append(Message('note_on',  note=n, velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=root,  velocity=0, time=int(ticks_per_step)))
            chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
            chord_track.append(Message('note_off', note=third, velocity=0, time=0))

        elif emotion_cat == 'fear':
            # drone_pedal: Very soft, sustained low pedal point
            root = max(0, int(chord_notes[0]) - 12)
            vel = max(10, chord_vel - 20)
            
            # Spike surge: purely dynamic (velocity boost), no fast melodic trills
            micro_a_val = state['micro_a']
            is_surge = abs(micro_a_val) > tracker.spike_threshold
            if is_surge:
                vel = min(127, vel + int(abs(micro_a_val) * 40))
                # Occasionally toss in a very quiet tense 5th just to thicken the drone
                fifth = min(127, root + 7)
                chord_track.append(Message('note_on',  note=root, velocity=vel, time=0))
                chord_track.append(Message('note_on',  note=fifth, velocity=max(10, vel - 15), time=0))
                chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
            else:
                chord_track.append(Message('note_on',  note=root, velocity=vel, time=0))
                chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))

        elif emotion_cat == 'neutral':
            # quartal_float: Chord built in fourths
            root   = int(chord_notes[0])
            fourth = min(127, root + 5)
            flat7  = min(127, fourth + 5)
            for n in [root, fourth, flat7]:
                chord_track.append(Message('note_on',  note=n, velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=root,  velocity=0, time=int(ticks_per_step)))
            chord_track.append(Message('note_off', note=fourth, velocity=0, time=0))
            chord_track.append(Message('note_off', note=flat7, velocity=0, time=0))


        # -------------------------------------------------------------------------
        # 4. MELODY GENERATION (Step 3: Motif Memory)
        # 40 % of the time: replay the saved motif (transposed ±1-2 scale degrees).
        # 60 % of the time: generate a fresh contour phrase and save it.

        melody_notes_and_durations = []
        active_pool = melody_pool  # pentatonic when neutral, diatonic otherwise

        def _pool_idx_nearest(target, p):
            """Return index of note in pool p closest to target pitch."""
            return min(range(len(p)), key=lambda k: abs(p[k] - target))

        use_motif = len(motif_buffer) > 0 and random.random() < 0.40

        if use_motif:
            # ── Replay motif transposed ±1-2 scale degrees ──
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, duration in motif_buffer:
                new_deg = melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note    = int(active_pool[new_deg]) + register_shift
                note    = max(0, min(127, note))
                melody_notes_and_durations.append((note, duration))
        else:
            # ── Generate a fresh phrase using VGMIDI Markov Chain ──
            # Instead of a hardcoded contour, we query the transition matrix
            # for the current emotion quadrant to build a dynamically realistic phrase.
            
            chosen_contour = []
            prev_interval = 0  # start stationary
            
            for _ in range(len(chosen_rhythm)):
                next_interval = markov_engine.query_next_interval(emotion_cat, prev_interval)
                chosen_contour.append(next_interval)
                prev_interval = next_interval
                
            new_motif = []

            # Harmonic adherence: macro valence anchors to chord, micro valence nudges
            chord_adherence_prob = max(0.4, min(0.95, 0.7 + (v * 0.20) + (state['micro_v'] * 0.10)))

            for i, duration in enumerate(chosen_rhythm):
                if i < len(chosen_contour):
                    melody_idx += chosen_contour[i]
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                # Occasionally snap to a chord tone for harmonic grounding
                if random.random() < chord_adherence_prob:
                    target_note = random.choice(chord_notes) + random.choice([12, 24])
                    melody_idx  = _pool_idx_nearest(target_note, active_pool)
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                note = int(active_pool[melody_idx]) + register_shift

                # Micro valence expression (same as realtime_generator)
                if state['micro_v'] > 0.4 and random.random() < 0.35:
                    note += 12
                if state['micro_v'] < -0.4 and random.random() < 0.35:
                    note += random.choice([-1, 1])
                note = max(0, min(127, note))

                # Record motif as degree offset from the START melody_idx of this phrase
                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, duration))

                # Phrasing rests for neutral states
                if is_neutral and random.random() < 0.3:
                    melody_notes_and_durations.append((None, duration))
                else:
                    melody_notes_and_durations.append((note, duration))

            # Save the new phrase to the motif buffer (replace old one)
            motif_buffer[:] = new_motif

        for note, duration in melody_notes_and_durations:
            if note is None:
                melody_track.append(Message('note_off', note=0, velocity=0, time=int(duration)))
            else:
                melody_track.append(Message('note_on',  note=int(note), velocity=int(velocity), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0,              time=int(duration)))

    mid.save(filename)
    print(f"🎵 Saved Final Cohesive MIDI: {filename}")
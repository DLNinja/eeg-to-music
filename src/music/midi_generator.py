import json
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np
from .emotion_tracker import EmotionTracker
from .markov_engine import MarkovEngine

# ── CHORD TRANSITION MATRIX (Markov Chain on Relative Scale Degrees) ──
# Maps current chord degree (0-6) to weighted next-degree options per emotion.
# Shared across both offline (midi_generator) and live (realtime_generator).
CHORD_TRANSITIONS = {
    'happy': {
        # Lydian/Ionian: Wants to move to IV (3) and V (4), strong resolutions to I (0)
        0: {'options': [0, 1, 3, 4, 5], 'weights': [10, 10, 30, 30, 20]},
        1: {'options': [4, 5, 0],       'weights': [60, 20, 20]},
        2: {'options': [3, 5],          'weights': [50, 50]},
        3: {'options': [0, 4, 1],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [70, 30]},
        5: {'options': [3, 1, 0],       'weights': [50, 30, 20]},
        6: {'options': [0, 5],          'weights': [80, 20]}
    },
    'sad': {
        # Aeolian: Heavy reliance on walking down (i -> VII -> VI -> v)
        0: {'options': [0, 3, 5, 6],    'weights': [20, 20, 30, 30]},
        1: {'options': [4, 6],          'weights': [70, 30]},
        2: {'options': [5, 3],          'weights': [60, 40]},
        3: {'options': [0, 4, 6],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [60, 40]},
        5: {'options': [3, 6, 0],       'weights': [40, 40, 20]},
        6: {'options': [0, 2, 5],       'weights': [50, 20, 30]}
    },
    'fear': {
        # Phrygian/Locrian: Tense, droning on the root, anxious half-steps to the flat 2nd (II)
        0: {'options': [0, 1, 2],       'weights': [70, 20, 10]},
        1: {'options': [0, 2],          'weights': [80, 20]},
        2: {'options': [0, 1],          'weights': [50, 50]},
        3: {'options': [0, 4],          'weights': [80, 20]},
        4: {'options': [0, 1],          'weights': [80, 20]},
        5: {'options': [0, 6],          'weights': [70, 30]},
        6: {'options': [0],             'weights': [100]}
    },
    'neutral': {
        # Dorian/Mixolydian: Wandering jazz/pop turnarounds, avoiding aggressive resolutions
        0: {'options': [0, 1, 3, 4],    'weights': [20, 30, 30, 20]},
        1: {'options': [0, 3, 4],       'weights': [20, 40, 40]},
        2: {'options': [1, 5],          'weights': [50, 50]},
        3: {'options': [0, 1, 4],       'weights': [40, 30, 30]},
        4: {'options': [0, 1, 3],       'weights': [30, 30, 40]},
        5: {'options': [1, 3],          'weights': [50, 50]},
        6: {'options': [0, 3],          'weights': [50, 50]}
    }
}

def get_mode_intervals(mode_name):
    # GREEK MODES (intervals relative to the root)
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
    # GENERATES THE POOL OF VALID NOTES BASED ON MODE AND ROOT (the key of the song)
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
        return [root_midi, root_midi + intervals[2], root_midi + 6]
    return [root_midi, root_midi + intervals[2], root_midi + intervals[4]]

def generate_midi_from_emotions(emotions_array, base_key_offset=0, filename="eeg_music.mid"):
    mid = MidiFile()

    # left hand CHORDS + right hand MELODY
    chord_track = MidiTrack()
    melody_track = MidiTrack()
    mid.tracks.extend([chord_track, melody_track])
    
    # Instrument = Acoustic Grand Piano (Program 0)
    chord_track.append(Message('program_change', program=0, time=0))
    melody_track.append(Message('program_change', program=0, time=0))

    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat * 2 # each time step holds for 2 beats

    # RHYTHM DICTIONARY:
    # lists of note durations in ticks adding up to ticks_per_step (960)
    rhythms = {
        'slow': [[960], [480, 480], [720, 240]],
        'med': [[480, 240, 240], [240, 240, 480], [320, 320, 320]], # triplets included
        'fast': [[240, 240, 240, 240], [120, 120, 240, 480], [240, 120, 120, 480]]
    }

    # TRACKING VARIABLES (Cohesion and Phrasing)
    current_bpm = None
    prev_dominant_idx = -1
    emotion_streak = 0
    
    # starting index 21 = root interval of the 3rd octave
    melody_idx = 21 

    # ── Dynamic Key Selection (Schubert) ──
    if base_key_offset == 0 and len(emotions_array) > 0:
        dom_idx = int(np.argmax(emotions_array[0]))
        # 0=Neutral, 1=Sad, 2=Fear, 3=Happy
        EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
        if dom_idx == 3: # Happy -> C Maj (0) or G Maj (7)
            base_key_offset = random.choice([0, 7])
        elif dom_idx == 1: # Sad -> D Min (2) or F Min (5)
            base_key_offset = random.choice([2, 5])
        elif dom_idx == 2: # Fear -> C# Min (1) or Eb Min (3)
            base_key_offset = random.choice([1, 3])
        else: # Neutral -> F Maj (5) or A Min (9)
            base_key_offset = random.choice([5, 9])
        print(f"[OfflineGenerator] Dynamic Key Set! Emotion: {EMOTION_LABELS[dom_idx]}, Offset: +{base_key_offset}")

    # shared root note across all modes for parallel cohesion 
    fundamental_bass_root = 36 + base_key_offset # (24 = C1, 36 = C2, 48 = C3)
    
    tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
    markov_engine = MarkovEngine()

    # MOTIF MEMORY (stores the last generated 3-4 note pattern)
    motif_buffer = []  # list of scale-degree offsets relative to melody_idx

    current_chord_degree = 0

    for t, p in enumerate(emotions_array):
        # Update emotion tracker
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx]
        tracker.update_from_discrete(dominant_idx, intensity)
        
        state = tracker.get_state()
        v = state['macro_v']
        a = state['macro_a']
        
        # Track MACRO MOOD streak to implement variations
        if dominant_idx == prev_dominant_idx:
            emotion_streak += 1
        else:
            emotion_streak = 0
        prev_dominant_idx = dominant_idx

        # --------------------------------------------------------------------------
        # 1. MODE SELECTION
        # VALENCE determines the "color" of the music (scale)
        is_neutral = abs(v) < 0.2 and abs(a) < 0.2

        if is_neutral:
            # _________________"CHAMELEON" NEUTRAL_________________ 
            # Use the nearest mode so neutral doesn't pick a random
            # color each bar — it "leans" toward whichever quadrant is closest.
            if v >= 0:
                current_mode = 'mixolydian'   # BRIGHT NEUTRAL
            else:
                current_mode = 'dorian'        # DARK NEUTRAL
            # Sus chords = tonally ambiguous, they work for any mood
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

        # Generate the pool of valid notes for the selected mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)

        # ── Neutral pentatonic restriction & register shift ──
        if is_neutral:
            # Use degrees 0, 1, 3, 4, 5 of the diatonic pool
            # (skip the 3rd and 7th — remove modal "color" notes for ambiguity)
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
            # Neutral Register Shift: 
            # negative V → drop one octave (dark ambient)
            # positive V → raise one octave (serene)
            register_shift = 12 if v > 0.05 else (-12 if v < -0.05 else 0)
        else:
            melody_pool = pool
            register_shift = 0
        
        # --------------------------------------------------------------------------
        # 2. DYNAMICS & TEMPO (Mapped to Arousal)
        # BPM scale: -1.0 (60) to 1.0 (140)
        target_bpm = 100 + (a * 40)
        
        # EXPONENTIAL MOVING AVERAGE (for smooth tempo transitions)
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

        # VELOCITY MAPPING: -1.0 (30) to 1.0 (110)
        velocity = int(70 + (a * 40))
        velocity = max(30, min(110, velocity))
        
        # RHYTHMIC DENSITY based on Arousal
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

        # Map MACRO V/A to discrete emotional categories for the progression
        if v > 0.4 and a > 0.0:
            emotion_cat = 'happy'
        elif v < -0.3 and a < 0.0:
            emotion_cat = 'sad'
        elif v < -0.3 and a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        # --- CHORD PROGRESSION (Markov Transition Matrix) ---

        # Harmonic Rhythm: How many steps to hold a chord based on Macro Arousal
        if a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        # Advance chord via Markov dice roll
        if t == 0 or emotion_streak == 0:
            # On start or emotion shift, force a safe root
            current_chord_degree = 0
        elif emotion_streak % harmonic_rhythm == 0:
            # Time to change the chord — look up the transition matrix
            matrix = CHORD_TRANSITIONS.get(emotion_cat, {})
            if current_chord_degree in matrix:
                options = matrix[current_chord_degree]['options']
                weights = matrix[current_chord_degree]['weights']
                current_chord_degree = random.choices(options, weights=weights, k=1)[0]
            else:
                current_chord_degree = 0  # Fallback

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

        # --- ACCOMPANIMENT STYLES ---
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
        # 4. MELODY GENERATION
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
                
                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12
                    
                note    = max(0, min(127, note))
                melody_notes_and_durations.append((note, duration))
        else:
            # ── Generate a fresh phrase using VGMIDI Markov Chain ──
            # Instead of a hardcoded contour, we query the transition matrix
            # for the current emotion quadrant to build a dynamically realistic phrase.
            
            chosen_contour = []
            prev_intervals = [0, 0, 0]  # start stationary with history
            
            for _ in range(len(chosen_rhythm)):
                next_interval = markov_engine.query_next_interval(emotion_cat, prev_intervals)
                chosen_contour.append(next_interval)
                prev_intervals.pop(0)
                prev_intervals.append(next_interval)
                
            new_motif = []

            # Harmonic adherence: MACRO VALENCE anchors to chord, MICRO VALENCE nudges
            chord_adherence_prob = max(0.4, min(0.95, 0.7 + (v * 0.20) + (state['micro_v'] * 0.10)))

            for i, duration in enumerate(chosen_rhythm):
                if i < len(chosen_contour):
                    melody_idx += chosen_contour[i]
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                # Occasionally snap to a chord tone for harmonic grounding
                if random.random() < chord_adherence_prob:
                    # Snap to safe triad tones only (Root, 3rd, 5th)
                    # to prevent dissonant clashes with any 7ths or extensions
                    safe_snap_notes = chord_notes[:3]
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    melody_idx  = _pool_idx_nearest(target_note, active_pool)
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                note = int(active_pool[melody_idx]) + register_shift

                # Micro valence expression (same as realtime_generator)
                if state['micro_v'] > 0.4 and random.random() < 0.35:
                    note += 12
                if state['micro_v'] < -0.4 and random.random() < 0.35:
                    note += random.choice([-1, 1])

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

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
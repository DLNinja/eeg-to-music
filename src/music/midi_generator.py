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

# ── SPIKE TRANSITION PROFILES ──
# Maps (macro_emotion, spike_emotion) → musical modifiers for spike moments.
# Each profile defines how the music is temporarily colored during a spike transition.
SPIKE_PROFILES = {
    ('happy', 'sad'):     {'name': 'BITTERSWEET',  'tempo_mult': 0.85, 'vel_shift': -15, 'chord_color': 'add_minor_3rd', 'melody_register': -6, 'rest_prob': 0.2},
    ('happy', 'neutral'): {'name': 'SERENITY',     'tempo_mult': 0.75, 'vel_shift': -20, 'chord_color': 'sus2',          'melody_register': 0,  'rest_prob': 0.35},
    ('happy', 'fear'):    {'name': 'ANXIETY',       'tempo_mult': 1.15, 'vel_shift': +10, 'chord_color': 'add_tritone',   'melody_register': +3, 'rest_prob': 0.0},
    ('sad', 'happy'):     {'name': 'HOPE',          'tempo_mult': 1.10, 'vel_shift': +15, 'chord_color': 'major_lift',    'melody_register': +6, 'rest_prob': 0.0},
    ('sad', 'neutral'):   {'name': 'ACCEPTANCE',    'tempo_mult': 0.90, 'vel_shift': -5,  'chord_color': 'resolve_root',  'melody_register': 0,  'rest_prob': 0.25},
    ('sad', 'fear'):      {'name': 'DISTURBED',     'tempo_mult': 1.05, 'vel_shift': +5,  'chord_color': 'add_quiet_dissonance', 'melody_register': 0, 'rest_prob': 0.1},
    ('fear', 'happy'):    {'name': 'COURAGE',       'tempo_mult': 1.20, 'vel_shift': +25, 'chord_color': 'major_power',   'melody_register': +6, 'rest_prob': 0.0},
    ('fear', 'sad'):      {'name': 'DESPAIR',       'tempo_mult': 0.70, 'vel_shift': +10, 'chord_color': 'thick_minor',   'melody_register': -3, 'rest_prob': 0.15},
    ('fear', 'neutral'):  {'name': 'RELIEF',        'tempo_mult': 0.75, 'vel_shift': -15, 'chord_color': 'resolve_major', 'melody_register': 0,  'rest_prob': 0.30},
    ('neutral', 'happy'): {'name': 'AWAKENING',     'tempo_mult': 1.15, 'vel_shift': +15, 'chord_color': 'bright_triad',  'melody_register': +3, 'rest_prob': 0.0},
    ('neutral', 'sad'):   {'name': 'MELANCHOLY',    'tempo_mult': 0.90, 'vel_shift': -10, 'chord_color': 'minor_color',   'melody_register': -3, 'rest_prob': 0.2},
    ('neutral', 'fear'):  {'name': 'UNEASE',        'tempo_mult': 1.10, 'vel_shift': +5,  'chord_color': 'add_tritone',   'melody_register': +3, 'rest_prob': 0.1},
}

def apply_spike_chord_color(chord_notes, color_type):
    """Apply spike-specific chord coloring to modify existing chord notes."""
    colored = list(chord_notes)
    if color_type == 'add_minor_3rd':
        # Replace major 3rd with minor 3rd (lower by 1 semitone) → bittersweet
        if len(colored) > 1:
            colored[1] = colored[1] - 1
    elif color_type == 'sus2':
        # Replace 3rd with 2nd → ethereal calm
        if len(colored) > 1:
            colored[1] = colored[0] + 2
    elif color_type == 'add_tritone':
        # Add a tritone (#4) → anxious tension
        colored.append(min(127, colored[0] + 6))
    elif color_type == 'major_lift':
        # Force major triad → hopeful lift
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'resolve_root':
        # Simplify to root + 5th → acceptance/resolution
        colored = [colored[0], min(127, colored[0] + 7)]
    elif color_type == 'add_quiet_dissonance':
        # Add half-step above root → subtle disturbance
        colored.append(min(127, colored[0] + 1))
    elif color_type == 'major_power':
        # Major triad with doubled root octave below → courageous power
        colored = [max(0, colored[0] - 12), colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'thick_minor':
        # Minor triad with low octave root + minor 7th → despairing thickness
        colored = [max(0, colored[0] - 12), colored[0], min(127, colored[0] + 3), min(127, colored[0] + 7), min(127, colored[0] + 10)]
    elif color_type == 'resolve_major':
        # Force major resolution → relief from tension
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'bright_triad':
        # Major triad in mid register → awakening energy
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'minor_color':
        # Minor triad → melancholic tinge
        colored = [colored[0], min(127, colored[0] + 3), min(127, colored[0] + 7)]
    return [max(0, min(127, int(n))) for n in colored]

def detect_trill(recent_pitches):
    """Detects rapid alternation between 2 adjacent notes to prevent annoying repetitive trills."""
    if len(recent_pitches) < 4:
        return False
    # Check last 4 notes pattern A B A B
    n1, n2, n3, n4 = [p[0] for p in recent_pitches[-4:]]
    # They must be alternating but distinct
    return n1 == n3 and n2 == n4 and n1 != n2

def resolve_dissonance(pitch, sounding_chord_notes):
    """
    Checks if a melody pitch creates a half-step clash against any sounding chord note.
    If so, snaps it to the closest consonant chord tone to maintain purely diatonic harmony.
    """
    for ct in sounding_chord_notes:
        if abs(pitch % 12 - ct % 12) == 1:
            nearest_ct = min(sounding_chord_notes, key=lambda c: abs((c % 12) - (pitch % 12)))
            return int((nearest_ct % 12) + (pitch // 12) * 12)
    return int(pitch)

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
    elif chord_type == "sus4":
        return [root_midi, root_midi + intervals[3], root_midi + intervals[4]]
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
    fundamental_bass_root = 36 + base_key_offset
    
    tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
    markov_engine = MarkovEngine()

    # MOTIF MEMORY (stores the last generated 3-4 note pattern)
    motif_buffer = []  # list of scale-degree offsets relative to melody_idx

    current_chord_degree = 0
    consecutive_trill_count = 0      # Phase 3A: anti-trill tracker
    spike_duration_counter = 0       # Phase 5: short spike tracking
    active_spike_profile = None      # Phase 4: current spike profile
    
    # Phase 7: Fear State Machine
    fear_submode = 'ambiguity'
    fear_tick_counter = 0
    fear_ramp_velocity = 35

    for t, p in enumerate(emotions_array):
        # Update emotion tracker
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx]
        tracker.update_from_discrete(dominant_idx, intensity)
        
        state = tracker.get_state()
        v = state['macro_v']
        a = state['macro_a']
        is_spike = state['is_spike']
        spike_label = state['spike_label']
        macro_label = state['macro_label']

        # Secondary emotion for neutral chameleon behavior
        sorted_emotions = np.argsort(p)
        secondary_idx = sorted_emotions[-2]
        LABEL_NAMES = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}
        secondary_label = LABEL_NAMES.get(secondary_idx, 'neutral')

        # Track MACRO MOOD streak to implement variations
        if dominant_idx == prev_dominant_idx:
            emotion_streak += 1
        else:
            emotion_streak = 0
        prev_dominant_idx = dominant_idx

        # ── SPIKE PROFILE MANAGEMENT ──
        if is_spike and macro_label != spike_label:
            spike_key = (macro_label, spike_label)
            active_spike_profile = SPIKE_PROFILES.get(spike_key)
            spike_duration_counter += 1
        else:
            spike_duration_counter = 0
            active_spike_profile = None

        # Map MACRO V/A to discrete emotional categories early
        if v > 0.0 and a > 0.0:
            emotion_cat = 'happy'
        elif v < 0.0 and a < 0.0:
            emotion_cat = 'sad'
        elif v < 0.0 and a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

        # --------------------------------------------------------------------------
        # 1. MODE SELECTION
        # VALENCE determines the "color" of the music (scale)
        is_neutral = (emotion_cat == 'neutral')
        
        if emotion_cat == 'neutral':
            current_mode = 'mixolydian'
            chord_type = 'sus2'
        elif emotion_cat == 'happy':
            current_mode = 'lydian' if v > 0.75 else 'ionian'
            chord_type = "triad"
        elif emotion_cat == 'sad':
            current_mode = 'aeolian'
            chord_type = "triad"
        else: # fear
            if v > -0.2:
                current_mode = 'dorian'
                chord_type = "sus2"
            elif v > -0.6:
                current_mode = 'aeolian'
                chord_type = "triad"
            else:
                # Use locrian extremely rarely, stick to phrygian mostly
                current_mode = 'phrygian' if v > -0.95 else 'locrian'
                chord_type = "dim"

        # Generate the pool of valid notes for the selected mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)

        # ── Neutral pentatonic restriction ──
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
            register_shift = 0  # Neutral stays in natural register
        else:
            melody_pool = pool
            register_shift = 0
        
        # --------------------------------------------------------------------------
        # 2. DYNAMICS & TEMPO (Mapped to Arousal)
        # BPM scale: -1.0 (60) to 1.0 (140)
        target_bpm = 100 + (a * 40)

        # Apply spike tempo modifier
        if active_spike_profile:
            if spike_duration_counter <= 2:
                # Short spike: subtle tempo nudge (5-8%)
                nudge = 0.06 if state['micro_a'] > 0 else -0.06
                target_bpm *= (1.0 + nudge)
            else:
                # Full spike: apply profile tempo multiplier
                target_bpm *= active_spike_profile['tempo_mult']

        # EXPONENTIAL MOVING AVERAGE (for smooth tempo transitions)
        if current_bpm is None:
            current_bpm = target_bpm
        else:
            current_bpm = 0.7 * current_bpm + 0.3 * target_bpm
            
        tempo = bpm2tempo(int(current_bpm))
        chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # Dynamic ticks_per_step to ensure 1 iteration = exactly 1 second
        ticks_per_step = int((current_bpm / 60.0) * ticks_per_beat)

        # VELOCITY MAPPING: -1.0 (30) to 1.0 (110)
        velocity = int(70 + (a * 40))
        velocity = max(30, min(110, velocity))

        # Sad needs more presence (higher velocity)
        if macro_label == 'sad':
            velocity = max(50, min(110, velocity + 15))

        # Apply spike velocity modifier
        if active_spike_profile:
            if spike_duration_counter <= 2:
                # Short spike: subtle velocity boost (12%)
                velocity = int(velocity * 1.12)
            else:
                velocity = velocity + active_spike_profile['vel_shift']
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

        # --- CHORD PROGRESSION (Markov Transition Matrix) ---

        # Harmonic Rhythm: How many steps to hold a chord based on Macro Arousal
        if a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        # Sad & Neutral chord variety: prevent boring repetition at slow tempos
        if emotion_cat in ('sad', 'neutral') and emotion_streak > 2:
            harmonic_rhythm = min(harmonic_rhythm, 2)  # Force chord change every 2 steps max
        # Even more aggressive for neutral — never hold more than 3 steps
        if emotion_cat == 'neutral':
            harmonic_rhythm = min(harmonic_rhythm, 3)

        # Advance chord via Markov dice roll
        if emotion_cat == 'fear':
            # Fear State Machine & chord override (eerie submodes)
            if t == 0 or emotion_streak == 0 or prev_dominant_idx != 2:
                # Initialize fear submode on entry to fear
                fear_submode = random.choices(['ambiguity', 'eerie_melodic', 'climax'], weights=[65, 30, 5])[0]
                fear_tick_counter = 0
                fear_ramp_velocity = 35
                current_chord_degree = 0

            fear_tick_counter += 1
            if fear_submode == 'eerie_melodic':
                # Descending harmony (e.g. 0 -> 6 -> 5 -> 4)
                if emotion_streak % harmonic_rhythm == 0:
                    current_chord_degree = (current_chord_degree - 1) % 7
            else:
                # Ambiguity/Climax stays mostly on tonic or steps slightly
                if emotion_streak % harmonic_rhythm == 0:
                    matrix = CHORD_TRANSITIONS.get(emotion_cat, {})
                    if current_chord_degree in matrix:
                        options = matrix[current_chord_degree]['options']
                        weights = matrix[current_chord_degree]['weights']
                        current_chord_degree = random.choices(options, weights=weights, k=1)[0]
                    else:
                        current_chord_degree = 0
                        
        elif t == 0 or emotion_streak == 0:
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

        # Harmonic Minor Alteration: Tension chord (Major V) for Sad emotion
        if emotion_cat == 'sad' and current_chord_degree == 4 and chord_type == "triad":
            chord_notes[1] += 1  # Raise minor 3rd to Major 3rd

        # Emotion-specific chord velocity
        if emotion_cat == 'sad':
            chord_vel = max(40, velocity + 5)  # Sad needs more presence
        else:
            chord_vel = max(20, velocity - 10)

        # Apply spike chord coloring (only for sustained spikes, 3+ samples)
        if active_spike_profile and spike_duration_counter > 2:
            chord_notes = apply_spike_chord_color(chord_notes, active_spike_profile['chord_color'])

        # Spike rest probability override for melody
        spike_rest_prob = active_spike_profile['rest_prob'] if active_spike_profile and spike_duration_counter > 2 else None

        # --- ACCOMPANIMENT STYLES ---
        # ALL voicings below use pool shifts inside chord_notes (+12/-12)
        # to guarantee 100% diatonic consistency without breaking the mode!
        
        if emotion_cat == 'happy':
            # sustained_block: Classic block chord right on the beat
            # Register cap: keep happy chords warm, not tense (max C4 = 72)
            chord_notes = [n - 12 if n > 72 else n for n in chord_notes]
            for n in chord_notes:
                chord_track.append(Message('note_on',  note=int(n), velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=int(chord_notes[0]), velocity=0, time=int(ticks_per_step)))
            for n in chord_notes[1:]:
                chord_track.append(Message('note_off', note=int(n), velocity=0, time=0))

        elif emotion_cat == 'sad':
            # open_wide (Diatonic): Root + 3rd (octave up) + 5th (octave down if possible)
            root  = int(chord_notes[0])
            third = int(chord_notes[1]) + 12
            fifth = int(chord_notes[2]) - 12
            
            # Ensure MIDI range safety
            fifth = fifth if fifth >= 0 else fifth + 12
            third = third if third <= 127 else third - 12
            
            for n in [root, fifth, third]:
                chord_track.append(Message('note_on',  note=n, velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=root,  velocity=0, time=int(ticks_per_step)))
            chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
            chord_track.append(Message('note_off', note=third, velocity=0, time=0))

        elif emotion_cat == 'fear':
            chord_track.append(Message('control_change', control=64, value=127, time=0))

            # Brings chords higher up to bridge the gap with the melody
            root = max(0, int(chord_notes[0]) - 12) 
            fifth_down = max(0, int(chord_notes[2]) - 12)

            # Route submode logic
            if fear_submode == 'climax':
                vel = 110 if random.random() < 0.2 else 0 # 20% chance of sudden stab
                if vel > 0:
                    minor_2nd = min(127, root + 1)
                    chord_track.append(Message('note_on', note=root, velocity=vel, time=0))
                    chord_track.append(Message('note_on', note=fifth_down, velocity=vel - 10, time=0))
                    chord_track.append(Message('note_on', note=minor_2nd, velocity=vel - 10, time=0))
                    chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step // 4)))
                    chord_track.append(Message('note_off', note=fifth_down, velocity=0, time=0))
                    chord_track.append(Message('note_off', note=minor_2nd, velocity=0, time=int(ticks_per_step - (ticks_per_step // 4))))
                else:
                    chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))
            else:
                # Ambiguity or Eerie Melodic
                if fear_submode == 'ambiguity':
                    # Ramp up slowly over 15 ticks
                    if fear_tick_counter % 15 == 0:
                        fear_ramp_velocity = 45 # Reset, increased base intensity
                    else:
                        fear_ramp_velocity += 4
                    vel = min(100, fear_ramp_velocity)
                else:
                    vel = random.randint(55, 75) # Constant brooding, higher base

                minor_2nd = min(127, root + 1)
                chord_track.append(Message('note_on', note=root, velocity=vel, time=0))
                if random.random() < 0.15:
                    chord_track.append(Message('note_on', note=minor_2nd, velocity=max(10, vel // 3), time=0))
                chord_track.append(Message('note_on', note=fifth_down, velocity=max(10, vel - 15), time=0))

                chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=minor_2nd, velocity=0, time=0))
                chord_track.append(Message('note_off', note=fifth_down, velocity=0, time=0))

            chord_track.append(Message('control_change', control=64, value=0, time=0))

        elif emotion_cat == 'neutral':
            # quartal_float (Diatonic): Stack 3 notes using pool indices to guarantee diatonic 4ths/3rds
            idx = chord_root_idx
            q1 = pool[idx]
            q2 = pool[idx + 3] if (idx + 3) < len(pool) else pool[-1]
            q3 = pool[idx + 6] if (idx + 6) < len(pool) else pool[-1]
            
            for n in [q1, q2, q3]:
                chord_track.append(Message('note_on',  note=int(n), velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=int(q1),  velocity=0, time=int(ticks_per_step)))
            chord_track.append(Message('note_off', note=int(q2), velocity=0, time=0))
            chord_track.append(Message('note_off', note=int(q3), velocity=0, time=0))


        # -------------------------------------------------------------------------
        # 4. MELODY GENERATION
        # 40 % of the time: replay the saved motif (transposed ±1-2 scale degrees).
        # 60 % of the time: generate a fresh contour phrase and save it.
        # Fear always generates fresh (sparse, unpredictable).

        melody_notes_and_durations = []
        
        # Create a mutable copy of the pool for this chunk
        active_pool = list(melody_pool)  # pentatonic when neutral, diatonic otherwise
        
        # Harmonic Minor Alteration: Align melody with the tension V chord
        if emotion_cat == 'sad' and current_chord_degree == 4:
            # Change the minor v's third in the melody pool to a major v's third (leading tone)
            # Since the pool is generated sequentially across octaves, every 7th note (idx % 7 == 6) is the 7th scale degree
            for idx in range(len(active_pool)):
                if (idx % 7) == 6:
                    active_pool[idx] += 1

        def _pool_idx_nearest(target, p):
            """Return index of note in pool p closest to target pitch."""
            return min(range(len(p)), key=lambda k: abs(p[k] - target))

        # Fear skips motif replay — sparse and unpredictable
        use_motif = len(motif_buffer) > 0 and random.random() < 0.40 and emotion_cat != 'fear'

        if use_motif:
            # ── Replay motif transposed ±1-2 scale degrees ──
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, duration in motif_buffer:
                new_deg = melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note    = int(active_pool[new_deg]) + register_shift

                # Apply spike melody register shift
                if active_spike_profile and spike_duration_counter > 2:
                    note += active_spike_profile['melody_register']

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12
                    
                note    = max(0, min(127, note))
                
                # Universal Dissonance Guard (Applies to replayed motif too!)
                if emotion_cat in ('happy', 'sad', 'neutral'):
                    note = resolve_dissonance(note, chord_notes)
                    
                melody_notes_and_durations.append((note, duration))
        else:
            # ── Generate a fresh phrase using VGMIDI Markov Chain ──
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
            # Happy/Sad: higher floor to prevent dissonant clashes
            if emotion_cat in ('happy', 'sad'):
                chord_adherence_prob = max(0.80, chord_adherence_prob)
            # Neutral: lower adherence to reduce tonic repetition, but prefer non-root tones
            if emotion_cat == 'neutral':
                chord_adherence_prob = min(0.55, chord_adherence_prob)

            for i, duration in enumerate(chosen_rhythm):
                if emotion_cat == 'fear':
                    # Sparse eerie right hand: mostly rests with occasional notes
                    if random.random() < 0.55:
                        melody_notes_and_durations.append((None, duration))
                        new_motif.append((0, duration))
                        continue
                    # Bring melody register down to approach the chords (C4-C5 range)
                    melody_idx = random.randint(28, min(len(active_pool) - 1, 36))
                    note = int(active_pool[melody_idx])
                    # Rare chromatic coloring for eeriness (sparing for musicality)
                    if random.random() < 0.15:
                        note += random.choice([-1, 1, 6, -6])
                else:
                    if i < len(chosen_contour):
                        melody_idx += chosen_contour[i]
                        melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                    # Occasionally snap to a chord tone for harmonic grounding
                    if random.random() < chord_adherence_prob:
                        # Snap to safe triad tones only (Root, 3rd, 5th)
                        safe_snap_notes = chord_notes[:3]
                        # Neutral: prefer non-root tones to reduce tonic repetition
                        if emotion_cat == 'neutral' and len(safe_snap_notes) > 1:
                            safe_snap_notes = safe_snap_notes[1:]  # skip root
                        target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                        melody_idx  = _pool_idx_nearest(target_note, active_pool)
                        melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                    note = int(active_pool[melody_idx]) + register_shift

                    # Micro valence expression — ONLY for fear/neutral transitions
                    if state['micro_v'] > 0.4 and random.random() < 0.35:
                        note += 12  # Octave jump is always safe
                    if emotion_cat == 'fear' and state['micro_v'] < -0.4 and random.random() < 0.25:
                        note += random.choice([-1, 1])  # Chromatic only for fear

                # Apply spike melody register shift
                if active_spike_profile and spike_duration_counter > 2:
                    note += active_spike_profile['melody_register']

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

                note = max(0, min(127, note))

                # Record motif as degree offset from the START melody_idx of this phrase
                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, duration))

                # Determine rest probability
                if spike_rest_prob is not None:
                    should_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.3:
                    should_rest = True
                else:
                    should_rest = False

                if should_rest:
                    melody_notes_and_durations.append((None, duration))
                else:
                    # Universal Dissonance Guard
                    if emotion_cat in ('happy', 'sad', 'neutral'):
                        note = resolve_dissonance(note, chord_notes)
                    melody_notes_and_durations.append((note, duration))

            # Save the new phrase to the motif buffer (replace old one)
            motif_buffer[:] = new_motif

        # ── Anti-trill guard (applies to happy, sad, neutral) ──
        # Rule: max 1 trill per 5 seconds — if ANY trill detected, replace immediately
        if emotion_cat in ('happy', 'sad', 'neutral'):
            if detect_trill(melody_notes_and_durations):
                consecutive_trill_count += 1
            else:
                consecutive_trill_count = 0
            # Trigger on the FIRST trill detection (was >= 2, now >= 1)
            if consecutive_trill_count >= 1:
                # Force non-trill: ascending arpeggio through chord tones
                melody_notes_and_durations = []
                for i_r, duration in enumerate(chosen_rhythm):
                    ct_idx = i_r % len(chord_notes)
                    note = int(chord_notes[ct_idx]) + 12
                    while note > 84:
                        note -= 12
                    note = max(0, min(127, note))
                    melody_notes_and_durations.append((note, duration))
                consecutive_trill_count = 0

        for note, duration in melody_notes_and_durations:
            if note is None:
                melody_track.append(Message('note_off', note=0, velocity=0, time=int(duration)))
            else:
                # Fear melody: separate velocity for eerie dynamics
                if emotion_cat == 'fear':
                    mel_vel = random.randint(90, 110) if random.random() < 0.12 else random.randint(55, 75)
                else:
                    mel_vel = velocity
                melody_track.append(Message('note_on',  note=int(note), velocity=int(mel_vel), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0,              time=int(duration)))

    mid.save(filename)
    print(f"🎵 Saved Final Cohesive MIDI: {filename}")
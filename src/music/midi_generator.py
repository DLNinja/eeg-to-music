import json
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np
from .emotion_tracker import EmotionTracker
from .markov_engine import MarkovEngine

# CHORD TRANSITION MATRIX 
# Markov Chain on weighted next relative scale degrees numbered 0-6
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
        # Aeolian: relies on downward progressions (i -> VII -> VI -> v)
        0: {'options': [0, 3, 5, 6],    'weights': [8, 25, 33, 34]},
        1: {'options': [4, 6],          'weights': [70, 30]},
        2: {'options': [5, 3],          'weights': [60, 40]},
        3: {'options': [0, 4, 6],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [60, 40]},
        5: {'options': [3, 6, 0],       'weights': [40, 40, 20]},
        6: {'options': [0, 2, 5],       'weights': [50, 20, 30]}
    },
    'fear': {
        # Phrygian: relies on i↔bII tension and dissonance
        0: {'options': [1, 6],          'weights': [60, 40]},
        1: {'options': [0, 2],          'weights': [70, 30]},
        2: {'options': [0, 1],          'weights': [45, 55]},
        3: {'options': [0, 1],          'weights': [35, 65]},
        4: {'options': [1, 0],          'weights': [70, 30]},
        5: {'options': [1, 6],          'weights': [60, 40]},
        6: {'options': [5, 0],          'weights': [65, 35]}        
    },
    'neutral_dorian': {
        # Dorian: focuses on i → IV transitions (minor scale with major IV chord)
        0: {'options': [3, 6, 1],       'weights': [55, 30, 15]},
        1: {'options': [3, 0],          'weights': [65, 35]},
        2: {'options': [6, 3],          'weights': [60, 40]},
        3: {'options': [0, 6, 1, 5],    'weights': [40, 30, 15, 15]},
        4: {'options': [3, 0],          'weights': [60, 40]},
        5: {'options': [6, 3, 0],       'weights': [45, 35, 20]},
        6: {'options': [0, 3, 5],       'weights': [40, 35, 25]}
    },
    'neutral_mixolydian': {
        # Mixolydian: focuses on I → ♭VII transitions (major scale with a lowered 7th)
        0: {'options': [6, 3],          'weights': [55, 45]},
        1: {'options': [3, 6, 0],       'weights': [50, 35, 15]},
        2: {'options': [6, 3],          'weights': [60, 40]},
        3: {'options': [6, 5, 1, 0],    'weights': [40, 25, 20, 15]},
        4: {'options': [3, 6, 0],       'weights': [45, 40, 15]},
        5: {'options': [6, 3, 0],       'weights': [55, 35, 10]},
        6: {'options': [3, 5, 1, 0],    'weights': [35, 30, 20, 15]}
    }
}

# SPIKE TRANSITION PROFILES
# Maps specific (macro_emotion, spike_emotion) pairs to musical modifiers for highlighting the micro spikes
SPIKE_PROFILES = {
    ('happy', 'sad'):     {'name': 'BITTERSWEET',  'tempo_mult': 0.85, 'vel_shift': -15, 'chord_color': 'add_minor_3rd',       'melody_register': -6, 'rest_prob': 0.2},
    ('happy', 'neutral'): {'name': 'SERENITY',     'tempo_mult': 0.75, 'vel_shift': -20, 'chord_color': 'sus2',                'melody_register': 0,  'rest_prob': 0.35},
    ('happy', 'fear'):    {'name': 'ANXIETY',       'tempo_mult': 1.30, 'vel_shift': +15, 'chord_color': 'anxious_creep',       'melody_register': 0,  'rest_prob': 0.0},
    ('sad', 'happy'):     {'name': 'HOPE',          'tempo_mult': 1.10, 'vel_shift': +15, 'chord_color': 'major_lift',          'melody_register': +6, 'rest_prob': 0.0},
    ('sad', 'neutral'):   {'name': 'ACCEPTANCE',    'tempo_mult': 0.90, 'vel_shift': -5,  'chord_color': 'picardy_lift',        'melody_register': 0,  'rest_prob': 0.25},
    ('sad', 'fear'):      {'name': 'DISTURBED',     'tempo_mult': 1.05, 'vel_shift': +8,  'chord_color': 'disturbed_tension',  'melody_register': 0,  'rest_prob': 0.1},
    ('fear', 'happy'):    {'name': 'COURAGE',       'tempo_mult': 1.20, 'vel_shift': +25, 'chord_color': 'epic_modal',          'melody_register': +6, 'rest_prob': 0.0},
    ('fear', 'sad'):      {'name': 'DESOLATION',    'tempo_mult': 0.70, 'vel_shift': +10, 'chord_color': 'hollow_madd9',        'melody_register': -3, 'rest_prob': 0.15},
    ('fear', 'neutral'):  {'name': 'RELIEF',        'tempo_mult': 0.75, 'vel_shift': -15, 'chord_color': 'resolve_major',       'melody_register': 0,  'rest_prob': 0.30},
    ('neutral', 'happy'): {'name': 'AWAKENING',     'tempo_mult': 1.15, 'vel_shift': +15, 'chord_color': 'bright_triad',        'melody_register': +3, 'rest_prob': 0.0},
    ('neutral', 'sad'):   {'name': 'MELANCHOLY',    'tempo_mult': 0.90, 'vel_shift': -10, 'chord_color': 'minor_color',         'melody_register': -3, 'rest_prob': 0.2},
    ('neutral', 'fear'):  {'name': 'UNEASE',        'tempo_mult': 1.10, 'vel_shift': +12, 'chord_color': 'suspended_tension',   'melody_register': +3, 'rest_prob': 0.05},
}

def apply_spike_chord_color(chord_notes, color_type):
    """Apply spike-specific chord coloring to modify existing chord notes."""
    colored = list(chord_notes)
    if color_type == 'add_minor_3rd':
        # BITTERSWEET: major 3rd replaced with minor 3rd
        if len(colored) > 1:
            colored[1] = colored[1] - 1
    elif color_type == 'sus2':
        # SERENITY: Replace 3rd with 2nd
        if len(colored) > 1:
            colored[1] = colored[0] + 2
    elif color_type == 'suspended_tension':
        # UNEASE: Replace 3rd with Perfect 4th
        colored = [colored[0], colored[0] + 5, colored[0] + 7]
    elif color_type == 'anxious_creep':
        # ANXIETY: lowered root, dominant 7th added for tension 
        bass = colored[0] - 12 if colored[0] > 36 else colored[0]
        dom7 = min(127, colored[0] + 10) 
        colored = [bass] + colored[1:] + [dom7]
    elif color_type == 'major_lift':
        # HOPE: introduce major triad
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'picardy_lift':
        # ACCEPTANCE: major chord in a minor context
        colored = [colored[0], colored[0] + 7, colored[0] + 4 + 12]
    elif color_type == 'disturbed_tension':
        # DISTURBED: drop the bass one octave, add a quiet ♭6 an octave up
        bass = colored[0] - 12 if colored[0] > 36 else colored[0]
        tension_note = min(127, colored[0] + 8 + 12)
        colored = [bass] + colored[1:] + [tension_note]
    elif color_type == 'epic_modal':
        # COURAGE: Open 5th power chord (Root + P5 + Root octave up)
        colored = [colored[0], colored[0] + 7, colored[0] + 12]
    elif color_type == 'hollow_madd9':
        # DESOLATION: Root + minor 3rd and 9th clustered an octave up
        colored = [colored[0], colored[0] + 7, colored[0] + 14, colored[0] + 15]
    elif color_type == 'resolve_major':
        # RELIEF: force major resolution
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'bright_triad':
        # AWAKENING: major triad in mid register
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'minor_color':
        # MELANCHOLY: minor triad
        colored = [colored[0], min(127, colored[0] + 3), min(127, colored[0] + 7)]
    return [max(0, min(127, int(n))) for n in colored]


def detect_trill(recent_pitches):
    # Detects rapid alternation between 2 adjacent notes to prevent repetitive trills.
    if len(recent_pitches) < 4:
        return False
    n1, n2, n3, n4 = [p[0] for p in recent_pitches[-4:]]
    return n1 == n3 and n2 == n4 and n1 != n2


def resolve_dissonance(pitch, sounding_chord_notes):
    # Checks for any clashing notes and snaps to the nearest diatonic note.
    for ct in sounding_chord_notes:
        if abs(pitch % 12 - ct % 12) == 1:
            nearest_ct = min(sounding_chord_notes, key=lambda c: abs((c % 12) - (pitch % 12)))
            return int((nearest_ct % 12) + (pitch // 12) * 12)
    return int(pitch)

def get_mode_intervals(mode_name):
    # GREEK MODES (intervals relative to the root)
    modes = {
        'lydian':            [0, 2, 4, 6, 7, 9, 11],  # HAPPY
        'ionian':            [0, 2, 4, 5, 7, 9, 11],  # HAPPY
        'mixolydian':        [0, 2, 4, 5, 7, 9, 10],  # NEUTRAL
        'dorian':            [0, 2, 3, 5, 7, 9, 10],  # NEUTRAL
        'aeolian':           [0, 2, 3, 5, 7, 8, 10],  # SAD
        'phrygian':          [0, 1, 3, 5, 7, 8, 10],  # FEAR
        'locrian':           [0, 1, 3, 5, 6, 8, 10],  # NOT USED (may use for FEAR)
        'harmonic_minor':    [0, 2, 3, 5, 7, 8, 11],  # FEAR (also used for SAD chord logic, see line 648)
        'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10],  # FEAR
    }
    return modes.get(mode_name, modes['ionian'])

def get_mode_pool(mode_name, root_midi=24, octaves=8):
    # Generates the pool of valid notes based on mode and root (the key of the song)
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

    # Left hand CHORDS + right hand MELODY
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

    # Dynamic Key Selection (Schubert)
    if base_key_offset == 0 and len(emotions_array) > 0:
        dom_idx = int(np.argmax(emotions_array[0]))
        # 0 = NEUTRAL, 1 = SAD, 2 = FEAR, 3 = HAPPY
        EMOTION_LABELS = ["NEUTRAL", "SAD", "FEAR", "HAPPY"]
        if dom_idx == 3: # HAPPY -> C Major (0) or G Major (7)
            base_key_offset = random.choice([0, 7])
        elif dom_idx == 1: # SAD -> D Minor (2) or F Minor (5)
            base_key_offset = random.choice([2, 5])
        elif dom_idx == 2: # FEAR -> C# Minor (1) or Eb Minor (3)
            base_key_offset = random.choice([1, 3])
        else: # NEUTRAL -> F Major (5) or A Minor (9)
            base_key_offset = random.choice([5, 9])
        print(f"[OfflineGenerator] Dynamic Key Set! Emotion: {EMOTION_LABELS[dom_idx]}, Offset: +{base_key_offset}")


    # shared root note across all modes for parallel cohesion
    fundamental_bass_root = 36 + base_key_offset
    
    tracker = EmotionTracker(window_size=10, spike_threshold=0.3)
    markov_engine = MarkovEngine()

    # MOTIF MEMORY (last generated 3-4 note pattern)
    motif_buffer = []

    current_chord_degree = 0
    consecutive_trill_count = 0      # anti-trill tracker
    spike_duration_counter = 0       # short spike tracking
    active_spike_profile = None      # current spike profile
    neutral_locked_mode = None       # NEUTRAL: lock Dorian/Mixolydian per passage

    for t, p in enumerate(emotions_array):
        # Update emotion tracker
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx]
        tracker.update_from_discrete(dominant_idx, intensity)
        
        state = tracker.get_state()
        v = state['macro_v']
        a = state['macro_a']
        is_spike = state['is_spike']
        spike_intensity = state['spike_intensity']
        spike_label = state['spike_label']
        macro_label = state['macro_label']

        # Secondary emotion for neutral "chameleon" behavior
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

        # SPIKE PROFILE MANAGEMENT
        if is_spike and macro_label != spike_label:
            spike_key = (macro_label, spike_label)
            active_spike_profile = SPIKE_PROFILES.get(spike_key)
            spike_duration_counter += 1
        else:
            spike_duration_counter = 0
            active_spike_profile = None

        # Map MACRO V/A to discrete emotional categories 
        # Valence = the "color" of the music (positive = major modes, negative = minor modes)
        # Arousal = the "intensity" of the music (high = fast, low = slow)
        if v > 0.0 and a > 0.0:
            emotion_cat = 'happy'
        elif v < 0.0 and a < 0.0:
            emotion_cat = 'sad'
        elif v < 0.0 and a >= 0.0:
            emotion_cat = 'fear'
        else:
            emotion_cat = 'neutral'

#--------------------------------------------------------------------------------------------------------------
        # ____________________
        #| 1. MODE SELECTION  |
        #|____________________|

        is_neutral = (emotion_cat == 'neutral')
        
        if is_neutral:
            # Mode selection for "chameleon" neutral (DORIAN = sad leaning, MIXOLYDIAN = happy leaning)
            if emotion_streak == 0 or neutral_locked_mode is None:
                if p[1] >= p[3]:  # sad probability >= happy probability
                    neutral_locked_mode = 'dorian' 
                else:
                    neutral_locked_mode = 'mixolydian'  
            current_mode = neutral_locked_mode
            chord_type = 'triad'
        elif emotion_cat == 'happy':
            current_mode = 'lydian' if v > 0.75 else 'ionian'
            chord_type = "triad"
        elif emotion_cat == 'sad':
            current_mode = 'aeolian'
            chord_type = "triad"
        else: # FEAR
            # Non-tertian voicings (no 3rd = no major/minor identity)
            if v > -0.4:
                current_mode = 'phrygian'        
            elif v > -0.7:
                current_mode = 'harmonic_minor'   
            else:
                current_mode = 'phrygian_dominant' 
            chord_type = "fear_open"  

        # Generate the pool of valid notes for the selected mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)

        melody_pool = pool
        register_shift = 0
#--------------------------------------------------------------------------------------------------------------
        # ____________________
        #| 2. DYNAMICS & TEMPO|
        #|____________________|

        # BPM scale: 60 to 140
        target_bpm = 100 + (a * 40)

        # Spike tempo modifier 
        if active_spike_profile and spike_intensity > 0:
            tempo_mult = active_spike_profile['tempo_mult']
            blended_mult = 1.0 + (tempo_mult - 1.0) * min(1.0, spike_intensity)
            target_bpm *= blended_mult

        # Exponential moving average
        if current_bpm is None:
            current_bpm = target_bpm
        else:
            current_bpm = 0.7 * current_bpm + 0.3 * target_bpm
            
        tempo = bpm2tempo(int(current_bpm))
        chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # ticks_per_step ensures 1 iteration = exactly 1 second
        ticks_per_step = int((current_bpm / 60.0) * ticks_per_beat)

        # Velocity mapping
        velocity = int(70 + (a * 40))
        velocity = max(30, min(110, velocity))

        # SAD/NEUTRAL velocity boost
        if macro_label == 'sad':
            velocity = max(50, min(110, velocity + 15))
        if emotion_cat == 'neutral':
            velocity = max(60, velocity)

        # Spike Velocity Modifier
        if active_spike_profile and spike_intensity > 0:
            # Graduated scaling
            if spike_intensity < 0.3:
                effect_scale = 0.3
            elif spike_intensity < 0.6:
                effect_scale = 0.6
            else:
                effect_scale = 1.0
            vel_shift = int(active_spike_profile['vel_shift'] * effect_scale)
            velocity = velocity + vel_shift
            velocity = max(30, min(110, velocity))

        # Rhythmic density based on arousal
        if a > 0.6:
            chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
        elif a > 0.0:
            chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
        else:
            chosen_rhythm_ratios = [1.0] if random.random() > 0.5 else [0.5, 0.5]


        # ANXIETY spike (happy→fear): rhythmic density doubling for nervous energy
        if active_spike_profile and active_spike_profile.get('name') == 'ANXIETY' and spike_intensity > 0.3:
            if chosen_rhythm_ratios == [1.0] or chosen_rhythm_ratios == [0.5, 0.5]:
                chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
            elif len(chosen_rhythm_ratios) == 3:
                chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]

        # SAD/NEUTRAL micro-timing humanization
        if emotion_cat in ('sad', 'neutral') and len(chosen_rhythm_ratios) > 1:
            jittered = []
            for i, r in enumerate(chosen_rhythm_ratios):
                jitter = random.uniform(-0.12, 0.12) * r
                jittered.append(r + jitter)
            # Normalize to preserve total duration
            total = sum(jittered)
            chosen_rhythm_ratios = [r / total for r in jittered]

        # Convert ratios to actual ticks
        chosen_rhythm = [int(r * ticks_per_step) for r in chosen_rhythm_ratios]
        # Adjust last note to fix rounding errors and ensure sum is exactly ticks_per_step
        if sum(chosen_rhythm) != ticks_per_step:
            chosen_rhythm[-1] += (ticks_per_step - sum(chosen_rhythm))

# ---------------------------------------------------------------------------------------------------------------
        # ___________________
        #|   3. CHORDS &    |
        #|   ACCOMPANIMENT  |
        #|__________________|

        # Harmonic Rhythm
        if a > 0.6:
            harmonic_rhythm = random.choice([1, 2])
        elif a > 0.0:
            harmonic_rhythm = 2
        else:
            harmonic_rhythm = 4

        # SAD/NEUTRAL chord variety
        if emotion_cat in ('sad', 'neutral') and emotion_streak > 2:
            harmonic_rhythm = min(harmonic_rhythm, 2) 
        # NEUTRAL: more chord movement
        if emotion_cat == 'neutral':
            harmonic_rhythm = min(harmonic_rhythm, 2)
        # FEAR: keep chords moving
        if emotion_cat == 'fear':
            harmonic_rhythm = min(harmonic_rhythm, 2)

        # Advance chord via Markov dice roll
        if t == 0 or emotion_streak == 0:
            current_chord_degree = 0
        elif emotion_streak % harmonic_rhythm == 0:
            # Time to change the chord — look up the transition matrix
            if emotion_cat == 'neutral':
                matrix_key = 'neutral_dorian' if current_mode == 'dorian' else 'neutral_mixolydian'
            else:
                matrix_key = emotion_cat
            matrix = CHORD_TRANSITIONS.get(matrix_key, {})
            if current_chord_degree in matrix:
                options = matrix[current_chord_degree]['options']
                weights = matrix[current_chord_degree]['weights']
                current_chord_degree = random.choices(options, weights=weights, k=1)[0]
            else:
                current_chord_degree = 0

        # FEAR: ~30% of the time when intensity > 0.90, force a move to iv, bVI, or bvii
        if emotion_cat == 'fear' and intensity > 0.90:
            if random.random() < 0.30:
                current_chord_degree = random.choice([3, 5, 6])

        base_chord_pool_idx = 14
        chord_root_idx = base_chord_pool_idx + current_chord_degree

        # Diatonic mapping
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "sus4":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 3], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        elif chord_type == "fear_open":
            # Non-tertian: avoids major/minor chord identity
            # PHRYGIAN: adds dissonance
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
        elif chord_type == "cinematic_open":
            # Cinematic modal: triad + mode signature note
            # DORIAN: minor add6 (m3 + M6)
            # MIXOLYDIAN: major add♭7 (M3 + ♭7)
            if current_mode == 'dorian':
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
            else:  # mixolydian
                flat7 = pool[chord_root_idx + 6] if (chord_root_idx + 6) < len(pool) else min(127, pool[chord_root_idx] + 10)
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], flat7]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # HARMONIC MINOR Alteration: Tension chord (Major V) for SAD emotion
        if emotion_cat == 'sad' and current_chord_degree == 4 and chord_type == "triad":
            chord_notes[1] += 1  # raises minor 3rd to major 3rd

        # Emotion-specific chord velocity
        if emotion_cat == 'sad':
            chord_vel = max(40, velocity + 5)
        elif emotion_cat == 'neutral':
            chord_vel = max(35, min(70, velocity - 10))
        else:
            chord_vel = max(20, velocity - 10)

        # Apply spike chord coloring (graduated by spike_intensity)
        if active_spike_profile and spike_intensity > 0:
            # COURAGE spike profile override
            if active_spike_profile.get('chord_color') == 'epic_modal' and spike_intensity > 0.6:
                epic_sequence = [0, 5, 6] 
                current_chord_degree = epic_sequence[emotion_streak % 3]
                chord_root_idx = base_chord_pool_idx + current_chord_degree
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
            # Applied chord coloring at any spike intensity (subtle at low, full at high)
            chord_notes = apply_spike_chord_color(chord_notes, active_spike_profile['chord_color'])

            # UNEASE spike profile override
            spike_name = active_spike_profile.get('name', '')
            if spike_name in ('UNEASE',) and spike_intensity > 0.5:
                if chord_notes[0] > 36:  
                    chord_notes[0] = chord_notes[0] - 12
                ghost_note = min(127, chord_notes[0] + 1)
                ghost_vel = max(15, int(chord_vel * 0.25))  
                chord_track.append(Message('note_on', note=int(ghost_note), velocity=ghost_vel, time=0))
                chord_track.append(Message('note_off', note=int(ghost_note), velocity=0, time=int(ticks_per_step // 2)))

        # Spike rest probability override for melody
        if active_spike_profile and spike_intensity > 0.3:
            spike_rest_prob = active_spike_profile['rest_prob'] * min(1.0, spike_intensity / 0.6)
        else:
            spike_rest_prob = None
        
        # --- ACCOMPANIMENT STYLES ---
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


# ---------------------------------------------------------------------------------------------------------------------
        
        # ______________________
        #| 4. MELODY GENERATION |
        #|______________________|

        # 40 % of the time: replay the saved motif (transposed ±1-2 scale degrees).
        # 60 % of the time: generate a fresh contour phrase and save it.

        melody_notes_and_durations = []
        
        # Create a mutable copy of the pool for this chunk
        active_pool = list(melody_pool)  
        
        # HARMONIC MINOR alteration to SAD macromood
        if emotion_cat == 'sad' and current_chord_degree == 4:
            for idx in range(len(active_pool)):
                if (idx % 7) == 6:
                    active_pool[idx] += 1

        def _pool_idx_nearest(target, p):
            """Return index of note in pool p closest to target pitch."""
            return min(range(len(p)), key=lambda k: abs(p[k] - target))

        use_motif = len(motif_buffer) > 0 and random.random() < 0.40

        if use_motif:
            # Use saved motif transposed ±1-2 scale degrees
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, duration in motif_buffer:
                new_deg = melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note    = int(active_pool[new_deg]) + register_shift

                # Apply spike melody register shift
                if active_spike_profile and spike_intensity > 0.3:
                    note += int(active_spike_profile['melody_register'] * min(1.0, spike_intensity))

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12
                    
                note    = max(0, min(127, note))
                
                # Universal Dissonance Guard (applies to all emotions but FEAR)
                if emotion_cat in ('happy', 'sad', 'neutral'):
                    note = resolve_dissonance(note, chord_notes)
                    
                melody_notes_and_durations.append((note, duration))
        else:
            # Generate a fresh musical phrase using VGMIDI Markov Chain
            chosen_contour = []
            prev_intervals = [0, 0, 0]  # empty history
            
            for _ in range(len(chosen_rhythm)):
                next_interval = markov_engine.query_next_interval(emotion_cat, prev_intervals)
                chosen_contour.append(next_interval)
                prev_intervals.pop(0)
                prev_intervals.append(next_interval)
                
            new_motif = []

            # Harmonic adherence: MACRO VALENCE anchors to chord, MICRO VALENCE nudges
            chord_adherence_prob = max(0.4, min(0.95, 0.7 + (v * 0.20) + (state['micro_v'] * 0.10)))
            # HAPPY/SAD: higher floor to prevent dissonant clashes
            if emotion_cat in ('happy', 'sad'):
                chord_adherence_prob = max(0.80, chord_adherence_prob)
            # NEUTRAL: strong adherence to expose the modal chord color (add6/add9)
            if emotion_cat == 'neutral':
                chord_adherence_prob = max(0.80, chord_adherence_prob)
            # FEAR: high adherence so melody follows the dark non-tertian chords
            if emotion_cat == 'fear':
                chord_adherence_prob = max(0.75, chord_adherence_prob)
            # Spike transitions: boost adherence so melody follows the spike chord coloring
            if active_spike_profile and spike_intensity > 0.3:
                chord_adherence_prob = max(0.90, chord_adherence_prob)

            for i, duration in enumerate(chosen_rhythm):
                if i < len(chosen_contour):
                    melody_idx += chosen_contour[i]
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                # FEAR: lower register (C3-C5 range) with descending bias
                if emotion_cat == 'fear':
                    melody_idx = max(14, min(len(active_pool) - 1, melody_idx))
                    if random.random() < 0.30 and melody_idx > 16:
                        melody_idx -= 1

                # Snap to a chord tone for harmonic grounding
                if random.random() < chord_adherence_prob:
                    # Snap to safe triad tones only (Root, 3rd, 5th)
                    safe_snap_notes = chord_notes[:3]
                    # NEUTRAL: prefer non-root tones for melodic variety
                    if emotion_cat == 'neutral' and len(safe_snap_notes) > 1 and random.random() < 0.4:
                        safe_snap_notes = safe_snap_notes[1:]
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    # FEAR: snap to lower octave for darker register
                    if emotion_cat == 'fear':
                        target_note = random.choice(safe_snap_notes) + random.choice([0, 12])
                    melody_idx  = _pool_idx_nearest(target_note, active_pool)
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                note = int(active_pool[melody_idx]) + register_shift

                # Micro valence expression
                if state['micro_v'] > 0.4 and random.random() < 0.35:
                    note += 12  # Octave jump

                # Apply spike melody register shift
                if active_spike_profile and spike_intensity > 0.3:
                    note += int(active_spike_profile['melody_register'] * min(1.0, spike_intensity))

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

                note = max(0, min(127, note))

                # Motifs are recorded as scale degree offsets from the starting melody_idx
                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, duration))

                # Rest probabilities (pauses in the music)
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

            # Save the new phrase to the motif buffer (replace old one)
            motif_buffer[:] = new_motif

        # Anti-trill guard (applies to all emotions)
        if detect_trill(melody_notes_and_durations):
            consecutive_trill_count += 1
        else:
            consecutive_trill_count = 0

        if consecutive_trill_count >= 1:
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
                # Fear melody: slightly lower velocity for building tension
                if emotion_cat == 'fear':
                    mel_vel = max(45, min(85, velocity - 5))
                else:
                    mel_vel = velocity
                melody_track.append(Message('note_on',  note=int(note), velocity=int(mel_vel), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0,              time=int(duration)))

    mid.save(filename)
    print(f"Saved Final Cohesive MIDI: {filename}")
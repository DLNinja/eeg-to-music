
# harmony.py — Chord Progression, Voicing & Accompaniment Helpers

import random
from .music_theory import (
    CHORD_TRANSITIONS, apply_spike_chord_color,
    get_mode_pool, get_mode_intervals
)


# ────────────────────────────────────────────────────────────────────────
# MODE SELECTION
# ────────────────────────────────────────────────────────────────────────

def select_mode(emotion_cat, macro_v, p, emotion_streak, neutral_locked_mode):

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
        current_mode = 'lydian' if macro_v > 0.75 else 'ionian'
        chord_type = "triad"
    elif emotion_cat == 'sad':
        current_mode = 'aeolian'
        chord_type = "triad"
    else: # FEAR
        # Non-tertian voicings (no 3rd = no major/minor identity)
        if macro_v > -0.4:
            current_mode = 'phrygian'         
        elif macro_v > -0.7:
            current_mode = 'harmonic_minor'   
        else:
            current_mode = 'phrygian_dominant' 
        chord_type = "fear_open"  

    return current_mode, chord_type, neutral_locked_mode



# ────────────────────────────────────────────────────────────────────────
# HARMONIC RHYTHM
# ────────────────────────────────────────────────────────────────────────

def compute_harmonic_rhythm(macro_a, emotion_cat, emotion_streak):
    
    # Harmonic Rhythm
    if macro_a > 0.6:
        harmonic_rhythm = random.choice([1, 2])
    elif macro_a > 0.0:
        harmonic_rhythm = 2
    else:
        harmonic_rhythm = 4

    if emotion_cat in ('sad', 'neutral') and emotion_streak > 2:
        harmonic_rhythm = min(harmonic_rhythm, 2) 
    if emotion_cat == 'neutral':
        harmonic_rhythm = min(harmonic_rhythm, 2)
    if emotion_cat == 'fear':
        harmonic_rhythm = min(harmonic_rhythm, 2)

    return harmonic_rhythm


# ────────────────────────────────────────────────────────────────────────
# CHORD DEGREE ADVANCEMENT (Markov Transition Matrix)
# ────────────────────────────────────────────────────────────────────────

def advance_chord_degree(emotion_cat, current_mode, emotion_streak,
                         harmonic_rhythm, current_chord_degree, is_first_step):
    if is_first_step or emotion_streak == 0:
        return 0
    elif emotion_streak % harmonic_rhythm == 0:
        if emotion_cat == 'neutral':
            matrix_key = 'neutral_dorian' if current_mode == 'dorian' else 'neutral_mixolydian'
        else:
            matrix_key = emotion_cat
        matrix = CHORD_TRANSITIONS.get(matrix_key, {})
        if current_chord_degree in matrix:
            options = matrix[current_chord_degree]['options']
            weights = matrix[current_chord_degree]['weights']
            return random.choices(options, weights=weights, k=1)[0]
        else:
            return 0
    return current_chord_degree


# ────────────────────────────────────────────────────────────────────────
# CHORD NOTE CONSTRUCTION (Diatonic Voicings)
# ────────────────────────────────────────────────────────────────────────

def build_chord_notes(pool, chord_root_idx, chord_type, current_mode):

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
        chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
    elif chord_type == "cinematic_open":
        # Neutral chord structures for DORIAN and MIXOLYDIAN modes
        if current_mode == 'dorian':
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                           pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
        else:  # mixolydian
            flat7 = pool[chord_root_idx + 6] if (chord_root_idx + 6) < len(pool) else min(127, pool[chord_root_idx] + 10)
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                           pool[chord_root_idx + 4], flat7]
    else:
        chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
    return chord_notes


def apply_harmonic_minor_chord(chord_notes, emotion_cat, current_chord_degree, chord_type):
    if emotion_cat == 'sad' and current_chord_degree == 4 and chord_type == "triad":
        chord_notes[1] += 1  # raises minor 3rd to major 3rd
    return chord_notes


# ────────────────────────────────────────────────────────────────────────
# SPIKE CHORD OVERRIDES
# ────────────────────────────────────────────────────────────────────────

def apply_chord_spike_overrides(pool, base_chord_pool_idx, chord_notes,
                                spike_chord_color, spike_name, spike_intensity,
                                emotion_streak, current_chord_degree):
    if spike_chord_color:
        # COURAGE spike chord coloring
        if spike_chord_color == 'epic_modal' and spike_intensity > 0.6:
            epic_sequence = [0, 5, 6]  
            current_chord_degree = epic_sequence[emotion_streak % 3]
            chord_root_idx = base_chord_pool_idx + current_chord_degree
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        # Applied chord coloring at any spike intensity (subtle at low, full at high)
        chord_notes = apply_spike_chord_color(chord_notes, spike_chord_color)

    return chord_notes, current_chord_degree

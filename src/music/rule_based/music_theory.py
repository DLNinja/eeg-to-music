
# music_theory.py — Scale & Chord Foundation

import random

# CHORD TRANSITION MATRIX 
CHORD_TRANSITIONS = {
    'happy': {
        # Lydian/Ionian
        0: {'options': [0, 1, 3, 4, 5], 'weights': [10, 10, 30, 30, 20]},
        1: {'options': [4, 5, 0],       'weights': [60, 20, 20]},
        2: {'options': [3, 5],          'weights': [50, 50]},
        3: {'options': [0, 4, 1],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [70, 30]},
        5: {'options': [3, 1, 0],       'weights': [50, 30, 20]},
        6: {'options': [0, 5],          'weights': [80, 20]}
    },
    'sad': {
        # Aeolian
        0: {'options': [0, 3, 5, 6],    'weights': [8, 25, 33, 34]},
        1: {'options': [4, 6],          'weights': [70, 30]},
        2: {'options': [5, 3],          'weights': [60, 40]},
        3: {'options': [0, 4, 6],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [60, 40]},
        5: {'options': [3, 6, 0],       'weights': [40, 40, 20]},
        6: {'options': [0, 2, 5],       'weights': [50, 20, 30]}
    },
    'fear': {
        # Phrygian
        0: {'options': [1, 6],          'weights': [60, 40]},
        1: {'options': [0, 2],          'weights': [70, 30]},
        2: {'options': [0, 1],          'weights': [45, 55]},
        3: {'options': [0, 1],          'weights': [35, 65]},
        4: {'options': [1, 0],          'weights': [70, 30]},
        5: {'options': [1, 6],          'weights': [60, 40]},
        6: {'options': [5, 0],          'weights': [65, 35]}        
    },
    'neutral_dorian': {
        # Dorian
        0: {'options': [3, 6, 1],       'weights': [55, 30, 15]},
        1: {'options': [3, 0],          'weights': [65, 35]},
        2: {'options': [6, 3],          'weights': [60, 40]},
        3: {'options': [0, 6, 1, 5],    'weights': [40, 30, 15, 15]},
        4: {'options': [3, 0],          'weights': [60, 40]},
        5: {'options': [6, 3, 0],       'weights': [45, 35, 20]},
        6: {'options': [0, 3, 5],       'weights': [40, 35, 25]}
    },
    'neutral_mixolydian': {
        # Mixolydian
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
SPIKE_PROFILES = {
    ('happy', 'sad'):     {'name': 'BITTERSWEET',  'tempo_mult': 0.85, 'vel_shift': -15, 'chord_color': 'add_minor_3rd',       'melody_register': -6, 'rest_prob': 0.2},
    ('happy', 'neutral'): {'name': 'SERENITY',     'tempo_mult': 0.75, 'vel_shift': -20, 'chord_color': 'sus2',                'melody_register': 0,  'rest_prob': 0.35},
    ('happy', 'fear'):    {'name': 'ANXIETY',       'tempo_mult': 1.30, 'vel_shift': +15, 'chord_color': 'anxious_creep',       'melody_register': 0,  'rest_prob': 0.0},
    ('sad', 'happy'):     {'name': 'HOPE',          'tempo_mult': 1.10, 'vel_shift': +15, 'chord_color': 'major_lift',          'melody_register': +6, 'rest_prob': 0.0},
    ('sad', 'neutral'):   {'name': 'ACCEPTANCE',    'tempo_mult': 0.90, 'vel_shift': -5,  'chord_color': 'picardy_lift',        'melody_register': 0,  'rest_prob': 0.25},
    ('sad', 'fear'):      {'name': 'DISTURBED',     'tempo_mult': 1.05, 'vel_shift': +8,  'chord_color': 'disturbed_tension',   'melody_register': 0,  'rest_prob': 0.1},
    ('fear', 'happy'):    {'name': 'COURAGE',       'tempo_mult': 1.20, 'vel_shift': +25, 'chord_color': 'epic_modal',          'melody_register': +6, 'rest_prob': 0.0},
    ('fear', 'sad'):      {'name': 'DESOLATION',    'tempo_mult': 0.70, 'vel_shift': +10, 'chord_color': 'hollow_madd9',        'melody_register': -3, 'rest_prob': 0.15},
    ('fear', 'neutral'):  {'name': 'RELIEF',        'tempo_mult': 0.75, 'vel_shift': -15, 'chord_color': 'resolve_major',       'melody_register': 0,  'rest_prob': 0.30},
    ('neutral', 'happy'): {'name': 'AWAKENING',     'tempo_mult': 1.15, 'vel_shift': +15, 'chord_color': 'bright_triad',        'melody_register': +3, 'rest_prob': 0.0},
    ('neutral', 'sad'):   {'name': 'MELANCHOLY',    'tempo_mult': 0.90, 'vel_shift': -10, 'chord_color': 'minor_color',         'melody_register': -3, 'rest_prob': 0.2},
    ('neutral', 'fear'):  {'name': 'UNEASE',        'tempo_mult': 1.10, 'vel_shift': +12, 'chord_color': 'suspended_tension',   'melody_register': +3, 'rest_prob': 0.05},
}


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
        'harmonic_minor':    [0, 2, 3, 5, 7, 8, 11],  # FEAR (also used for SAD harmonic minor chord alteration)
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

def select_key_offset(dominant_idx): 
    # Dynamic Key Selection (Schubert) -> pick a musically appropriate key

    if dominant_idx == 3:    # HAPPY -> C Major (0) or G Major (7)
        return random.choice([0, 7])
    elif dominant_idx == 1:  # SAD -> D Minor (2) or F Minor (5)
        return random.choice([2, 5])
    elif dominant_idx == 2:  # FEAR -> C# Minor (1) or Eb Minor (3)
        return random.choice([1, 3])
    else:                    # NEUTRAL -> F Major (5) or A Minor (9)
        return random.choice([5, 9])


def apply_spike_chord_color(chord_notes, color_type):
    # Apply spike-specific chord coloring to modify existing chord notes.

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

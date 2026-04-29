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
        # Root self-loop reduced (20→8) to prevent stagnation at slow tempos
        0: {'options': [0, 3, 5, 6],    'weights': [8, 25, 33, 34]},
        1: {'options': [4, 6],          'weights': [70, 30]},
        2: {'options': [5, 3],          'weights': [60, 40]},
        3: {'options': [0, 4, 6],       'weights': [40, 40, 20]},
        4: {'options': [0, 5],          'weights': [60, 40]},
        5: {'options': [3, 6, 0],       'weights': [40, 40, 20]},
        6: {'options': [0, 2, 5],       'weights': [50, 20, 30]}
    },
    'fear': {
        # Phrygian: i↔bII oscillation is the backbone, bvii→bVI descent is secondary
        # Fewer options + heavy weights = clear directional momentum
        0: {'options': [1, 6],          'weights': [60, 40]},       # i → bII (horror!) or bvii (descent)
        1: {'options': [0, 2],          'weights': [70, 30]},       # bII → i (loop back) or bIII (mysterious)
        2: {'options': [0, 1],          'weights': [45, 55]},       # bIII → bII or i
        3: {'options': [0, 1],          'weights': [35, 65]},       # iv → bII (pull toward tension)
        4: {'options': [1, 0],          'weights': [70, 30]},       # v° → bII (tension → dread)
        5: {'options': [1, 6],          'weights': [60, 40]},       # bVI → bII (doom descent end) or bvii
        6: {'options': [5, 0],          'weights': [65, 35]}        # bvii → bVI (doom descent) or i
    },
    'neutral_dorian': {
        # THE FORCE THEME (Binary Sunset) — John Williams
        # Dorian signature: i → IV is THE moment. The major IV chord in a minor context
        # creates the spiritual, noble, yearning quality. The raised 6th is the magic.
        # Chains: i → IV → i (yearning oscillation), i → IV → ♭VII → i (noble arc)
        0: {'options': [3, 6, 1],       'weights': [55, 30, 15]},      # i → IV (the lift!) or ♭VII or ii
        1: {'options': [3, 0],          'weights': [65, 35]},          # ii → IV (pull toward the light) or i
        2: {'options': [6, 3],          'weights': [60, 40]},          # ♭III → ♭VII or IV
        3: {'options': [0, 6, 1, 5],    'weights': [40, 30, 15, 15]},  # IV → i (yearning return) or ♭VII or ii or ♭VI
        4: {'options': [3, 0],          'weights': [60, 40]},          # v → IV (toward the lift) or i
        5: {'options': [6, 3, 0],       'weights': [45, 35, 20]},      # ♭VI → ♭VII or IV or i
        6: {'options': [0, 3, 5],       'weights': [40, 35, 25]}       # ♭VII → i (resolution) or IV or ♭VI
    },
    'neutral_mixolydian': {
        # JOURNEY TO THE ISLAND (Jurassic Park) — John Williams
        # Mixolydian signature: I → ♭VII is THE moment. The flat 7th takes the sweetness
        # out of major and replaces it with prehistoric scale and power.
        # Chains: I → ♭VII → IV → ♭VII → ... (long exploration before returning home)
        0: {'options': [6, 3],          'weights': [55, 45]},          # I → ♭VII (the grandeur drop!) or IV
        1: {'options': [3, 6, 0],       'weights': [50, 35, 15]},      # ii → IV or ♭VII (keeps moving, root rare)
        2: {'options': [6, 3],          'weights': [60, 40]},          # ♭III → ♭VII or IV
        3: {'options': [6, 5, 1, 0],    'weights': [40, 25, 20, 15]},  # IV → ♭VII (continuation!) or ♭VI or ii or I(rare)
        4: {'options': [3, 6, 0],       'weights': [45, 40, 15]},      # v → IV or ♭VII (root rare)
        5: {'options': [6, 3, 0],       'weights': [55, 35, 10]},      # ♭VI → ♭VII (grandeur) or IV (root rare)
        6: {'options': [3, 5, 1, 0],    'weights': [35, 30, 20, 15]}   # ♭VII → IV or ♭VI or ii (root rare, long chain)
    }
}

# ── SPIKE TRANSITION PROFILES ──
# Maps (macro_emotion, spike_emotion) → musical modifiers for spike moments.
# Each profile defines how the music is temporarily colored during a spike transition.
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
        # Replace major 3rd with minor 3rd (lower by 1 semitone) → bittersweet
        if len(colored) > 1:
            colored[1] = colored[1] - 1
    elif color_type == 'sus2':
        # Replace 3rd with 2nd → ethereal calm
        if len(colored) > 1:
            colored[1] = colored[0] + 2
    elif color_type == 'suspended_tension':
        # Sus4: Replace 3rd with Perfect 4th → unresolved yearning without tritone harshness
        # [Root, P4, P5] — the classic "something's about to happen" sound
        colored = [colored[0], colored[0] + 5, colored[0] + 7]
    elif color_type == 'anxious_creep':
        # Anxiety = restlessness + a fear sprinkle. Keep the happy chord intact,
        # drop the root one octave for weight, and add a minor 7th (dominant 7th)
        # for that unresolved "something is slightly off" tension.
        # The b7 wants to resolve but never does — perfect anxious restlessness.
        bass = colored[0] - 12 if colored[0] > 36 else colored[0]
        dom7 = min(127, colored[0] + 10)  # minor 7th = 10 semitones above root
        colored = [bass] + colored[1:] + [dom7]
    elif color_type == 'major_lift':
        # Force major triad → hopeful lift
        colored = [colored[0], min(127, colored[0] + 4), min(127, colored[0] + 7)]
    elif color_type == 'picardy_lift':
        # Picardy Third: Force MAJOR triad in a minor context.
        # Open voicing: Root, P5, Major 3rd (octave up) for spacious cinematic feel.
        # Melody dissonance guard auto-snaps to these notes — no clash possible.
        colored = [colored[0], colored[0] + 7, colored[0] + 4 + 12]
    elif color_type == 'phrygian_shadow':
        # Phrygian Shadow: flatten the 2nd degree → b9 tension against root
        # Creates suffocating, claustrophobic feel without harsh tritones
        # Subtle: just the flattened 2nd, no extra dissonant voice
        if len(colored) > 1:
            colored[1] = colored[1] - 1  # Flatten toward b2/b9
    elif color_type == 'disturbed_tension':
        # Keep the original sad chord intact, drop the bass one octave,
        # and add a quiet ♭6 (half-step above the 5th) an octave up
        # for subtle, non-abrasive tension — like a shadow creeping in.
        bass = colored[0] - 12 if colored[0] > 36 else colored[0]
        tension_note = min(127, colored[0] + 8 + 12)  # ♭6 an octave up (gentle)
        colored = [bass] + colored[1:] + [tension_note]
    elif color_type == 'add_quiet_dissonance':
        # Add half-step above root → subtle disturbance
        colored.append(min(127, colored[0] + 1))
    elif color_type == 'epic_modal':
        # Epic Modal: Open 5th power chord (Root + P5 + Root octave up)
        # No 3rd = massive, emotionally ambiguous brass sound
        colored = [colored[0], colored[0] + 7, colored[0] + 12]
    elif color_type == 'hollow_madd9':
        # Hollow Minor Add 9: Root + P5 + (minor 3rd + 9th clustered an octave up)
        # The 9th (14 semitones) sits right next to minor 3rd (15 semitones)
        # creating the painful "ache" rub. Open spacing = emptiness.
        colored = [colored[0], colored[0] + 7, colored[0] + 14, colored[0] + 15]
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
        'lydian':            [0, 2, 4, 6, 7, 9, 11],
        'ionian':            [0, 2, 4, 5, 7, 9, 11],
        'mixolydian':        [0, 2, 4, 5, 7, 9, 10],
        'dorian':            [0, 2, 3, 5, 7, 9, 10],
        'aeolian':           [0, 2, 3, 5, 7, 8, 10],
        'phrygian':          [0, 1, 3, 5, 7, 8, 10],
        'locrian':           [0, 1, 3, 5, 6, 8, 10],
        'harmonic_minor':    [0, 2, 3, 5, 7, 8, 11],  # Augmented 2nd (♭6→M7) = gothic, eerie
        'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10],  # ♭2 + M3 = alien, extreme
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
    neutral_locked_mode = None       # Neutral: lock Dorian/Mixolydian per passage

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
            # Lock mode at start of each neutral passage based on sad vs happy probability
            if emotion_streak == 0 or neutral_locked_mode is None:
                if p[1] >= p[3]:  # sad probability >= happy probability
                    neutral_locked_mode = 'dorian'   # Melancholic but dignified
                else:
                    neutral_locked_mode = 'mixolydian'  # Wonder and discovery
            current_mode = neutral_locked_mode
            # Pure triads only — modal character comes from the scale and progression,
            # not from fancy voicings. Force Theme and Jurassic Park are powerful
            # BECAUSE of simple, clear triads that let the chord movement speak.
            chord_type = 'triad'
        elif emotion_cat == 'happy':
            current_mode = 'lydian' if v > 0.75 else 'ionian'
            chord_type = "triad"
        elif emotion_cat == 'sad':
            current_mode = 'aeolian'
            chord_type = "triad"
        else: # fear
            # All fear uses non-tertian voicings (no 3rd = no major/minor identity)
            if v > -0.4:
                current_mode = 'phrygian'        # ♭2 = brooding, oppressive, claustrophobic
            elif v > -0.7:
                current_mode = 'harmonic_minor'   # Augmented 2nd = gothic, eerie, Dracula
            else:
                current_mode = 'phrygian_dominant' # ♭2 + M3 = alien, extreme, snake-charmer
            chord_type = "fear_open"  # Root + P5 + 6th (non-tertian, avoids happy/sad territory)

        # Generate the pool of valid notes for the selected mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)

        melody_pool = pool
        register_shift = 0
        
        # --------------------------------------------------------------------------
        # 2. DYNAMICS & TEMPO (Mapped to Arousal)
        # BPM scale: -1.0 (60) to 1.0 (140)
        target_bpm = 100 + (a * 40)

        # Apply spike tempo modifier (graduated by spike_intensity)
        if active_spike_profile and spike_intensity > 0:
            # Graduated: scale tempo multiplier by spike intensity
            tempo_mult = active_spike_profile['tempo_mult']
            # Blend toward profile tempo based on intensity (subtle at low, full at high)
            blended_mult = 1.0 + (tempo_mult - 1.0) * min(1.0, spike_intensity)
            target_bpm *= blended_mult

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
        # Neutral needs enough presence to feel emotionally engaged (matched closer to sad)
        if emotion_cat == 'neutral':
            velocity = max(60, velocity)

        # Apply spike velocity modifier (graduated by spike_intensity)
        if active_spike_profile and spike_intensity > 0:
            # Graduated scaling: 0-30% → 30% effect, 30-60% → 60% effect, 60-100% → full effect
            if spike_intensity < 0.3:
                effect_scale = 0.3
            elif spike_intensity < 0.6:
                effect_scale = 0.6
            else:
                effect_scale = 1.0
            vel_shift = int(active_spike_profile['vel_shift'] * effect_scale)
            velocity = velocity + vel_shift
            velocity = max(30, min(110, velocity))

        # RHYTHMIC DENSITY based on Arousal
        if a > 0.6:
            chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
        elif a > 0.0:
            chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
        else:
            chosen_rhythm_ratios = [1.0] if random.random() > 0.5 else [0.5, 0.5]

        # ANXIETY spike (happy→fear): rhythmic density doubling for nervous energy
        if active_spike_profile and active_spike_profile.get('name') == 'ANXIETY' and spike_intensity > 0.3:
            # Jump to the next-faster rhythm category
            if chosen_rhythm_ratios == [1.0] or chosen_rhythm_ratios == [0.5, 0.5]:
                chosen_rhythm_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
            elif len(chosen_rhythm_ratios) == 3:
                chosen_rhythm_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]

        # Sad & Neutral micro-timing humanization: ±5-12% timing jitter on note durations
        # Breaks the mechanical grid feel at slower tempos
        if emotion_cat in ('sad', 'neutral') and len(chosen_rhythm_ratios) > 1:
            jittered = []
            for i, r in enumerate(chosen_rhythm_ratios):
                jitter = random.uniform(-0.12, 0.12) * r
                jittered.append(r + jitter)
            # Normalize to preserve total duration
            total = sum(jittered)
            chosen_rhythm_ratios = [r / total for r in jittered]

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
        # Neutral: more chord movement — never hold more than 2 steps
        if emotion_cat == 'neutral':
            harmonic_rhythm = min(harmonic_rhythm, 2)
        # Fear: keep chords moving at a steady moderate pace
        if emotion_cat == 'fear':
            harmonic_rhythm = min(harmonic_rhythm, 2)

        # Advance chord via Markov dice roll
        if t == 0 or emotion_streak == 0:
            # On start or emotion shift, force a safe root
            current_chord_degree = 0
        elif emotion_streak % harmonic_rhythm == 0:
            # Time to change the chord — look up the transition matrix
            # Neutral uses mode-specific matrices for distinct character
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
                current_chord_degree = 0  # Fallback

        # Fear near 100%: enhanced chord movement to prevent stagnation
        # ~30% of the time when intensity > 0.90, force a move to iv, bVI, or bvii
        if emotion_cat == 'fear' and intensity > 0.90:
            if random.random() < 0.30:
                # Break the i↔bII oscillation with doom descent or tension pull
                current_chord_degree = random.choice([3, 5, 6])  # iv, bVI, bvii

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
        elif chord_type == "fear_open":
            # Non-tertian: Root + P5 + 6th degree (skips the 3rd entirely)
            # Avoids major/minor chord identity — sounds heavy, ambiguous, dread-inducing
            # Phrygian: [root, P5, ♭6] creates semitone rub (B↔C). Harmonic Minor: exposes aug 2nd
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
        elif chord_type == "cinematic_open":
            # Cinematic modal: triad + mode signature note for distinct character
            # Dorian: minor add6 (m3 + M6) — warm nostalgia, spiritual, noble (Force Theme)
            # Mixolydian: major add♭7 (M3 + ♭7) — prehistoric power, grandeur (Jurassic Park)
            if current_mode == 'dorian':
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], pool[chord_root_idx + 5]]
            else:  # mixolydian
                # ♭7 as color note instead of 9th — exposes the Mixolydian signature directly
                # Creates a dominant-quality tonic (power without resolution need)
                flat7 = pool[chord_root_idx + 6] if (chord_root_idx + 6) < len(pool) else min(127, pool[chord_root_idx] + 10)
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2],
                               pool[chord_root_idx + 4], flat7]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]

        # Harmonic Minor Alteration: Tension chord (Major V) for Sad emotion
        if emotion_cat == 'sad' and current_chord_degree == 4 and chord_type == "triad":
            chord_notes[1] += 1  # Raise minor 3rd to Major 3rd

        # Emotion-specific chord velocity
        if emotion_cat == 'sad':
            chord_vel = max(40, velocity + 5)  # Sad needs more presence
        elif emotion_cat == 'neutral':
            chord_vel = max(35, min(70, velocity - 10))  # C418 delicate touch — soft, contemplative
        else:
            chord_vel = max(20, velocity - 10)

        # Apply spike chord coloring (graduated by spike_intensity)
        if active_spike_profile and spike_intensity > 0:
            # EPIC MODAL override: force heroic i→bVI→bVII cycle before coloring
            if active_spike_profile.get('chord_color') == 'epic_modal' and spike_intensity > 0.6:
                epic_sequence = [0, 5, 6]  # i → bVI → bVII heroic loop
                current_chord_degree = epic_sequence[emotion_streak % 3]
                chord_root_idx = base_chord_pool_idx + current_chord_degree
                # Rebuild chord from the forced epic degree
                chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
            # Apply chord coloring at any spike intensity (subtle at low, full at high)
            chord_notes = apply_spike_chord_color(chord_notes, active_spike_profile['chord_color'])

            # INTO-FEAR transitions: drop bass + half-step ghost note for darkness
            # Only for spikes transitioning INTO fear (ANXIETY, DISTURBED, UNEASE)
            spike_name = active_spike_profile.get('name', '')
            if spike_name in ('UNEASE',) and spike_intensity > 0.5:
                # Drop the bass note one octave for weight/darkness
                if chord_notes[0] > 36:  # Don't drop below C2
                    chord_notes[0] = chord_notes[0] - 12
                # Ghost note: minor 2nd above root, very quiet, first half of step only
                ghost_note = min(127, chord_notes[0] + 1)
                ghost_vel = max(15, int(chord_vel * 0.25))  # ~25% of chord velocity
                chord_track.append(Message('note_on', note=int(ghost_note), velocity=ghost_vel, time=0))
                chord_track.append(Message('note_off', note=int(ghost_note), velocity=0, time=int(ticks_per_step // 2)))
                # The ghost note ends at the half-step mark; the main chord continues below

        # Spike rest probability override for melody (graduated)
        if active_spike_profile and spike_intensity > 0.3:
            spike_rest_prob = active_spike_profile['rest_prob'] * min(1.0, spike_intensity / 0.6)
        else:
            spike_rest_prob = None

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
            
            # Simple subtle arpeggio (~10%): bass note first, then each voice enters in quick succession
            if random.random() < 0.10:
                roll_voices = [root, fifth, third]
                roll_gap = int(ticks_per_step * 0.03)  # ~3% of step between each note entry
                for vi, n in enumerate(roll_voices):
                    vel_taper = max(30, chord_vel - (vi * 3))
                    chord_track.append(Message('note_on', note=n, velocity=vel_taper, time=(roll_gap if vi > 0 else 0)))
                
                # All notes off together at end of step
                sustain_remaining = int(ticks_per_step - roll_gap * (len(roll_voices) - 1))
                chord_track.append(Message('note_off', note=roll_voices[0], velocity=0, time=sustain_remaining))
                for n in roll_voices[1:]:
                    chord_track.append(Message('note_off', note=n, velocity=0, time=0))
            else:
                # Standard full block chord
                for n in [root, fifth, third]:
                    chord_track.append(Message('note_on', note=n, velocity=chord_vel, time=0))
                chord_track.append(Message('note_off', note=root, velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))

        elif emotion_cat == 'neutral':
            # C418 Minecraft voicing (Sweden / Danny style):
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
                arp_gap = int(ticks_per_step * 0.08)  # ~8% of step between each entry
                
                # Bass note (first, soft)
                chord_track.append(Message('note_on', note=bass, velocity=bass_vel, time=0))
                # Third (after gap)
                chord_track.append(Message('note_on', note=third, velocity=upper_vel, time=arp_gap))
                # Fifth (after another gap)
                chord_track.append(Message('note_on', note=fifth, velocity=max(25, upper_vel - 3), time=arp_gap))
                
                # All sustain together until end of step
                sustain = int(ticks_per_step - arp_gap * 2)
                chord_track.append(Message('note_off', note=bass,  velocity=0, time=sustain))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))
            else:
                # Full block chord (standard ~90%)
                chord_track.append(Message('note_on', note=bass, velocity=bass_vel, time=0))
                chord_track.append(Message('note_on', note=third, velocity=upper_vel, time=0))
                chord_track.append(Message('note_on', note=fifth, velocity=max(25, upper_vel - 3), time=0))
                
                chord_track.append(Message('note_off', note=bass,  velocity=0, time=int(ticks_per_step)))
                chord_track.append(Message('note_off', note=third, velocity=0, time=0))
                chord_track.append(Message('note_off', note=fifth, velocity=0, time=0))

        elif emotion_cat == 'fear':
            # Fear: heavy non-tertian voicing (Root + P5 + 6th) + sustain pedal
            chord_track.append(Message('control_change', control=64, value=127, time=0))

            # Deep bass root with spread upper tension for cinematic horror voicing
            fear_chord = list(chord_notes)
            while fear_chord[0] > 48:  # Root in deep bass (below C3)
                fear_chord[0] -= 12
            fear_chord[1] = fear_chord[1] - 12 if fear_chord[1] > 69 else fear_chord[1]
            fear_chord[2] = fear_chord[2] - 12 if fear_chord[2] > 69 else fear_chord[2]
            vel = max(50, velocity)

            for n in fear_chord:
                chord_track.append(Message('note_on', note=int(n), velocity=vel, time=0))

            # RARE dissonance: ~5% chance of a quiet tritone or minor 2nd
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

        # Fear CAN replay motifs — dark, repeating themes are scarier than randomness
        use_motif = len(motif_buffer) > 0 and random.random() < 0.40

        if use_motif:
            # ── Replay motif transposed ±1-2 scale degrees ──
            shift = random.choice([-2, -1, 1, 2])
            for deg_offset, duration in motif_buffer:
                new_deg = melody_idx + deg_offset + shift
                new_deg = max(21, min(len(active_pool) - 1, new_deg))
                note    = int(active_pool[new_deg]) + register_shift

                # Apply spike melody register shift (graduated)
                if active_spike_profile and spike_intensity > 0.3:
                    note += int(active_spike_profile['melody_register'] * min(1.0, spike_intensity))

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
            # Neutral: strong adherence to expose the modal chord color (add6/add9)
            if emotion_cat == 'neutral':
                chord_adherence_prob = max(0.80, chord_adherence_prob)
            # Fear: high adherence so melody follows the dark non-tertian chords
            if emotion_cat == 'fear':
                chord_adherence_prob = max(0.75, chord_adherence_prob)
            # Spike transitions: boost adherence so melody follows the spike chord coloring
            # Prevents modal clashes during mode changes (e.g., fear→happy, sad→happy)
            if active_spike_profile and spike_intensity > 0.3:
                chord_adherence_prob = max(0.90, chord_adherence_prob)

            for i, duration in enumerate(chosen_rhythm):
                if i < len(chosen_contour):
                    melody_idx += chosen_contour[i]
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                # Fear: lower register (C3-C5 range) with descending bias
                if emotion_cat == 'fear':
                    melody_idx = max(14, min(len(active_pool) - 1, melody_idx))
                    # Gentle descending bias: 30% chance to step down
                    if random.random() < 0.30 and melody_idx > 16:
                        melody_idx -= 1

                # Occasionally snap to a chord tone for harmonic grounding
                if random.random() < chord_adherence_prob:
                    # Snap to safe triad tones only (Root, 3rd, 5th)
                    safe_snap_notes = chord_notes[:3]
                    # Neutral: sometimes prefer non-root tones for melodic variety
                    if emotion_cat == 'neutral' and len(safe_snap_notes) > 1 and random.random() < 0.4:
                        safe_snap_notes = safe_snap_notes[1:]  # skip root occasionally
                    target_note = random.choice(safe_snap_notes) + random.choice([12, 24])
                    # Fear: snap to lower octave for darker register
                    if emotion_cat == 'fear':
                        target_note = random.choice(safe_snap_notes) + random.choice([0, 12])
                    melody_idx  = _pool_idx_nearest(target_note, active_pool)
                    melody_idx  = max(21, min(len(active_pool) - 1, melody_idx))

                note = int(active_pool[melody_idx]) + register_shift

                # Micro valence expression
                if state['micro_v'] > 0.4 and random.random() < 0.35:
                    note += 12  # Octave jump is always safe

                # Apply spike melody register shift (graduated)
                if active_spike_profile and spike_intensity > 0.3:
                    note += int(active_spike_profile['melody_register'] * min(1.0, spike_intensity))

                # Cap super high notes to prevent whistle notes (C6 is 84)
                while note > 84:
                    note -= 12

                note = max(0, min(127, note))

                # Record motif as degree offset from the START melody_idx of this phrase
                new_motif.append((chosen_contour[i] if i < len(chosen_contour) else 0, duration))

                # Determine rest probability
                if spike_rest_prob is not None:
                    should_rest = random.random() < spike_rest_prob
                elif is_neutral and random.random() < 0.10:
                    should_rest = True  # Occasional breaths, not constant gaps
                elif emotion_cat == 'fear' and random.random() < 0.15:
                    should_rest = True  # Some breathing room, not emptiness
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

        # ── Anti-trill guard (applies to all emotions) ──
        # Rule: max 1 trill per 5 seconds — if ANY trill detected, replace immediately
        if detect_trill(melody_notes_and_durations):
            consecutive_trill_count += 1
        else:
            consecutive_trill_count = 0
        # Trigger on the FIRST trill detection
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
                # Fear melody: slightly lower velocity for brooding feel
                if emotion_cat == 'fear':
                    mel_vel = max(45, min(85, velocity - 5))
                else:
                    mel_vel = velocity
                melody_track.append(Message('note_on',  note=int(note), velocity=int(mel_vel), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0,              time=int(duration)))

    mid.save(filename)
    print(f"Saved Final Cohesive MIDI: {filename}")
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np

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

    for t, p in enumerate(emotions_array):
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx] # confidence of the dominant emotion (0.0 to 1.0)
        
        # Track streak to implement variations within identical sustained emotion periods
        if dominant_idx == prev_dominant_idx:
            emotion_streak += 1
        else:
            emotion_streak = 0
        prev_dominant_idx = dominant_idx

        # --------------------------------------------------------------------------
        # 1. MODE SELECTION
        # We now map intensities strictly to modes of the SAME root rather than relative keys
        if dominant_idx == 3: # Happy
            current_mode = 'lydian' if intensity > 0.65 else 'ionian'
            chord_type = "triad"
        elif dominant_idx == 0: # Neutral
            current_mode = 'mixolydian' if p[3] > p[1] else 'dorian'
            chord_type = "sus2"
        elif dominant_idx == 1: # Sad
            current_mode = 'phrygian' if intensity > 0.65 else 'aeolian'
            chord_type = "triad"
        elif dominant_idx == 2: # Fear
            current_mode = 'locrian' if intensity > 0.65 else 'phrygian'
            chord_type = "dim"

        # Generate the pool of valid diatonic notes for this exact parallel mode
        pool = get_mode_pool(current_mode, root_midi=(24 + base_key_offset), octaves=8)
        
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
            
        # 2. DYNAMICS & TEMPO (Smoothed Arousal)
        arousal = p[3] + p[2]
        
        # Calculate Target BPM for this tick 
        # Narrowed default constraints: Sad/Fear bottoms out around ~80 BPM, High Happy peaks around ~130 BPM
        target_bpm = 80 + (arousal * 50)
        
        # Exponential Moving Average for smooth cinematic tempo transitions
        if current_bpm is None:
            current_bpm = target_bpm
        else:
            current_bpm = 0.7 * current_bpm + 0.3 * target_bpm
            
        tempo = bpm2tempo(int(current_bpm))
        chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # Velocity mapping (30 to 110)
        velocity = max(30, min(110, int(40 + (arousal * 70))))
        
        # Rhythmic Selection (Streak based diversification)
        if emotion_streak > 4 and random.random() < 0.3:
            # Introduce a rhythmic breakdown / pause measure every so often
            chosen_rhythm = [480, 480] if arousal < 0.5 else [240, 240, 480]
        else:
            if arousal > 0.7:
                chosen_rhythm = random.choice(rhythms['fast'])
            elif arousal > 0.4:
                chosen_rhythm = random.choice(rhythms['med'])
            else:
                chosen_rhythm = random.choice(rhythms['slow'])

        # 3. CHORD & ACCOMPANIMENT GENERATION (Differentiation)
        
        # C3 starting fundamental (C3 is exactly 2 octaves up from C1 which is index 0. 2 octaves = index 14 in a 7-note diatonic pool)
        base_chord_pool_idx = 14
        
        # Dynamic Chord Progression Engine: Move the bass note around the scale DEGREES (0 = I, 1 = ii, etc.)
        # Happy/Neutral lean on I, IV, V, vi. Sad/Fear lean on i, iv, v, VI, diminished, etc.
        progression_degrees = {
            'lydian':     [0, 3, 4, 5],    # I, IV, V, vi
            'ionian':     [0, 3, 4, 5],    # I, IV, V, vi
            'mixolydian': [0, 3, 6, 4],    # I, IV, VII, V
            'dorian':     [0, 2, 3, 4],    # i, III, IV, v
            'aeolian':    [0, 5, 2, 3],    # i, VI, III, iv
            'phrygian':   [0, 1, 5, 3],    # i, II, VI, iv
            'locrian':    [0, 4, 5, 2]     # i(dim), v(dim), VI, III
        }
        
        # Pick the chord based on emotion streak, probabilities, and the mode
        # Rather than a rigid looping pattern, we'll smoothly change chords to break monotony randomly
        # We only change chords occasionally (every ~4 seconds) unless the arousal is super high
        change_chord_prob = 0.2 if arousal < 0.6 else 0.4
        
        # Initialize or update the active chord degree for this emotion chunk
        if t == 0 or emotion_streak == 0 or (emotion_streak % 4 == 0 and random.random() < change_chord_prob):
            prog_pattern = progression_degrees.get(current_mode, [0, 3, 4, 5])
            # Randomly pick a new chord from the mode's progression pool
            current_chord_degree = random.choice(prog_pattern)
            
        chord_root_idx = base_chord_pool_idx + current_chord_degree
        
        # Map strictly from diatonic pool to guarantee perfect harmony and destroy dissonance!
        if chord_type == "triad":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        elif chord_type == "sus2":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 1], pool[chord_root_idx + 4]]
        elif chord_type == "dim":
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx] + 6]
        else:
            chord_notes = [pool[chord_root_idx], pool[chord_root_idx + 2], pool[chord_root_idx + 4]]
        
        # Apply highly distinct musical accompaniment figures per emotion
        if dominant_idx == 3: # HAPPY: Majestic, slower chords (2x slower)
            chord_vel = max(20, velocity - 10) 
            
            # 1 steady, sustained pulse per step (lasting the full ticks_per_step)
            step_time = int(ticks_per_step) 
            
            # Turn all notes in the chord ON
            for n in chord_notes: 
                chord_track.append(Message('note_on', note=n, velocity=chord_vel, time=0))
            
            # Advance the clock by step_time and turn the first note OFF
            chord_track.append(Message('note_off', note=chord_notes[0], velocity=0, time=int(step_time)))
            
            # Turn the remaining notes OFF at exactly the same time (time=0)
            for n in chord_notes[1:]: 
                chord_track.append(Message('note_off', note=n, velocity=0, time=0))
            
        elif dominant_idx == 1: # SAD: Cascading Broken Arpeggios (Downward focus)
            chord_vel = max(20, velocity - 10)
            step_time = int(ticks_per_step // 4) # 4 notes, explicitly integer division
            
            # Root, 3rd, 5th, 3rd (Shifted down one octave diatonically)
            arpeggio = [
                pool[max(0, chord_root_idx - 7)], 
                pool[max(0, chord_root_idx - 7 + 2)], 
                pool[max(0, chord_root_idx - 7 + 4)], 
                pool[max(0, chord_root_idx - 7 + 2)]
            ]
            
            for n in arpeggio:
                chord_track.append(Message('note_on', note=n, velocity=chord_vel, time=0)) 
                chord_track.append(Message('note_off', note=n, velocity=0, time=int(step_time)))
                
        elif dominant_idx == 2: # FEAR: Subtle, eerie, creeping long sustains
            chord_vel = max(10, velocity - 30)  # Very quiet, subtle tension
            
            # Instead of a jarring 3-note stab, creep in one dissonant note very softly, very low
            # Shift down two octaves diatonically (index - 14) 
            eerie_idx = random.choice([chord_root_idx, chord_root_idx + 2, chord_root_idx + 4]) - 14
            eerie_note = pool[max(0, eerie_idx)]
            
            chord_track.append(Message('note_on', note=eerie_note, velocity=chord_vel, time=0)) 
            chord_track.append(Message('note_off', note=eerie_note, velocity=0, time=int(ticks_per_step))) 
            
        else: # NEUTRAL: Long floating sustained block chord
            chord_vel = max(20, velocity - 25)
            for n in chord_notes: 
                chord_track.append(Message('note_on', note=n, velocity=chord_vel, time=0))
            chord_track.append(Message('note_off', note=chord_notes[0], velocity=0, time=int(ticks_per_step)))
            for n in chord_notes[1:]: 
                chord_track.append(Message('note_off', note=n, velocity=0, time=0))


        # -------------------------------------------------------------------------
        # 4. MELODY GENERATION (Pure Random Walk with Harmonic Adherence)
        
        melody_notes_and_durations = []
        
        for i, duration in enumerate(chosen_rhythm):
            # HARMONIC FIX: Map Right Hand exactly to Left Hand chord notes
            
            # In Happy (Vivaldi style), melodies are highly structured arpeggios. 
            # We force 95% chord tone adherence for Happy to eliminate "randomness", 70% for others.
            chord_adherence_prob = 0.95 if dominant_idx == 3 else 0.70
            
            if random.random() < chord_adherence_prob:  # Play an exact chord tone to eliminate dissonance
                # Shift chord tones up into the melody range safely
                target_note = random.choice(chord_notes) + random.choice([12, 24])
                note = target_note
                # Ensure melody_idx tracks roughly where we are so random walks still work afterwards
                try:
                    melody_idx = min(range(len(pool)), key=lambda k: abs(pool[k]-note))
                except ValueError:
                    pass
            else:
                # Random Walk: Step diatonically as a passing tone
                step = random.choices([-1, 0, 1], weights=[30, 40, 30])[0]
                melody_idx += step
                
                # Bound the melody index to stay within vocal range
                melody_idx = max(21, min(35, melody_idx))
                note = pool[melody_idx]
            
            # Phrasing: Periodically jump an octave up if the feeling sustains happily
            if emotion_streak > 6 and dominant_idx == 3:
                note += 12 if random.random() < 0.5 else 0

            # Chromaticism: Occasional "wrong notes" (sharps/flats) in Fear for dissonant jarring impact
            if dominant_idx == 2 and random.random() < 0.3:
                note += random.choice([-1, 1])

            # Phrasing: Introduce rests into Neutral to make it sparser and less robotic
            if dominant_idx == 0 and random.random() < 0.3:
                melody_notes_and_durations.append((None, duration)) # Note None = Rest
            else:
                melody_notes_and_durations.append((note, duration))
        
        for note, duration in melody_notes_and_durations:
            if note is None:
                melody_track.append(Message('note_off', note=0, velocity=0, time=int(duration)))
            else:
                melody_track.append(Message('note_on', note=int(note), velocity=int(velocity), time=0))
                melody_track.append(Message('note_off', note=int(note), velocity=0, time=int(duration)))

    mid.save(filename)
    print(f"🎵 Saved Final Cohesive MIDI: {filename}")
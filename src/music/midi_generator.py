from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import random
import numpy as np

def get_diatonic_pool(base_offset=0, octaves=6):
    # generate the 7 diatonic notes for a given base key (across multiple octaves).

    # major scale intervals: W, W, H, W, W, W, H, where W = +2, H = +1
    intervals = [0, 2, 4, 5, 7, 9, 11]

    pool = []
    base_midi = 24 # C1

    for oct in range(octaves):
        for interval in intervals:
            note = base_midi + base_offset + (oct * 12) + interval
            if note <= 127:
                pool.append(note)
    return pool

def generate_midi_from_emotions(emotions_array, base_key_offset=0, filename="eeg_music.mid"):
    mid = MidiFile()

    # left hand chords + right hand melody
    chord_track = MidiTrack()
    melody_track = MidiTrack()
    mid.tracks.extend([chord_track, melody_track])
    
    # Explicitly set the instrument to Acoustic Grand Piano (Program 0) for both tracks
    # This fixes Timidity "No instrument mapped to tone bank 0, program 0" errors on Linux
    chord_track.append(Message('program_change', program=0, time=0))
    melody_track.append(Message('program_change', program=0, time=0))

    ticks_per_beat = 480
    ticks_per_step = ticks_per_beat * 2 # each time step holds for 2 beats

    # generate one single unified pool of notes for the song (ensures cohesion)
    diatonic = get_diatonic_pool(base_key_offset)

    # degree indices for the 7 greek musical modes within the diatonic scale
    mode_degrees = {
        'lydian': 3,     # IV (Brightest)
        'ionian': 0,     # I  (Major)
        'mixolydian': 4, # V
        'dorian': 1,     # II
        'aeolian': 5,    # VI (Minor)
        'phrygian': 2,   # III
        'locrian': 6     # VII (Darkest)
    }

    # rhythm pools to replace randomized note counts, ensuring musical timing
    rhythms = {
        'slow': [[960], [480, 480]],
        'med': [[480, 240, 240], [240, 240, 480]],
        'fast': [[240, 240, 240, 240]]
    }

    for t, p in enumerate(emotions_array):
        dominant_idx = np.argmax(p)
        intensity = p[dominant_idx] # confidence of the dominant emotion (0.0 to 1.0)

#--------------------------------------------------------------------------------------
        # 1. MODE SELECTION (based on dominant emotion)
        if dominant_idx == 3: # Happy
            #here we use either lydian or standard major scale (ionian)
            current_mode = 'lydian' if intensity > 0.65 else 'ionian'

        elif dominant_idx == 0: # Neutral
            # neutral leans towards happy => use mixolydian. if it leans sad => use dorian.
            current_mode = 'mixolydian' if p[3] > p[1] else 'dorian'

        elif dominant_idx == 1: # Sad
            # here we use either standard minor scale (aeolian) or phrygian
            current_mode = 'phrygian' if intensity > 0.65 else 'aeolian'

        elif dominant_idx == 2: # Fear
            # we use either phrygian or if the fear is high we use locrian mode.
            current_mode = 'locrian' if intensity > 0.65 else 'phrygian'

#--------------------------------------------------------------------------
        # 2. DYNAMICS + RHYTHM (Arousal Based)

        # Arousal = P_Happy + P_Fear (the high energy emotions)
        arousal = p[3] + p[2]

        # tempo mapping (60 BPM to 160 BPM)
        current_bpm = 60 + (arousal * 100)
        tempo = bpm2tempo(int(current_bpm))
        chord_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        # velocity mapping (40 to 110)
        velocity = int(40 + (arousal * 70))

        # note density and duration (Mapped to musical rhythm pools)
        if arousal > 0.7:
            chosen_rhythm = random.choice(rhythms['fast']) # 16th/fast 8th notes
        elif arousal > 0.4:
            chosen_rhythm = random.choice(rhythms['med'])  # standard 8th notes
        else:
            chosen_rhythm = random.choice(rhythms['slow']) # quarter/half notes

#-------------------------------------------------------------------------
        # 3. NOTE SELECTION & MODAL ALTERATIONS

        # Base Root in the 3rd octave
        root_idx = 14 + mode_degrees[current_mode]

        # Start with standard diatonic melody pool
        melody_pool = diatonic[21:35]
        chord_tones_octave_up = [diatonic[root_idx + 7], diatonic[root_idx + 9], diatonic[root_idx + 11]]

        # --- SPICE IT UP: Emotion-Specific Borrowed Notes & Chord Voicings ---

        if dominant_idx == 3: # HAPPY
            # Pure, stable Major triad (Root, 3rd, 5th)
            chord_notes = [diatonic[root_idx], diatonic[root_idx + 2], diatonic[root_idx + 4]]
            # Melody stays purely diatonic (bright and stable)

        elif dominant_idx == 0: # NEUTRAL
            # Sus2 Chord (Root, 2nd, 5th). Gives a floating, ambiguous, emotionless feel.
            chord_notes = [diatonic[root_idx], diatonic[root_idx + 1], diatonic[root_idx + 4]]

            # Pentatonic Melody: Remove the 4th and 7th degrees of the major scale
            # This makes the melody wander peacefully without any tense resolutions.
            melody_pool = [n for n in diatonic[21:35] if (n - base_key_offset) % 12 not in [5, 11]]

        elif dominant_idx == 1: # SAD
            # Standard Minor triad
            chord_notes = [diatonic[root_idx], diatonic[root_idx + 2], diatonic[root_idx + 4]]

            # Harmonic Minor Alteration: Raise the 7th degree of the minor scale
            # (which is the 5th degree of the relative major) by 1 semitone for tragic tension.
            altered_melody = []
            for n in diatonic[21:35]:
                if (n - base_key_offset) % 12 == 7: # If it's the 'G' in C Major
                    altered_melody.append(n + 1)    # Sharp it to 'G#'
                else:
                    altered_melody.append(n)
            melody_pool = altered_melody

        elif dominant_idx == 2: # FEAR
            # Diminished/Altered Chord: Flatten the 5th to create a terrifying Tritone clash
            chord_notes = [diatonic[root_idx], diatonic[root_idx + 2], diatonic[root_idx + 4] - 1]
            # Melody pool stays standard here, but we inject chromaticism below!

        # --- MELODY GENERATION ---
        melody_notes_and_durations = []
        for i, duration in enumerate(chosen_rhythm):
            if i == 0:
                # Anchor melody to the altered chord on the downbeat
                note = chord_notes[0] + 12 # Root note, one octave up
            else:
                note = random.choice(melody_pool)

                # FEAR CHROMATICISM: 40% chance to randomly sharp or flat a melody note
                # to create jarring, horror-movie dissonance.
                if dominant_idx == 2 and random.random() < 0.4:
                    note += random.choice([-1, 1])

            melody_notes_and_durations.append((note, duration))

#-------------------------------------------------------------------------
        # 4. WRITE NOTES TO MIDI

        # Write Left Hand Chord
        for note in chord_notes:
            chord_track.append(Message('note_on', note=note, velocity=max(30, velocity - 15), time=0))

        # Hold chord for the full 2 beats
        chord_track.append(Message('note_off', note=chord_notes[0], velocity=0, time=ticks_per_step))
        for note in chord_notes[1:]:
            chord_track.append(Message('note_off', note=note, velocity=0, time=0))

        # Write Right Hand Melody
        for note, duration in melody_notes_and_durations:
            melody_track.append(Message('note_on', note=note, velocity=velocity, time=0))
            melody_track.append(Message('note_off', note=note, velocity=0, time=duration))

    mid.save(filename)
    print(f"🎵 Saved Final Cohesive MIDI: {filename}")
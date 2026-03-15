# MIDI Generation Documentation

This project implements two distinct but harmonically aligned paths for generating music from EEG-derived emotion probabilities: **Pipeline (Offline) Generation** and **Real-time Synthesis**.

---

## 1. Pipeline Generation (`midi_generator.py`)

Used in the Pipeline View to generate `.mid` files from a full trial of classification results.

### Core Steps:
1.  **Metric Calculation**:
    - **Dominant Emotion**: Index of the highest probability (`np.argmax`).
    - **Intensity**: The raw probability of the dominant emotion.
    - **Arousal**: Sum of 'Happy' and 'Fear' probabilities, used to drive tempo and velocity.
2.  **Mode & Key Mapping**:
    - Uses 8-octave parallel Greek modes (Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian).
    - Maps emotions to modes based on intensity (e.g., Happy/High → Lydian, Sad/Low → Aeolian).
    - Ensures all modes share a fixed root (e.g., C2) for melodic cohesion.
3.  **Rhythmic Engine**:
    - Divides time into "Steps" (2 beats per classified second).
    - Selects rhythmic patterns from three pools (Slow, Medium, Fast) based on Arousal.
4.  **Accompaniment Logic**:
    - **Happy**: Majestic sustained block chords.
    - **Sad**: Cascading broken arpeggios (downward focused).
    - **Fear**: Creeping/eerie long sustains on dissonant notes.
    - **Neutral**: Soft, floating sustained block chords.
5.  **Melody Random Walk**:
    - Restricted to the current mode's diatonic pool.
    - High **Harmonic Adherence**: 95% of notes in Happy mode are forced chord tones; 70% in others.
6.  **Texturing (Optional)**:
    - If EEG features are passed, maps bands (Delta..Gamma) to MIDI Control Change (CC) messages for Chorus, Reverb, Attack, and Brightness.

---

## 2. Real-time Generation (`realtime_generator.py`)

Used in the Real-time View to synthesize audio with low latency (1-second chunks) during live data streaming.

### Core Steps:
1.  **FluidSynth lazy-initialization**: Starts the audio driver and loads the soundfont only when first used.
2.  **1-Second Chunk Processing**:
    - Pulls the latest classification result from a thread-safe update queue.
    - Updates target BPM and note velocity via arousal smoothing.
3.  **Stateful Music Engine**:
    - Tracks "Emotion Streak" to avoid jarred transitions; only changes chord progression every 4 seconds or on major emotion shifts.
4.  **Synchronized Playback**:
    - Uses high-resolution `time.time()` tracking to prevent drift between chunks.
    - Executes sequential accompaniment notes (arpeggios) using blocking sleeps within a dedicated thread.
5.  **Alignment Fix**:
    - **CC 11 over CC 1**: Uses Expression (CC 11) for texturing instead of Modulation (CC 1) to avoid unwanted LFO-vibrato on piano patches.
    - **Attack Clamping**: Limits Attack (CC 73) to avoid muffled note-starts.
6.  **UI Feedback Loop**:
    - Emits Qt signals (`note_played`, `state_update`) to drive the Piano Roll and UI labels in real-time.

---

## Shared Music Theory Logic

Both generators share the same fixed rules for:
- **Mode Pool Generation**: `get_mode_pool`
- **Chord Construction**: `get_chord`
- **Progression Logic**: Mode-specific degree patterns (e.g., I-IV-V-vi for Ionian).
- **Parallel Movement**: Always anchored to the same root MIDI note to ensure that even as the feeling changes, the harmonic resolution remains satisfying.

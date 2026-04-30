# Piano Music Generator MVP - Technical Guide

This guide explains the core music generation logic implemented in `src/music/midi_generator.py`. The system translates EEG-derived emotion probabilities (Valence and Arousal) into professional-grade piano compositions using a blend of music theory, Markov chains, and dynamic state management.

---

## 1. Mode Selection (Emotional Color)
The engine maps macro-emotional states to specific Greek modes to establish the "color" of the piece:
- **Valence-Arousal Mapping**: 
  - **Happy**: Ionian or Lydian (for high valence).
  - **Sad**: Aeolian.
  - **Fear**: Phrygian, Harmonic Minor, or Phrygian Dominant (depending on the depth of negative valence).
  - **Neutral**: Operates as a "Chameleon" mode, locking into Dorian (sad-leaning) or Mixolydian (happy-leaning) based on the secondary dominant emotion.
- **Diatonic Pools**: Each mode generates a restricted pool of valid MIDI notes (across 8 octaves) anchored to a shared root note, ensuring harmonic cohesion even during rapid emotional shifts.

## 2. Dynamics & Tempo (Energy and Intensity)
Musical energy is directly driven by the **Arousal** metric:
- **Dynamic BPM**: Scales between 60 and 140 BPM using an Exponential Moving Average (EMA) for smooth, natural tempo transitions.
- **Rhythmic Density**: High arousal triggers faster subdivisions (16th notes, triplets), while low arousal favors sustained half-notes and whole-notes.
- **Humanization**: Sad and Neutral modes include micro-timing jitter to simulate the subtle imperfections of a human pianist.
- **Spike Modifiers**: Sudden emotional spikes (e.g., ANXIETY or RELIEF) apply immediate multipliers to tempo and velocity to highlight the micro-event.

## 3. Chords and Accompaniment (Harmonic Foundation)
The harmonic progression is driven by a sophisticated state machine:
- **Markov Transitions**: Relative scale degrees are selected using an emotion-specific transition matrix, ensuring logical and "musical" chord movements (e.g., I-IV-V in Happy, i-bII in Fear).
- **Voicing Styles**: 
  - **Happy**: Majestic block triads.
  - **Sad**: Cascading rolled chords or "cascades" (downward focus).
  - **Neutral**: Arpeggiated patterns with a split between deep bass and mid-range upper voices.
  - **Fear**: Sustained non-tertian (quartal) voicings with added dissonant "ghost notes."
- **Spike Coloring**: During spikes, chords are "colored" with specific intervals (e.g., adding a minor 3rd for "Bittersweet" transitions or using open power chords for "Courage").

## 4. Melody Generation (Contour and Phrase)
The melody provides the narrative thread of the music:
- **VGMIDI Markov Engine**: Melodic intervals are generated using a Markov chain trained on video game music, ensuring natural-sounding contours rather than random walks.
- **Motif Memory**: The engine maintains a short-term buffer to repeat and transpose previous phrases (40% probability), creating thematic consistency and "catchy" motifs.
- **Harmonic Adherence**: A strict "Dissonance Guard" snaps melodic notes to the current chord tones (75-95% adherence) to prevent clashing, while allowing enough variety for modal expression.
- **Register Management**: Automatically shifts the melody register based on intensity—lower for brooding Fear passages and higher for "Awakening" or "Hope" spikes.

---
*This guide summarizes the logic for the "Completed Piano Music Generator MVP".*

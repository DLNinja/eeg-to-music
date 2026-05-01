# Translating Brainwaves into Music
**An Overview of the EEG-to-MIDI Generation Engine**

## The Goal
The purpose of this engine is to take raw emotional data—specifically brainwave parameters representing "Valence" (how positive or negative an emotion is) and "Arousal" (how intense or calm the emotion is)—and translate them smoothly into cohesive, cinematic music. Instead of simply stringing random notes together, the system dynamically reacts to the user's brain state, creating real-time "motifs" and chord progressions that sound intentionally composed.

## How It Works: The "Macro" vs "Micro" Emotions
Human emotion isn't just one static number; it ebbs, flows, and spikes. To capture this musically, our engine splits the data into two layers:
1. **Macro Mood (The Foundation):** A rolling average of the recent emotion data. This dictates the overall "scene." It determines the musical mode (e.g., a Happy major key vs a Sad minor key) and controls how fast the chords change (the harmonic rhythm).
2. **Micro Spikes (The Expressions):** Sudden, sharp changes in emotion (like a jump scare or a rush of joy). These don't change the underlying key, but instead act as immediate "decorations"—like a sudden velocity swell, a dissonant "tension" note, or a jump up the octave.

## The Four Emotional Quadrants
Based on the Valence/Arousal (V/A) coordinates, the music is bucketed into one of four distinct personalities:
- **Happy (High Valence, High Arousal):** Plays classic pop/triumphant chord progressions (I-V-vi-IV). The accompaniment features strong, sustained block chords.
- **Sad (Low Valence, Low Arousal):** Plays a tragic, descending "Andalusian" chord progression. The left hand plays wide, spacious chords with low, resonant bass.
- **Fear (Low Valence, High Arousal):** Scraps traditional chord changes and instead plays a dark, suspenseful "Drone." Sudden spikes in fear trigger swelling tension instead of melodic changes.
- **Neutral (Low Arousal, Central Valence):** Acts as a musical "chameleon." It shifts into mysterious progressions (like Dorian or Mixolydian modes), builds harmonies in open "fourths" (Quartal harmony), and restricts the melody strictly to a pentatonic scale so it flows without clashing.

## The "Composer Brain" (Markov Chain Model)
When playing the right-hand melody, a completely random walk sounds like a computer. To fix this, we trained a statistical model on the **VGMIDI dataset**—a library of hundreds of video game soundtracks spanning different emotions.

Instead of guessing what note to play next, the engine asks: *"When a real video game composer was writing a 'Happy' song, and they just went up two steps on the piano, what did they do next?"*

The system looks at the transition probabilities extracted from these real soundtracks and chooses the next melodic jump based on authentic human behavior.

## Pipeline vs. Live Generation
The engine comes in two flavors:
1. **The Offline Generator (`midi_generator.py`):** Takes a completely pre-recorded chunk of EEG data and renders a perfectly timed MIDI file all at once. It has the luxury of looking ahead, allowing it to build coherent musical phrases that line up perfectly with the emotion graph.
2. **The Live Synthesizer (`realtime_generator.py`):** Built for actual neurofeedback. As the brainwave headset streams live data, this synthesizer builds the music "one second at a time," scheduling MIDI notes ahead dynamically while ensuring no auditory stutters. It feeds into a software synthesizer (FluidSynth) to immediately play the audio out loud.

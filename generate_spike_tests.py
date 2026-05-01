import numpy as np
import os
import sys

# Ensure we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.music.midi_generator import generate_midi_from_emotions, SPIKE_PROFILES

EMOTION_VECS = {
    'neutral': [1.0, 0.0, 0.0, 0.0],
    'sad':     [0.0, 1.0, 0.0, 0.0],
    'fear':    [0.0, 0.0, 1.0, 0.0],
    'happy':   [0.0, 0.0, 0.0, 1.0]
}

os.makedirs('music', exist_ok=True)

for (macro_emo, spike_emo), profile in SPIKE_PROFILES.items():
    name = profile['name'].lower()
    filename = f"music/test_{name}.mid"
    print(f"Generating {filename} ({macro_emo} -> {spike_emo})")
    
    macro_vec = EMOTION_VECS[macro_emo]
    spike_vec = EMOTION_VECS[spike_emo]
    
    # 15 steps to stabilize macro state
    seq = [macro_vec] * 15
    # 5 steps for the spike to be fully registered and heard
    seq += [spike_vec] * 5
    # 5 steps to recover back to macro
    seq += [macro_vec] * 5
    
    arr = np.array(seq)
    # We set base_key_offset=0 so dynamic key selection operates naturally
    generate_midi_from_emotions(arr, filename=filename, base_key_offset=0)

print("All 12 test MIDI files have been generated successfully.")

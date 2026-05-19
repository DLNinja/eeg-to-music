import numpy as np
import os
import sys

# Ensure we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.music.midi_generator import generate_midi_from_emotions

EMOTION_VECS = {
    'neutral': [1.0, 0.0, 0.0, 0.0],
    'sad':     [0.0, 1.0, 0.0, 0.0],
    'fear':    [0.0, 0.0, 1.0, 0.0],
    'happy':   [0.0, 0.0, 0.0, 1.0]
}

os.makedirs('test-trials', exist_ok=True)

test_scenarios = {
    "TEST1": ['happy', 'sad', 'fear', 'neutral'],
    "TEST2": ['sad', 'happy', 'neutral', 'fear'],
    "TEST3": ['neutral', 'happy', 'fear', 'sad'],
    "TEST4": ['fear', 'neutral', 'sad', 'happy']
}

for name, sequence in test_scenarios.items():
    filename = f"test-trials/{name}.mid"
    print(f"Generating {filename} ({' -> '.join(sequence)})")
    
    seq_list = []
    for emo in sequence:
        # 15 steps per mood, roughly 15 seconds (assuming ~60-120 BPM mapping)
        seq_list += [EMOTION_VECS[emo]] * 15
        
    arr = np.array(seq_list)
    generate_midi_from_emotions(arr, filename=filename, base_key_offset=0)

print("All 4 test MIDI files have been generated successfully.")

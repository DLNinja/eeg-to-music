import os
import json
import random

class MarkovEngine:
    """
    Loads VGMIDI transition matrices into memory at startup.
    Provides robust query methods that fall back to sensible defaults
    if an unseen state is encountered, ensuring the real-time generator
    never crashes.
    """
    def __init__(self, models_dir="models/transitions"):
        self.matrices = {}
        quadrants = ['happy', 'sad', 'fear', 'neutral']
        
        # Absolute path resolution (assuming this script is in src/music)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        full_models_dir = os.path.join(base_path, models_dir)

        if not os.path.exists(full_models_dir):
            print(f"[MarkovEngine] WARNING: Models dir not found: {full_models_dir}")
            print("[MarkovEngine] Please run train_markov_midi.py first. Using empty models.")
            for q in quadrants:
                self.matrices[q] = {'pitch_interval': {}, 'duration': {}}
            return

        for q in quadrants:
            path = os.path.join(full_models_dir, f'transitions_{q}.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.matrices[q] = json.load(f)
            else:
                print(f"[MarkovEngine] WARNING: Missing {path}. Using empty.")
                self.matrices[q] = {'pitch_interval': {}, 'duration': {}}

        print(f"[MarkovEngine] Loaded {len(quadrants)} VGMIDI transition matrices.")

    def query_next_interval(self, emotion_cat, prev_interval):
        """
        Given the emotion quadrant ('happy', etc.) and the integer interval of 
        the last melodic jump, randomly sample the next interval based on composer data.
        Returns: integer interval (e.g., -2, 0, +4)
        """
        if emotion_cat not in self.matrices:
            emotion_cat = 'neutral'
            
        pitch_matrix = self.matrices[emotion_cat].get('pitch_interval', {})
        
        # Convert integer to string for JSON lookup
        prev_str = str(prev_interval)
        
        if prev_str in pitch_matrix:
            transitions = pitch_matrix[prev_str]
        else:
            # Fallback 1: Unseen state. Use the distribution for standing still ("0")
            if "0" in pitch_matrix:
                transitions = pitch_matrix["0"]
            else:
                # Fallback 2: Completely empty matrix (e.g. model not trained yet)
                return random.choice([-2, -1, 0, 1, 2])
        
        choices = list(transitions.keys())
        weights = list(transitions.values())
        
        next_interval_str = random.choices(choices, weights=weights, k=1)[0]
        return int(next_interval_str)

    def query_next_durations(self, emotion_cat, prev_dur_label, beats_remaining):
        """
        (Optional convenience query if we choose to use Markov for rhythm too).
        Returns a string duration label.
        """
        pass

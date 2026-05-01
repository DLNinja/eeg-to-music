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
        
        # Absolute path
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        full_models_dir = os.path.join(base_path, models_dir)

        if not os.path.exists(full_models_dir):
            print(f"[MarkovEngine] WARNING: Models dir not found: {full_models_dir}")
            print("[MarkovEngine] Please run train_markov_midi.py first. Using empty models.")
            for q in quadrants:
                self.matrices[q] = {'pitch_interval_1': {}, 'pitch_interval_2': {}, 'pitch_interval_3': {}, 'duration': {}}
            return

        for q in quadrants:
            path = os.path.join(full_models_dir, f'transitions_{q}.json')
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.matrices[q] = json.load(f)
            else:
                print(f"[MarkovEngine] WARNING: Missing {path}. Using empty.")
                self.matrices[q] = {'pitch_interval_1': {}, 'pitch_interval_2': {}, 'pitch_interval_3': {}, 'duration': {}}

        print(f"[MarkovEngine] Loaded {len(quadrants)} VGMIDI transition matrices.")

    def query_next_interval(self, emotion_cat, prev_intervals):
        """
        Given the emotion quadrant ('happy', etc.) and a list of previous integer intervals 
        (up to 3), search from 3rd-order down to 1st-order for a match.
        Returns: integer interval (e.g., -2, 0, +4)
        """
        if emotion_cat not in self.matrices:
            emotion_cat = 'neutral'
            
        mat = self.matrices[emotion_cat]
        p1 = mat.get('pitch_interval_1', {})
        p2 = mat.get('pitch_interval_2', {})
        p3 = mat.get('pitch_interval_3', {})
        
        # Ensure prev_intervals is a list
        if isinstance(prev_intervals, int):
            prev_intervals = [prev_intervals]
            
        # Pad with 0s if too short
        while len(prev_intervals) < 3:
            prev_intervals.insert(0, 0)
            
        i1, i2, i3 = prev_intervals[-3], prev_intervals[-2], prev_intervals[-1]
        
        state3 = f"{i1},{i2},{i3}"
        state2 = f"{i2},{i3}"
        state1 = str(i3)
        
        transitions = None
        
        # Variable-Order Fallback Logic
        if state3 in p3:
            transitions = p3[state3]
        elif state2 in p2:
            transitions = p2[state2]
        elif state1 in p1:
            transitions = p1[state1]
        elif "0" in p1:
            # Fallback 1: Unseen state. Use the distribution for standing still ("0")
            transitions = p1["0"]
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

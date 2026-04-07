from collections import deque
import numpy as np

class EmotionTracker:
    """
    Tracks continuous Valence-Arousal (V-A) values
    and identifies long-term mood (MACRO) and short-term response (MICRO SPIKES).
    """
    def __init__(self, window_size=10, spike_threshold=0.3):
        # CURRENT STATE
        self.v = 0.0
        self.a = 0.0
        
        # HISTORY FOR ROLLING AVERAGE (MACRO MOOD)
        self.history = deque(maxlen=window_size) # knows the last 10 states
        self.spike_threshold = spike_threshold
        
        # BACKWARD COMPATIBILITY: EMOTIONS -> V-A SPACE
        # 3: HAPPY, 1: SAD, 2: FEAR, 0: NEUTRAL
        self.LABEL_TO_VA = {
            3: (0.8, 0.8),   # Top-Right: High V, High A
            1: (-0.8, -0.8), # Bottom-Left: Low V, Low A
            2: (-0.5, 0.9),  # Top-Left: Low V, High A
            0: (0.0, 0.0)    # Center: Neutral
        }

    def update_from_discrete(self, label_idx, confidence=1.0):
        """
        Adapts the existing emotion classifier output into the V-A space.
        Weighting by confidence allows for more fluid movement between quadrants.
        """
        target_v, target_a = self.LABEL_TO_VA.get(label_idx, (0.0, 0.0))
        self.update_from_va(target_v * confidence, target_a * confidence)

    def update_from_va(self, v, a):
        """
        Directly update coordinates and add to rolling history.
        """
        self.v = float(v)
        self.a = float(a)
        self.history.append((self.v, self.a))

    def get_macro_state(self):
        """Returns the average Valence and Arousal of the history window."""
        if not self.history:
            return self.v, self.a
        
        avg_v = sum(h[0] for h in self.history) / len(self.history)
        avg_a = sum(h[1] for h in self.history) / len(self.history)
        return avg_v, avg_a

    def get_micro_state(self, current_v, current_a):
        """Subtracts the Macro Mood from the current raw V-A input to find the delta."""
        macro_v, macro_a = self.get_macro_state()
        micro_v = current_v - macro_v
        micro_a = current_a - macro_a
        return micro_v, micro_a
        
    def get_state(self):
        """Returns the current state of both streams for the generation engine."""
        macro_v, macro_a = self.get_macro_state()
        micro_v, micro_a = self.get_micro_state(self.v, self.a)
        
        return {
            "valence": self.v,
            "arousal": self.a,
            "macro_v": macro_v,
            "macro_a": macro_a,
            "micro_v": micro_v,
            "micro_a": micro_a,
            "is_spike": abs(micro_a) > self.spike_threshold
        }

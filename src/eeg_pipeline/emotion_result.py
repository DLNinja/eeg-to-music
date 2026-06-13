# Object containing a 1-second segment's results from the classification pipeline

from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class EmotionResult:

    de_features:  np.ndarray        # (n_channels, n_bands) differential entropy matrix
    probs:        np.ndarray        # softmax probabilities [Neutral, Sad, Fear, Happy]
    timestamp:    Optional[float]   # seconds into stream, None in SimulatorView
    band_powers:  dict              # {band: {mean, abs_mean, channels, abs_channels}, asymmetry}

    @property
    def dominant_idx(self) -> int:
        # Index of the highest-probability emotion class
        return int(np.argmax(self.probs))

    @property
    def asymmetry(self) -> float:
        # Smoothed FAA score
        return float(self.band_powers.get('asymmetry', 0.0))

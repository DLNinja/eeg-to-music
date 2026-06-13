import numpy as np

# Window size for Z-score calibration phase (in seconds)
ZSCORE_CALIBRATION_TIME = 30


class ZScoreTracker:    
    def __init__(self, window_seconds: int = ZSCORE_CALIBRATION_TIME):
        self.window_seconds = window_seconds
        self.history = []
        # Locked = calibration phase over / Unlocked = during calibration phase
        self.is_locked = False
        self.mean = None
        self.std = None
    
    def update(self, band_power_array: np.ndarray) -> np.ndarray:
        # If already locked, compute Z-score using static mean and std
        if self.is_locked:
            return (band_power_array - self.mean) / self.std
        
        # Accumulating phase
        self.history.append(band_power_array.copy())
        
        # Lock baseline if we reached target window size
        if len(self.history) >= self.window_seconds:
            history_arr = np.array(self.history)
            self.mean = history_arr.mean(axis=0)
            self.std = history_arr.std(axis=0) + 1e-8
            self.is_locked = True
            self.history.clear()
            return (band_power_array - self.mean) / self.std
            
        # During calibration phase, return zeros
        return np.zeros_like(band_power_array)
    
    def reset(self):
        self.history.clear()
        self.is_locked = False
        self.mean = None
        self.std = None


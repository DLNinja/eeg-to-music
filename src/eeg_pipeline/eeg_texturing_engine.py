# EEG Texturing Module

# Receives Z-score values to monitor and map to different CC parameters
# Band to CC mapping:
#     Alpha - Reverb      (CC 91) — Relaxation
#     Beta  - Brightness  (CC 74) — Alertness
#     Theta - Chorus      (CC 93) — Shimmer
#     Delta - Tremolo     (CC 92) — Fatigue
#     Gamma - Vibrato     (CC  1) — Focus
#     FAA   - Panning     (CC 10) — Approach / withdrawal spatial mapping

import numpy as np
from collections import deque
from src.eeg_pipeline.zscore_tracker import ZScoreTracker

BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class EEGTexturingEngine:

    def __init__(self):
        self.tracker = ZScoreTracker()
        self.band_z_scalars: dict[str, float] = {b: 0.5 for b in BAND_ORDER}
        self.z_history: dict[str, deque] = {b: deque(maxlen=5) for b in BAND_ORDER}
        self.last_z_scores: np.ndarray | None = None # for UI

    @property
    def is_calibrated(self) -> bool:
        return self.tracker.is_locked

    @property
    def calibration_progress(self) -> int:

        if self.tracker.is_locked:
            return self.tracker.window_seconds
        return len(self.tracker.history)

    def reset(self):
        self.tracker.reset()
        self.band_z_scalars = {b: 0.5 for b in BAND_ORDER}

        for d in self.z_history.values():
            d.clear()
        self.last_z_scores = None

    def process(self, band_powers: dict) -> None:
        
        # Build (n_channels, n_bands) raw power matrix
        exclude = {'asymmetry', 'baseline_locked', 'calibration_seconds'}
        band_keys = [k for k in band_powers if k not in exclude]

        if not band_keys:
            return

        raw_power = np.column_stack([band_powers[b]['channels'] for b in band_keys])

        # Z-score baseline
        z_scores = self.tracker.update(raw_power)  # zeros during calibration
        self.last_z_scores = z_scores

        # Sigmoid normalisation + channel averaging
        if self.tracker.is_locked and z_scores.size > 0:
            z_norm = 1.0 / (1.0 + np.exp(-z_scores))
            for i, band in enumerate(BAND_ORDER):
                if i < z_norm.shape[1]:
                    self.band_z_scalars[band] = float(z_norm[:, i].mean())
                else:
                    self.band_z_scalars[band] = 0.5
        else:
            # Not yet calibrated —> stay at neutral midpoint
            self.band_z_scalars = {b: 0.5 for b in BAND_ORDER}

        # Update rolling trend history
        for band in BAND_ORDER:
            self.z_history[band].append(self.band_z_scalars[band])

    def apply_cc(
        self,
        emotion_label: str,
        band_z_scalars: dict,
        asymmetry: float,
        synth,
    ) -> None:
        
        if synth is None:
            return

        alpha_z = band_z_scalars.get('alpha', 0.5)
        beta_z  = band_z_scalars.get('beta',  0.5)
        theta_z = band_z_scalars.get('theta', 0.5)
        delta_z = band_z_scalars.get('delta', 0.5)
        gamma_z = band_z_scalars.get('gamma', 0.5)

        alpha_trend = self.trend('alpha')
        beta_trend  = self.trend('beta')
        delta_trend = self.trend('delta')

        for ch in [0, 1]:
            
            # Brightness (CC 74) & Reverb (CC 91) — Valence + Arousal texture
            if emotion_label == 'happy':
                if alpha_z > 0.5:   # Contentment / Cozy (conflicting arousal)
                    bright_val = 40 + beta_z * 30
                    reverb_val = 60 + alpha_z * 60
                else:               # Excitement (aligned arousal)
                    bright_val = 80 + beta_z * 47
                    reverb_val = max(0, 50 - beta_z * 50)
            elif emotion_label == 'fear':
                if beta_z > 0.5:    # Hyper-vigilance / Panic
                    bright_val = 90 + beta_z * 37
                    reverb_val = 20 + alpha_z * 40
                else:               # Paralyzed Dread
                    bright_val = 30 + beta_z * 40
                    reverb_val = 40 + alpha_z * 60
            elif emotion_label == 'sad':
                bright_val = 60 + beta_z * 40 if beta_z > 0.5 else 30 + beta_z * 30
                reverb_val = 40 + alpha_z * 80
            else:  # neutral
                bright_val = 50 + beta_z * 60
                reverb_val = 40 + alpha_z * 80

            # Trend modifiers
            reverb_val += alpha_trend * 40
            bright_val += beta_trend  * 25

            # Chorus (CC 93) — Theta shimmer
            if emotion_label == 'fear':
                chorus_val = 30 + theta_z * 90
            elif emotion_label == 'neutral':
                chorus_val = theta_z * 100
            else:
                chorus_val = theta_z * 60

            # Vibrato (CC 1) — Gamma focus
            if emotion_label == 'happy':
                vibrato_val = gamma_z * 100
            elif emotion_label == 'fear':
                vibrato_val = 20 + gamma_z * 80
            else:
                vibrato_val = gamma_z * 60

            # Tremolo (CC 92) — Delta fatigue / heaviness
            if emotion_label in ('sad', 'fear'):
                tremolo_val = 20 + delta_z * 80
            elif emotion_label == 'happy':
                tremolo_val = delta_z * 50
            else:
                tremolo_val = delta_z * 60
            tremolo_val += delta_trend * 50

            # 5. FAA - Panning (CC 10) + Chorus / Reverb spatial modifiers
            pan_val = 64 - np.clip(asymmetry * 20, -32, 32)
            if asymmetry > 0.1:     # Approach / Engagement
                chorus_mod = asymmetry * 25
                reverb_mod = -asymmetry * 20
            elif asymmetry < -0.1:  # Withdrawal / Avoidance
                chorus_mod = abs(asymmetry) * 55
                reverb_mod = abs(asymmetry) * 35
            else:
                chorus_mod = 0
                reverb_mod = 0

            # Neural EQ (CC 71) & Expression / Compression (CC 11)
            res_val        = 30 + gamma_z * 70
            expression_val = 70 + ((beta_z + gamma_z) / 2.0) * 57

            self.send_cc(synth, ch, 74, bright_val)
            self.send_cc(synth, ch, 91, reverb_val + reverb_mod)
            self.send_cc(synth, ch, 93, chorus_val + chorus_mod)
            self.send_cc(synth, ch, 1,  vibrato_val)
            self.send_cc(synth, ch, 92, tremolo_val)
            self.send_cc(synth, ch, 10, pan_val)
            self.send_cc(synth, ch, 71, res_val)
            self.send_cc(synth, ch, 11, expression_val)

    def send_cc(self, synth, ch: int, ctrl: int, val: float) -> None:
        # Clamp the CC value to MIDI interval [0, 127]
        synth.cc(ch, ctrl, int(max(0, min(127, val))))

    def trend(self, band: str) -> float:
        # Trend of z-score changes
        h = self.z_history[band]
        
        if len(h) < 3:
            return 0.0
        
        recent = np.mean(list(h)[-2:])
        older  = np.mean(list(h)[:2])
        return float(recent - older)

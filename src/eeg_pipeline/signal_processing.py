# Signal processing related methods
# Split pipeline methods in OfflineProcessor and RealtimeProcessor

import numpy as np
import mne
from scipy.signal import butter, sosfilt, sosfilt_zi, get_window, iirnotch, tf2sos
from scipy.fft import fft

n_channels = 62
sf         = 200

bands = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  14),
    'beta':  (14, 31),
    'gamma': (31, 50),
}

# Frontal electrode indices in the SEED-62 layout (10-20 system):
#   Left:  FP1(0), AF3(3), F7(5), F3(7)
#   Right: FP2(2), AF4(4), F8(13), F4(11)
FAA_LEFT_IDX  = [0, 3, 5, 7]
FAA_RIGHT_IDX = [2, 4, 13, 11]

def compute_features(segment_data: np.ndarray, fs: int, stft_n: int = 256) -> tuple:
    # Feature extraction for a 1-second segment
    # Computes band powers, DE features and FAA score

    band_list   = list(bands.items())
    n_bands     = len(band_list)
    n_ch, win   = segment_data.shape

    f_start_idx = np.array([(lo / fs * stft_n) for _, (lo, _) in band_list], dtype=int)
    f_end_idx   = np.array([(hi / fs * stft_n) for _, (_, hi) in band_list], dtype=int)

    window   = get_window("hann", win)
    windowed = segment_data * window
    fft_data = fft(windowed, n=stft_n, axis=1)
    mag_sq   = np.abs(fft_data[:, :stft_n // 2]) ** 2

    total_pow   = mag_sq.sum(axis=1) + 1e-12
    de_features = np.zeros((n_ch, n_bands))
    band_powers = {}

    for b, (bname, _) in enumerate(band_list):
        lo, hi        = f_start_idx[b], f_end_idx[b]
        band_mag      = mag_sq[:, lo:hi + 1]
        band_energy   = band_mag.mean(axis=1)
        de_features[:, b] = np.log2(100 * band_energy + 1e-12)

        band_sum      = band_mag.sum(axis=1)
        abs_pow_mean  = float(np.log2(band_energy.mean() + 1e-12))

        band_powers[bname] = {
            'mean':         float((band_sum / total_pow).mean()),
            'abs_mean':     abs_pow_mean,
            'channels':     band_sum / total_pow,
            'abs_channels': band_energy,
        }

    # Frontal Alpha Asymmetry
    if 'alpha' in band_powers:
        abs_alpha  = band_powers['alpha']['abs_channels']
        left_pow   = np.mean(abs_alpha[FAA_LEFT_IDX])  + 1e-12
        right_pow  = np.mean(abs_alpha[FAA_RIGHT_IDX]) + 1e-12
        band_powers['asymmetry'] = float(np.log(right_pow) - np.log(left_pow))
    else:
        band_powers['asymmetry'] = 0.0

    return de_features, band_powers


# Offline Pipeline

class OfflineProcessor:
    # Feature extraction for the whole trial at once

    def extract_de_features(
        self,
        signal: np.ndarray,
        segment_len: float,
        stft_n: int,
        fs: int,
    ) -> np.ndarray:
        
        # Bandpass + Notch filters for the whole trial
        ch_names = [f'CH{i}' for i in range(n_channels)]
        info     = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg', verbose=0)
        raw      = mne.io.RawArray(signal, info, verbose=0)
        raw.filter(l_freq=0.1, h_freq=75.0, fir_design='firwin', verbose=0)
        raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=0)
        filtered = raw.get_data()

        # Segment the trial
        _, n_samples  = filtered.shape
        window_len    = int(segment_len * fs)
        n_windows     = n_samples // window_len
        n_bands       = len(bands)
        features      = np.zeros((n_windows, n_channels, n_bands))

        # Extract per segment features
        for w in range(n_windows):
            start   = w * window_len
            segment = filtered[:, start:start + window_len]
            de, _   = compute_features(segment, fs, stft_n)
            features[w] = de

        return features

    def smooth(self, features: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply causal moving-average smoothing along the time axis."""
        T, C, B   = features.shape
        smoothed  = np.zeros_like(features)
        for c in range(C):
            for b in range(B):
                x        = features[:, c, b]
                pad      = window - 1
                x_padded = np.pad(x, (pad, 0), mode='edge')
                kernel   = np.ones(window) / window
                smoothed[:, c, b] = np.convolve(x_padded, kernel, mode='valid')
        return smoothed


# Real-time Pipeline

class RealtimeProcessor:
    """Stateful bandpass filter + per-segment FFT feature extraction."""

    def __init__(self, fs: int = 200):

        # 0.1-75Hz filter for artifact removal
        self.fs  = fs
        nyq      = fs / 2.0
        self.sos = butter(4, [0.1 / nyq, 75.0 / nyq], btype='band', output='sos')
        self._zi_template = sosfilt_zi(self.sos)
        self._zi: np.ndarray | None = None

        # 50Hz notch filter for mainline noise removal
        b, a = iirnotch(50.0, 30.0, fs)
        self.sos_notch = tf2sos(b, a)
        self._zi_notch_template = sosfilt_zi(self.sos_notch)
        self._zi_notch: np.ndarray | None = None

    def reset(self):
        # Reset filter state for the start of a new session

        n_sections = self._zi_template.shape[0]
        self._zi   = np.zeros((n_channels, n_sections, 2))
        for ch in range(n_channels):
            self._zi[ch] = self._zi_template.copy()

        n_sections_notch = self._zi_notch_template.shape[0]
        self._zi_notch = np.zeros((n_channels, n_sections_notch, 2))
        for ch in range(n_channels):
            self._zi_notch[ch] = self._zi_notch_template.copy()

    def filter(self, segment: np.ndarray) -> np.ndarray:
        # Apply both filters on the 1-second segment
        
        # Bandpass filter
        zi_in                     = np.transpose(self._zi, (1, 0, 2))
        filtered, zi_out_T        = sosfilt(self.sos, segment, axis=1, zi=zi_in)
        self._zi                  = np.transpose(zi_out_T, (1, 0, 2))

        # 50Hz Notch filter
        zi_notch_in               = np.transpose(self._zi_notch, (1, 0, 2))
        filtered_notch, zi_notch_out_T = sosfilt(self.sos_notch, filtered, axis=1, zi=zi_notch_in)
        self._zi_notch            = np.transpose(zi_notch_out_T, (1, 0, 2))

        return filtered_notch

    def analyze(self, filtered_segment: np.ndarray, stft_n: int = 256) -> tuple:
        return compute_features(filtered_segment, self.fs, stft_n)

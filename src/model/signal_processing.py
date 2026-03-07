import numpy as np
import mne
from scipy.signal import stft, welch, get_window
from scipy.fft import fft

n_channels = 62
sf = 200
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 50)
}
band_names = list(bands.keys())

def get_de_stft(signal, segment_len, stft_n, fs):
    ch_names = [f'CH{i}' for i in range(n_channels)]
    eeg_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg', verbose=0)
    raw = mne.io.RawArray(signal, eeg_info, verbose=0)
    raw.filter(l_freq=0.1, h_freq=75.0, fir_design='firwin', verbose=0)

    band_starts = np.array([v[0] for v in bands.values()])
    band_ends   = np.array([v[1] for v in bands.values()])

    signal = raw.get_data()
    _, n_samples = signal.shape
    window_len = int(segment_len * fs)
    n_windows = n_samples // window_len
    n_bands = len(band_starts)

    f_start_idx = (np.array(band_starts) / fs * stft_n).astype(int)
    f_end_idx   = (np.array(band_ends)   / fs * stft_n).astype(int)

    window = get_window("hann", window_len)

    features = np.zeros((n_windows, n_channels, n_bands))

    for w in range(n_windows):
        start = w * window_len
        end = start + window_len
        segment = signal[:, start:end]

        segment = segment * window

        fft_data = fft(segment, n=stft_n, axis=1)
        mag_sq = np.abs(fft_data[:, :stft_n // 2]) ** 2

        for b in range(n_bands):
            low = f_start_idx[b]
            high = f_end_idx[b]

            band_energy = mag_sq[:, low:high+1].mean(axis=1)
            features[w, :, b] =  np.log2(100* band_energy + 1e-12)

    return features

def moving_average(x, window=3):
    pad = window - 1
    x_padded = np.pad(x, (pad, 0), mode='edge')
    kernel = np.ones(window) / window
    y = np.convolve(x_padded, kernel, mode='valid')
    return y

def smooth_features(features, window=5):
    smoothed = np.zeros_like(features)

    T, C, B = features.shape

    for c in range(C):
        for b in range(B):
            smoothed[:, c, b] = moving_average(features[:, c, b], window)

    return smoothed


# Code to apply the processing and de features extraction on a trial
# features = stft_data = get_de_stft(trials[20], 1, 256, sf)
# smoothed = smooth_features(stft_data)


# ──────────────────────────────────────────────────────
# Real-Time Streaming Functions
# ──────────────────────────────────────────────────────

from scipy.signal import butter, sosfilt, sosfilt_zi

def create_bandpass_filter(fs=200, low=0.1, high=75.0, order=4):
    """Design a Butterworth bandpass filter (SOS form) and compute initial conditions.
    
    Call this ONCE at initialization. Returns (sos, zi_template) where zi_template
    has shape (n_sections, 2) — scale it per channel via np.tile.
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype='band', output='sos')
    zi_single = sosfilt_zi(sos)  # shape: (n_sections, 2)
    return sos, zi_single


def filter_segment(segment, sos, zi):
    """Apply bandpass filter to one segment with state continuity.
    
    Args:
        segment: np.array of shape (n_channels, n_samples) — one 1-second window
        sos: filter coefficients from create_bandpass_filter
        zi: filter state, shape (n_channels, n_sections, 2)
    
    Returns:
        filtered: np.array (n_channels, n_samples) — filtered segment
        zi_out: updated filter state to pass into the next call
    """
    n_ch = segment.shape[0]
    filtered = np.zeros_like(segment)
    zi_out = np.zeros_like(zi)
    
    for ch in range(n_ch):
        filtered[ch], zi_out[ch] = sosfilt(sos, segment[ch], zi=zi[ch])
    
    return filtered, zi_out


def extract_single_window_features(filtered_segment, stft_n=256, fs=200):
    """Extract DE features from one already-filtered 1-second segment.
    
    Args:
        filtered_segment: shape (n_channels, window_len) — already bandpass-filtered
        stft_n: FFT size
        fs: sampling frequency
    
    Returns:
        features: shape (n_channels, n_bands) — DE features for this window
    """
    band_starts = np.array([v[0] for v in bands.values()])
    band_ends   = np.array([v[1] for v in bands.values()])
    n_bands = len(band_starts)
    
    f_start_idx = (band_starts / fs * stft_n).astype(int)
    f_end_idx   = (band_ends   / fs * stft_n).astype(int)
    
    n_ch, window_len = filtered_segment.shape
    window = get_window("hann", window_len)
    
    windowed = filtered_segment * window
    fft_data = fft(windowed, n=stft_n, axis=1)
    mag_sq = np.abs(fft_data[:, :stft_n // 2]) ** 2
    
    features = np.zeros((n_ch, n_bands))
    for b in range(n_bands):
        low = f_start_idx[b]
        high = f_end_idx[b]
        band_energy = mag_sq[:, low:high+1].mean(axis=1)
        features[:, b] = np.log2(100 * band_energy + 1e-12)
    
    return features
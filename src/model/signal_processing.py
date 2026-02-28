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
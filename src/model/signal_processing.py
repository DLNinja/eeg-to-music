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
    # use axis=1 to filter along samples for each channel concurrently
    # zi must be transposed to (n_sections, n_channels, 2) for sosfilt with axis=1
    # or we can iterate if the overhead is small, but axis=1 is usually faster.
    # Actually, sosfilt with axis=1 Expects zi of shape (n_sections, n_channels, 2)
    zi_in = np.transpose(zi, (1, 0, 2))
    filtered, zi_out_transposed = sosfilt(sos, segment, axis=1, zi=zi_in)
    zi_out = np.transpose(zi_out_transposed, (1, 0, 2))
    
    return filtered, zi_out


def analyze_eeg_segment(filtered_segment, stft_n=256, fs=200):
    """Unified spectral analysis: DE features + band powers from a single FFT.
    
    Performs ONE FFT pass and derives both:
    1. Differential Entropy (DE) features for classification
    2. Relative band powers for music texturing and topomaps
    3. Frontal Alpha Asymmetry (FAA) for spatial sonification
    
    Args:
        filtered_segment: shape (n_channels, window_len) — already bandpass-filtered
        stft_n: FFT size
        fs: sampling frequency
    
    Returns:
        tuple: (de_features, band_powers)
            - de_features: shape (n_channels, n_bands)
            - band_powers: dict with 'delta'...'gamma' (each {'mean', 'channels'}) + 'asymmetry'
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
    
    # Total power per channel (for relative power normalization)
    total_pow = mag_sq.sum(axis=1) + 1e-12
    
    de_features = np.zeros((n_ch, n_bands))
    band_powers = {}
    band_name_list = list(bands.keys())
    
    for b in range(n_bands):
        low = f_start_idx[b]
        high = f_end_idx[b]
        band_mag = mag_sq[:, low:high+1]
        
        # DE features: log of mean band energy
        band_energy = band_mag.mean(axis=1)
        de_features[:, b] = np.log2(100 * band_energy + 1e-12)
        
        # Relative band power: sum of band energy / total energy per channel
        band_sum = band_mag.sum(axis=1)
        rel_pow_channels = band_sum / total_pow
        
        # Absolute band power: log-scaled, independent of other bands
        # This prevents a strong Alpha from masking genuine Theta activations
        abs_pow_mean = float(np.log2(band_energy.mean() + 1e-12))
        
        band_powers[band_name_list[b]] = {
            'mean': float(rel_pow_channels.mean()),      # relative (sums to ~1)
            'abs_mean': abs_pow_mean,                      # absolute (independent)
            'channels': rel_pow_channels                   # per-channel relative (for topomaps)
        }
    
    # --- Frontal Alpha Asymmetry (FAA) ---
    # SEED-62 Frontal Indices (International 10-20):
    # Left: FP1(0), AF3(3), F7(5), F3(7)
    # Right: FP2(2), AF4(4), F8(13), F4(11)
    if 'alpha' in band_powers:
        alpha_channels = band_powers['alpha']['channels']
        left_alpha = np.mean(alpha_channels[[0, 3, 5, 7]]) + 1e-12
        right_alpha = np.mean(alpha_channels[[2, 4, 13, 11]]) + 1e-12
        band_powers['asymmetry'] = float(np.log(right_alpha) - np.log(left_alpha))
    else:
        band_powers['asymmetry'] = 0.0
    
    return de_features, band_powers


# Legacy wrappers (for any other callers)
def extract_single_window_features(filtered_segment, stft_n=256, fs=200):
    features, _ = analyze_eeg_segment(filtered_segment, stft_n, fs)
    return features

def extract_band_powers(filtered_segment, fs=200):
    _, bp = analyze_eeg_segment(filtered_segment, fs=fs)
    return bp
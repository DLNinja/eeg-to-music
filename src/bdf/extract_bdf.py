import os
import glob
import numpy as np
import mne

from src.model.signal_processing import get_de_stft, smooth_features

def get_label_from_filename(filename):
    filename = filename.lower()
    if 'sad' in filename:
        return 1
    elif 'fear' in filename:
        return 2
    elif 'happy' in filename:
        return 3
    elif 'neutral' in filename:
        return 0
    else:
        return 0

def extract_features_from_bdf(bdf_path):
    print(f"Processing: {bdf_path}")
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    
    # We need to map BioSemi 64 to SEED 62 channels.
    # We load the first 64 channels (the actual EEG channels, dropping Status)
    raw.pick(raw.ch_names[:64])
    signal_64 = raw.get_data()
    
    # Sampling frequency
    sfreq = raw.info['sfreq']
    
    print(f"  Resampling from {sfreq} Hz to 200 Hz...")
    raw.resample(200.0, npad="auto")
    signal_64_200 = raw.get_data()
    
    # ELECTRODE MAPPING FROM SEED (62) to BIOSEMI (64)
    # -1 means the channel exists in SEED but not in BioSemi (PO5, PO6, CB1, CB2)
    mapping = [0, 32, 33, 2, 35, 6, 5, 4, 3, 37, 38, 39, 40, 41, 7, 8, 9, 10, 46, 45, 44, 43, 42, 14, 13, 12, 11, 47, 48, 49, 50, 51, 15, 16, 17, 18, 31, 55, 54, 53, 52, 22, 21, 20, 19, 30, 56, 57, 58, 59, 24, -1, 25, 29, 62, -1, 61, -1, 26, 28, 63, -1]
    
    seed_channels = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
    proper_casing = [c.replace('Z', 'z').replace('FP', 'Fp') for c in seed_channels]
    
    signal_200 = np.zeros((62, signal_64_200.shape[1]))
    for i, m in enumerate(mapping):
        if m != -1:
            signal_200[i] = signal_64_200[m]
            
    # __________________________________________________
    # TOPOGRAPHICAL SPHERICAL SPLINE INTERPOLATION (MNE)
    # __________________________________________________

    print("  Applying Topographical Spherical Spline Interpolation for missing electrodes...")
    info = mne.create_info(ch_names=proper_casing, sfreq=200, ch_types='eeg')
    raw_seed = mne.io.RawArray(signal_200, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1005')
    ch_pos = montage.get_positions()['ch_pos']
    
    # Manually added CB1 and CB2 (approximate near O1/O2 but lower z/y)
    if 'O1' in ch_pos:
        cb1_pos = ch_pos['O1'].copy()
        cb1_pos[2] -= 0.02 # shift down 2cm
        cb1_pos[1] -= 0.01 # shift back 1cm
        ch_pos['CB1'] = cb1_pos
    if 'O2' in ch_pos:
        cb2_pos = ch_pos['O2'].copy()
        cb2_pos[2] -= 0.02
        cb2_pos[1] -= 0.01
        ch_pos['CB2'] = cb2_pos
        
    custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw_seed.set_montage(custom_montage)
    
    # The 4 missing channels
    raw_seed.info['bads'] = ['PO5', 'PO6', 'CB1', 'CB2']
    
    # Interpolation using surrounding topographic values
    raw_seed.interpolate_bads(reset_bads=True, verbose=False)
    signal_200 = raw_seed.get_data()
        
    
    print("  Extracting DE STFT features...")
    # (n_windows, 62, 5)
    stft_data = get_de_stft(signal_200, segment_len=1.0, stft_n=256, fs=200.0)
    
    print("  Smoothing features...")
    smoothed_data = smooth_features(stft_data, window=5)
    
    return smoothed_data

def main():
    bdf_dir = os.path.join("models", "bpd")
    bdf_files = glob.glob(os.path.join(bdf_dir, "*.bdf"))
    
    if not bdf_files:
        print(f"No BDF files found in {bdf_dir}")
        return
        
    X_list = []
    y_list = []
    
    for bdf_file in bdf_files:
        label = get_label_from_filename(os.path.basename(bdf_file))
        features = extract_features_from_bdf(bdf_file)
        
        n_windows = features.shape[0]
        y_labels = np.full((n_windows,), label)
        
        X_list.append(features)
        y_list.append(y_labels)
        print(f"  -> Extracted {n_windows} windows of shape {features.shape[1:]}. Label: {label}")
        
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    out_file = os.path.join(bdf_dir, "extracted_features.npz")
    np.savez(out_file, X=X_all, y=y_all)
    print(f"\nSuccessfully saved {X_all.shape[0]} total windows to {out_file}")

if __name__ == "__main__":
    main()

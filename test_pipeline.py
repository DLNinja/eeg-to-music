import os
import scipy.io
import numpy as np
import torch
from src.model.signal_processing import get_de_stft, smooth_features, sf
from src.model.emotion_classifier import EEGResNet

def load_trial_data(dataset_path, subject_id, session_id, trial_idx):
    """
    Loads EEG data for a specific trial from a dataset.
    
    Args:
        dataset_path (str): Path to the dataset (e.g., 'data/raw/eeg_seed')
        subject_id (int): Subject ID (e.g., 1)
        session_id (int): Session ID (1, 2, or 3)
        trial_idx (int): Trial index to load (0-indexed)
        
    Returns:
        np.ndarray: EEG trial data of shape (channels, samples)
    """
    subject_dir = os.path.join(dataset_path, str(session_id))
    
    # Check if directory exists
    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Directory not found: {subject_dir}")
        
    # Find the .mat file for the subject
    mat_files = [f for f in os.listdir(subject_dir) if f.startswith(f"{subject_id}_") and f.endswith('.mat')]
    
    if not mat_files:
        raise FileNotFoundError(f"No .mat file found for subject {subject_id} in {subject_dir}")
        
    mat_path = os.path.join(subject_dir, mat_files[0])
    print(f"Loading data from: {mat_path}")
    
    # Load .mat file
    mat_data = scipy.io.loadmat(mat_path)
    
    # Extract trial keys (ignoring metadata keys starting with '__')
    trial_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    # Sort or parse keys if necessary. Usually they might be named like 'djc_eeg1', 'djc_eeg2', etc.
    # For now, just taking them in the order they appear or by sorting
    trial_keys.sort()
    
    if trial_idx >= len(trial_keys):
        raise IndexError(f"Trial index {trial_idx} out of range. Found {len(trial_keys)} trials.")
        
    target_key = trial_keys[trial_idx]
    print(f"Extracting trial: {target_key}")
    
    trial_data = mat_data[target_key]
    print(f"Trial data shape: {trial_data.shape}")
    
    return trial_data

def test_pipeline():
    # Parameters
    dataset_path = 'data/raw/eeg_seed'
    subject_id = 9
    session_id = 3
    trial_idx = 0
    
    segment_len = 1  # 1 second windows
    stft_n = 256
    
    try:
        # 1. Load Data
        print("--- Step 1: Loading Data ---")
        trial_data = load_trial_data(dataset_path, subject_id, session_id, trial_idx)
        
        # 2. Extract Features
        print("\n--- Step 2: Feature Extraction ---")
        print("Extracting DE STFT features...")
        features = get_de_stft(trial_data, segment_len, stft_n, sf)
        print(f"Extracted features shape (Time, Channels, Bands): {features.shape}")
        
        # 3. Smooth Features
        print("\n--- Step 3: Feature Smoothing ---")
        smoothed_features = smooth_features(features)
        print(f"Smoothed features shape: {smoothed_features.shape}")
        
        # 4. Prepare for Model
        print("\n--- Step 4: Model Prediction ---")
        # Format for ResNet: (Batch, Channels, Height, Width)
        # Here we have 1 trial, 1 sequence. 
        # Let's reshape/transpose features to be treated as an image for ResNet
        # One option: Treat bands or channels as spatial dimensions
        
        # Currently features are (T, C, B).
        # We need (B, C_in, H, W). 
        # C_in=1, H=Channels=62, W=Bands=5
        T, C, B = smoothed_features.shape
        import matplotlib.pyplot as plt

        print(f"Number of time frames: {T}")

        # Convert to tensor: shape (T, 1, 62, 5)
        input_tensor = torch.tensor(smoothed_features, dtype=torch.float32).unsqueeze(1)
        print(f"Model input tensor shape: {input_tensor.shape}")
        
        model = EEGResNet(num_classes=4) 
        checkpoint = torch.load("models/best_model_stft_smooth.pt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            
        print(f"Model output shape: {output.shape}")
        
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
        
        print("\nPlotting emotion probabilities over time...")
        emotion_labels = ['Emotion 0', 'Emotion 1', 'Emotion 2', 'Emotion 3']
        time_axis = np.arange(T) * segment_len

        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.plot(time_axis, probabilities[:, i], label=emotion_labels[i])
            
        plt.title(f"Emotion Probabilities Over Time (Subject {subject_id}, Session {session_id}, Trial {trial_idx})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Probability")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"emotion_probs_s{subject_id}_sess{session_id}_t{trial_idx}.png")
        print(f"Plot saved as emotion_probs_s{subject_id}_sess{session_id}_t{trial_idx}.png")
        
        print("\nPipeline test completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_pipeline()

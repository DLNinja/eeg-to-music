# EEG to Music

An end-to-end desktop application that takes raw EEG signals, classifies the underlying emotional state using a deep learning model (ResNet-18), and procedurally generates expressive MIDI music reflecting that emotion — all through a custom-built PyQt5 interface with real-time visualization.

Built as a research/university project exploring the intersection of Brain-Computer Interfaces (BCI), affective computing, and algorithmic music composition.

---

## Overview

The application implements a full signal processing and classification pipeline:

1. **EEG Signal Loading** — reads `.mat` files containing multi-channel EEG recordings (62 channels, 200 Hz sampling rate, SEED dataset format)
2. **Feature Extraction** — computes Differential Entropy (DE) features via STFT across 5 frequency bands (delta, theta, alpha, beta, gamma)
3. **Temporal Smoothing** — applies a moving average convolution to reduce noise in the feature space
4. **Emotion Classification** — feeds smoothed features into a ResNet-18 model (modified for single-channel 62×5 inputs) trained to classify 4 emotional states: Neutral, Sad, Fear, Happy
5. **Music Generation** — maps classified emotions to musical parameters (scale, tempo, velocity, density) and generates MIDI sequences
6. **Synthesized Playback** — renders MIDI to audio in real time via FluidSynth with a GM soundfont, synchronized with a visual piano roll and timeline

---

## Project Structure

```
EEGtoMusic/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies (pip freeze)
├── models/                          # Pre-trained model weights (not tracked by git)
│   └── best_model_stft_smooth.pt
├── data/                            # Raw EEG .mat files (not tracked by git)
│   └── raw/eeg_seed/
├── music/                           # Generated MIDI output (not tracked by git)
└── src/
    ├── model/
    │   ├── signal_processing.py     # STFT, DE features, bandpass filtering, smoothing
    │   └── emotion_classifier.py    # ResNet-18 architecture definition
    ├── music/
    │   └── midi_generator.py        # Emotion → MIDI parameter mapping & generation
    └── ui/
        ├── main_window.py           # Main window, view stack, theme system
        └── views/
            ├── home_view.py         # Home menu with navigation + theme toggle
            ├── plot_view.py         # Standalone EEG signal plotter
            ├── pipeline_view.py     # Full pipeline: load → classify → generate → play
            ├── realtime_view.py     # Real-time streaming classifier (threaded)
            └── music_view.py        # DAW-style MIDI player with piano roll
```

---

## Setup & Installation

### 1. System Dependencies

The application uses **FluidSynth** for MIDI-to-audio synthesis. Install it via your package manager:

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install fluidsynth fluid-soundfont-gm
```

**macOS (Homebrew):**
```bash
brew install fluidsynth
```

### 2. Python Environment

Create and activate a virtual environment (Python 3.8+):
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages from the requirements file:
```bash
pip install -r requirements.txt
```

### 4. Data & Model Setup

Before running the app you need to place:
- **EEG data**: raw `.mat` trial files in `data/raw/eeg_seed/`
- **Model weights**: the trained checkpoint file at `models/best_model_stft_smooth.pt`

> These files are not included in the repository due to their size.

---

## Running the Application

```bash
python main.py
```

---

## Application Pages

### Home

The main menu. From here you can navigate to any of the four pages below, and switch between **Dark** and **Light** themes using the dropdown at the bottom.

### EEG Signal Plotter

A standalone viewer for exploring raw EEG signals from loaded `.mat` files.

- Load a `.mat` file and select a trial from the dropdown
- Choose channel display mode: **Single Channel**, **Channel Range**, or **All 62 Channels**
- Toggle between **Full Signal** view and **Windowed View** with adjustable window size and a scrollbar for navigation

### Emotion Pipeline

The full classification and music generation workflow:

1. Open a `.mat` file → select a trial
2. Click **Run Classification** — the app extracts STFT features, smooths them, and runs the ResNet-18 classifier
3. View the EEG signal and emotion probability curves on synchronized plots
4. Click **Generate Music** — maps the classified emotions to MIDI parameters and creates a `.mid` file
5. The embedded **Music Player** appears with playback controls, a piano roll visualization, and a playhead synced to the emotion timeline

### Real-Time Emotion Classifier

Simulates live EEG streaming with per-second classification:

1. Load a `.mat` file → select a trial → press **▶ Play**
2. The EEG signal streams live on the top plot (5-second sliding window)
3. Every 1 second (200 samples), the system automatically:
   - Bandpass filters the segment using `scipy.signal.sosfilt` with persistent filter state
   - Extracts DE features via windowed FFT
   - Applies moving average smoothing
   - Classifies with the ResNet-18 model
4. The emotion probability plot updates in real time as classifications come in
5. **All classification runs in a background thread** — the EEG plot never stutters, even at 10× speed
6. After playback finishes, a **Review Mode** appears with Full/Windowed view and scrollbar to browse the complete recording

**Controls:** Play / Pause / Stop, Speed (1×/2×/5×/10×), Channel selection (Single / Range / All)

### Music Player & Visualizer

A DAW-style MIDI player with:
- File browser for `.mid` files
- Transport controls (Play / Pause / Stop) with a time slider
- Piano roll visualization showing notes over time
- Real-time audio synthesis via FluidSynth

---

## Tech Stack

| Component | Technology |
|---|---|
| GUI Framework | PyQt5 with custom QPainter rendering |
| Signal Processing | NumPy, SciPy (STFT, bandpass filtering) |
| EEG Preprocessing | MNE-Python (batch), SciPy sosfilt (real-time) |
| Deep Learning | PyTorch (ResNet-18) |
| Music Generation | Mido (MIDI creation) |
| Audio Synthesis | FluidSynth via pyfluidsynth |
| Plotting | Custom QPainter widgets (no Matplotlib dependency) |

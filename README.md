# EEG to Music Application

A complete GUI pipeline for classifying emotions from EEG Signal Data and procedurally generating Music (MIDI) based on emotional states, complete with a DAW-style interface for synthesized playback and visual timeline synchronization.

## Setup & Installation

This project requires Python 3.8+ and standard system dependencies for audio playback.

### 1. System Dependencies
In order to synthesize and playback audio cleanly, the application uses **FluidSynth**. You will need to install it and a base soundfont using your system's package manager:

**Ubuntu / Debian Linux**:
```bash
sudo apt update
sudo apt install fluidsynth fluid-soundfont-gm
```

**MacOS** (via Homebrew):
```bash
brew install fluidsynth
```

### 2. Python Environment Setup
Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
# (On Windows use: venv\Scripts\activate)
```

### 3. Install Required Packages
Install the required packages directly into your virtual environment:

```bash
pip install PyQt5 matplotlib numpy scipy torch mido pyfluidsynth
```

## Running the Application

Before running the app, ensure you have:
1. Placed your raw `.mat` EEG data trials in the `data/raw/eeg_seed/` directory.
2. Placed the compiled PyTorch model inside the `models/` directory (e.g., `models/best_model_stft_smooth.pt`).

Run the application:
```bash
python main.py
```

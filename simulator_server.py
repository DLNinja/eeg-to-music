import socket
import struct
import time
import threading
import scipy.io
import os
import numpy as np
import math
import scipy.signal

HOST = '0.0.0.0'
PORT = 8888
CHANNELS = 62          # Channels in the data file
STREAM_CHANNELS = 64   # Channels actually sent over TCP (extra ones are zeros)
SAMPLE_RATE = 256   # Hz — rate at which data is streamed to the client
DATA_RATE = 200     # Hz — original sample rate of the .mat files
                    #       set SAMPLE_RATE > DATA_RATE to upsample before streaming
SAMPLES_PER_PACKET = 2  # Number of samples bundled into each TCP packet
INTERVAL = SAMPLES_PER_PACKET / SAMPLE_RATE  # Sleep time between sends
MAT_FILE = 'data/raw/eeg_seed/1/1_20131027.mat'

# BioSemi protocol constants
# BioSemi ADC resolution: 1 bit = 31.25 nV = 0.03125 µV
# So 1 µV = 1 / 0.03125 = 32 bits
UV_TO_BITS = 32
BYTES_PER_SAMPLE = 3  # 24-bit samples


def float_to_biosemi_24bit(value_uv):
    """
    Convert a float µV value into a 3-byte little-endian signed integer,
    matching the BioSemi ActiView TCP/IP format.
    """
    raw_int = int(round(value_uv * UV_TO_BITS))
    raw_int = max(-8388608, min(8388607, raw_int))
    raw_bytes = raw_int.to_bytes(4, byteorder='little', signed=True)
    return raw_bytes[:3]


def load_eeg_data(filepath):
    """Loads all trials of EEG data, upsampling from DATA_RATE to SAMPLE_RATE."""
    print(f"Loading {filepath} into memory...")
    mat = scipy.io.loadmat(filepath)

    # Auto-detect trial keys (e.g. 'cz_eeg1', 'djc_eeg1', etc.)
    trial_keys = sorted(
        [k for k in mat.keys() if not k.startswith('_') and 'eeg' in k],
        key=lambda x: int(x.split('eeg')[1])
    )

    # Compute upsample ratio
    if SAMPLE_RATE != DATA_RATE:
        g = math.gcd(SAMPLE_RATE, DATA_RATE)
        up = SAMPLE_RATE // g
        down = DATA_RATE // g
        print(f"Upsampling trials from {DATA_RATE} Hz → {SAMPLE_RATE} Hz (ratio {up}:{down})")
    else:
        up, down = 1, 1

    trials = []
    for key in trial_keys:
        data_trial = mat[key][:CHANNELS, :]  # (62, N_original)
        if up != 1 or down != 1:
            # resample_poly operates along last axis by default
            data_trial = scipy.signal.resample_poly(data_trial, up, down, axis=1).astype(np.float32)
        trials.append(data_trial)

    orig_len = mat[trial_keys[0]].shape[1] if trial_keys else 0
    new_len = trials[0].shape[1] if trials else 0
    print(f"Loaded {len(trials)} trials: {orig_len} → {new_len} samples/trial after upsampling.")
    return trials


def handle_client(conn, addr, all_trials):
    print(f"Connected by {addr}")
    try:
        trial_index = 0
        while True:
            current_trial = all_trials[trial_index]
            num_samples = current_trial.shape[1]
            print(f"Streaming trial {trial_index + 1} ({num_samples} samples) to {addr}")

            sample_idx = 0
            while sample_idx < num_samples:
                # Collect up to SAMPLES_PER_PACKET samples into one packet
                packet = bytearray()
                samples_in_packet = 0

                while samples_in_packet < SAMPLES_PER_PACKET and sample_idx < num_samples:
                    data = current_trial[:, sample_idx]
                    for ch in range(CHANNELS):
                        packet.extend(float_to_biosemi_24bit(data[ch]))
                    # Pad with zero-value dummy channels to reach STREAM_CHANNELS
                    zero_bytes = float_to_biosemi_24bit(0.0)
                    for _ in range(STREAM_CHANNELS - CHANNELS):
                        packet.extend(zero_bytes)
                    sample_idx += 1
                    samples_in_packet += 1

                conn.sendall(packet)
                # Sleep for exactly the real-time duration of this packet
                time.sleep(samples_in_packet / SAMPLE_RATE)

            # Loop to next trial
            trial_index = (trial_index + 1) % len(all_trials)

    except (ConnectionResetError, BrokenPipeError):
        print(f"\nClient {addr} disconnected.")
    except Exception as e:
        print(f"\nError handling client {addr}: {e}")
    finally:
        conn.close()


def main():
    if not os.path.exists(MAT_FILE):
        print(f"Error: {MAT_FILE} not found. Please ensure it is in the same directory.")
        return

    all_trials = load_eeg_data(MAT_FILE)

    packet_bytes = STREAM_CHANNELS * BYTES_PER_SAMPLE * SAMPLES_PER_PACKET

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        print(f"EEG Simulator Server (BioSemi Protocol) listening on {HOST}:{PORT}")
        print(f"Streaming {CHANNELS} real + {STREAM_CHANNELS - CHANNELS} dummy channels at {SAMPLE_RATE} Hz")
        print(f"Packet format: {SAMPLES_PER_PACKET} samples x {STREAM_CHANNELS} ch x {BYTES_PER_SAMPLE} bytes = {packet_bytes} bytes/packet")
        print(f"(Client should read {STREAM_CHANNELS} ch and discard channels {CHANNELS+1}–{STREAM_CHANNELS})")
        print("Waiting for clients to connect...")

        try:
            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=handle_client, args=(conn, addr, all_trials))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()

"""
train_markov_midi.py — Time-Aligned Markov Chain Extraction from VGMIDI

Reads the VGMIDI dataset (continuous V-A annotations + MIDI files),
time-aligns MIDI events with the 32-sample V-A curves, and builds
per-quadrant Markov transition matrices for:
  1. Melody pitch intervals  (current_interval → next_interval)
  2. Note durations           (current_duration_bin → next_duration_bin)
  3. Quadrant transitions     (current_quadrant → next_quadrant)

Output: JSON files in models/transitions/ for each of the 4 quadrants.

Usage:
  python src/music/train_markov_midi.py [--download]

  --download    Clone the VGMIDI repo into datasets/vgmidi/ first.
"""

import os
import sys
import json
import math
import subprocess
from collections import defaultdict
import mido

# ─── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR  = os.path.join(PROJECT_ROOT, 'datasets', 'vgmidi')
ANNO_DIR     = os.path.join(DATASET_DIR, 'labelled', 'annotations')
MIDI_DIR     = os.path.join(DATASET_DIR, 'labelled', 'midi')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'models', 'transitions')

ANNO_FILES   = ['vgmidi_raw_1.json', 'vgmidi_raw_2.json']
N_SAMPLES    = 32  # Each annotation has exactly 32 time-aligned V-A samples

# Duration bins (in seconds) — coarse bucketing for Markov state
DURATION_BINS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # edges
DURATION_LABELS = ['sixteenth', 'eighth', 'quarter', 'half', 'whole', 'long']

# ─── Helpers ────────────────────────────────────────────────────────────────────

def download_dataset():
    """Clone the VGMIDI repo from GitHub."""
    if os.path.exists(DATASET_DIR):
        print(f"Dataset directory already exists: {DATASET_DIR}")
        return
    os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)
    print(f"Cloning VGMIDI into {DATASET_DIR}...")
    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/lucasnfe/vgmidi.git', DATASET_DIR],
        check=True
    )
    print("Download complete.")

def get_quadrant(v, a):
    """Map continuous V-A to a discrete quadrant label."""
    if abs(v) < 0.15 and abs(a) < 0.15:
        return 'neutral'
    if v >= 0 and a >= 0:
        return 'happy'
    if v < 0 and a < 0:
        return 'sad'
    if v < 0 and a >= 0:
        return 'fear'
    # v >= 0, a < 0 → serene/peaceful, map to neutral
    return 'neutral'

def bin_duration(seconds):
    """Quantise a note duration (seconds) into a named bin."""
    for i, edge in enumerate(DURATION_BINS):
        if seconds <= edge:
            return DURATION_LABELS[i]
    return DURATION_LABELS[-1]

def midi_to_timed_notes(filepath):
    """
    Parse a MIDI file and return a flat list of
    (onset_seconds, pitch, duration_seconds) sorted by onset.
    """
    try:
        mid = mido.MidiFile(filepath)
    except Exception as e:
        print(f"  Skipping {os.path.basename(filepath)}: {e}")
        return []

    tempo = 500000  # default 120 BPM
    ticks_per_beat = mid.ticks_per_beat

    # Flatten all tracks into absolute-time events
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.type in ('note_on', 'note_off'):
                abs_sec = mido.tick2second(abs_tick, ticks_per_beat, tempo)
                events.append((abs_sec, msg.type, msg.note, msg.velocity))

    events.sort(key=lambda x: x[0])

    # Pair note_on → note_off to get durations
    active = {}   # pitch → onset_time
    notes = []
    for t, typ, pitch, vel in events:
        if typ == 'note_on' and vel > 0:
            active[pitch] = t
        elif typ == 'note_off' or (typ == 'note_on' and vel == 0):
            if pitch in active:
                onset = active.pop(pitch)
                dur = max(0.01, t - onset)
                notes.append((onset, pitch, dur))

    notes.sort(key=lambda x: x[0])
    return notes

def get_piece_duration(filepath):
    """Get total duration of a MIDI file in seconds."""
    try:
        mid = mido.MidiFile(filepath)
        return mid.length
    except:
        return 0

def average_annotations(annotations_for_piece):
    """
    Given a list of (valence_array, arousal_array) from multiple annotators,
    return the element-wise averaged (valence, arousal) arrays.
    """
    n = len(annotations_for_piece)
    if n == 0:
        return None, None
    length = len(annotations_for_piece[0][0])
    avg_v = [0.0] * length
    avg_a = [0.0] * length
    for v_arr, a_arr in annotations_for_piece:
        for i in range(length):
            avg_v[i] += v_arr[i]
            avg_a[i] += a_arr[i]
    avg_v = [x / n for x in avg_v]
    avg_a = [x / n for x in avg_a]
    return avg_v, avg_a

# ─── Main Extraction ───────────────────────────────────────────────────────────

def extract_transitions():
    """
    Core algorithm:
    1. For each annotated piece, load the averaged 32-sample V-A curve.
    2. Load the corresponding MIDI file and extract timed notes.
    3. Divide the MIDI timeline into 32 equal windows.
    4. For each window, assign the V-A value → quadrant.
    5. For each consecutive pair of notes within a window,
       record (pitch_interval, duration_bin) transitions under that quadrant.
    6. Also record quadrant-to-quadrant transitions across windows.
    """
    # Step 1: Parse annotations — group by piece index
    piece_annotations = defaultdict(list)  # piece_idx → [(v_array, a_array), ...]

    for anno_file in ANNO_FILES:
        path = os.path.join(ANNO_DIR, anno_file)
        if not os.path.exists(path):
            print(f"WARNING: Annotation file not found: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations = data.get('annotations', data)
        for key, entry in annotations.items():
            # key format: "pieceX_Y" where X = piece index, Y = annotator
            parts = key.split('_')
            piece_idx = parts[0].replace('piece', '')
            v_arr = entry.get('valence', [])
            a_arr = entry.get('arousal', [])
            if len(v_arr) == N_SAMPLES and len(a_arr) == N_SAMPLES:
                piece_annotations[piece_idx].append((v_arr, a_arr))

    print(f"Found annotations for {len(piece_annotations)} pieces.")

    # Step 2: Build the MIDI filename mapping
    # VGMIDI CSV maps piece index → filename
    csv_path = os.path.join(DATASET_DIR, 'vgmidi_labelled.csv')
    piece_to_midi = {}
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                midi_path = row.get('midi', '')
                # The CSV path is relative: labelled/phrases/...
                # We need the full MIDI from labelled/midi/
                basename = os.path.basename(midi_path)
                # Strip the trailing _0.mid, _1.mid suffix to get the original filename
                name_parts = basename.rsplit('_', 1)
                if len(name_parts) == 2:
                    original_name = name_parts[0] + '.mid'
                else:
                    original_name = basename
                piece_to_midi[str(i)] = original_name

    # Initialise transition counters
    quadrant_transitions = {
        q: {
            'pitch_interval_1': defaultdict(lambda: defaultdict(int)),  # 1st order
            'pitch_interval_2': defaultdict(lambda: defaultdict(int)),  # 2nd order
            'pitch_interval_3': defaultdict(lambda: defaultdict(int)),  # 3rd order
            'duration':         defaultdict(lambda: defaultdict(int)),
        }
        for q in ['happy', 'sad', 'fear', 'neutral']
    }
    quadrant_flow = defaultdict(lambda: defaultdict(int))  # from_quad → to_quad
    stats = {'pieces_processed': 0, 'notes_counted': 0, 'windows_counted': 0}

    # Step 3: Process each piece
    for piece_idx, anno_list in piece_annotations.items():
        midi_filename = piece_to_midi.get(piece_idx)
        if not midi_filename:
            continue

        midi_path = os.path.join(MIDI_DIR, midi_filename)
        if not os.path.exists(midi_path):
            # Try with spaces in filename
            continue

        # Average annotations across all annotators
        avg_v, avg_a = average_annotations(anno_list)
        if avg_v is None:
            continue

        # Parse MIDI
        notes = midi_to_timed_notes(midi_path)
        if len(notes) < 4:
            continue

        total_dur = get_piece_duration(midi_path)
        if total_dur <= 0:
            continue

        window_dur = total_dur / N_SAMPLES
        stats['pieces_processed'] += 1

        # Step 4: Assign notes to windows and record transitions
        prev_quadrant = None
        for w in range(N_SAMPLES):
            v = avg_v[w]
            a = avg_a[w]
            quadrant = get_quadrant(v, a)

            # Record quadrant flow
            if prev_quadrant is not None:
                quadrant_flow[prev_quadrant][quadrant] += 1
            prev_quadrant = quadrant

            # Find notes in this window
            win_start = w * window_dur
            win_end = (w + 1) * window_dur
            window_notes = [n for n in notes if win_start <= n[0] < win_end]

            if len(window_notes) < 2:
                continue

            stats['windows_counted'] += 1

            # Maintain a sliding window of last 3 intervals (initialized to 0)
            prev_intervals = [0, 0, 0]

            # Record note-to-note transitions within this window
            for i in range(len(window_notes) - 1):
                _, p1, d1 = window_notes[i]
                _, p2, d2 = window_notes[i + 1]

                interval_next = p2 - p1

                state1 = str(prev_intervals[-1])
                state2 = f"{prev_intervals[-2]},{prev_intervals[-1]}"
                state3 = f"{prev_intervals[-3]},{prev_intervals[-2]},{prev_intervals[-1]}"

                dur_bin_curr = bin_duration(d1)
                dur_bin_next = bin_duration(d2)

                q_data = quadrant_transitions[quadrant]
                q_data['pitch_interval_1'][state1][str(interval_next)] += 1
                q_data['pitch_interval_2'][state2][str(interval_next)] += 1
                q_data['pitch_interval_3'][state3][str(interval_next)] += 1
                q_data['duration'][dur_bin_curr][dur_bin_next] += 1
                stats['notes_counted'] += 1
                
                # Update history
                prev_intervals.pop(0)
                prev_intervals.append(interval_next)

    return quadrant_transitions, quadrant_flow, stats

def normalise(counts_dict):
    """Convert count dict → probability dict."""
    probs = {}
    for state_from, transitions in counts_dict.items():
        total = sum(transitions.values())
        if total > 0:
            probs[state_from] = {k: round(v / total, 4) for k, v in transitions.items()}
    return probs

def main():
    if '--download' in sys.argv:
        download_dataset()

    # Verify dataset exists
    if not os.path.exists(ANNO_DIR):
        print(f"ERROR: Dataset not found at {DATASET_DIR}")
        print("Run with --download to automatically clone the VGMIDI repository.")
        return

    print("=" * 60)
    print("  VGMIDI Time-Aligned Markov Extraction")
    print("=" * 60)

    quadrant_transitions, quadrant_flow, stats = extract_transitions()

    print(f"\nPieces processed: {stats['pieces_processed']}")
    print(f"Windows counted:  {stats['windows_counted']}")
    print(f"Note transitions: {stats['notes_counted']}")

    # Save per-quadrant transition matrices
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for q in ['happy', 'sad', 'fear', 'neutral']:
        output = {
            'pitch_interval_1': normalise(quadrant_transitions[q]['pitch_interval_1']),
            'pitch_interval_2': normalise(quadrant_transitions[q]['pitch_interval_2']),
            'pitch_interval_3': normalise(quadrant_transitions[q]['pitch_interval_3']),
            'duration':         normalise(quadrant_transitions[q]['duration']),
        }

        out_file = os.path.join(OUTPUT_DIR, f'transitions_{q}.json')
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        n_pitch_states = len(output['pitch_interval_1'])
        n_dur_states   = len(output['duration'])
        print(f"  {q:8s}: {n_pitch_states:3d} 1st-order pitch states, {n_dur_states:3d} duration states → {out_file}")

    # Save quadrant flow matrix (emotion transition probabilities)
    flow_output = normalise(quadrant_flow)
    flow_file = os.path.join(OUTPUT_DIR, 'quadrant_flow.json')
    with open(flow_file, 'w') as f:
        json.dump(flow_output, f, indent=2)
    print(f"\n  Quadrant flow matrix → {flow_file}")

    print("\n" + "=" * 60)
    print("  DONE — Transition matrices saved to models/transitions/")
    print("=" * 60)

if __name__ == '__main__':
    main()

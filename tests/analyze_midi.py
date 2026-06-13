"""Analyze a generated MIDI file for chord-melody clashes."""
import mido
import sys

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_name(midi_note):
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"

def pitch_class(midi_note):
    return midi_note % 12

def analyze_midi(filepath):
    mid = mido.MidiFile(filepath)
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath}")
    print(f"Tracks: {len(mid.tracks)}, Ticks/beat: {mid.ticks_per_beat}")
    print(f"{'='*70}\n")

    if len(mid.tracks) < 2:
        print("ERROR: Expected 2 tracks (chords + melody)")
        return

    # Parse both tracks into timed events
    chord_track = mid.tracks[0]
    melody_track = mid.tracks[1]

    def parse_track(track):
        """Returns list of (abs_tick, 'on'/'off', pitch, velocity)"""
        events = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                events.append((abs_tick, 'on', msg.note, msg.velocity))
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                events.append((abs_tick, 'off', msg.note, 0))
        return events

    chord_events = parse_track(chord_track)
    melody_events = parse_track(melody_track)

    # Build time-sliced snapshots: at each tick, what chord notes and melody notes are sounding?
    all_ticks = sorted(set(
        [e[0] for e in chord_events] + [e[0] for e in melody_events]
    ))

    active_chords = set()
    active_melody = set()
    chord_idx = 0
    melody_idx = 0

    clashes = []
    total_steps = 0
    clash_steps = 0

    # Group by "step" (each chord onset = new step)
    chord_onsets = [e for e in chord_events if e[1] == 'on']
    melody_onsets = [e for e in melody_events if e[1] == 'on']

    # For each melody note, find what chord notes are active at that time
    # Build chord intervals: (start_tick, end_tick, [pitches])
    chord_spans = []
    active = {}
    for evt in chord_events:
        tick, typ, pitch, vel = evt
        if typ == 'on':
            active[pitch] = tick
        elif typ == 'off' and pitch in active:
            chord_spans.append((active.pop(pitch), tick, pitch))

    # For each melody note, check against concurrent chord notes
    melody_spans = []
    active = {}
    for evt in melody_events:
        tick, typ, pitch, vel = evt
        if typ == 'on':
            active[pitch] = (tick, vel)
        elif typ == 'off' and pitch in active:
            start, v = active.pop(pitch)
            melody_spans.append((start, tick, pitch, v))

    print(f"Chord notes: {len(chord_spans)}, Melody notes: {len(melody_spans)}")
    print(f"\n--- CHORD-MELODY CLASH ANALYSIS ---\n")

    clash_count = 0
    for m_start, m_end, m_pitch, m_vel in melody_spans:
        # Find all chord notes sounding during this melody note
        concurrent_chords = []
        for c_start, c_end, c_pitch in chord_spans:
            # Overlap check
            if c_start < m_end and c_end > m_start:
                concurrent_chords.append(c_pitch)

        if not concurrent_chords:
            continue

        # Check for clashes: half-step (1 semitone) or full-step (2 semitones) between
        # melody pitch class and any chord pitch class
        m_pc = pitch_class(m_pitch)
        for c_pitch in concurrent_chords:
            c_pc = pitch_class(c_pitch)
            interval = min(abs(m_pc - c_pc), 12 - abs(m_pc - c_pc))
            if interval == 1:  # half-step clash
                clash_count += 1
                tick_sec = m_start / mid.ticks_per_beat / 2  # approximate seconds
                print(f"  [!] HALF-STEP CLASH at ~{tick_sec:.1f}s: "
                      f"melody={note_name(m_pitch)}(vel{m_vel}) vs chord={note_name(c_pitch)} "
                      f"(interval={interval} semitones)")
            elif interval == 2:  # whole-step — less severe but can sound off
                tick_sec = m_start / mid.ticks_per_beat / 2
                # Only report if this is low register (more noticeable)
                if m_pitch < 72:
                    clash_count += 1
                    print(f"  [~] WHOLE-STEP tension at ~{tick_sec:.1f}s: "
                          f"melody={note_name(m_pitch)}(vel{m_vel}) vs chord={note_name(c_pitch)}")

    print(f"\n--- SUMMARY ---")
    print(f"Total clashes found: {clash_count}")
    print(f"Melody notes analyzed: {len(melody_spans)}")
    if len(melody_spans) > 0:
        print(f"Clash rate: {clash_count/len(melody_spans)*100:.1f}%")

    # Also check for trill patterns
    print(f"\n--- TRILL ANALYSIS ---")
    melody_pitches = [p for _, _, p, _ in melody_spans]
    trill_runs = 0
    current_run = 1
    max_run = 1
    for i in range(1, len(melody_pitches)):
        if len(set(melody_pitches[max(0,i-3):i+1])) <= 2 and i >= 3:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            if current_run >= 4:
                trill_runs += 1
                tick_sec = melody_spans[i][0] / mid.ticks_per_beat / 2
                print(f"  [T] Trill run of {current_run} notes ending at ~{tick_sec:.1f}s")
            current_run = 1
    if current_run >= 4:
        trill_runs += 1
        print(f"  [T] Trill run of {current_run} notes at end")
    print(f"Total trill runs (4+ alternating notes): {trill_runs}")
    print(f"Longest trill run: {max_run} notes")

    # Repeated note analysis
    print(f"\n--- REPETITION ANALYSIS ---")
    repeat_count = 0
    max_repeat = 1
    cur_repeat = 1
    for i in range(1, len(melody_pitches)):
        if melody_pitches[i] == melody_pitches[i-1]:
            cur_repeat += 1
            if cur_repeat > max_repeat:
                max_repeat = cur_repeat
        else:
            if cur_repeat >= 4:
                repeat_count += 1
                tick_sec = melody_spans[i][0] / mid.ticks_per_beat / 2
                print(f"  [R] Same note repeated {cur_repeat}x at ~{tick_sec:.1f}s: {note_name(melody_pitches[i-1])}")
            cur_repeat = 1
    if cur_repeat >= 4:
        repeat_count += 1
        print(f"  [R] Same note repeated {cur_repeat}x at end: {note_name(melody_pitches[-1])}")
    print(f"Total repetition runs (4+ same note): {repeat_count}")
    print(f"Longest same-note run: {max_repeat}")

if __name__ == '__main__':
    import glob
    files = glob.glob("music/*.mid")
    for f in files:
        analyze_midi(f)

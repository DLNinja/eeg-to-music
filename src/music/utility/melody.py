"""
melody.py — Melody Generation Utilities

Provides the building blocks used by both the offline MIDI generator
and the real-time synthesiser when constructing note-by-note melodies:

  - Trill detection (4-note oscillation guard)
  - Dissonance resolution (snap clashing notes to nearest chord tone)
  - Pool nearest-note lookup (find closest pitch in the mode pool)
  - Harmonic minor pool alteration (raise 7th degree for SAD dominant chords)
  - Chord adherence probability (how strongly the melody snaps to chord tones)

The actual melody loops (motif replay, fresh Markov phrase generation)
live in each orchestrator because the MIDI and real-time generators
schedule notes differently (MIDI ticks vs wall-clock timestamps).
"""

import random


def detect_trill(recent_pitches):
    # Detects rapid alternation between 2 adjacent notes to prevent repetitive trills.
    if len(recent_pitches) < 4:
        return False
    n1, n2, n3, n4 = [p[0] for p in recent_pitches[-4:]]
    return n1 == n3 and n2 == n4 and n1 != n2


def resolve_dissonance(pitch, sounding_chord_notes):
    # Checks for any clashing notes and snaps to the nearest diatonic note.
    for ct in sounding_chord_notes:
        if abs(pitch % 12 - ct % 12) == 1:
            nearest_ct = min(sounding_chord_notes, key=lambda c: abs((c % 12) - (pitch % 12)))
            return int((nearest_ct % 12) + (pitch // 12) * 12)
    return int(pitch)


def pool_idx_nearest(target, pool):
    """Return index of note in pool closest to target pitch."""
    return min(range(len(pool)), key=lambda k: abs(pool[k] - target))


def apply_harmonic_minor_to_pool(active_pool, emotion_cat, current_chord_degree):
    """
    HARMONIC MINOR alteration to SAD macromood:
    When the chord is on scale degree 4 (the dominant V), raise the 7th degree
    in the melody pool to create the leading tone of the harmonic minor scale.
    This makes the V chord a true major dominant for stronger tension-resolution.
    """
    if emotion_cat == 'sad' and current_chord_degree == 4:
        for idx in range(len(active_pool)):
            if (idx % 7) == 6:
                active_pool[idx] += 1
    return active_pool


def compute_chord_adherence(emotion_cat, macro_v, micro_v,
                            active_spike_profile, spike_intensity, force_snap):
    """
    Compute the probability that a melody note snaps to a chord tone.

    Harmonic adherence: MACRO VALENCE anchors to chord, MICRO VALENCE nudges.
    Higher values = melody sticks closer to chord tones (more consonant).
    Lower values  = melody wanders between scale degrees (more melodic freedom).

    Returns: (chord_adherence_prob, force_snap_consumed)
      force_snap_consumed is True if force_snap was active and has now been used.
    """
    chord_adherence_prob = max(0.4, min(0.95, 0.7 + (macro_v * 0.20) + (micro_v * 0.10)))
    # HAPPY/SAD: higher floor to prevent dissonant clashes
    if emotion_cat in ('happy', 'sad'):
        chord_adherence_prob = max(0.80, chord_adherence_prob)
    # NEUTRAL: strong adherence to expose the modal chord color (add6/add9)
    if emotion_cat == 'neutral':
        chord_adherence_prob = max(0.80, chord_adherence_prob)
    # FEAR: high adherence so melody follows the dark non-tertian chords
    if emotion_cat == 'fear':
        chord_adherence_prob = max(0.75, chord_adherence_prob)
    # Spike transitions: boost adherence so melody follows the spike chord coloring
    if active_spike_profile and spike_intensity > 0.3:
        chord_adherence_prob = max(0.90, chord_adherence_prob)

    # force_snap: guarantee first note anchors to chord on emotion shift
    force_snap_consumed = False
    if force_snap:
        chord_adherence_prob = 1.0
        force_snap_consumed = True

    return chord_adherence_prob, force_snap_consumed

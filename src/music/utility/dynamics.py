
# dynamics.py — Rhythm Density & Micro-Timing

import random


def apply_anxiety_rhythm_doubling(chosen_ratios, spike_name, spike_intensity):
    if spike_name == 'ANXIETY' and spike_intensity > 0.3:
        if chosen_ratios == [1.0] or chosen_ratios == [0.5, 0.5]:
            chosen_ratios = [0.5, 0.25, 0.25] if random.random() > 0.5 else [0.333, 0.333, 0.333]
        elif len(chosen_ratios) == 3:
            chosen_ratios = [0.25, 0.25, 0.25, 0.25] if random.random() > 0.5 else [0.125, 0.125, 0.25, 0.5]
    return chosen_ratios


def apply_humanization_jitter(chosen_ratios, emotion_cat):
    if emotion_cat in ('sad', 'neutral') and len(chosen_ratios) > 1:
        jittered = []
        for i, r in enumerate(chosen_ratios):
            jitter = random.uniform(-0.12, 0.12) * r
            jittered.append(r + jitter)
        # Normalize to preserve total duration
        total = sum(jittered)
        chosen_ratios = [r / total for r in jittered]
    return chosen_ratios

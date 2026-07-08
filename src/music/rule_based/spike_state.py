from .music_theory import SPIKE_PROFILES

class SpikeState:
    def __init__(self):
        self.active_profile = None
        self.intensity = 0.0
        self.duration_counter = 0

    def update(self, is_spike, macro_label, spike_label, intensity):
        # Activate or deactivate the spike profile based on the emotion tracker state.
        if is_spike and macro_label != spike_label:
            spike_key = (macro_label, spike_label)
            self.active_profile = SPIKE_PROFILES.get(spike_key)
            self.intensity = intensity
            self.duration_counter += 1
        else:
            self.active_profile = None
            self.intensity = 0.0
            self.duration_counter = 0

    @property
    def is_active(self):
        return self.active_profile is not None and self.intensity > 0

    @property
    def name(self):
        return self.active_profile.get('name', '') if self.is_active else ''

    def get_rest_probability(self):
        if self.is_active and self.intensity > 0.3:
            return self.active_profile.get('rest_prob', 0) * min(1.0, self.intensity / 0.6)
        return None

    def get_melody_register_shift(self):
        if self.is_active and self.intensity > 0.3:
            return int(self.active_profile.get('melody_register', 0) * min(1.0, self.intensity))
        return 0

    def get_tempo_multiplier(self):
        if self.is_active and self.intensity > 0:
            tempo_mult = self.active_profile.get('tempo_mult', 1.0)
            return 1.0 + (tempo_mult - 1.0) * min(1.0, self.intensity)
        return 1.0
    
    def get_velocity_shift(self):
        if self.is_active and self.intensity > 0:
            if self.intensity < 0.3:
                effect_scale = 0.3
            elif self.intensity < 0.6:
                effect_scale = 0.6
            else:
                effect_scale = 1.0
            return int(self.active_profile.get('vel_shift', 0) * effect_scale)
        return 0

    def get_chord_color(self):
        if self.is_active and self.intensity > 0.0:
            return self.active_profile.get('chord_color')
        return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def resolve_spike_profile(is_spike, macro_label, spike_label, duration_counter, current_profile):
    if is_spike and macro_label != spike_label:
        spike_key = (macro_label, spike_label)
        profile = SPIKE_PROFILES.get(spike_key)
        return profile, duration_counter + 1
    else:
        return None, 0

def compute_spike_rest_prob(active_spike_profile, spike_intensity):
    if active_spike_profile and spike_intensity > 0.3:
        return active_spike_profile.get('rest_prob', 0) * min(1.0, spike_intensity / 0.6)
    return None

"""Random patch sampling over Vital's synthesis-relevant parameter subset.

Strategy: drop MIDI CC mappings (2080) and Modulation routing matrix (320). From the
remaining ~455 params, exclude bypass/active switches. Each patch mutates a random
subset of the remaining params, drawn uniformly in [0,1].
"""

import numpy as np


# Substrings whose params get held at default — they can zero out the signal or stretch
# the envelope past our render window, producing silent renders.
HOLD_AT_DEFAULT_SUBSTRINGS = (
    "level",
    "volume",
    "gain",      # filter drive is "Drive" not "Gain", and EQ Gain at 0 is mid not silent — but be conservative
    "amount",
    "attack",    # envelope attack time — long values mean the note never hits sustain
    "release",   # release time — long values can clip our 2s window in weird ways
    "polyphony",
    "legato",
    "oversampling",
    "portamento",
)


def synthesis_param_indices(synth) -> list[int]:
    """Return indices of params that meaningfully shape sound (not host plumbing)."""
    indices = []
    n = synth.get_plugin_parameter_size()
    for i in range(n):
        name = synth.get_parameter_name(i)
        if name.startswith("MIDI"):
            continue
        # Skip the modulation routing matrix entirely — leave Vital's default routes
        if name.startswith("Modulation "):
            parts = name.split()
            if len(parts) >= 2 and parts[1].isdigit():
                continue
        lower = name.lower()
        if "bypass" in lower or "active" in lower:
            continue
        if any(s in lower for s in HOLD_AT_DEFAULT_SUBSTRINGS):
            continue
        indices.append(i)
    return indices


def capture_state(synth) -> dict[int, float]:
    """Snapshot every parameter's current value, so we can reset between patches."""
    return {i: synth.get_parameter(i) for i in range(synth.get_plugin_parameter_size())}


def random_overrides(
    rng: np.random.Generator,
    candidates: list[int],
    mutation_fraction: float = 0.4,
) -> dict[int, float]:
    """Pick a random subset of candidate params and draw uniform [0,1] values."""
    k = int(len(candidates) * mutation_fraction)
    chosen = rng.choice(candidates, size=k, replace=False)
    return {int(idx): float(rng.random()) for idx in chosen}


def apply_state(synth, defaults: dict[int, float], overrides: dict[int, float]) -> None:
    """Reset all params to defaults, then apply this patch's overrides."""
    for idx, val in defaults.items():
        synth.set_parameter(idx, val)
    for idx, val in overrides.items():
        synth.set_parameter(idx, val)

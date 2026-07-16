"""Mutate-around-real-presets sampler.

Given a synth that has been loaded with a real preset's state (full state
including wavetables, LFO shapes, modulations), perturb a random subset of
its flat scalar synthesis params by small amounts to produce a variant.

The base preset's wavetables/LFOs/modulations are inherited — variants
share their parent's wavetable identity and explore the parameter space
*around* that identity.
"""

import numpy as np


def variant_overrides(
    rng: np.random.Generator,
    base_state: dict[int, float],
    candidates: list[int],
    mutation_fraction: float | None = None,
    perturbation_strength: float | None = None,
) -> dict[int, float]:
    """Pick a random subset of candidate params and perturb each from its base value.

    If mutation_fraction / perturbation_strength are None, samples them randomly
    per call to produce variants at multiple scales (some close to parent, some
    far from it). This is what we want for VAE training data — diversity of
    distances from each anchor.

    Each chosen param shifts by Normal(0, perturbation_strength) clamped to [0,1].
    """
    if mutation_fraction is None:
        mutation_fraction = float(rng.uniform(0.05, 0.30))
    if perturbation_strength is None:
        perturbation_strength = float(rng.uniform(0.05, 0.40))

    k = max(1, int(len(candidates) * mutation_fraction))
    chosen = rng.choice(candidates, size=k, replace=False)
    overrides: dict[int, float] = {}
    for idx in chosen:
        base_val = base_state[int(idx)]
        delta = float(rng.normal(0.0, perturbation_strength))
        new_val = max(0.0, min(1.0, base_val + delta))
        overrides[int(idx)] = new_val
    return overrides

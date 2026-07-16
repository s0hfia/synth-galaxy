"""Render N random patches: WAVs + parquet of param vectors.

Usage:
    uv run python scripts/render_dataset.py --n 100
"""

import argparse
import time

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from synth_galaxy.config import AUDIO_DIR, DATA_DIR, PATCHES_DIR, SAMPLE_RATE
from synth_galaxy.render import load_vital, make_engine, render_note
from synth_galaxy.sampler import (
    apply_state,
    capture_state,
    random_overrides,
    synthesis_param_indices,
)


def main(n: int, seed: int, mutation_fraction: float) -> None:
    rng = np.random.default_rng(seed)
    engine = make_engine()
    synth = load_vital(engine)

    defaults = capture_state(synth)
    candidates = synthesis_param_indices(synth)
    print(f"Total params: {len(defaults)}   Synthesis candidates: {len(candidates)}")
    print(f"Per patch: mutating {int(len(candidates) * mutation_fraction)} params "
          f"(~{mutation_fraction:.0%})")

    rows = []
    t0 = time.time()
    for i in tqdm(range(n), desc="Rendering"):
        overrides = random_overrides(rng, candidates, mutation_fraction)
        apply_state(synth, defaults, overrides)
        audio = render_note(engine, synth)

        out_wav = AUDIO_DIR / f"patch_{i:06d}.wav"
        sf.write(out_wav, audio.T, SAMPLE_RATE)

        param_values = [synth.get_parameter(idx) for idx in candidates]
        rms = float(np.sqrt((audio ** 2).mean()))
        peak = float(np.abs(audio).max())
        rows.append({
            "patch_id": i,
            "wav_path": str(out_wav.relative_to(DATA_DIR)),
            "rms": rms,
            "peak": peak,
            "n_mutated": len(overrides),
            **{f"p{idx}": val for idx, val in zip(candidates, param_values)},
        })

    df = pd.DataFrame(rows)
    out_pq = PATCHES_DIR / f"dataset_n{n}_seed{seed}.parquet"
    df.to_parquet(out_pq, index=False)

    elapsed = time.time() - t0
    print(f"\nDone. {n} patches in {elapsed:.1f}s  ({elapsed / n:.2f}s/patch)")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Param parquet: {out_pq}")
    print(f"Mean RMS: {df['rms'].mean():.4f}   "
          f"silent patches (rms<0.001): {(df['rms'] < 0.001).sum()}   "
          f"clipping patches (peak>=0.99): {(df['peak'] >= 0.99).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mutation-fraction", type=float, default=0.4)
    args = ap.parse_args()
    main(args.n, args.seed, args.mutation_fraction)

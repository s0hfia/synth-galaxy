"""UMAP-project wavetable spectral fingerprints to 3D — galaxy clustered by
*timbre skeleton*: which wavetable harmonic shapes each preset uses.

Reads:
  data/mutations_v1/wt_fingerprints.npy            (n_bases, 96) log-mag bins x 3 oscs
  data/mutations_v1/wt_fingerprints_preset_ids.npy

Writes:
  data/mutations_v1/galaxy_coords_wt.parquet  (preset_id, x, y, z)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from synth_galaxy.config import DATA_DIR


def main(in_dir: Path, n_neighbors: int, min_dist: float, seed: int) -> None:
    fps = np.load(in_dir / "wt_fingerprints.npy")              # (n, 96)
    pids = np.load(in_dir / "wt_fingerprints_preset_ids.npy")  # (n,)
    print(f"Loaded {len(fps)} fingerprints, dim={fps.shape[1]}")

    X = fps.astype(np.float32)

    # Tiny jitter so all-zero (init-osc-2-3) rows don't collapse to one point.
    rng = np.random.default_rng(seed)
    X = X + rng.normal(0, 1e-3, X.shape).astype(np.float32)

    reducer = UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, metric="cosine",
    )
    coords = reducer.fit_transform(X)
    print(f"UMAP coords: {coords.shape}")
    print(f"  x: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

    out_df = pd.DataFrame({
        "preset_id": pids,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })
    out_path = in_dir / "galaxy_coords_wt.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DATA_DIR / "mutations_v1")
    ap.add_argument("--n-neighbors", type=int, default=20)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.in_dir, args.n_neighbors, args.min_dist, args.seed)

"""UMAP-project modulation fingerprints to 3D — a galaxy clustered by *how the
patches were wired up*, not how they sound or what synth params they use.

Reads:
  data/mutations_v1/mod_fingerprints.npy            (n_bases, 4, 4)
  data/mutations_v1/mod_fingerprints_preset_ids.npy (n_bases,) preset_ids in same order

Writes:
  data/mutations_v1/galaxy_coords_mod.parquet  (preset_id, x, y, z)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from synth_galaxy.config import DATA_DIR


def main(in_dir: Path, n_neighbors: int, min_dist: float, seed: int) -> None:
    fps = np.load(in_dir / "mod_fingerprints.npy")             # (n, 4, 4)
    pids = np.load(in_dir / "mod_fingerprints_preset_ids.npy") # (n,)
    print(f"Loaded {len(fps)} fingerprints, flattening to {fps.shape[1] * fps.shape[2]}D")

    X = fps.reshape(len(fps), -1).astype(np.float32)

    # Cap the dynamic range a bit so a single huge-amount routing doesn't dominate.
    X = np.log1p(X * 4.0)

    # If a preset has zero modulations, the fingerprint is all-zero — UMAP would
    # collapse those into a single point. That's actually the right behavior
    # (they ARE structurally identical), but we add a tiny jitter so they don't
    # render as one overdrawn dot.
    rng = np.random.default_rng(seed)
    X = X + rng.normal(0, 1e-3, X.shape).astype(np.float32)

    reducer = UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, metric="euclidean",
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
    out_path = in_dir / "galaxy_coords_mod.parquet"
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

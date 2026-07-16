"""UMAP the 66,904-wavetable corpus fingerprints to 3D for the wavetable atlas viewer.

Reads:
  data/wt_corpus/wt_corpus_fingerprints.npy   (66904, 32) log-mag bands
  data/wt_corpus/wt_corpus_meta.parquet       source/bank/name/index

Writes:
  data/wt_corpus/wt_atlas_coords.parquet      meta + x,y,z UMAP coords
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from synth_galaxy.config import DATA_DIR


def main(in_dir: Path, n_neighbors: int, min_dist: float, seed: int) -> None:
    fps = np.load(in_dir / "wt_corpus_fingerprints.npy")
    meta = pd.read_parquet(in_dir / "wt_corpus_meta.parquet")
    print(f"Loaded {len(fps)} fingerprints, dim={fps.shape[1]}")

    # Tiny jitter to break ties between identical (all-zero) tables.
    rng = np.random.default_rng(seed)
    X = fps.astype(np.float32) + rng.normal(0, 1e-3, fps.shape).astype(np.float32)

    print(f"Fitting UMAP ({n_neighbors} neighbors, min_dist={min_dist}, cosine)...")
    t0 = time.time()
    reducer = UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, metric="cosine", low_memory=True,
    )
    coords = reducer.fit_transform(X)
    print(f"UMAP done in {time.time()-t0:.1f}s. coords shape: {coords.shape}")

    out = meta.copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out["z"] = coords[:, 2]
    # idx field = row index into the corpus arrays (.npy slices)
    out["idx"] = np.arange(len(out), dtype=np.int32)

    out_path = in_dir / "wt_atlas_coords.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"  per-source counts:")
    print(out["source"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DATA_DIR / "wt_corpus")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.in_dir, args.n_neighbors, args.min_dist, args.seed)

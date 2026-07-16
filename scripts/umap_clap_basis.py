"""UMAP-project CLAP audio embeddings to 3D for an alternate galaxy basis.

This produces a galaxy organized by perceptual / semantic similarity (since
CLAP was trained on audio<->text pairs) — distinct from the VAE-latent
galaxy which clusters by synthesis-parameter structure.

Reads:
  data/mutations_v1/clap_audio_embeddings.npy   (n_bases, 512)
  data/mutations_v1/clap_scores.parquet         metadata + scores

Writes:
  data/mutations_v1/galaxy_coords_clap.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

from synth_galaxy.config import DATA_DIR


def main(in_dir: Path, n_neighbors: int, min_dist: float, seed: int) -> None:
    audio_embs = np.load(in_dir / "clap_audio_embeddings.npy")
    scores_df = pd.read_parquet(in_dir / "clap_scores.parquet")
    assert len(audio_embs) == len(scores_df), \
        f"shape mismatch: embs={audio_embs.shape} scores={len(scores_df)}"
    print(f"Loaded {len(audio_embs)} CLAP audio embeddings (dim={audio_embs.shape[1]})")

    reducer = UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, metric="cosine",  # CLAP embeddings are normalized -> cosine is natural
    )
    coords = reducer.fit_transform(audio_embs)
    print(f"UMAP coords: {coords.shape}")
    print(f"  x: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

    out_df = pd.DataFrame({
        "preset_id": scores_df["preset_id"].values,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })
    # carry over scalar features so the existing viz code can color by them
    # (we'll separately handle CLAP descriptor columns)
    out_path = in_dir / "galaxy_coords_clap.parquet"
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

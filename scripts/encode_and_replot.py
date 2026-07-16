"""Encode every patch through the trained VAE, UMAP the 12D latent to 3D,
save coords, and regenerate the interactive plotly galaxy on top of the VAE
latent space (instead of the librosa-feature space)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from umap import UMAP

from synth_galaxy.config import DATA_DIR
from synth_galaxy.vae import LATENT_DIM, ParamVAE


def main(model_path: Path, dataset_dir: Path, out_dir: Path) -> None:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    n_params = ckpt["n_params"]
    latent_dim = ckpt["latent_dim"]
    print(f"Loaded checkpoint: epoch={ckpt['epoch']} val_recon={ckpt['val_recon']:.5f}")
    print(f"  n_params={n_params}  latent_dim={latent_dim}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ParamVAE(n_params=n_params, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params = np.load(dataset_dir / "params.npy")
    meta = pd.read_parquet(dataset_dir / "metadata.parquet")
    print(f"Encoding {len(params)} patches...")

    # Batch-encode all params -> deterministic latent mu
    batch = 1024
    latents_list = []
    with torch.no_grad():
        for i in range(0, len(params), batch):
            x = torch.from_numpy(params[i : i + batch].astype(np.float32)).to(device)
            z = model.encode(x, deterministic=True)
            latents_list.append(z.cpu().numpy())
    latents = np.concatenate(latents_list, axis=0).astype(np.float32)
    print(f"Latent shape: {latents.shape}  range: [{latents.min():.2f}, {latents.max():.2f}]")
    np.save(out_dir / "latents.npy", latents)

    print("Running UMAP 12D -> 3D ...")
    reducer = UMAP(n_components=3, n_neighbors=30, min_dist=0.15, random_state=42, metric="euclidean")
    coords = reducer.fit_transform(latents)
    print(f"3D coords range:  x[{coords[:,0].min():.2f},{coords[:,0].max():.2f}]  "
          f"y[{coords[:,1].min():.2f},{coords[:,1].max():.2f}]  "
          f"z[{coords[:,2].min():.2f},{coords[:,2].max():.2f}]")

    coords_df = pd.DataFrame({
        "preset_id": meta["preset_id"].values,
        "patch_id": meta["patch_id"].values,
        "variant_id": meta["variant_id"].values,
        "is_base": meta["is_base"].values,
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "spec_centroid": meta["spec_centroid"].values,
        "rms": meta["rms"].values,
        "harmonic_ratio": meta.get("harmonic_ratio", pd.Series([0.5]*len(meta))).values
            if "harmonic_ratio" in meta.columns else np.full(len(meta), 0.5),
        "stereo_width": meta["stereo_width"].values,
        "spec_flatness": meta["spec_flatness"].values,
    })
    out_coords = out_dir / "galaxy_coords_vae.parquet"
    coords_df.to_parquet(out_coords, index=False)
    print(f"Wrote {out_coords}")
    print(f"Now regenerate interactive HTML with:")
    print(f"  uv run python scripts/galaxy_interactive.py --coords {out_coords} --dataset {dataset_dir/'metadata.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DATA_DIR / "models/vae_mutations_v1/vae_best.pt")
    ap.add_argument("--dataset", type=Path, default=DATA_DIR / "mutations_v1")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "mutations_v1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main(args.model, args.dataset, args.out)

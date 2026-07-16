"""Fit UMAP to 3D, save coords, render a sanity-check matplotlib scatter."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from synth_galaxy.config import DATA_DIR, FEATURES_DIR
from synth_galaxy.embed import fit_umap


def main(features_parquet: Path) -> None:
    feat_df = pd.read_parquet(features_parquet)
    print(f"Loaded {len(feat_df)} feature rows from {features_parquet.name}")

    coords, _, _ = fit_umap(feat_df)
    print(f"UMAP coords shape: {coords.shape}")
    print(f"  x: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

    id_col = "preset_id" if "preset_id" in feat_df.columns else "patch_id"
    coords_df = pd.DataFrame({
        id_col: feat_df[id_col].values,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "spec_centroid": feat_df["spec_centroid"].values,
        "rms": feat_df["rms"].values,
        "harmonic_ratio": feat_df["harmonic_ratio"].values,
        "stereo_width": feat_df["stereo_width"].values,
        "spec_flatness": feat_df["spec_flatness"].values,
    })
    out_pq = DATA_DIR / f"galaxy_coords_{features_parquet.stem}.parquet"
    coords_df.to_parquet(out_pq, index=False)
    print(f"Wrote {out_pq}")

    # Filter silent patches for visualization (they collapse to a single point)
    plot_df = coords_df[coords_df["rms"] > 0.001].copy()
    print(f"Plotting {len(plot_df)} of {len(coords_df)} (skipping {len(coords_df) - len(plot_df)} silent)")

    fig = plt.figure(figsize=(11, 9), facecolor="#0a0a14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a14")

    sizes = 30 + 180 * (plot_df["rms"] / max(0.001, plot_df["rms"].max()))
    scat = ax.scatter(
        plot_df["x"], plot_df["y"], plot_df["z"],
        c=plot_df["spec_centroid"], cmap="plasma",
        s=sizes, alpha=0.85, edgecolors="none",
    )
    cb = plt.colorbar(scat, label="spectral centroid (Hz)  →  brightness", pad=0.1)
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")
    cb.outline.set_edgecolor("white")

    ax.set_xlabel("UMAP 1", color="white")
    ax.set_ylabel("UMAP 2", color="white")
    ax.set_zlabel("UMAP 3", color="white")
    ax.set_title("synth-galaxy  v0  ·  size=loudness  ·  color=brightness", color="white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.03, 0.03, 0.08, 1.0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)
    ax.tick_params(colors="white")

    out_png = DATA_DIR / f"galaxy_{features_parquet.stem}.png"
    plt.savefig(out_png, dpi=140, facecolor="#0a0a14", bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path, nargs="?", default=None)
    args = ap.parse_args()
    if args.features is None:
        candidates = sorted(FEATURES_DIR.glob("features_*.parquet"))
        if not candidates:
            raise SystemExit("No features parquet found")
        args.features = candidates[-1]
    main(args.features)

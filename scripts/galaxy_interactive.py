"""Build an interactive 3D plotly scatter of the galaxy with UI controls.

Controls (top-left of the figure):
- "Color by" dropdown: swap which feature drives marker color
  - brightness (spectral centroid), loudness (RMS), noisiness (flatness),
    harmonic ratio, stereo width, pack (categorical)
- "Show" dropdown: bases only / variants only / both

Click a point to preview its audio (only for base presets — variants don't
keep WAV files). Run a local HTTP server to enable preview:

    cd ~/projects/synth-galaxy && python -m http.server 8000
    open http://localhost:8000/data/galaxy_interactive.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synth_galaxy.config import DATA_DIR, FEATURES_DIR, PATCHES_DIR


CLICK_AUDIO_JS = """
(function() {
  let audio = document.createElement('audio');
  audio.id = 'preset-preview-audio';
  audio.style.position = 'fixed';
  audio.style.bottom = '12px';
  audio.style.left = '12px';
  audio.style.zIndex = '9999';
  audio.controls = true;
  audio.preload = 'none';
  document.body.appendChild(audio);

  let label = document.createElement('div');
  label.id = 'preset-preview-label';
  label.style.position = 'fixed';
  label.style.bottom = '60px';
  label.style.left = '12px';
  label.style.zIndex = '9999';
  label.style.color = '#cfd';
  label.style.background = 'rgba(0,0,0,0.6)';
  label.style.padding = '6px 10px';
  label.style.borderRadius = '4px';
  label.style.fontFamily = 'monospace';
  label.style.fontSize = '12px';
  label.style.maxWidth = '480px';
  label.textContent = 'click a base preset (large marker) to preview audio';
  document.body.appendChild(label);

  function wire() {
    let plot = document.querySelector('.plotly-graph-div');
    if (!plot) { setTimeout(wire, 100); return; }
    plot.on('plotly_click', function(data) {
      if (!data || !data.points || !data.points.length) return;
      let p = data.points[0];
      let cd = p.customdata;
      if (!cd) return;
      let wav = cd[6];
      if (!wav) {
        label.textContent = '(variant — no audio; only base presets are playable)';
        return;
      }
      audio.src = wav;
      audio.play().catch(e => { console.warn('audio play failed', e); });
      label.textContent = cd[0] + ' \\u2014 ' + cd[1] + ' \\u00b7 ' + cd[2] + ' \\u00b7 ' + cd[3];
    });
  }
  wire();
})();
"""


COLOR_FIELDS = [
    ("Brightness (spectral centroid)", "spec_centroid", "Plasma"),
    ("Loudness (RMS)", "rms", "Viridis"),
    ("Noisiness (spectral flatness)", "spec_flatness", "Cividis"),
    ("Harmonic ratio", "harmonic_ratio", "Magma"),
    ("Stereo width", "stereo_width", "Turbo"),
]


def _pack_to_index(packs: pd.Series) -> tuple[np.ndarray, dict]:
    """Map pack name -> integer index for categorical coloring."""
    uniq = sorted(packs.dropna().unique().tolist())
    lookup = {p: i for i, p in enumerate(uniq)}
    return packs.map(lookup).fillna(-1).to_numpy(), lookup


def build_trace(df: pd.DataFrame, name: str, size_mult: float,
                show_legend: bool = True) -> go.Scatter3d:
    """Build one Scatter3d trace; color defaults to spectral centroid."""
    customdata = df[
        ["preset_name", "author", "pack", "preset_style", "rms", "spec_centroid", "wav_path"]
    ].fillna("").values

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "by %{customdata[1]}<br>"
        "pack: %{customdata[2]}<br>"
        "style: %{customdata[3]}<br>"
        "rms: %{customdata[4]:.3f}<br>"
        "centroid: %{customdata[5]:.0f} Hz<br>"
        "<extra></extra>"
    )

    rms_clipped = df["rms"].clip(0.001, df["rms"].quantile(0.98))
    size = size_mult * (4 + 22 * (rms_clipped / max(0.001, rms_clipped.max())))

    return go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        name=name,
        mode="markers",
        marker=dict(
            size=size,
            color=df["spec_centroid"],
            colorscale="Plasma",
            cmin=df["spec_centroid"].min(),
            cmax=df["spec_centroid"].max(),
            colorbar=dict(
                title=dict(text="brightness", font=dict(color="white")),
                tickfont=dict(color="white"),
            ),
            opacity=0.85,
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=hover,
        showlegend=show_legend,
    )


def build_color_buttons(df_list: list[pd.DataFrame]) -> list[dict]:
    """Build a 'Color by' dropdown with restyle actions for all visible traces."""
    buttons = []
    for label, col, cmap in COLOR_FIELDS:
        vals_per_trace = [d[col].to_numpy() for d in df_list]
        cmin = min(v.min() for v in vals_per_trace)
        cmax = max(v.max() for v in vals_per_trace)
        buttons.append(dict(
            label=label,
            method="restyle",
            args=[{
                "marker.color": vals_per_trace,
                "marker.colorscale": [cmap] * len(df_list),
                "marker.cmin": [cmin] * len(df_list),
                "marker.cmax": [cmax] * len(df_list),
            }],
        ))

    # "Color by pack" — categorical
    all_packs = pd.concat([d["pack"] for d in df_list]).fillna("")
    uniq = sorted([p for p in all_packs.unique() if p])
    lookup = {p: i for i, p in enumerate(uniq)}
    pack_colors_per_trace = [
        d["pack"].fillna("").map(lookup).fillna(-1).to_numpy()
        for d in df_list
    ]
    buttons.append(dict(
        label="Pack (categorical)",
        method="restyle",
        args=[{
            "marker.color": pack_colors_per_trace,
            "marker.colorscale": ["Rainbow"] * len(df_list),
            "marker.cmin": [0] * len(df_list),
            "marker.cmax": [max(1, len(uniq) - 1)] * len(df_list),
        }],
    ))
    return buttons


def main(coords_path: Path, dataset_path: Path, out_path: Path,
         max_variants: int = 6000, seed: int = 42) -> None:
    coords = pd.read_parquet(coords_path)
    dataset = pd.read_parquet(dataset_path)

    # Pick the right id column. If both have patch_id, use that; else preset_id.
    if "patch_id" in coords.columns and "patch_id" in dataset.columns:
        id_col = "patch_id"
    else:
        id_col = "preset_id"

    # Merge metadata; resolve is_base from coords (canonical) if present.
    cols_wanted = [id_col, "preset_name", "author", "pack", "preset_style", "wav_path"]
    cols_present = [c for c in cols_wanted if c in dataset.columns]
    df = coords.merge(dataset[cols_present], on=id_col, how="left")

    # is_base flag (from coords if present, else infer from variant_id)
    if "is_base" in df.columns:
        df["is_base"] = df["is_base"].fillna(False).astype(bool)
    elif "variant_id" in df.columns:
        df["is_base"] = df["variant_id"] == 0
    else:
        df["is_base"] = True  # treat all as base if we have no info

    # Fill defaults for any missing string columns
    for c in ("preset_name", "author", "pack", "preset_style"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    if "wav_path" not in df.columns:
        df["wav_path"] = ""
    df["wav_path"] = df["wav_path"].fillna("").astype(str)

    bases = df[df["is_base"]].copy().reset_index(drop=True)
    variants = df[~df["is_base"]].copy()
    if len(variants) > max_variants:
        variants = variants.sample(n=max_variants, random_state=seed).reset_index(drop=True)
    print(f"Bases: {len(bases)}, variants (sampled): {len(variants)}")

    bases_trace = build_trace(bases, name=f"Real presets ({len(bases)})",
                              size_mult=1.0)
    variants_trace = build_trace(variants, name=f"Mutations ({len(variants)})",
                                 size_mult=0.4)
    variants_trace.marker.opacity = 0.35

    fig = go.Figure(data=[bases_trace, variants_trace])

    color_buttons = build_color_buttons([bases, variants])
    show_buttons = [
        dict(label="All", method="restyle", args=[{"visible": [True, True]}]),
        dict(label="Real presets only", method="restyle", args=[{"visible": [True, False]}]),
        dict(label="Mutations only", method="restyle", args=[{"visible": [False, True]}]),
    ]

    fig.update_layout(
        title=dict(
            text=f"synth-galaxy · {len(bases)} real + {len(variants)} mutation samples · size=loudness",
            font=dict(color="white"),
        ),
        paper_bgcolor="#0a0a14",
        scene=dict(
            xaxis=dict(title="UMAP 1", color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            yaxis=dict(title="UMAP 2", color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            zaxis=dict(title="UMAP 3", color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            bgcolor="#0a0a14",
        ),
        margin=dict(l=0, r=0, t=120, b=0),
        font=dict(color="white"),
        legend=dict(font=dict(color="white"), x=1.0, y=0.5),
        updatemenus=[
            dict(
                buttons=color_buttons,
                direction="down",
                showactive=True,
                x=0.0, y=1.15,
                xanchor="left", yanchor="top",
                pad=dict(r=10, t=10),
                bgcolor="#1a1a30",
                bordercolor="#444",
                font=dict(color="white"),
            ),
            dict(
                buttons=show_buttons,
                direction="down",
                showactive=True,
                x=0.25, y=1.15,
                xanchor="left", yanchor="top",
                pad=dict(r=10, t=10),
                bgcolor="#1a1a30",
                bordercolor="#444",
                font=dict(color="white"),
            ),
        ],
        annotations=[
            dict(text="Color by:", showarrow=False, x=0.0, y=1.20,
                 xref="paper", yref="paper", xanchor="left",
                 font=dict(color="#cfd", size=12)),
            dict(text="Show:", showarrow=False, x=0.25, y=1.20,
                 xref="paper", yref="paper", xanchor="left",
                 font=dict(color="#cfd", size=12)),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=CLICK_AUDIO_JS,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"View at:  http://localhost:8000/data/{out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=None,
                    help="Galaxy coords parquet (defaults to latest)")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Dataset metadata parquet")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "galaxy_interactive.html")
    ap.add_argument("--max-variants", type=int, default=6000,
                    help="Cap on variant points (browser perf). 0 = bases only.")
    args = ap.parse_args()

    if args.coords is None:
        candidates = (
            sorted((DATA_DIR / "mutations_v1").glob("galaxy_coords_*.parquet"))
            + sorted(DATA_DIR.glob("galaxy_coords_*.parquet"))
        )
        if not candidates:
            raise SystemExit("No galaxy_coords_*.parquet found")
        args.coords = candidates[0]

    if args.dataset is None:
        for cand in (DATA_DIR / "mutations_v1" / "metadata.parquet",
                     *sorted(PATCHES_DIR.glob("presets_*.parquet"), reverse=True),
                     *sorted(PATCHES_DIR.glob("dataset_*.parquet"), reverse=True)):
            if cand.exists():
                args.dataset = cand
                break

    print(f"coords:  {args.coords}")
    print(f"dataset: {args.dataset}")
    main(args.coords, args.dataset, args.out, max_variants=args.max_variants)

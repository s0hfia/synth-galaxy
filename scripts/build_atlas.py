"""Build atlas.html — the wavetable atlas viewer.

Renders 66,904 wavetables as a 3D plotly scatter, colored by source pack,
hover for source/bank/name, click → fetch /wt/<idx>.wav from the Flask
backend and play it (single C4 cycle looped to ~1.2s).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synth_galaxy.config import DATA_DIR


WT_DIR = DATA_DIR / "wt_corpus"


CLICK_AUDIO_JS = """
(function() {
  let style = document.createElement('style');
  style.textContent = `
    #atlas-panel {
      position: fixed; top: 12px; right: 12px;
      width: 320px;
      background: rgba(10,10,20,0.94);
      color: #cfd;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      border: 1px solid #333;
      border-radius: 6px;
      padding: 12px 14px;
      z-index: 9999;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    #atlas-title { font-size: 14px; color: #fff; font-weight: bold; line-height: 1.2; }
    #atlas-subtitle { font-size: 11px; color: #889; margin: 2px 0 8px 0; }
    #atlas-audio { width: 100%; margin-top: 6px; height: 28px; }
    .atlas-spec {
      height: 40px; background: #0a0a14; border-radius: 3px;
      margin-top: 8px;
    }
    .atlas-foot { color: #557; font-size: 9px; margin-top: 8px; letter-spacing: 0.05em; }
  `;
  document.head.appendChild(style);

  let panel = document.createElement('div');
  panel.id = 'atlas-panel';
  panel.innerHTML = `
    <div id="atlas-title">click a wavetable</div>
    <div id="atlas-subtitle">66,904 single-cycle/bank tables</div>
    <canvas id="atlas-wave" class="atlas-spec" width="320" height="40"></canvas>
    <audio id="atlas-audio" controls preload="none"></audio>
    <div class="atlas-foot">audio: 1 cycle resampled to C4, looped ~1.2s</div>
  `;
  document.body.appendChild(panel);

  let titleEl = panel.querySelector('#atlas-title');
  let subEl = panel.querySelector('#atlas-subtitle');
  let audio = panel.querySelector('#atlas-audio');
  let waveCanvas = panel.querySelector('#atlas-wave');
  let ctx = waveCanvas.getContext('2d');

  function drawWavePlaceholder() {
    ctx.fillStyle = '#0a0a14'; ctx.fillRect(0, 0, 320, 40);
    ctx.strokeStyle = 'rgba(180,200,220,0.10)';
    ctx.beginPath(); ctx.moveTo(0, 20); ctx.lineTo(320, 20); ctx.stroke();
  }
  drawWavePlaceholder();

  function drawWaveFromAudio(url) {
    // fetch as ArrayBuffer, decode via WebAudio API for nice tracing
    let actx = window._sgAudioCtx || (window._sgAudioCtx = new (window.AudioContext || window.webkitAudioContext)());
    fetch(url).then(r => r.arrayBuffer()).then(buf => actx.decodeAudioData(buf, function(audioBuf) {
      let ch = audioBuf.getChannelData(0);
      // sample down to canvas width
      let W = 320, H = 40;
      ctx.fillStyle = '#0a0a14'; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#6cf';
      ctx.beginPath();
      let stride = Math.max(1, Math.floor(ch.length / W));
      for (let x = 0; x < W; x++) {
        let s = ch[x * stride] || 0;
        let y = (H/2) + s * (H/2 - 2);
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }, function(e) { console.warn('decode failed', e); }));
  }

  function wire() {
    let plot = document.querySelector('.plotly-graph-div');
    if (!plot) { setTimeout(wire, 100); return; }
    plot.on('plotly_click', function(data) {
      if (!data || !data.points || !data.points.length) return;
      let p = data.points[0];
      let cd = p.customdata;
      if (!cd) return;
      // cd = [name, source, bank, idx]
      let name = cd[0], source = cd[1], bank = cd[2], idx = cd[3];
      titleEl.textContent = name || '(unnamed)';
      subEl.innerHTML = `${source} <span style="color:#557">·</span> ${bank} <span style="color:#557">·</span> #${idx}`;
      let url = '/wt/' + idx + '.wav';
      audio.src = url;
      audio.play().catch(e => console.warn('audio play failed', e));
      drawWaveFromAudio(url);
    });
  }
  wire();
})();
"""


def main(out_path: Path, max_points: int) -> None:
    coords_path = WT_DIR / "wt_atlas_coords.parquet"
    if not coords_path.exists():
        raise SystemExit(f"Missing {coords_path} — run umap_wt_atlas.py first.")
    df = pd.read_parquet(coords_path)
    print(f"Loaded {len(df)} atlas points")

    if max_points and len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df)} for browser perf")

    # categorical color: source pack
    sources = sorted(df["source"].unique())
    source_to_id = {s: i for i, s in enumerate(sources)}
    color_vals = df["source"].map(source_to_id).to_numpy()

    customdata = df[["name", "source", "bank", "idx"]].fillna("").values

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]} · %{customdata[2]}<br>"
        "click to play"
        "<extra></extra>"
    )

    fig = go.Figure(data=go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers",
        marker=dict(
            size=2.4,
            color=color_vals,
            colorscale="Rainbow",
            cmin=0, cmax=len(sources) - 1 if len(sources) > 1 else 1,
            opacity=0.85,
            line=dict(width=0),
        ),
        customdata=customdata,
        hovertemplate=hover,
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"wavetable atlas · {len(df):,} timbre skeletons (CC0 corpus) · color = source pack",
            font=dict(color="white", size=13),
        ),
        paper_bgcolor="#050510",
        scene=dict(
            xaxis=dict(title="", color="white", backgroundcolor="#050510",
                       gridcolor="rgba(255,255,255,0.07)", showbackground=True,
                       showticklabels=False),
            yaxis=dict(title="", color="white", backgroundcolor="#050510",
                       gridcolor="rgba(255,255,255,0.07)", showbackground=True,
                       showticklabels=False),
            zaxis=dict(title="", color="white", backgroundcolor="#050510",
                       gridcolor="rgba(255,255,255,0.07)", showbackground=True,
                       showticklabels=False),
            bgcolor="#050510",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(color="white"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True,
                   post_script=CLICK_AUDIO_JS)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"\nServe with: cd {DATA_DIR.parent} && uv run python scripts/search_server.py")
    print(f"Then open: http://localhost:8000/atlas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "atlas.html")
    ap.add_argument("--max-points", type=int, default=66904,
                    help="Cap the number of orbs (browser perf). Default = all 66904.")
    args = ap.parse_args()
    main(args.out, args.max_points)

"""Interactive 3D galaxy v2: multi-basis + CLAP descriptor colors.

Three dropdowns:
  - Basis:    VAE latent (param-similarity) vs CLAP audio (perceptual-similarity)
  - Color by: quantitative features, pack (categorical), all CLAP descriptors
  - Show:     all / bases only / variants only

Variants only exist in the VAE basis (CLAP wasn't run on the 44k variant audio).
Switching to CLAP basis auto-hides variants.

Click a base preset (large marker) -> audio preview via local HTTP server.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from synth_galaxy.clap_descriptors import all_descriptors
from synth_galaxy.config import DATA_DIR


CLICK_AUDIO_JS = """
(function() {
  // ---------- styles ----------
  let style = document.createElement('style');
  style.textContent = `
    #sg-panel {
      position: fixed; top: 12px; right: 12px;
      width: 360px; max-height: calc(100vh - 24px);
      overflow-y: auto;
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
    #sg-title { font-size: 14px; color: #fff; font-weight: bold; line-height: 1.2; }
    #sg-subtitle { font-size: 11px; color: #889; margin: 2px 0 8px 0; }
    .sg-section { margin-top: 14px; }
    .sg-section h4 {
      margin: 0 0 6px 0;
      font-size: 10px;
      letter-spacing: 0.15em;
      color: #7a8;
      font-weight: 600;
    }
    .sg-row { display: flex; justify-content: space-between; padding: 2px 0; gap: 8px; }
    .sg-row .lbl { color: #cfd; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sg-row .val { color: #ada; font-weight: bold; white-space: nowrap; }
    .sg-row.neighbor { cursor: pointer; }
    .sg-row.neighbor:hover { background: rgba(120,200,160,0.10); }
    .sg-bar {
      height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px;
      margin: 1px 0 4px 0; overflow: hidden;
    }
    .sg-bar > div { height: 100%; background: linear-gradient(90deg, #6cf, #afc); }
    .sg-empty { color: #557; font-style: italic; }
    #sg-audio { width: 100%; margin-top: 6px; height: 28px; }
  `;
  document.head.appendChild(style);

  // ---------- panel ----------
  let panel = document.createElement('div');
  panel.id = 'sg-panel';
  panel.innerHTML = `
    <div id="sg-title">click a preset to inspect</div>
    <div id="sg-subtitle">drop a click on any large marker</div>
    <audio id="sg-audio" controls preload="none"></audio>
    <div class="sg-section"><h4>SOUNDS LIKE</h4><div id="sg-clap" class="sg-empty">—</div></div>
    <div class="sg-section"><h4>SONIC NEIGHBORS</h4><div id="sg-neighbors" class="sg-empty">—</div></div>
    <div class="sg-section"><h4>STANDOUT PARAMS</h4><div id="sg-params" class="sg-empty">—</div></div>
    <div class="sg-section"><h4>MOD FINGERPRINT</h4><div id="sg-fp" class="sg-empty">—</div><div id="sg-fp-legend" style="font-size:9px;color:#557;margin-top:4px;letter-spacing:0.05em;">rows: <b style="color:#7a8">sources</b> · cols: <b style="color:#7a8">destinations</b> · color: routing strength</div></div>
    <div class="sg-section"><h4>MODULATIONS</h4><div id="sg-mods" class="sg-empty">—</div></div>
  `;
  document.body.appendChild(panel);

  let audio = panel.querySelector('#sg-audio');
  let titleEl = panel.querySelector('#sg-title');
  let subEl = panel.querySelector('#sg-subtitle');
  let clapEl = panel.querySelector('#sg-clap');
  let neighEl = panel.querySelector('#sg-neighbors');
  let paramsEl = panel.querySelector('#sg-params');
  let fpEl = panel.querySelector('#sg-fp');
  let modsEl = panel.querySelector('#sg-mods');

  // ---------- details fetch ----------
  let DETAILS = null;
  fetch('/data/mutations_v1/details.json')
    .then(r => r.json())
    .then(d => { DETAILS = d; })
    .catch(e => console.warn('details.json fetch failed:', e));

  let presetIdToIndex = null;

  function fmt(n, d) { return Number(n).toFixed(d); }

  function renderClap(arr) {
    if (!arr || !arr.length) { clapEl.innerHTML = '<span class="sg-empty">—</span>'; return; }
    let max = Math.max(...arr.map(x => x.score));
    let min = Math.min(...arr.map(x => x.score));
    let range = Math.max(1e-6, max - min);
    clapEl.innerHTML = arr.map(x => {
      let pct = Math.max(3, Math.round(100 * (x.score - min) / range));
      return `<div><div class="sg-row"><span class="lbl">${x.label}</span><span class="val">${fmt(x.score,2)}</span></div><div class="sg-bar"><div style="width:${pct}%"></div></div></div>`;
    }).join('');
  }

  function renderNeighbors(arr) {
    if (!arr || !arr.length) { neighEl.innerHTML = '<span class="sg-empty">—</span>'; return; }
    neighEl.innerHTML = arr.map(x =>
      `<div class="sg-row neighbor" data-pid="${x.preset_id}"><span class="lbl">${x.name} <span style="color:#557">· ${x.pack}</span></span><span class="val">${fmt(x.sim,2)}</span></div>`
    ).join('');
    neighEl.querySelectorAll('.neighbor').forEach(el => {
      el.addEventListener('click', () => focusPreset(parseInt(el.dataset.pid)));
    });
  }

  function renderParams(arr) {
    if (!arr || !arr.length) { paramsEl.innerHTML = '<span class="sg-empty">—</span>'; return; }
    paramsEl.innerHTML = arr.map(x => {
      let sign = x.z >= 0 ? '+' : '';
      let zColor = Math.abs(x.z) > 2 ? '#fc7' : '#ada';
      return `<div class="sg-row"><span class="lbl">${x.name}</span><span class="val" style="color:${zColor}">${fmt(x.value,2)} <span style="color:#557">(z=${sign}${fmt(x.z,1)})</span></span></div>`;
    }).join('');
  }

  function renderMods(arr) {
    if (!arr || !arr.length) { modsEl.innerHTML = '<span class="sg-empty">— (none active)</span>'; return; }
    modsEl.innerHTML = arr.map(x => {
      let sign = x.amount >= 0 ? '+' : '';
      return `<div class="sg-row"><span class="lbl">${x.source} → ${x.dest}</span><span class="val">${sign}${fmt(x.amount,2)}</span></div>`;
    }).join('');
  }

  function renderFp(svg) {
    if (!svg) { fpEl.innerHTML = '<span class="sg-empty">—</span>'; return; }
    fpEl.innerHTML = svg;
  }

  function focusPreset(pid) {
    if (!presetIdToIndex) return;
    let idx = presetIdToIndex[String(pid)];
    if (idx == null) return;
    let plot = document.querySelector('.plotly-graph-div');
    if (!plot || !plot.data || !plot.data[0]) return;
    let cd = plot.data[0].customdata[idx];
    onClick({ points: [{ customdata: cd, pointNumber: idx, curveNumber: 0 }] });
  }

  function onClick(data) {
    if (!data || !data.points || !data.points.length) return;
    let p = data.points[0];
    let cd = p.customdata;
    if (!cd) return;
    let name = cd[0], author = cd[1], pack = cd[2], style = cd[3];
    let wavPath = cd[6], presetId = cd[7];
    titleEl.textContent = name || '(unnamed)';
    subEl.innerHTML = `by ${author || '?'} <span style="color:#557">·</span> ${pack || '?'} <span style="color:#557">·</span> ${style || '?'}`;
    if (wavPath) {
      audio.src = wavPath;
      audio.play().catch(e => console.warn('audio play failed', e));
    } else {
      audio.removeAttribute('src');
    }
    let d = DETAILS && DETAILS[String(presetId)];
    if (d) {
      renderClap(d.clap);
      renderNeighbors(d.neighbors);
      renderParams(d.params);
      renderFp(d.fp_svg);
      renderMods(d.mods);
    } else if (!DETAILS) {
      clapEl.innerHTML = neighEl.innerHTML = paramsEl.innerHTML = fpEl.innerHTML = modsEl.innerHTML = '<span class="sg-empty">(loading details…)</span>';
    } else {
      clapEl.innerHTML = neighEl.innerHTML = paramsEl.innerHTML = fpEl.innerHTML = modsEl.innerHTML = '<span class="sg-empty">(no details for this point — variant)</span>';
    }
  }

  function wire() {
    let plot = document.querySelector('.plotly-graph-div');
    if (!plot) { setTimeout(wire, 100); return; }
    if (plot.data && plot.data[0] && plot.data[0].customdata) {
      presetIdToIndex = {};
      plot.data[0].customdata.forEach((cd, i) => {
        let pid = cd[7];
        if (pid !== '' && pid != null) presetIdToIndex[String(pid)] = i;
      });
    }
    plot.on('plotly_click', onClick);
  }
  wire();
})();

// ============================================================
// Audio-drop: drag an audio file onto the page -> POST /upload
// -> show CLAP preset neighbors + 67k-corpus wavetable matches
// ============================================================
(function() {
  let style = document.createElement('style');
  style.textContent = `
    #sg-drop-overlay {
      position: fixed; inset: 0; z-index: 10000;
      background: rgba(10,20,40,0.85);
      display: none; align-items: center; justify-content: center;
      pointer-events: none;
      border: 4px dashed rgba(120,200,255,0.5);
      backdrop-filter: blur(4px);
    }
    #sg-drop-overlay.active { display: flex; }
    #sg-drop-msg {
      font-family: ui-monospace, monospace;
      font-size: 22px;
      color: #cdf;
      text-align: center;
      padding: 40px;
      letter-spacing: 0.05em;
    }
    #sg-drop-msg .hint { font-size: 12px; color: #88a; margin-top: 12px; letter-spacing: 0.1em; }
    #drop-panel {
      position: fixed; top: 12px; left: 12px;
      width: 380px; max-height: calc(100vh - 24px);
      overflow-y: auto;
      background: rgba(10,10,20,0.94);
      color: #cfd;
      font-family: ui-monospace, monospace;
      font-size: 11px;
      border: 1px solid #345;
      border-radius: 6px;
      padding: 14px 16px;
      z-index: 9998;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
      display: none;
    }
    #drop-panel.visible { display: block; }
    #drop-title { font-size: 14px; color: #cdf; font-weight: bold; }
    #drop-filename { font-size: 11px; color: #889; margin: 2px 0 8px 0; word-break: break-all; }
    #drop-audio { width: 100%; height: 28px; margin: 6px 0; }
    .drop-section { margin-top: 14px; }
    .drop-section h4 {
      margin: 0 0 6px 0;
      font-size: 10px;
      letter-spacing: 0.15em;
      color: #7ac;
      font-weight: 600;
    }
    .drop-row {
      display: flex; justify-content: space-between; padding: 3px 0;
      gap: 8px; cursor: pointer;
    }
    .drop-row:hover { background: rgba(120,200,255,0.08); border-radius: 2px; }
    .drop-row .lbl { color: #cfd; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .drop-row .val { color: #ada; font-weight: bold; }
    #drop-close {
      float: right; cursor: pointer; color: #557; font-size: 16px;
      line-height: 14px; padding: 0 4px;
    }
    #drop-close:hover { color: #cdf; }
    #drop-spinner {
      display: none; color: #cdf; font-size: 11px;
      letter-spacing: 0.1em; margin-top: 12px;
    }
    .spinner.show { display: block; }
  `;
  document.head.appendChild(style);

  let overlay = document.createElement('div');
  overlay.id = 'sg-drop-overlay';
  overlay.innerHTML = `<div id="sg-drop-msg">drop audio<div class="hint">CLAP-encoded, mapped, matched against 66,904 wavetables</div></div>`;
  document.body.appendChild(overlay);

  let panel = document.createElement('div');
  panel.id = 'drop-panel';
  panel.innerHTML = `
    <span id="drop-close">×</span>
    <div id="drop-title">audio drop</div>
    <div id="drop-filename"></div>
    <audio id="drop-audio" controls preload="none"></audio>
    <div id="drop-spinner" class="spinner">analyzing…</div>
    <div class="drop-section"><h4>SOUNDS LIKE THESE PRESETS</h4><div id="drop-neighbors">—</div></div>
    <div class="drop-section"><h4>WAVETABLE MATCHES (67k CORPUS)</h4><div id="drop-corpus">—</div></div>
  `;
  document.body.appendChild(panel);

  let panelTitle = panel.querySelector('#drop-title');
  let panelFn = panel.querySelector('#drop-filename');
  let panelAudio = panel.querySelector('#drop-audio');
  let neighEl = panel.querySelector('#drop-neighbors');
  let corpusEl = panel.querySelector('#drop-corpus');
  let spinner = panel.querySelector('#drop-spinner');
  panel.querySelector('#drop-close').addEventListener('click', () => panel.classList.remove('visible'));

  // current upload state
  let lastUploadId = null;
  let lastTopPid = null;
  let chosenBasePid = null;
  let lastFilename = '';

  // Save UI (added after corpus section)
  let saveSection = document.createElement('div');
  saveSection.className = 'drop-section';
  saveSection.innerHTML = `
    <h4>EXPORT</h4>
    <div style="font-size:10px;color:#88a;margin-bottom:6px">
      writes a real .vital to <span style="color:#7ac">~/Music/Vital/User/Presets/</span> using the extracted
      wavetable + the chosen neighbor's FX/modulations as a base.
    </div>
    <div style="font-size:10px;color:#88a;margin-bottom:4px">base preset:</div>
    <div id="drop-base-pid" style="font-size:11px;color:#cfd;margin-bottom:6px">(top neighbor)</div>
    <input id="drop-save-name" type="text" placeholder="preset name (filename)" style="
      width:100%;background:#0a0a14;border:1px solid #345;border-radius:3px;
      padding:6px 8px;color:#cfd;font-family:inherit;font-size:11px;margin-bottom:6px;
      box-sizing:border-box;">
    <button id="drop-save-btn" style="
      width:100%;background:#2a3a5a;border:1px solid #5af;color:#cdf;
      padding:8px;border-radius:3px;cursor:pointer;font-family:inherit;font-size:11px;
      letter-spacing:0.1em">
      💾  save .vital
    </button>
    <div id="drop-save-result" style="font-size:10px;color:#7ca;margin-top:8px;
      word-break:break-all"></div>
  `;
  panel.appendChild(saveSection);
  let baseBadge = saveSection.querySelector('#drop-base-pid');
  let saveBtn = saveSection.querySelector('#drop-save-btn');
  let saveName = saveSection.querySelector('#drop-save-name');
  let saveResult = saveSection.querySelector('#drop-save-result');

  saveBtn.addEventListener('click', async () => {
    if (!lastUploadId) {
      saveResult.innerHTML = '<span style="color:#f88">no upload to save</span>';
      return;
    }
    saveBtn.disabled = true; saveBtn.textContent = 'saving…';
    saveResult.textContent = '';
    try {
      let resp = await fetch('/save', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          upload_id: lastUploadId,
          base_preset_id: chosenBasePid || lastTopPid,
          name: saveName.value || lastFilename.replace(/\\.[^.]+$/, ''),
        }),
      });
      let data = await resp.json();
      if (!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
      saveResult.innerHTML = '<span style="color:#7ca">✓ saved to</span> <span style="color:#cdf">' + data.path + '</span><br>'
        + '<span style="color:#557">reload Vital\\'s preset browser to find it</span>';
    } catch (err) {
      saveResult.innerHTML = '<span style="color:#f88">error: ' + err.message + '</span>';
    } finally {
      saveBtn.disabled = false; saveBtn.textContent = '💾  save .vital';
    }
  });

  let dragDepth = 0;
  function isAudio(types) {
    return types && (Array.from(types).indexOf('Files') >= 0);
  }
  document.addEventListener('dragenter', (e) => {
    if (!isAudio(e.dataTransfer && e.dataTransfer.types)) return;
    dragDepth++;
    overlay.classList.add('active');
    e.preventDefault();
  });
  document.addEventListener('dragover', (e) => { e.preventDefault(); });
  document.addEventListener('dragleave', (e) => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.remove('active');
  });
  document.addEventListener('drop', async (e) => {
    e.preventDefault();
    dragDepth = 0;
    overlay.classList.remove('active');
    let files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    let file = files[0];
    lastFilename = file.name;
    saveName.value = file.name.replace(/\\.[^.]+$/, '');
    panel.classList.add('visible');
    panelTitle.textContent = 'audio drop';
    panelFn.textContent = file.name;
    panelAudio.src = URL.createObjectURL(file);
    neighEl.innerHTML = '<span style="color:#557;font-style:italic">—</span>';
    corpusEl.innerHTML = '<span style="color:#557;font-style:italic">—</span>';
    saveResult.textContent = '';
    spinner.classList.add('show');

    let fd = new FormData();
    fd.append('file', file);
    try {
      let resp = await fetch('/upload', { method: 'POST', body: fd });
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      let data = await resp.json();
      spinner.classList.remove('show');
      lastUploadId = data.upload_id || null;
      lastTopPid = (data.neighbors && data.neighbors[0]) ? data.neighbors[0].preset_id : null;
      chosenBasePid = lastTopPid;
      if (lastTopPid != null && data.neighbors[0]) {
        baseBadge.textContent = data.neighbors[0].name + ' (#' + lastTopPid + ')';
      }
      renderResults(data);
    } catch (err) {
      spinner.classList.remove('show');
      neighEl.innerHTML = '<span style="color:#f88">error: ' + err.message + '</span>';
    }
  });

  function fmt(n) { return Number(n).toFixed(2); }

  function renderResults(data) {
    let n = data.neighbors || [];
    if (!n.length) {
      neighEl.innerHTML = '<span style="color:#557;font-style:italic">no matches</span>';
    } else {
      neighEl.innerHTML = n.map(x => {
        return `<div class="drop-row neighbor" data-pid="${x.preset_id}" data-wav="${x.wav || ''}">
                  <span class="lbl">${x.name || '?'} <span style="color:#557">· ${x.pack || ''}</span></span>
                  <span class="val">${fmt(x.sim)}</span>
                </div>`;
      }).join('');
      neighEl.querySelectorAll('.neighbor').forEach(el => {
        el.addEventListener('click', () => {
          let pid = parseInt(el.dataset.pid);
          let wav = el.dataset.wav;
          if (wav) { panelAudio.src = '/' + wav; panelAudio.play().catch(()=>{}); }
          // Also pick this as the base preset for /save export.
          chosenBasePid = pid;
          baseBadge.textContent = (el.querySelector('.lbl').textContent.trim()) + ' (#' + pid + ')';
          // Trigger the existing preset-click handler if available so the
          // detail panel on the right also updates.
          let plotDiv = document.querySelector('.plotly-graph-div');
          if (plotDiv && plotDiv.data && plotDiv.data[0]) {
            let cdArr = plotDiv.data[0].customdata;
            for (let i = 0; i < cdArr.length; i++) {
              if (parseInt(cdArr[i][7]) === pid) {
                plotDiv.emit('plotly_click', { points: [{ customdata: cdArr[i], pointNumber: i, curveNumber: 0 }] });
                break;
              }
            }
          }
        });
      });
    }

    let c = data.corpus_matches || [];
    if (!c.length) {
      corpusEl.innerHTML = '<span style="color:#557;font-style:italic">no matches</span>';
    } else {
      corpusEl.innerHTML = c.map(x => {
        return `<div class="drop-row corpus" data-idx="${x.idx}">
                  <span class="lbl">${x.name} <span style="color:#557">· ${x.source} / ${x.bank}</span></span>
                  <span class="val">${fmt(x.sim)}</span>
                </div>`;
      }).join('');
      corpusEl.querySelectorAll('.corpus').forEach(el => {
        el.addEventListener('click', () => {
          let idx = parseInt(el.dataset.idx);
          panelAudio.src = '/wt/' + idx + '.wav';
          panelAudio.play().catch(()=>{});
        });
      });
    }
  }
})();
"""


QUANTITATIVE_COLOR_FIELDS = [
    ("brightness",  "spec_centroid", "Plasma"),
    ("loudness",    "rms",           "Viridis"),
    ("noisiness",   "spec_flatness", "Cividis"),
    ("harmonics",   "harmonic_ratio","Magma"),
    ("stereo width","stereo_width",  "Turbo"),
]


def _customdata(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    if "preset_id" not in out.columns:
        out["preset_id"] = -1
    # wav_path is stored relative to DATA_DIR (e.g. "mutations_v1/base_wavs/...")
    # Prefix it with the /data/ route so the browser resolves to Flask's data endpoint.
    if "wav_path" in out.columns:
        wp = out["wav_path"].fillna("").astype(str)
        out["wav_path"] = wp.apply(
            lambda p: ("/data/" + p) if (p and not p.startswith(("/", "http://", "https://"))) else p
        )
    # preset_id last so it doesn't shift earlier indices used elsewhere
    return out[["preset_name", "author", "pack", "preset_style",
                "rms", "spec_centroid", "wav_path", "preset_id"]].fillna("").values


def _hover() -> str:
    return ("<b>%{customdata[0]}</b><br>"
            "by %{customdata[1]}<br>"
            "pack: %{customdata[2]}<br>"
            "style: %{customdata[3]}<br>"
            "rms: %{customdata[4]:.3f}<br>"
            "centroid: %{customdata[5]:.0f} Hz<br>"
            "<extra></extra>")


def _sizes(df: pd.DataFrame, mult: float) -> np.ndarray:
    rms_clipped = df["rms"].clip(0.001, max(0.001, df["rms"].quantile(0.98)))
    return mult * (4 + 22 * (rms_clipped / max(0.001, rms_clipped.max()))).to_numpy()


# Feature candidates for interpreting UMAP axes. Each axis label is chosen by
# strongest |Pearson correlation| with these — producer-meaningful shortlist,
# not all 60 CLAP descriptors (too noisy).
_INTERP_FEATURE_COLS = {
    "brightness":   "spec_centroid",
    "loudness":     "rms",
    "noisiness":    "spec_flatness",
    "harmonics":    "harmonic_ratio",
    "stereo width": "stereo_width",
}
_INTERP_CLAP_LABELS = [
    "bright", "dark", "warm", "hard", "soft", "deep", "sharp", "booming",
    "rough", "metallic", "hollow", "sparkling", "airy", "ringing",
    "percussive", "sustained", "reverberant",
    "sub bass", "supersaw lead", "ethereal pad", "bell-like",
    "atmospheric pad", "aggressive", "gentle", "dreamy",
]


def _axis_labels_for_basis (bases: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> tuple[str, str, str]:
    """For each axis, find its strongest |Pearson correlation| feature."""
    # Build candidate series
    cands: dict[str, pd.Series] = {}
    for lbl, col in _INTERP_FEATURE_COLS.items():
        if col in bases.columns:
            cands[lbl] = bases[col]
    for lbl in _INTERP_CLAP_LABELS:
        col = "clap_" + lbl
        if col in bases.columns:
            cands[lbl] = bases[col]

    labels: list[str] = []
    for axis_col in (x_col, y_col, z_col):
        vals = bases[axis_col]
        best_lbl, best_r = "unaligned", 0.0
        for lbl, series in cands.items():
            r = vals.corr(series)
            if pd.isna (r):
                continue
            if abs (r) > abs (best_r):
                best_r, best_lbl = r, lbl
        if best_lbl == "unaligned":
            labels.append ("(unaligned)")
        else:
            arrow = "↑" if best_r > 0 else "↓"
            labels.append (f"{best_lbl} {arrow}  (r={abs(best_r):.2f})")
    return (labels[0], labels[1], labels[2])


def main(out_path: Path, max_variants: int, seed: int) -> None:
    in_dir = DATA_DIR / "mutations_v1"
    vae_coords = pd.read_parquet(in_dir / "galaxy_coords_vae.parquet")
    metadata = pd.read_parquet(in_dir / "metadata.parquet")
    clap_coords = pd.read_parquet(in_dir / "galaxy_coords_clap.parquet")
    clap_scores = pd.read_parquet(in_dir / "clap_scores.parquet")
    mod_coords = pd.read_parquet(in_dir / "galaxy_coords_mod.parquet")
    wt_coords = pd.read_parquet(in_dir / "galaxy_coords_wt.parquet")
    print(f"Loaded VAE: {len(vae_coords)}, CLAP: {len(clap_coords)}, "
          f"MOD: {len(mod_coords)}, WT: {len(wt_coords)}, "
          f"metadata: {len(metadata)}, CLAP scores: {len(clap_scores)}")

    # ensure metadata has the columns we expect
    for c in ("preset_name", "author", "pack", "preset_style", "wav_path"):
        if c not in metadata.columns:
            metadata[c] = ""
        metadata[c] = metadata[c].fillna("").astype(str)
    # CLAP scores has preset_id but lacks wav_path etc; rely on metadata for those
    score_cols = [c for c in clap_scores.columns if c.startswith("clap_")]

    # ---- BASE TRACE: merge VAE coords + metadata + CLAP coords + CLAP scores ----
    base_meta = metadata[metadata["is_base"]] if "is_base" in metadata.columns \
        else metadata[metadata["variant_id"] == 0]
    base_meta = base_meta.copy()
    bases = base_meta.merge(vae_coords, on="patch_id", how="left",
                            suffixes=("", "_vae"))
    bases = bases.rename(columns={"x": "x_vae", "y": "y_vae", "z": "z_vae"})
    bases = bases.merge(
        clap_coords.rename(columns={"x": "x_clap", "y": "y_clap", "z": "z_clap"})[
            ["preset_id", "x_clap", "y_clap", "z_clap"]
        ],
        on="preset_id", how="left",
    )
    bases = bases.merge(
        mod_coords.rename(columns={"x": "x_mod", "y": "y_mod", "z": "z_mod"})[
            ["preset_id", "x_mod", "y_mod", "z_mod"]
        ],
        on="preset_id", how="left",
    )
    bases = bases.merge(
        wt_coords.rename(columns={"x": "x_wt", "y": "y_wt", "z": "z_wt"})[
            ["preset_id", "x_wt", "y_wt", "z_wt"]
        ],
        on="preset_id", how="left",
    )
    bases = bases.merge(clap_scores[["preset_id"] + score_cols],
                        on="preset_id", how="left")

    # Drop bases missing any coord (defensive — there should be ~few of these)
    bases = bases.dropna(subset=["x_clap", "x_vae", "x_mod", "x_wt"]).reset_index(drop=True)
    print(f"Joined bases: {len(bases)}")

    # ---- VARIANT TRACE: VAE basis only ----
    var_mask = (~metadata["is_base"]) if "is_base" in metadata.columns \
        else (metadata["variant_id"] > 0)
    # Use the same suffix policy as bases: keep left-side metadata column names
    # for rms / spec_centroid / etc., suffix the duplicates from vae_coords.
    variants = metadata[var_mask].merge(
        vae_coords[["patch_id", "x", "y", "z"]], on="patch_id", how="left",
    )
    if len(variants) > max_variants:
        variants = variants.sample(n=max_variants, random_state=seed).reset_index(drop=True)
    print(f"Variants in plot (sampled): {len(variants)}")

    # ---- traces (default basis = VAE, default color = brightness) ----
    bases_trace = go.Scatter3d(
        x=bases["x_vae"], y=bases["y_vae"], z=bases["z_vae"],
        name=f"Real presets ({len(bases)})",
        mode="markers",
        marker=dict(
            size=_sizes(bases, 1.0),
            color=bases["spec_centroid"],
            colorscale="Plasma",
            cmin=bases["spec_centroid"].min(),
            cmax=bases["spec_centroid"].max(),
            colorbar=dict(
                title=dict(text="brightness", font=dict(color="white")),
                tickfont=dict(color="white"),
            ),
            opacity=0.9,
            line=dict(width=0),
        ),
        customdata=_customdata(bases),
        hovertemplate=_hover(),
    )
    variants_trace = go.Scatter3d(
        x=variants["x"], y=variants["y"], z=variants["z"],
        name=f"Mutations ({len(variants)})",
        mode="markers",
        marker=dict(
            size=_sizes(variants, 0.35),
            color=variants["spec_centroid"],
            colorscale="Plasma",
            cmin=bases["spec_centroid"].min(),
            cmax=bases["spec_centroid"].max(),
            opacity=0.25,
            line=dict(width=0),
            showscale=False,
        ),
        customdata=_customdata(variants),
        hovertemplate=_hover(),
        showlegend=True,
    )
    fig = go.Figure(data=[bases_trace, variants_trace])

    # ---- Interpret each basis's axes ----
    vae_labels  = _axis_labels_for_basis (bases, "x_vae",  "y_vae",  "z_vae")
    clap_labels = _axis_labels_for_basis (bases, "x_clap", "y_clap", "z_clap")
    mod_labels  = _axis_labels_for_basis (bases, "x_mod",  "y_mod",  "z_mod")
    wt_labels   = _axis_labels_for_basis (bases, "x_wt",   "y_wt",   "z_wt")
    print ("Axis interpretations:")
    for name, ls in [("VAE", vae_labels), ("CLAP", clap_labels),
                     ("MOD", mod_labels), ("WT", wt_labels)]:
        print (f"  {name}: x={ls[0]}  y={ls[1]}  z={ls[2]}")

    def _axis_layout (labels: tuple[str, str, str]) -> dict:
        return {
            "scene.xaxis.title.text": labels[0],
            "scene.yaxis.title.text": labels[1],
            "scene.zaxis.title.text": labels[2],
        }

    # ---- Basis dropdown (swaps x/y/z arrays, toggles variants visibility) ----
    basis_buttons = [
        dict(
            label="by synth design (VAE)",
            method="update",
            args=[{
                "x": [bases["x_vae"].to_numpy(), variants["x"].to_numpy()],
                "y": [bases["y_vae"].to_numpy(), variants["y"].to_numpy()],
                "z": [bases["z_vae"].to_numpy(), variants["z"].to_numpy()],
                "visible": [True, True],
            }, _axis_layout (vae_labels)],
        ),
        dict(
            label="by how it sounds (CLAP)",
            method="update",
            args=[{
                "x": [bases["x_clap"].to_numpy(), variants["x"].to_numpy()],
                "y": [bases["y_clap"].to_numpy(), variants["y"].to_numpy()],
                "z": [bases["z_clap"].to_numpy(), variants["z"].to_numpy()],
                "visible": [True, False],
            }, _axis_layout (clap_labels)],
        ),
        dict(
            label="by mod structure (engineering DNA)",
            method="update",
            args=[{
                "x": [bases["x_mod"].to_numpy(), variants["x"].to_numpy()],
                "y": [bases["y_mod"].to_numpy(), variants["y"].to_numpy()],
                "z": [bases["z_mod"].to_numpy(), variants["z"].to_numpy()],
                "visible": [True, False],
            }, _axis_layout (mod_labels)],
        ),
        dict(
            label="by wavetable (timbre skeleton)",
            method="update",
            args=[{
                "x": [bases["x_wt"].to_numpy(), variants["x"].to_numpy()],
                "y": [bases["y_wt"].to_numpy(), variants["y"].to_numpy()],
                "z": [bases["z_wt"].to_numpy(), variants["z"].to_numpy()],
                "visible": [True, False],
            }, _axis_layout (wt_labels)],
        ),
    ]

    # ---- Color buttons ----
    color_buttons: list[dict] = []

    def section(label: str) -> dict:
        # Header row in the dropdown — method='skip' makes it a no-op.
        return dict(label=label, method="skip", args=[{}])

    # ── basic features ──
    color_buttons.append(section("── BASIC ──"))
    for label, col, cmap in QUANTITATIVE_COLOR_FIELDS:
        if col not in bases.columns or col not in variants.columns:
            continue
        cmin = min(bases[col].min(), variants[col].min())
        cmax = max(bases[col].max(), variants[col].max())
        color_buttons.append(dict(
            label=label, method="restyle",
            args=[{
                "marker.color": [bases[col].to_numpy(), variants[col].to_numpy()],
                "marker.colorscale": [cmap, cmap],
                "marker.cmin": [cmin, cmin],
                "marker.cmax": [cmax, cmax],
            }],
        ))

    # categorical (pack)
    all_packs = pd.concat([bases["pack"], variants["pack"]]).fillna("")
    uniq = sorted([p for p in all_packs.unique() if p])
    lookup = {p: i for i, p in enumerate(uniq)}
    color_buttons.append(dict(
        label="pack", method="restyle",
        args=[{
            "marker.color": [
                bases["pack"].fillna("").map(lookup).fillna(-1).to_numpy(),
                variants["pack"].fillna("").map(lookup).fillna(-1).to_numpy(),
            ],
            "marker.colorscale": ["Rainbow", "Rainbow"],
            "marker.cmin": [0, 0],
            "marker.cmax": [max(1, len(uniq) - 1)] * 2,
        }],
    ))

    # CLAP descriptors — group by section with headers, skip ones missing from parquet
    variants_zero = np.zeros(len(variants), dtype=np.float32)
    last_group = None
    for group, lbl, _ in all_descriptors():
        col = f"clap_{lbl}"
        if col not in bases.columns:
            continue
        if group != last_group:
            color_buttons.append(section(f"── {group.upper()} ──"))
            last_group = group
        base_vals = bases[col].to_numpy()
        cmin, cmax = float(base_vals.min()), float(base_vals.max())
        color_buttons.append(dict(
            label=lbl,
            method="restyle",
            args=[{
                "marker.color": [base_vals, variants_zero],
                "marker.colorscale": ["Inferno", "Greys"],
                "marker.cmin": [cmin, cmin],
                "marker.cmax": [cmax, cmax],
            }],
        ))

    # ---- Show dropdown ----
    show_buttons = [
        dict(label="everything", method="restyle", args=[{"visible": [True, True]}]),
        dict(label="real presets only", method="restyle", args=[{"visible": [True, False]}]),
        dict(label="mutations only", method="restyle", args=[{"visible": [False, True]}]),
    ]

    fig.update_layout(
        title=dict(
            text=f"synth-galaxy v2 · {len(bases)} bases + {len(variants)} variants "
                 f"· multi-basis + CLAP descriptors",
            font=dict(color="white", size=14),
        ),
        paper_bgcolor="#0a0a14",
        scene=dict(
            xaxis=dict(title=vae_labels[0], color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            yaxis=dict(title=vae_labels[1], color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            zaxis=dict(title=vae_labels[2], color="white", backgroundcolor="#0a0a14",
                       gridcolor="rgba(255,255,255,0.1)", showbackground=True),
            bgcolor="#0a0a14",
        ),
        margin=dict(l=0, r=0, t=120, b=0),
        font=dict(color="white"),
        legend=dict(font=dict(color="white"), x=1.0, y=0.5),
        updatemenus=[
            dict(buttons=basis_buttons, direction="down", showactive=True,
                 x=0.0, y=1.15, xanchor="left", yanchor="top",
                 bgcolor="#1a1a30", bordercolor="#444", font=dict(color="white")),
            dict(buttons=color_buttons, direction="down", showactive=True,
                 x=0.30, y=1.15, xanchor="left", yanchor="top",
                 bgcolor="#1a1a30", bordercolor="#444", font=dict(color="white")),
            dict(buttons=show_buttons, direction="down", showactive=True,
                 x=0.60, y=1.15, xanchor="left", yanchor="top",
                 bgcolor="#1a1a30", bordercolor="#444", font=dict(color="white")),
        ],
        annotations=[
            dict(text="Basis:", showarrow=False, x=0.0, y=1.20,
                 xref="paper", yref="paper", xanchor="left",
                 font=dict(color="#cfd", size=12)),
            dict(text="Color by:", showarrow=False, x=0.30, y=1.20,
                 xref="paper", yref="paper", xanchor="left",
                 font=dict(color="#cfd", size=12)),
            dict(text="Show:", showarrow=False, x=0.60, y=1.20,
                 xref="paper", yref="paper", xanchor="left",
                 font=dict(color="#cfd", size=12)),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True,
                   post_script=CLICK_AUDIO_JS)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"http://localhost:8000/data/{out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=DATA_DIR / "galaxy_v2.html")
    ap.add_argument("--max-variants", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.out, args.max_variants, args.seed)

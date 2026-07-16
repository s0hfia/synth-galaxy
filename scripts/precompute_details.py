"""Precompute per-preset rich details for the click panel.

For each base preset writes:
  - top_5_clap        list of (label, score) sorted by score descending
  - top_5_neighbors   list of {preset_id, name, pack, sim} via CLAP cosine
  - top_10_params     list of {name, value, z, pop_mean} via z-score vs population
  - top_5_mods        list of {source, dest, amount, slot} from .vital JSON

Output:
  data/mutations_v1/details.parquet  (one row per base preset, JSON columns)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from synth_galaxy.config import DATA_DIR


PRESETS_ROOT = Path("/Users/sof/Music/Vital")
MUT_DIR = DATA_DIR / "mutations_v1"


def humanize(snake: str) -> str:
    """osc_2_distortion_amount -> Osc 2 Distortion Amount."""
    return " ".join(part.upper() if part in {"lfo", "fx", "eq"} else part.capitalize()
                    for part in snake.split("_"))


# ---------- modulation fingerprint ----------
SOURCE_FAMILIES = ["lfo", "env", "macro", "other"]
DEST_FAMILIES = ["osc", "filter", "fx", "mix"]
FX_DEST_PREFIXES = ("chorus", "delay", "reverb", "phaser", "flanger",
                    "distortion", "compressor", "eq_")


def source_family_idx(src: str) -> int:
    s = src.lower()
    if s.startswith("lfo_") or s == "lfo":
        return 0
    head = s.split("_", 1)[0]
    if head in ("env", "envelope") or s.startswith("env_") or s.startswith("envelope_"):
        return 1
    if s.startswith("macro"):
        return 2
    return 3  # random_lfo_*, mod_wheel, pitch_wheel, velocity, note, aftertouch...


def dest_family_idx(dst: str) -> int:
    s = dst.lower()
    if s.startswith("osc_"):
        return 0
    if s.startswith("filter_"):
        return 1
    if any(s.startswith(p) for p in FX_DEST_PREFIXES):
        return 2
    return 3  # levels, voice, master, env timing, lfo timing, …


def fingerprint_svg(fp, w: int = 200, h: int = 96) -> str:
    """Render the 4x4 mod fingerprint as a self-contained SVG glyph."""
    pad = 6
    label_w = 32
    label_h = 12
    rows, cols = 4, 4
    cell_w = (w - pad * 2 - label_w) / cols
    cell_h = (h - pad * 2 - label_h) / rows

    vmax = float(fp.max()) if fp.max() > 0 else 1.0

    parts: list[str] = [f'<rect width="{w}" height="{h}" fill="#0a0a14" rx="4"/>']

    # cells
    for r in range(rows):
        for c in range(cols):
            v = float(fp[r, c])
            t = (v / vmax) if vmax > 0 else 0.0
            # Cool→warm: deep blue (low) → cyan (mid) → yellow (high)
            if t < 0.5:
                u = t / 0.5
                rr = int(20 + (90 - 20) * u)
                gg = int(40 + (200 - 40) * u)
                bb = int(120 + (220 - 120) * u)
            else:
                u = (t - 0.5) / 0.5
                rr = int(90 + (240 - 90) * u)
                gg = int(200 + (220 - 200) * u)
                bb = int(220 - (220 - 80) * u)
            color = f"rgb({rr},{gg},{bb})"
            alpha = 0.08 + 0.92 * t if v > 0 else 0.08
            x = pad + label_w + c * cell_w
            y = pad + r * cell_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 1:.1f}" '
                f'height="{cell_h - 1:.1f}" fill="{color}" fill-opacity="{alpha:.2f}" rx="2"/>'
            )
            if v > 0.05:
                cx, cy = x + cell_w / 2, y + cell_h / 2 + 3
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy:.1f}" fill="white" text-anchor="middle" '
                    f'font-size="8" font-family="ui-monospace,monospace">{v:.1f}</text>'
                )

    # row labels (sources, left)
    for r in range(rows):
        y = pad + r * cell_h + cell_h / 2 + 3
        parts.append(
            f'<text x="{pad + label_w - 4}" y="{y:.1f}" fill="#7a8" text-anchor="end" '
            f'font-size="9" font-family="ui-monospace,monospace">{SOURCE_FAMILIES[r]}</text>'
        )

    # col labels (destinations, bottom)
    for c in range(cols):
        x = pad + label_w + c * cell_w + cell_w / 2
        y = h - 2
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="#7a8" text-anchor="middle" '
            f'font-size="9" font-family="ui-monospace,monospace">{DEST_FAMILIES[c]}</text>'
        )

    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def main() -> None:
    print("Loading inputs...")
    meta = pd.read_parquet(MUT_DIR / "metadata.parquet")
    params_arr = np.load(MUT_DIR / "params.npy")           # (44099, 401)
    candidates = np.load(MUT_DIR / "synthesis_param_indices.npy")  # (401,)
    scores_df = pd.read_parquet(MUT_DIR / "clap_scores.parquet")
    audio_embs = np.load(MUT_DIR / "clap_audio_embeddings.npy")    # (2147, 512), normalized

    vital_param_list = json.loads((DATA_DIR / "vital_params.json").read_text())
    name_by_idx = {int(p["idx"]): p["name"] for p in vital_param_list}
    candidate_names = [name_by_idx[int(idx)] for idx in candidates]

    bases = meta[meta["is_base"]].copy().reset_index(drop=True)
    print(f"Bases: {len(bases)}   total in metadata: {len(meta)}")

    # Population stats (on bases only — variants would skew toward random)
    base_patch_ids = bases["patch_id"].to_numpy(dtype=int)
    base_params = params_arr[base_patch_ids]
    pop_mean = base_params.mean(axis=0)
    pop_std = base_params.std(axis=0) + 1e-6

    # CLAP column lookup
    clap_cols = [c for c in scores_df.columns if c.startswith("clap_")]
    clap_labels = [c.removeprefix("clap_") for c in clap_cols]
    clap_matrix = scores_df[clap_cols].to_numpy()              # (2147, 60)
    score_preset_ids = scores_df["preset_id"].to_numpy(dtype=int)
    pid_to_score_idx = {int(pid): i for i, pid in enumerate(score_preset_ids)}

    # For neighbor naming
    pid_to_name = dict(zip(bases["preset_id"], bases["preset_name"]))
    pid_to_pack = dict(zip(bases["preset_id"], bases["pack"]))

    rows = []
    fingerprints = np.zeros((len(bases), 4, 4), dtype=np.float32)
    fingerprint_order_pids: list[int] = []
    skipped_mods = 0
    for fp_row_idx, (_, r) in enumerate(tqdm(bases.iterrows(), total=len(bases), desc="Details")):
        pid = int(r["preset_id"])
        patch_id = int(r["patch_id"])

        # --- top 5 CLAP descriptors ---
        top5_clap = []
        if pid in pid_to_score_idx:
            si = pid_to_score_idx[pid]
            scores_row = clap_matrix[si]
            top_idx = np.argsort(scores_row)[::-1][:5]
            top5_clap = [
                {"label": clap_labels[i], "score": float(scores_row[i])}
                for i in top_idx
            ]

        # --- top 5 sonic neighbors via CLAP cosine ---
        top5_neighbors = []
        if pid in pid_to_score_idx:
            si = pid_to_score_idx[pid]
            sims = audio_embs @ audio_embs[si]
            sims[si] = -1.0
            top_idx = np.argsort(sims)[::-1][:5]
            for nidx in top_idx:
                n_pid = int(score_preset_ids[nidx])
                top5_neighbors.append({
                    "preset_id": n_pid,
                    "name": pid_to_name.get(n_pid, ""),
                    "pack": pid_to_pack.get(n_pid, ""),
                    "sim": float(sims[nidx]),
                })

        # --- top 10 differentiating params (|z-score|) ---
        p = params_arr[patch_id]
        z = (p - pop_mean) / pop_std
        top10_idx = np.argsort(np.abs(z))[::-1][:10]
        top10_params = [
            {
                "name": candidate_names[i],
                "value": float(p[i]),
                "z": float(z[i]),
                "pop_mean": float(pop_mean[i]),
            }
            for i in top10_idx
        ]

        # --- top 5 modulations + 4x4 fingerprint from .vital JSON ---
        top5_mods = []
        fp = np.zeros((4, 4), dtype=np.float32)
        preset_path = PRESETS_ROOT / r["preset_path"]
        if preset_path.exists():
            try:
                j = json.loads(preset_path.read_text(encoding="utf-8"))
                settings = j.get("settings", {})
                mods_list = settings.get("modulations", []) or []
                amts: dict[int, float] = {}
                for k, v in settings.items():
                    if k.startswith("modulation_") and k.endswith("_amount") and isinstance(v, (int, float)):
                        try:
                            slot = int(k.split("_")[1])
                            amts[slot] = float(v)
                        except (ValueError, IndexError):
                            pass

                rich = []
                for i, m in enumerate(mods_list):
                    slot = i + 1
                    amt = amts.get(slot, 0.0)
                    src = (m or {}).get("source", "") or ""
                    dst = (m or {}).get("destination", "") or ""
                    if src and dst and abs(amt) > 0.001:
                        rich.append({
                            "source": humanize(src),
                            "dest": humanize(dst),
                            "amount": amt,
                            "slot": slot,
                        })
                        si = source_family_idx(src)
                        di = dest_family_idx(dst)
                        fp[si, di] += abs(amt)
                rich.sort(key=lambda x: -abs(x["amount"]))
                top5_mods = rich[:5]
            except Exception:
                skipped_mods += 1

        fingerprints[fp_row_idx] = fp
        fingerprint_order_pids.append(pid)

        rows.append({
            "preset_id": pid,
            "patch_id": patch_id,
            "top5_clap": json.dumps(top5_clap),
            "top5_neighbors": json.dumps(top5_neighbors),
            "top10_params": json.dumps(top10_params),
            "top5_mods": json.dumps(top5_mods),
            "fingerprint_svg": fingerprint_svg(fp),
        })

    df = pd.DataFrame(rows)
    out = MUT_DIR / "details.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({out.stat().st_size / 1024:.1f} KB)")

    # Save fingerprint matrices for UMAP (modulation galaxy basis).
    np.save(MUT_DIR / "mod_fingerprints.npy", fingerprints)
    np.save(MUT_DIR / "mod_fingerprints_preset_ids.npy",
            np.array(fingerprint_order_pids, dtype=np.int64))
    print(f"Wrote mod_fingerprints.npy  shape={fingerprints.shape}")

    # Also write a flat JSON dict so the browser can fetch it once and look up by preset_id.
    details_json = {}
    for _, row in df.iterrows():
        details_json[str(int(row["preset_id"]))] = {
            "clap": json.loads(row["top5_clap"]),
            "neighbors": json.loads(row["top5_neighbors"]),
            "params": json.loads(row["top10_params"]),
            "mods": json.loads(row["top5_mods"]),
            "fp_svg": row["fingerprint_svg"],
        }
    out_json = MUT_DIR / "details.json"
    out_json.write_text(json.dumps(details_json))
    print(f"Wrote {out_json}  ({out_json.stat().st_size / 1024:.1f} KB)")
    print(f"Mod-parse failures: {skipped_mods}")

    # peek at one
    sample = df.iloc[0]
    print(f"\nSample (preset_id={sample['preset_id']}):")
    print(f"  top5_clap: {json.loads(sample['top5_clap'])[:3]}")
    print(f"  top5_neighbors[0]: {json.loads(sample['top5_neighbors'])[0] if json.loads(sample['top5_neighbors']) else None}")
    print(f"  top10_params[0:3]: {json.loads(sample['top10_params'])[:3]}")
    print(f"  top5_mods[:3]: {json.loads(sample['top5_mods'])[:3]}")


if __name__ == "__main__":
    main()

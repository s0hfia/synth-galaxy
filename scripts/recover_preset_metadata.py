"""Rebuild the dataset parquet from already-rendered WAVs.

When render_presets.py is killed before completion, the audio survives but the
metadata parquet never gets written. This walks the preset folder in the same
sorted order, matches each rendered WAV by its 6-digit filename prefix back to
the original preset path, and rebuilds the parquet without re-rendering.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from synth_galaxy.config import AUDIO_DIR, DATA_DIR, PATCHES_DIR


def find_presets(root: Path) -> list[Path]:
    return sorted(root.rglob("*.vital"))


def list_rendered(audio_dir: Path) -> dict[int, Path]:
    """Map preset_id (6-digit filename prefix) -> WAV path."""
    out: dict[int, Path] = {}
    pattern = re.compile(r"^(\d{6})_")
    for p in audio_dir.glob("*.wav"):
        m = pattern.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def main(preset_root: Path, audio_dir: Path) -> None:
    presets = find_presets(preset_root)
    rendered = list_rendered(audio_dir)
    print(f"Preset folder: {len(presets)} files")
    print(f"Rendered WAVs: {len(rendered)} files in {audio_dir}")

    rows = []
    skipped = 0
    for i, preset_path in enumerate(tqdm(presets, desc="Recovering")):
        if i not in rendered:
            skipped += 1
            continue
        wav = rendered[i]
        try:
            vital_json = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception:
            vital_json = {}

        # Compute audio stats from the WAV instead of re-rendering.
        audio, sr = sf.read(wav, always_2d=True)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        rms = float(np.sqrt((mono ** 2).mean()))
        peak = float(np.abs(mono).max())

        pack = preset_path.parent.relative_to(preset_root).parts[0] \
            if preset_path.parent != preset_root else "root"

        rows.append({
            "preset_id": i,
            "preset_path": str(preset_path.relative_to(preset_root)),
            "preset_name": vital_json.get("preset_name") or preset_path.stem,
            "author": vital_json.get("author") or "",
            "preset_style": vital_json.get("preset_style") or "",
            "pack": pack,
            "wav_path": str(wav.relative_to(DATA_DIR)),
            "rms": rms,
            "peak": peak,
        })

    df = pd.DataFrame(rows)
    out_pq = PATCHES_DIR / f"presets_n{len(rows)}.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"\nRecovered metadata for {len(df)} presets (skipped {skipped})")
    print(f"Wrote {out_pq}")
    print(f"Mean RMS: {df['rms'].mean():.4f}  "
          f"silent (rms<0.001): {(df['rms'] < 0.001).sum()}  "
          f"clipping (peak>=0.99): {(df['peak'] >= 0.99).sum()}")
    print("\nTop 10 packs by preset count:")
    print(df["pack"].value_counts().head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Users/sof/Music/Vital"))
    ap.add_argument("--audio-dir", type=Path, default=AUDIO_DIR / "presets")
    args = ap.parse_args()
    main(args.root, args.audio_dir)

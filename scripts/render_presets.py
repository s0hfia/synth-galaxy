"""Render every .vital preset under a folder, capturing audio + metadata."""

import argparse
import json
import re
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from synth_galaxy.config import AUDIO_DIR, DATA_DIR, PATCHES_DIR, SAMPLE_RATE
from synth_galaxy.preset_loader_full import vital_json_to_state_file
from synth_galaxy.render import load_vital, make_engine, render_note


def safe_filename(stem: str) -> str:
    """Make a filename-safe stem, preserving readability."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return cleaned[:80] or "unnamed"


def find_presets(root: Path) -> list[Path]:
    return sorted(root.rglob("*.vital"))


def main(preset_root: Path, limit: int | None, fail_fast: bool) -> None:
    presets = find_presets(preset_root)
    print(f"Found {len(presets)} .vital files in {preset_root}")
    if limit:
        presets = presets[:limit]
        print(f"Limiting to first {limit}")

    engine = make_engine()
    synth = load_vital(engine)
    print("Vital loaded.")

    preset_audio_dir = AUDIO_DIR / "presets"
    preset_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="synth-galaxy-state-") as tmp:
        tmp_dir = Path(tmp)
        for i, preset_path in enumerate(tqdm(presets, desc="Rendering")):
            try:
                vital_json = json.loads(preset_path.read_text(encoding="utf-8"))
                state_path = tmp_dir / f"state_{i:06d}.bin"
                vital_json_to_state_file(vital_json, state_path)
                synth.load_state(str(state_path))
                audio = render_note(engine, synth)

                safe = safe_filename(preset_path.stem)
                out_wav = preset_audio_dir / f"{i:06d}_{safe}.wav"
                sf.write(out_wav, audio.T, SAMPLE_RATE)

                pack = preset_path.parent.relative_to(preset_root).parts[0] \
                    if preset_path.parent != preset_root else "root"
                rms = float(np.sqrt((audio ** 2).mean()))
                peak = float(np.abs(audio).max())

                rows.append({
                    "preset_id": i,
                    "preset_path": str(preset_path.relative_to(preset_root)),
                    "preset_name": vital_json.get("preset_name") or preset_path.stem,
                    "author": vital_json.get("author") or "",
                    "preset_style": vital_json.get("preset_style") or "",
                    "pack": pack,
                    "wav_path": str(out_wav.relative_to(DATA_DIR)),
                    "rms": rms,
                    "peak": peak,
                })
            except Exception as e:
                failures.append((str(preset_path), f"{type(e).__name__}: {e}"))
                if fail_fast:
                    traceback.print_exc()
                    raise

    df = pd.DataFrame(rows)
    out_pq = PATCHES_DIR / f"presets_n{len(rows)}.parquet"
    df.to_parquet(out_pq, index=False)

    elapsed = time.time() - t0
    print(f"\nRendered {len(rows)}/{len(presets)} in {elapsed:.1f}s "
          f"({elapsed / max(1, len(presets)):.2f}s/preset)")
    print(f"Failures: {len(failures)}")
    for path, err in failures[:5]:
        print(f"  {path}: {err}")

    print(f"\nDataset parquet: {out_pq}")
    print(f"Mean RMS: {df['rms'].mean():.4f}  "
          f"silent (rms<0.001): {(df['rms'] < 0.001).sum()}  "
          f"clipping (peak>=0.99): {(df['peak'] >= 0.99).sum()}")
    if len(df) and "pack" in df.columns:
        print("\nTop 10 packs by preset count:")
        print(df["pack"].value_counts().head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Users/sof/Music/Vital"))
    ap.add_argument("--limit", type=int, default=None, help="Render only first N presets")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()
    main(args.root, args.limit, args.fail_fast)

"""Probe which dawdreamer load method works on a .vital preset."""

import json
import sys
from pathlib import Path

from synth_galaxy.render import load_vital, make_engine, render_note
from synth_galaxy.config import SAMPLE_RATE
import numpy as np
import soundfile as sf

PRESET = Path("/Users/sof/Music/Vital/Organisms/BS Cyclops.vital")


def render_and_stat(synth, engine, label: str) -> None:
    audio = render_note(engine, synth)
    rms = float(np.sqrt((audio ** 2).mean()))
    peak = float(np.abs(audio).max())
    print(f"  [{label}] shape={audio.shape}  rms={rms:.4f}  peak={peak:.4f}")
    out = Path(__file__).parent.parent / "data" / "audio" / f"preset_test_{label}.wav"
    sf.write(out, audio.T, SAMPLE_RATE)
    print(f"    wrote {out.name}")


def main() -> None:
    print(f"Preset: {PRESET}  size={PRESET.stat().st_size} bytes\n")

    engine = make_engine()
    synth = load_vital(engine)

    # Baseline: render before loading anything to compare
    print("Baseline render (Vital default state):")
    render_and_stat(synth, engine, "00_default")

    # Method 1: load_preset(path)
    print("\nMethod 1: synth.load_preset(str(path))")
    try:
        result = synth.load_preset(str(PRESET))
        print(f"  load_preset returned: {result!r}")
        render_and_stat(synth, engine, "01_load_preset")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Reset
    synth = load_vital(make_engine())

    # Method 2: load_state(bytes) — feed raw file bytes
    print("\nMethod 2: synth.load_state(file_bytes)")
    try:
        data = PRESET.read_bytes()
        result = synth.load_state(data)
        print(f"  load_state returned: {result!r}")
        engine2 = make_engine()  # need same engine as synth
        # Actually we can't switch engines mid-stream. Just render with the existing one.
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Method 3: parse JSON, walk params, call set_parameter
    print("\nMethod 3: parse JSON, inspect structure")
    try:
        data = PRESET.read_bytes()
        # Try plain JSON first, then gzip
        try:
            preset = json.loads(data)
            print("  Format: plain JSON")
        except json.JSONDecodeError:
            import gzip
            preset = json.loads(gzip.decompress(data))
            print("  Format: gzipped JSON")
        print(f"  Top-level keys: {list(preset.keys())}")
        if "settings" in preset:
            settings = preset["settings"]
            print(f"  settings keys (first 30): {list(settings.keys())[:30]}")
            print(f"  total settings entries: {len(settings)}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

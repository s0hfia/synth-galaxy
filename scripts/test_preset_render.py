"""Render a handful of .vital presets via the flat-params loader. Listen to verify."""

from pathlib import Path

import numpy as np
import soundfile as sf

from synth_galaxy.config import AUDIO_DIR, SAMPLE_RATE
from synth_galaxy.preset_loader import (
    apply_preset_to_synth,
    build_mapping,
    load_vital_json,
)
from synth_galaxy.render import load_vital, make_engine, render_note
from synth_galaxy.sampler import capture_state


TEST_PRESETS = [
    "/Users/sof/Music/Vital/Organisms/BS Cyclops.vital",
    "/Users/sof/Music/Vital/Organisms/FX Prowler.vital",
    "/Users/sof/Music/Vital/Organisms/KEYS Galaxy 01 (C0 16th).vital",
    "/Users/sof/Music/Vital/Organisms/LD FMMODE.vital",
    "/Users/sof/Music/Vital/Organisms/ARP Eclipse 02.vital",
]


def main() -> None:
    engine = make_engine()
    synth = load_vital(engine)
    defaults = capture_state(synth)
    print(f"Loaded Vital. {len(defaults)} host params.")

    # Build mapping from one preset's keys (they should all share the same schema)
    sample_preset = load_vital_json(Path(TEST_PRESETS[0]))
    all_keys = list(sample_preset["settings"].keys())
    mapping, unmatched = build_mapping(synth, all_keys)
    print(f"\nMapping coverage: {len(mapping)}/{len(all_keys)} keys mapped to host params")
    print(f"Unmatched (first 20): {unmatched[:20]}")
    print()

    # Default render for baseline
    audio = render_note(engine, synth)
    base_rms = float(np.sqrt((audio ** 2).mean()))
    print(f"BASELINE  rms={base_rms:.4f}")

    out_dir = AUDIO_DIR
    for preset_path in TEST_PRESETS:
        path = Path(preset_path)
        preset = load_vital_json(path)
        n_written = apply_preset_to_synth(synth, preset["settings"], mapping, defaults)
        audio = render_note(engine, synth)
        rms = float(np.sqrt((audio ** 2).mean()))
        peak = float(np.abs(audio).max())

        safe = path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        out = out_dir / f"preset_{safe}.wav"
        sf.write(out, audio.T, SAMPLE_RATE)
        print(f"  {path.name:50s} written={n_written:3d}  rms={rms:.4f}  peak={peak:.4f}  → {out.name}")


if __name__ == "__main__":
    main()

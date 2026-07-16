"""Smoke test: load Vital with default state, render middle C, dump WAV + stats."""

import numpy as np

from synth_galaxy.config import AUDIO_DIR, SAMPLE_RATE, VITAL_VST3_PATH
from synth_galaxy.render import load_vital, make_engine, render_note, write_wav


def main() -> None:
    print(f"Vital.vst3 at: {VITAL_VST3_PATH}  exists={VITAL_VST3_PATH.exists()}")

    engine = make_engine()
    print("Engine created.")

    synth = load_vital(engine)
    print(f"Vital loaded: {type(synth).__name__}")

    public_attrs = [a for a in dir(synth) if not a.startswith("_")]
    print(f"PluginProcessor public API ({len(public_attrs)} attrs):")
    for a in sorted(public_attrs):
        print(f"  - {a}")

    audio = render_note(engine, synth)
    print(f"Rendered audio shape: {audio.shape}  dtype: {audio.dtype}")
    print(f"  min={audio.min():.4f}  max={audio.max():.4f}  rms={np.sqrt((audio**2).mean()):.4f}")

    out_path = AUDIO_DIR / "smoke_test.wav"
    write_wav(out_path, audio, SAMPLE_RATE)
    print(f"Wrote {out_path}  ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

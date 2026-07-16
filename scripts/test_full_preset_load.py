"""Verify `preset_loader_full.vital_json_to_state_file` actually loads a
preset into dawdreamer-hosted Vital.vst3 and produces audible differences.

Renders middle-C on three known presets via the wrapped-state path and
asserts that the rendered audio differs meaningfully from the default-state
render (RMS and spectral centroid clearly differ between them, and pairwise
between the three presets).

Run via:
    cd /Users/sof/projects/synth-galaxy && uv run python scripts/test_full_preset_load.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from synth_galaxy.config import AUDIO_DIR, DATA_DIR, SAMPLE_RATE
from synth_galaxy.preset_loader_full import (
    juce_base64_decode,
    juce_base64_encode,
    parse_state_file,
    vital_json_to_state_file,
)
from synth_galaxy.render import load_vital, make_engine, render_note


PRESETS = [
    Path("/Users/sof/Music/Vital/Organisms/BS Cyclops.vital"),
    Path("/Users/sof/Music/Vital/Organisms/FX Prowler.vital"),
    Path("/Users/sof/Music/Vital/Organisms/KEYS Galaxy 01 (C0 16th).vital"),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute single-number spectral centroid (Hz) over the full clip.

    Operates on mono mix-down. Returns 0 if audio is silent.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    if mono.size == 0 or float(np.abs(mono).max()) < 1e-9:
        return 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    if spec.sum() < 1e-12:
        return 0.0
    return float((freqs * spec).sum() / spec.sum())


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt((audio**2).mean()))


# ---------------------------------------------------------------------------
# Self-tests on the format helpers (cheap, no engine needed)
# ---------------------------------------------------------------------------

def selftest_juce_base64() -> None:
    """Round-trip arbitrary byte payloads and verify the encoder/decoder match.

    Also re-encodes the IComponent base64 from the captured default_state.bin
    and checks we reproduce it byte-for-byte (this is the ground-truth
    cross-check against actual JUCE-produced output).
    """
    rng = np.random.default_rng(42)
    for n in [0, 1, 2, 3, 5, 7, 64, 1000, 100_000]:
        payload = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        encoded = juce_base64_encode(payload)
        decoded = juce_base64_decode(encoded)
        assert decoded == payload, f"round-trip failed at n={n}"

    default_path = DATA_DIR / "default_state.bin"
    if default_path.exists():
        parsed = parse_state_file(default_path.read_bytes())
        if parsed["icomponent_b64"]:
            recoded = juce_base64_encode(parsed["icomponent_bytes"])
            assert recoded == parsed["icomponent_b64"], (
                "JUCE base64 re-encoding does NOT match captured ground truth — "
                f"got {recoded[:40]!r} vs {parsed['icomponent_b64'][:40]!r}"
            )
            print("  base64 ground-truth check: OK")
        else:
            print("  default_state.bin has no IComponent? skipped ground-truth check")
    else:
        print(f"  no default_state.bin at {default_path}; skipped ground-truth check")

    print("  base64 round-trip tests: OK")


# ---------------------------------------------------------------------------
# End-to-end render test
# ---------------------------------------------------------------------------

def render_one(synth, engine, label: str, wav_dir: Path) -> tuple[float, float]:
    """Render middle-C; write a wav for sanity-listening; return (rms, centroid)."""
    audio = render_note(engine, synth)
    rms = _rms(audio)
    cent = _spectral_centroid(audio, SAMPLE_RATE)
    out = wav_dir / f"full_state_{label}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio.T, SAMPLE_RATE)
    print(f"  [{label}]  rms={rms:.5f}  centroid={cent:7.1f} Hz  -> {out.name}")
    return rms, cent


def main() -> None:
    print("=" * 70)
    print("selftest: JUCE base64 round-trip + ground-truth re-encode")
    print("=" * 70)
    selftest_juce_base64()

    print()
    print("=" * 70)
    print("render: default state (baseline)")
    print("=" * 70)
    engine = make_engine()
    synth = load_vital(engine)

    wav_dir = AUDIO_DIR
    tmp_dir = Path(tempfile.mkdtemp(prefix="vital_state_"))

    default_rms, default_cent = render_one(synth, engine, "00_default", wav_dir)

    results: list[tuple[str, float, float]] = [
        ("default", default_rms, default_cent)
    ]

    for i, preset in enumerate(PRESETS, start=1):
        print()
        print(f"--- preset {i}: {preset.name} ---")
        if not preset.exists():
            print(f"  MISSING: {preset}")
            continue
        vital_data = json.loads(preset.read_text(encoding="utf-8"))
        state_path = tmp_dir / f"{preset.stem}.state.bin"
        vital_json_to_state_file(vital_data, state_path)
        print(f"  wrote state file: {state_path.name}  ({state_path.stat().st_size:,} bytes)")

        # Round-trip sanity: parse our own file and confirm JSON re-decodes.
        round = parse_state_file(state_path.read_bytes())
        assert round["magic_ok"], "self-written state file failed magic check"
        assert round["icomponent_json"] is not None, (
            "self-written state file's IComponent did not decode back to JSON"
        )
        rt = round["icomponent_json"]
        assert rt.get("preset_name") == vital_data.get("preset_name"), (
            "preset_name mismatch after round-trip"
        )
        print(
            f"  round-trip JSON ok: preset_name={rt.get('preset_name')!r}, "
            f"author={rt.get('author')!r}"
        )

        # Fresh engine + plugin each time so we know the prior preset isn't lingering.
        engine = make_engine()
        synth = load_vital(engine)
        loaded = synth.load_state(str(state_path))
        print(f"  synth.load_state -> {loaded!r}")
        # dawdreamer's load_state returns None on success; only False/0 should fail us.
        if loaded is False:
            raise RuntimeError(f"load_state explicitly returned False for {preset.name}")

        label = f"{i:02d}_{preset.stem.replace(' ', '_')[:30]}"
        rms, cent = render_one(synth, engine, label, wav_dir)
        results.append((preset.name, rms, cent))

    print()
    print("=" * 70)
    print("summary")
    print("=" * 70)
    for name, rms, cent in results:
        print(f"  {name:<45}  rms={rms:.5f}  centroid={cent:7.1f} Hz")

    # Assertions: every preset render should differ from default and from
    # every other preset by a clear margin in at least one feature.
    preset_results = results[1:]
    print()
    print("verifying audio actually changed...")

    failures: list[str] = []

    # Each preset vs default
    for name, rms, cent in preset_results:
        d_rms = abs(rms - default_rms)
        d_cent = abs(cent - default_cent)
        rel_rms = d_rms / max(default_rms, 1e-6)
        if rel_rms < 0.05 and d_cent < 50.0:
            failures.append(
                f"{name}: too similar to default "
                f"(d_rms_rel={rel_rms:.3%}, d_cent={d_cent:.1f}Hz)"
            )
        else:
            print(
                f"  OK  {name} vs default: d_rms_rel={rel_rms:.1%}, "
                f"d_centroid={d_cent:.0f}Hz"
            )

    # Each preset vs every other preset
    for i in range(len(preset_results)):
        for j in range(i + 1, len(preset_results)):
            a, ar, ac = preset_results[i]
            b, br, bc = preset_results[j]
            d_rms = abs(ar - br)
            d_cent = abs(ac - bc)
            rel_rms = d_rms / max(ar, br, 1e-6)
            if rel_rms < 0.05 and d_cent < 50.0:
                failures.append(
                    f"{a} vs {b}: indistinguishable "
                    f"(d_rms_rel={rel_rms:.3%}, d_cent={d_cent:.1f}Hz)"
                )
            else:
                print(
                    f"  OK  {a} vs {b}: d_rms_rel={rel_rms:.1%}, "
                    f"d_centroid={d_cent:.0f}Hz"
                )

    if failures:
        print()
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)

    print()
    print("all good — full preset state loading works.")


if __name__ == "__main__":
    main()

"""Extract per-preset wavetable spectral fingerprint vectors.

For each of the 2100 base presets:
  - for each of 3 oscillators, find the most-representative keyframe / window,
    decode to 2048 float32 samples
  - compute FFT magnitude, bin to 32 log-spaced frequency bands, log-scale
  - concatenate 3 oscillator spectra -> 96-D fingerprint per preset

Output:
  data/mutations_v1/wt_fingerprints.npy             (n_bases, 96)
  data/mutations_v1/wt_fingerprints_preset_ids.npy  (n_bases,)
  data/mutations_v1/wt_per_osc.npy                  (n_bases, 3, 2048)
    raw waves for the in-panel viewer (next step)
"""

import base64
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from synth_galaxy.config import DATA_DIR

PRESETS_ROOT = Path("/Users/sof/Music/Vital")
MUT_DIR = DATA_DIR / "mutations_v1"

FRAME_SIZE = 2048
N_BANDS = 32
SR = 44100  # nominal — doesn't really matter for our purposes


def decode_base64_to_floats(b64: str) -> np.ndarray:
    """Decode base64 → float32 little-endian samples. NaN/Inf cleaned to 0."""
    if not b64:
        return np.zeros(0, dtype=np.float32)
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    n = len(raw) // 4
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.array(struct.unpack(f"<{n}f", raw[:n * 4]), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip absurd values (audio_file blobs sometimes have garbage bytes)
    np.clip(arr, -10.0, 10.0, out=arr)
    return arr


def _to_fixed(samples: np.ndarray, n: int) -> np.ndarray:
    """Force samples to exactly n via truncation or zero-pad."""
    if len(samples) == n:
        return samples
    if len(samples) > n:
        return samples[:n]
    return np.pad(samples, (0, n - len(samples))).astype(np.float32)


def window_from_component(c: dict, n: int = FRAME_SIZE) -> np.ndarray:
    """Get a representative n-sample window from a component dict."""
    t = (c.get("type") or "").lower()
    keyframes = c.get("keyframes", []) or []

    # Wave Source: keyframes have wave_data
    if "wave" in t and "source" in t:
        for kf in keyframes:
            wd = kf.get("wave_data", "")
            if wd:
                samples = decode_base64_to_floats(wd)
                if samples.size:
                    return _to_fixed(samples, n)

    # Audio File Source: one big audio_file blob
    af = c.get("audio_file", "")
    if af:
        samples = decode_base64_to_floats(af)
        if samples.size:
            # Pick a window roughly mid-file (skip transients at the start)
            start = min(max(0, samples.size // 4), max(0, samples.size - n))
            return _to_fixed(samples[start:start + n], n)

    return np.zeros(n, dtype=np.float32)


def osc_wave(wt_entry: dict, n: int = FRAME_SIZE) -> np.ndarray:
    """Pull a representative window for one oscillator."""
    for g in wt_entry.get("groups", []) or []:
        for c in g.get("components", []) or []:
            w = window_from_component(c, n)
            if w.any():
                return w
    return np.zeros(n, dtype=np.float32)


def spectrum_bands(wave: np.ndarray, n_bands: int = N_BANDS,
                   n: int = FRAME_SIZE) -> np.ndarray:
    """Compute log-magnitude binned spectrum (log-spaced freq bands)."""
    if not wave.any():
        return np.zeros(n_bands, dtype=np.float32)
    # Cast to float64 for the RMS pass so over-the-top samples don't overflow.
    w64 = wave.astype(np.float64)
    w64 = w64 - w64.mean()
    rms = float(np.sqrt((w64 ** 2).mean())) or 1.0
    w64 = w64 / rms
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(w64 * window))  # (n/2+1,)
    # Log-spaced band edges from bin 1 up to N/2
    edges = np.geomspace(1, n // 2, n_bands + 1).astype(int)
    edges = np.clip(edges, 1, n // 2)
    bands = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        bands[i] = spec[a:b].mean()
    return np.log1p(bands * 50.0).astype(np.float32)


def main() -> None:
    print("Loading bases metadata...")
    meta = pd.read_parquet(MUT_DIR / "metadata.parquet")
    bases = meta[meta["is_base"]].copy().reset_index(drop=True)
    print(f"Bases: {len(bases)}")

    n_bases = len(bases)
    fps = np.zeros((n_bases, 3 * N_BANDS), dtype=np.float32)
    waves = np.zeros((n_bases, 3, FRAME_SIZE), dtype=np.float32)
    pids = np.zeros(n_bases, dtype=np.int64)
    n_missing = 0

    for i, (_, r) in enumerate(tqdm(bases.iterrows(), total=n_bases, desc="Fingerprints")):
        pids[i] = int(r["preset_id"])
        path = PRESETS_ROOT / r["preset_path"]
        if not path.exists():
            n_missing += 1
            continue
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            n_missing += 1
            continue
        wts = j.get("settings", {}).get("wavetables", []) or []
        for osc_idx in range(3):
            wt = wts[osc_idx] if osc_idx < len(wts) else {}
            w = osc_wave(wt)
            waves[i, osc_idx] = w
            fps[i, osc_idx * N_BANDS:(osc_idx + 1) * N_BANDS] = spectrum_bands(w)

    np.save(MUT_DIR / "wt_fingerprints.npy", fps)
    np.save(MUT_DIR / "wt_fingerprints_preset_ids.npy", pids)
    np.save(MUT_DIR / "wt_per_osc.npy", waves)
    print(f"\nFingerprints: shape {fps.shape}  dtype {fps.dtype}")
    print(f"Per-osc waves: shape {waves.shape}  size {waves.nbytes/1024/1024:.1f} MB")
    print(f"Missing presets: {n_missing}")
    print(f"Wrote {MUT_DIR/'wt_fingerprints.npy'}")
    print(f"Wrote {MUT_DIR/'wt_per_osc.npy'}")

    # Quick sanity: how many oscillators ended up silent?
    silent = (waves.sum(axis=-1) == 0).sum()
    print(f"Silent oscillators: {silent}/{n_bases*3}  ({100*silent/(n_bases*3):.1f}%)")


if __name__ == "__main__":
    main()

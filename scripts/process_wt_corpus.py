"""Process downloaded wavetable corpus → Vital-compatible 2048-frame audio + spectral fingerprints.

Sources:
  WaveEdit Online (CC0):  ~/Downloads/vital-generator/external_wavetables/wavedit-online/samples
                          705 banks × 64 tables × 256 samples
  AKWF (CC0):             ~/Downloads/vital-generator/external_wavetables/AKWF-FREE/AKWF
                          4358 single-cycle × 600 samples
  AKWF--Surge:            13074 single-cycle × 512 samples
  AKWF--Synthesis-Tech:   4427 banks × 16384 samples (64×256 packed)

Output (in data/wt_corpus/):
  wt_corpus.npy            (N, 2048) float16 — all frames resampled to Vital frame size
  wt_corpus_fingerprints.npy (N, 32) float32 — log-mag spectral bands per frame
  wt_corpus_meta.parquet   metadata: source, bank, index, name, orig_samples
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from synth_galaxy.config import DATA_DIR

FRAME_SIZE = 2048
N_BANDS = 32

EXTERNAL_ROOT = Path("/Users/sof/Downloads/vital-generator/external_wavetables")
WAVEEDIT_DIR = EXTERNAL_ROOT / "wavedit-online/samples"
AKWF_MAIN = EXTERNAL_ROOT / "AKWF-FREE/AKWF"
AKWF_SURGE = EXTERNAL_ROOT / "AKWF-FREE/AKWF--Surge"
AKWF_ST = EXTERNAL_ROOT / "AKWF-FREE/AKWF--Synthesis-Technology"

OUT_DIR = DATA_DIR / "wt_corpus"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_wav_mono(path: Path) -> np.ndarray:
    try:
        data, _sr = sf.read(str(path), always_2d=True)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    return data.mean(axis=1).astype(np.float32)


def resample_to_n(samples: np.ndarray, n: int = FRAME_SIZE) -> np.ndarray:
    L = len(samples)
    if L == n:
        return samples.astype(np.float32)
    if L < 2:
        return np.zeros(n, dtype=np.float32)
    # scipy resample_poly is much faster than scipy.signal.resample for small N
    from math import gcd
    g = gcd(n, L)
    up, down = n // g, L // g
    out = resample_poly(samples, up, down).astype(np.float32)
    if len(out) > n:
        out = out[:n]
    elif len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    # Normalize NaN/Inf
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def spectrum_bands(wave: np.ndarray, n_bands: int = N_BANDS,
                   n: int = FRAME_SIZE) -> np.ndarray:
    if not wave.any():
        return np.zeros(n_bands, dtype=np.float32)
    w64 = wave.astype(np.float64)
    w64 = w64 - w64.mean()
    rms = float(np.sqrt((w64 ** 2).mean())) or 1.0
    w64 = w64 / rms
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(w64 * window))
    edges = np.geomspace(1, n // 2, n_bands + 1).astype(int)
    edges = np.clip(edges, 1, n // 2)
    bands = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        bands[i] = spec[a:b].mean()
    return np.log1p(bands * 50.0).astype(np.float32)


def main() -> None:
    frames: list[np.ndarray] = []
    meta: list[dict] = []

    # ---- WaveEdit (banks of 64 × 256) ----
    we_files = sorted(WAVEEDIT_DIR.glob("*.WAV"))
    print(f"WaveEdit banks: {len(we_files)}")
    for path in tqdm(we_files, desc="WaveEdit"):
        samples = read_wav_mono(path)
        if len(samples) < 16384:
            continue
        for i in range(64):
            chunk = samples[i * 256:(i + 1) * 256]
            if len(chunk) != 256:
                continue
            frames.append(resample_to_n(chunk))
            meta.append({
                "source": "waveedit",
                "bank": path.stem,
                "index": i,
                "name": f"{path.stem}_{i:02d}",
                "orig_samples": 256,
            })

    # ---- AKWF main (4358 × 600 single-cycle) ----
    akwf_files = sorted(AKWF_MAIN.rglob("*.wav"))
    print(f"\nAKWF main: {len(akwf_files)}")
    for path in tqdm(akwf_files, desc="AKWF main"):
        samples = read_wav_mono(path)
        if len(samples) == 0:
            continue
        frames.append(resample_to_n(samples))
        meta.append({
            "source": "akwf",
            "bank": path.parent.name,
            "index": 0,
            "name": path.stem,
            "orig_samples": len(samples),
        })

    # ---- AKWF--Surge (13074 × 512 single-cycle) ----
    surge_files = sorted(AKWF_SURGE.rglob("*.wav"))
    print(f"\nAKWF--Surge: {len(surge_files)}")
    for path in tqdm(surge_files, desc="AKWF Surge"):
        samples = read_wav_mono(path)
        if len(samples) == 0:
            continue
        frames.append(resample_to_n(samples))
        meta.append({
            "source": "akwf-surge",
            "bank": path.parent.name,
            "index": 0,
            "name": path.stem,
            "orig_samples": len(samples),
        })

    # ---- AKWF--Synthesis-Technology (4427 × 16384 banks) ----
    st_files = sorted(AKWF_ST.rglob("*.wav"))
    print(f"\nAKWF--Synthesis-Technology banks: {len(st_files)}")
    for path in tqdm(st_files, desc="AKWF ST"):
        samples = read_wav_mono(path)
        if len(samples) < 16384:
            continue
        # 64 tables × 256 samples per file
        for i in range(64):
            chunk = samples[i * 256:(i + 1) * 256]
            if len(chunk) != 256:
                continue
            frames.append(resample_to_n(chunk))
            meta.append({
                "source": "akwf-st",
                "bank": path.stem,
                "index": i,
                "name": f"{path.stem}_{i:02d}",
                "orig_samples": 256,
            })

    print(f"\n=== Total frames: {len(frames)} ===")

    arr = np.stack(frames).astype(np.float16)  # ~half the storage of float32
    print(f"Frames array: shape={arr.shape}  size={arr.nbytes / 1024 / 1024:.1f} MB")

    # Fingerprints — compute on float32 promotion
    print(f"\nComputing {N_BANDS}-band fingerprints for {len(arr)} frames...")
    fps = np.zeros((len(arr), N_BANDS), dtype=np.float32)
    for i in tqdm(range(len(arr)), desc="Fingerprints"):
        fps[i] = spectrum_bands(arr[i].astype(np.float32))

    np.save(OUT_DIR / "wt_corpus.npy", arr)
    np.save(OUT_DIR / "wt_corpus_fingerprints.npy", fps)
    pd.DataFrame(meta).to_parquet(OUT_DIR / "wt_corpus_meta.parquet", index=False)

    print(f"\nSaved to {OUT_DIR}")
    print(f"  wt_corpus.npy              {arr.nbytes/1024/1024:.1f} MB")
    print(f"  wt_corpus_fingerprints.npy {fps.nbytes/1024/1024:.1f} MB")

    # Per-source breakdown
    df = pd.DataFrame(meta)
    print("\nBy source:")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()

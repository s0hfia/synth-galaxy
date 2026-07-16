"""Audio feature extraction: per-patch fixed-size descriptor for UMAP.

13 MFCCs (mean + std over time) + 9 spectral/stereo/loudness scalars = 35-dim per patch.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def extract_features(wav_path: Path, sr: int = 44100) -> dict:
    audio, file_sr = sf.read(wav_path, always_2d=True)  # (samples, channels)
    audio = audio.T  # → (channels, samples)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr, axis=-1)

    mono = audio.mean(axis=0).astype(np.float32)

    # MFCCs — 13 coeffs, summary stats over time
    mfccs = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)
    mfcc_std = mfccs.std(axis=1)

    # Spectral stats (time-averaged)
    spec_centroid = float(librosa.feature.spectral_centroid(y=mono, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=mono, sr=sr).mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=mono, sr=sr).mean())
    spec_flatness = float(librosa.feature.spectral_flatness(y=mono).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=mono).mean())

    # Stereo width — 1 - corr(L,R). Wider stereo image → lower correlation
    if audio.shape[0] >= 2 and audio[0].std() > 0 and audio[1].std() > 0:
        stereo_corr = float(np.corrcoef(audio[0], audio[1])[0, 1])
        stereo_width = max(0.0, 1.0 - max(0.0, stereo_corr))
    else:
        stereo_width = 0.0

    # Harmonic vs percussive energy ratio
    h, p = librosa.effects.hpss(mono)
    h_e = float((h ** 2).sum())
    p_e = float((p ** 2).sum())
    harmonic_ratio = h_e / max(1e-12, h_e + p_e)

    rms = float(np.sqrt((mono ** 2).mean()))
    peak = float(np.abs(mono).max())

    feats = {
        "rms": rms,
        "peak": peak,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff,
        "spec_flatness": spec_flatness,
        "zcr": zcr,
        "stereo_width": stereo_width,
        "harmonic_ratio": harmonic_ratio,
    }
    for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std)):
        feats[f"mfcc_mean_{i:02d}"] = float(m)
        feats[f"mfcc_std_{i:02d}"] = float(s)
    return feats


FEATURE_COLUMNS = [
    "rms", "peak", "spec_centroid", "spec_bandwidth", "spec_rolloff",
    "spec_flatness", "zcr", "stereo_width", "harmonic_ratio",
] + [f"mfcc_mean_{i:02d}" for i in range(13)] + [f"mfcc_std_{i:02d}" for i in range(13)]

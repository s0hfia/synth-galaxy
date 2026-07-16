"""Render mutation-variant patches from real Vital presets.

Per real preset: load full state (wavetables + LFO shapes + mod routings),
render the base, then render N variants by perturbing flat synthesis params.
Save mel-spectrograms + summary features + param vectors only (no WAVs for
variants — saves ~17GB disk for 50k variants). Base WAVs are kept.

Features:
- per-preset timeout via signal.alarm (so a hung load_state can't stall the run)
- checkpoint parquet every CHECKPOINT_EVERY presets (incremental save)
"""

import argparse
import json
import re
import signal
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from synth_galaxy.config import (
    AUDIO_DIR, DATA_DIR, FEATURES_DIR, PATCHES_DIR, SAMPLE_RATE,
)
from synth_galaxy.mutation_sampler import variant_overrides
from synth_galaxy.preset_loader_full import vital_json_to_state_file
from synth_galaxy.render import load_vital, make_engine, render_note
from synth_galaxy.sampler import capture_state, synthesis_param_indices


class PresetTimeout(Exception):
    """Raised when a single preset's load+render block exceeds the timeout."""


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise PresetTimeout()


signal.signal(signal.SIGALRM, _timeout_handler)


# Mel-spectrogram settings tuned for compact storage at decent fidelity.
N_MELS = 64
HOP = 1024
MELSPEC_DTYPE = np.float16  # halves storage vs float32; ample dynamic range


def audio_to_features(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Return (mel-spectrogram float16, summary features dict). audio is (ch, samp)."""
    mono = audio.mean(axis=0).astype(np.float32)
    melspec = librosa.feature.melspectrogram(
        y=mono, sr=sr, n_mels=N_MELS, hop_length=HOP, power=2.0
    )
    # Log-mel, then per-clip min-max to [0,1] for compact float16 storage.
    log_mel = librosa.power_to_db(melspec, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / max(1e-6, (log_mel.max() - log_mel.min()))
    log_mel = log_mel.astype(MELSPEC_DTYPE)

    spec_centroid = float(librosa.feature.spectral_centroid(y=mono, sr=sr).mean())
    spec_bandwidth = float(librosa.feature.spectral_bandwidth(y=mono, sr=sr).mean())
    spec_rolloff = float(librosa.feature.spectral_rolloff(y=mono, sr=sr).mean())
    spec_flatness = float(librosa.feature.spectral_flatness(y=mono).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y=mono).mean())
    rms = float(np.sqrt((mono ** 2).mean()))
    peak = float(np.abs(mono).max())

    if audio.shape[0] >= 2 and audio[0].std() > 0 and audio[1].std() > 0:
        stereo_corr = float(np.corrcoef(audio[0], audio[1])[0, 1])
        stereo_width = max(0.0, 1.0 - max(0.0, stereo_corr))
    else:
        stereo_width = 0.0

    return log_mel, {
        "rms": rms,
        "peak": peak,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff,
        "spec_flatness": spec_flatness,
        "zcr": zcr,
        "stereo_width": stereo_width,
    }


def safe_filename(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return cleaned[:60] or "unnamed"


def find_presets(root: Path) -> list[Path]:
    return sorted(root.rglob("*.vital"))


def main(
    preset_root: Path,
    variants_per_preset: int,
    limit: int | None,
    preset_timeout: int,
    checkpoint_every: int,
    seed: int,
    dataset_name: str,
) -> None:
    presets = find_presets(preset_root)
    if limit:
        presets = presets[:limit]
    print(f"Real presets: {len(presets)};  variants per preset: {variants_per_preset}")
    print(f"Target dataset size: {len(presets) * (1 + variants_per_preset):,}")

    engine = make_engine()
    synth = load_vital(engine)
    candidates = synthesis_param_indices(synth)
    n_total_params = synth.get_plugin_parameter_size()
    print(f"Vital host params: {n_total_params};  synthesis candidates: {len(candidates)}")

    out_dir = DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_wav_dir = out_dir / "base_wavs"
    base_wav_dir.mkdir(exist_ok=True)
    melspecs_path = out_dir / "melspecs.npy"
    params_path = out_dir / "params.npy"
    rows_path = out_dir / "metadata.parquet"
    candidates_path = out_dir / "synthesis_param_indices.npy"
    np.save(candidates_path, np.array(candidates, dtype=np.int32))

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    melspecs: list[np.ndarray] = []
    params: list[np.ndarray] = []  # each entry: params over the candidates subset
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    def checkpoint() -> None:
        pd.DataFrame(rows).to_parquet(rows_path, index=False)
        if melspecs:
            np.save(melspecs_path, np.stack(melspecs))
        if params:
            np.save(params_path, np.stack(params))

    with tempfile.TemporaryDirectory(prefix="synth-galaxy-mut-") as tmp:
        tmp_dir = Path(tmp)
        for i, preset_path in enumerate(tqdm(presets, desc="Presets")):
            try:
                signal.alarm(preset_timeout)

                vital_json = json.loads(preset_path.read_text(encoding="utf-8"))
                state_path = tmp_dir / f"state_{i:06d}.bin"
                vital_json_to_state_file(vital_json, state_path)
                synth.load_state(str(state_path))

                base_state = capture_state(synth)
                base_param_vec = np.array(
                    [base_state[idx] for idx in candidates], dtype=np.float32
                )

                pack = (
                    preset_path.parent.relative_to(preset_root).parts[0]
                    if preset_path.parent != preset_root else "root"
                )
                preset_name = vital_json.get("preset_name") or preset_path.stem
                author = vital_json.get("author") or ""
                style = vital_json.get("preset_style") or ""
                base_safe = safe_filename(preset_path.stem)

                # base render
                audio = render_note(engine, synth)
                melspec, feats = audio_to_features(audio, SAMPLE_RATE)
                base_wav = base_wav_dir / f"{i:06d}_{base_safe}.wav"
                sf.write(base_wav, audio.T, SAMPLE_RATE)

                rows.append({
                    "patch_id": len(rows),
                    "preset_id": i,
                    "variant_id": 0,
                    "is_base": True,
                    "preset_path": str(preset_path.relative_to(preset_root)),
                    "preset_name": preset_name,
                    "author": author,
                    "preset_style": style,
                    "pack": pack,
                    "wav_path": str(base_wav.relative_to(DATA_DIR)),
                    **feats,
                })
                melspecs.append(melspec)
                params.append(base_param_vec)

                # variants
                for v in range(1, variants_per_preset + 1):
                    overrides = variant_overrides(rng, base_state, candidates)
                    for idx, val in overrides.items():
                        synth.set_parameter(idx, val)
                    audio = render_note(engine, synth)
                    melspec, feats = audio_to_features(audio, SAMPLE_RATE)
                    variant_param_vec = base_param_vec.copy()
                    for j, idx in enumerate(candidates):
                        if idx in overrides:
                            variant_param_vec[j] = overrides[idx]
                    rows.append({
                        "patch_id": len(rows),
                        "preset_id": i,
                        "variant_id": v,
                        "is_base": False,
                        "preset_path": str(preset_path.relative_to(preset_root)),
                        "preset_name": preset_name,
                        "author": author,
                        "preset_style": style,
                        "pack": pack,
                        "wav_path": "",  # variants don't keep WAVs
                        **feats,
                    })
                    melspecs.append(melspec)
                    params.append(variant_param_vec)

                    # reset to base for next variant
                    for idx, val in overrides.items():
                        synth.set_parameter(idx, base_state[idx])

                signal.alarm(0)

            except PresetTimeout:
                failures.append((str(preset_path), "TIMEOUT"))
                signal.alarm(0)
            except Exception as e:
                failures.append((str(preset_path), f"{type(e).__name__}: {e}"))
                signal.alarm(0)

            if (i + 1) % checkpoint_every == 0:
                checkpoint()

    checkpoint()

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    print(f"\nRendered {len(df)} patches from {len(presets) - len(failures)} of "
          f"{len(presets)} presets in {elapsed:.1f}s  "
          f"({elapsed / max(1, len(presets)):.2f}s/preset)")
    print(f"Failures: {len(failures)}")
    for path, err in failures[:8]:
        print(f"  {path}: {err}")
    print(f"\nMetadata:   {rows_path}")
    print(f"Mel-specs:  {melspecs_path}  shape={np.load(melspecs_path).shape if melspecs_path.exists() else 'none'}")
    print(f"Params:     {params_path}    shape={np.load(params_path).shape if params_path.exists() else 'none'}")
    silent = int((df["rms"] < 0.001).sum()) if len(df) else 0
    clipping = int((df["peak"] >= 0.99).sum()) if len(df) else 0
    print(f"\nMean RMS: {df['rms'].mean():.4f}  silent: {silent}  clipping: {clipping}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Users/sof/Music/Vital"))
    ap.add_argument("--variants", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preset-timeout", type=int, default=15)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", type=str, default="mutations")
    args = ap.parse_args()
    main(
        args.root, args.variants, args.limit, args.preset_timeout,
        args.checkpoint_every, args.seed, args.name,
    )

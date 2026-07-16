"""Tiny Flask server for the synth-galaxy front-end.

Endpoints:
  GET  /                          galaxy_v2.html (preset galaxy)
  GET  /atlas                     atlas.html (wavetable atlas)
  GET  /data/<path>               static files
  POST /search                    CLAP text query -> {preset_id: score}
  GET  /wt/<int:idx>.wav          single corpus wavetable looped at C4, ~1s

Usage:
  cd ~/projects/synth-galaxy && uv run python scripts/search_server.py
  open http://localhost:8000/         # preset galaxy
  open http://localhost:8000/atlas    # wavetable atlas
"""

# ----- monkey-patches for laion-clap on torch>=2.6 -----
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):  # noqa: ANN001
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load  # type: ignore[assignment]

import laion_clap  # noqa: E402
_orig_load_ckpt = laion_clap.CLAP_Module.load_ckpt
def _patched_load_ckpt(self, *args, **kwargs):  # noqa: ANN001
    inner = self.model.load_state_dict
    def lsd(state_dict, strict=True, **kw):
        return inner(state_dict, strict=False, **kw)
    self.model.load_state_dict = lsd  # type: ignore[method-assign]
    return _orig_load_ckpt(self, *args, **kwargs)
laion_clap.CLAP_Module.load_ckpt = _patched_load_ckpt  # type: ignore[method-assign]
# ----- end patches -----

import argparse  # noqa: E402
import io  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from synth_galaxy.config import DATA_DIR  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUT_DIR = DATA_DIR / "mutations_v1"
WT_DIR = DATA_DIR / "wt_corpus"

print("Loading CLAP model (one-time, ~3s)...")
_model = laion_clap.CLAP_Module(enable_fusion=False)
_model.load_ckpt()
print("  CLAP model ready.")

print("Loading audio embeddings + preset-id ordering...")
_audio_embs = np.load(MUT_DIR / "clap_audio_embeddings.npy")  # (2147, 512), L2-normalized
_scores_df = pd.read_parquet(MUT_DIR / "clap_scores.parquet")
_preset_ids_sorted = _scores_df["preset_id"].to_numpy()
print(f"  {len(_preset_ids_sorted)} embeddings ready.")

print("Loading wavetable corpus (mmap)...")
_wt_corpus = np.load(WT_DIR / "wt_corpus.npy", mmap_mode="r")  # (66904, 2048) float16
_wt_fps = np.load(WT_DIR / "wt_corpus_fingerprints.npy")        # (66904, 32) float32
_wt_meta = pd.read_parquet(WT_DIR / "wt_corpus_meta.parquet")
# L2-normalize fingerprints once for cosine-search
_wt_fps_norm = _wt_fps / (np.linalg.norm(_wt_fps, axis=1, keepdims=True) + 1e-9)
print(f"  corpus shape={_wt_corpus.shape}  dtype={_wt_corpus.dtype}")
print(f"  fingerprints={_wt_fps.shape}  meta={len(_wt_meta)}")

# Preset metadata for the /upload neighbor lookup (preset_id -> name/pack)
_preset_meta = pd.read_parquet(MUT_DIR / "metadata.parquet")
_preset_meta = _preset_meta[_preset_meta["is_base"]].set_index("preset_id")
# CLAP basis coords for placing the dropped audio in the galaxy
_clap_coords = pd.read_parquet(MUT_DIR / "galaxy_coords_clap.parquet").set_index("preset_id")


# ---- audio render helpers ----
SR = 44100
C4_HZ = 261.6256
SAMPLES_PER_CYCLE_C4 = int(round(SR / C4_HZ))   # ~169
PLAYBACK_SECONDS = 1.2
PLAYBACK_LEN = int(SR * PLAYBACK_SECONDS)


def _wavetable_as_c4_wav(frame: np.ndarray) -> bytes:
    """Resample one 2048-sample wavetable cycle down to C4-pitch and tile to ~1.2s.
    Apply a tiny fade in/out to avoid clicks. Return as WAV bytes (PCM 16-bit)."""
    frame = frame.astype(np.float32, copy=True)
    # Resample the cycle to the C4 cycle length.
    from math import gcd
    g = gcd(SAMPLES_PER_CYCLE_C4, len(frame))
    cycle = resample_poly(frame, SAMPLES_PER_CYCLE_C4 // g, len(frame) // g)
    cycle = cycle.astype(np.float32)
    if len(cycle) != SAMPLES_PER_CYCLE_C4:
        cycle = cycle[:SAMPLES_PER_CYCLE_C4] if len(cycle) > SAMPLES_PER_CYCLE_C4 \
            else np.pad(cycle, (0, SAMPLES_PER_CYCLE_C4 - len(cycle)))

    # Normalize amplitude so all wavetables play at a comparable level.
    peak = float(np.abs(cycle).max())
    if peak > 1e-6:
        cycle = cycle * (0.6 / peak)  # peak ~0.6, leaves headroom

    n_repeats = PLAYBACK_LEN // SAMPLES_PER_CYCLE_C4 + 1
    audio = np.tile(cycle, n_repeats)[:PLAYBACK_LEN]

    # Fade in/out 5ms
    fade = SR // 200
    audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB cap on uploaded audio


@app.route("/")
def root():
    return send_from_directory(DATA_DIR, "galaxy_v2.html")


@app.route("/atlas")
def atlas_root():
    return send_from_directory(DATA_DIR, "atlas.html")


@app.route("/data/<path:p>")
def data_files(p):
    return send_from_directory(DATA_DIR, p)


@app.route("/wt/<int:idx>.wav")
def wavetable_audio(idx: int):
    if idx < 0 or idx >= len(_wt_corpus):
        return Response(f"bad index {idx}", status=404)
    frame = np.asarray(_wt_corpus[idx], dtype=np.float32)
    wav_bytes = _wavetable_as_c4_wav(frame)
    return Response(wav_bytes, mimetype="audio/wav",
                    headers={"Cache-Control": "public, max-age=86400"})


# ---------- /upload + /save: drop audio -> CLAP neighbors + corpus matches + .vital export ----------
import base64  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import uuid  # noqa: E402

import librosa  # noqa: E402

N_BANDS = 32
FRAME_SIZE = 2048
N_WT_FRAMES = 16          # number of wavetable keyframes to extract per audio drop

UPLOAD_CACHE_DIR = DATA_DIR / "uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

VITAL_USER_PRESETS = Path.home() / "Music" / "Vital" / "User" / "Presets"
PRESETS_ROOT = Path("/Users/sof/Music/Vital")


def _bands_from_frame(frame: np.ndarray) -> np.ndarray:
    """32-band log-mag spectrum on a 2048-sample frame."""
    w64 = frame.astype(np.float64)
    w64 = w64 - w64.mean()
    rms = float(np.sqrt((w64 ** 2).mean())) or 1.0
    w64 = w64 / rms
    window = np.hanning(FRAME_SIZE)
    spec = np.abs(np.fft.rfft(w64 * window))
    edges = np.geomspace(1, FRAME_SIZE // 2, N_BANDS + 1).astype(int)
    edges = np.clip(edges, 1, FRAME_SIZE // 2)
    bands = np.zeros(N_BANDS, dtype=np.float32)
    for i in range(N_BANDS):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        bands[i] = spec[a:b].mean()
    return np.log1p(bands * 50.0).astype(np.float32)


def _audio_to_wavetable_frames(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract 16 sequential 2048-sample wavetable frames + representative fingerprint.

    Returns:
      frames: (16, 2048) float32 — Vital-format keyframe data
      fp:     (32,) float32     — fingerprint for corpus matching (loudest frame's spectrum)
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 44100:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(audio, -10.0, 10.0, out=audio)

    # Find active region: RMS envelope, gate at 10% of peak.
    if len(audio) < FRAME_SIZE * 2:
        # Tiny clip — just pad and repeat.
        if len(audio) < FRAME_SIZE:
            audio = np.pad(audio, (0, FRAME_SIZE - len(audio)))
        active_start, active_end = 0, len(audio)
    else:
        hop = FRAME_SIZE // 4
        n_chunks = (len(audio) - FRAME_SIZE) // hop + 1
        rms_env = np.array([
            np.sqrt((audio[i*hop:i*hop + FRAME_SIZE] ** 2).mean())
            for i in range(n_chunks)
        ], dtype=np.float32)
        peak = float(rms_env.max() or 1.0)
        active_mask = rms_env > peak * 0.1
        if active_mask.any():
            active_start = int(np.argmax(active_mask)) * hop
            active_end = (len(active_mask) - int(np.argmax(active_mask[::-1])) - 1) * hop + FRAME_SIZE
        else:
            active_start, active_end = 0, len(audio)

    # Extract 16 evenly-spaced 2048-sample windows from the active region.
    span = max(FRAME_SIZE, active_end - active_start)
    if span < FRAME_SIZE * N_WT_FRAMES:
        # Audio too short for non-overlapping windows — overlap is fine.
        positions = np.linspace(active_start, max(active_start, len(audio) - FRAME_SIZE), N_WT_FRAMES).astype(int)
    else:
        positions = np.linspace(active_start, active_end - FRAME_SIZE, N_WT_FRAMES).astype(int)

    frames = np.zeros((N_WT_FRAMES, FRAME_SIZE), dtype=np.float32)
    rms_per_frame = np.zeros(N_WT_FRAMES, dtype=np.float32)
    for i, p in enumerate(positions):
        chunk = audio[p:p + FRAME_SIZE]
        if len(chunk) < FRAME_SIZE:
            chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
        # DC-remove + soft normalize to ±1 (helps Vital play it cleanly)
        chunk = chunk - chunk.mean()
        peak = float(np.abs(chunk).max())
        if peak > 1e-6:
            chunk = chunk * (0.9 / peak)
        frames[i] = chunk
        rms_per_frame[i] = float(np.sqrt((chunk ** 2).mean()))

    # Fingerprint from the loudest frame
    loudest = int(np.argmax(rms_per_frame))
    fp = _bands_from_frame(frames[loudest])
    return frames, fp


def _encode_frame_b64(frame: np.ndarray) -> str:
    """Encode a 2048-sample float32 frame to Vital's base64 wave_data format."""
    arr = np.asarray(frame, dtype="<f4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]+", "_", name).strip().strip("._")
    return cleaned[:80] or "extracted_preset"


def _inject_extracted_wavetable(base_preset: dict, frames: np.ndarray,
                                source_name: str) -> dict:
    """Mutate base_preset's wavetables[0] to use our 16 extracted frames as keyframes."""
    preset = json.loads(json.dumps(base_preset))  # deep copy
    settings = preset.get("settings", {})

    # Build the Wave Source component with our 16 keyframes (positions 0..255).
    positions = np.linspace(0, 255, len(frames)).astype(int).tolist()
    keyframes = []
    for pos, frame in zip(positions, frames):
        keyframes.append({
            "wave_data": _encode_frame_b64(frame),
            "position": int(pos),
            "start_position": 0.0,
            "window_size": 0.5,
            "window_fade": 0.5,
        })

    new_component = {
        "type": "Wave Source",
        "keyframes": keyframes,
        "interpolation_style": 1,
        "phase_style": 0,
        "fade_style": 0,
        "normalize_gain": False,
        "normalize_mult": 1.0,
        "random_seed": 0,
        "window_size": 2048,
        "audio_file": "",
        "audio_sample_rate": 44100,
    }
    new_wt = {
        "author": "synth-galaxy / extracted",
        "name": f"extracted: {source_name}"[:60],
        "full_normalize": True,
        "remove_all_dc": True,
        "version": "1.5.5",
        "groups": [{"components": [new_component]}],
    }

    wts = settings.get("wavetables") or [{}, {}, {}]
    while len(wts) < 3:
        wts.append({})
    wts[0] = new_wt
    settings["wavetables"] = wts

    # Bias the audible-osc params so osc 1 is on and sweeping through the wavetable.
    settings["osc_1_on"] = 1.0
    settings["osc_1_level"] = max(float(settings.get("osc_1_level", 0.7)), 0.7)
    # Place wave_frame at the middle so the user hears the central timbre by default.
    settings["osc_1_wave_frame"] = 128.0

    preset["preset_name"] = source_name[:60]
    preset["settings"] = settings
    return preset


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" in request.files:
        f = request.files["file"]
        data = f.read()
        filename = f.filename or "upload.wav"
    else:
        data = request.get_data()
        filename = request.headers.get("X-Filename", "upload.bin")

    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in (".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"):
        suffix = ".wav"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    tmp_path = tmp.name

    try:
        # ---- decode + CLAP ----
        import soundfile as sf
        try:
            audio, sr = sf.read(tmp_path, always_2d=False)
        except Exception:
            audio, sr = librosa.load(tmp_path, sr=44100, mono=False)
            if audio.ndim == 2:
                audio = audio.T

        embs = _model.get_audio_embedding_from_filelist([tmp_path])
        if hasattr(embs, "detach"):
            embs = embs.detach().cpu().numpy()
        emb = np.asarray(embs[0], dtype=np.float32)
        emb /= max(1e-9, float(np.linalg.norm(emb)))

        # ---- CLAP neighbors (presets) ----
        sims = (_audio_embs @ emb).astype(np.float32)
        top = np.argsort(sims)[::-1][:8]
        neighbors = []
        for i in top:
            pid = int(_preset_ids_sorted[i])
            meta = _preset_meta.loc[pid] if pid in _preset_meta.index else None
            name = str(meta["preset_name"]) if meta is not None else ""
            pack = str(meta["pack"]) if meta is not None else ""
            wav = f"data/{meta['wav_path']}" if meta is not None and "wav_path" in meta else ""
            neighbors.append({
                "preset_id": pid,
                "name": name,
                "pack": pack,
                "sim": float(sims[i]),
                "wav": wav,
            })

        # ---- 3D position in CLAP basis: nearest preset's xyz ----
        marker_xyz = None
        for n in neighbors:
            if n["preset_id"] in _clap_coords.index:
                row = _clap_coords.loc[n["preset_id"]]
                marker_xyz = [float(row["x"]), float(row["y"]), float(row["z"])]
                break

        # ---- multi-window extraction + corpus match ----
        frames, fp = _audio_to_wavetable_frames(audio, sr)
        fp_n = fp / max(1e-9, float(np.linalg.norm(fp)))
        cs = (_wt_fps_norm @ fp_n).astype(np.float32)
        top_c = np.argsort(cs)[::-1][:8]
        corpus_matches = []
        for i in top_c:
            corpus_matches.append({
                "idx": int(i),
                "source": str(_wt_meta.iloc[i]["source"]),
                "bank": str(_wt_meta.iloc[i]["bank"]),
                "name": str(_wt_meta.iloc[i]["name"]),
                "sim": float(cs[i]),
            })

        # ---- cache 16 frames keyed by upload UUID so /save can build a .vital later ----
        upload_id = uuid.uuid4().hex[:12]
        cache_dir = UPLOAD_CACHE_DIR / upload_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "frames.npy", frames)
        (cache_dir / "meta.json").write_text(json.dumps({
            "filename": filename,
            "top_neighbor_preset_id": neighbors[0]["preset_id"] if neighbors else None,
        }))

        return jsonify({
            "upload_id": upload_id,
            "filename": filename,
            "neighbors": neighbors,
            "corpus_matches": corpus_matches,
            "marker_xyz_clap": marker_xyz,
            "n_frames": int(len(frames)),
        })
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _find_preset_path_by_id(preset_id: int) -> Path | None:
    if preset_id not in _preset_meta.index:
        return None
    rel = _preset_meta.loc[preset_id, "preset_path"]
    return PRESETS_ROOT / rel


@app.route("/save", methods=["POST"])
def save_extracted_preset():
    """Build a .vital file from a cached upload + chosen base preset, write to disk."""
    body = request.get_json(force=True, silent=True) or {}
    upload_id = body.get("upload_id", "").strip()
    base_pid = body.get("base_preset_id")
    preset_name = body.get("name", "").strip()
    out_dir = body.get("out_dir")  # optional override

    if not upload_id:
        return jsonify({"error": "missing upload_id"}), 400
    cache_dir = UPLOAD_CACHE_DIR / upload_id
    frames_path = cache_dir / "frames.npy"
    if not frames_path.exists():
        return jsonify({"error": f"unknown upload_id {upload_id}"}), 404
    frames = np.load(frames_path)

    if base_pid is None:
        # default to the top neighbor cached in meta
        meta_blob = json.loads((cache_dir / "meta.json").read_text())
        base_pid = meta_blob.get("top_neighbor_preset_id")
    if base_pid is None:
        return jsonify({"error": "no base_preset_id and no cached top neighbor"}), 400

    base_path = _find_preset_path_by_id(int(base_pid))
    if base_path is None or not base_path.exists():
        return jsonify({"error": f"base preset {base_pid} not found"}), 404
    base_preset = json.loads(base_path.read_text(encoding="utf-8"))

    src_name = preset_name or json.loads((cache_dir / "meta.json").read_text()).get("filename") or "extracted"
    src_name = Path(src_name).stem if "." in src_name else src_name
    new_preset = _inject_extracted_wavetable(base_preset, frames, src_name)

    safe_name = _safe_filename(preset_name or src_name)
    out_dir_path = Path(out_dir).expanduser() if out_dir else VITAL_USER_PRESETS
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"{safe_name}.vital"
    # if exists, suffix with -N
    n = 1
    while out_path.exists():
        out_path = out_dir_path / f"{safe_name}-{n}.vital"
        n += 1
    out_path.write_text(json.dumps(new_preset), encoding="utf-8")

    return jsonify({
        "path": str(out_path),
        "preset_name": new_preset.get("preset_name"),
        "base_preset_id": int(base_pid),
    })


@app.route("/search", methods=["POST"])
def search():
    body = request.get_json(force=True, silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"query": "", "scores": {}})

    text_embs = _model.get_text_embedding([query])
    if hasattr(text_embs, "detach"):
        text_embs = text_embs.detach().cpu().numpy()
    t = np.asarray(text_embs[0], dtype=np.float32)
    t /= max(1e-9, float(np.linalg.norm(t)))

    sims = (_audio_embs @ t).astype(np.float32)  # (n_bases,)
    out = {int(pid): float(s) for pid, s in zip(_preset_ids_sorted, sims)}
    return jsonify({"query": query, "scores": out,
                    "min": float(sims.min()), "max": float(sims.max())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"\nServing at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

"""Encode base preset WAVs + descriptor vocabulary with LAION-CLAP.

Outputs (in data/mutations_v1/):
  clap_audio_embeddings.npy  -- (n_bases, 512) normalized
  clap_text_embeddings.npy   -- (n_descriptors, 512) normalized
  clap_scores.parquet        -- (n_bases x n_descriptors) cosine similarity
                                plus preset metadata
"""

import argparse
import re
import time
from pathlib import Path

# PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the
# laion-clap checkpoint (contains numpy.core.multiarray.scalar). We trust the
# checkpoint, so patch torch.load before laion_clap imports it.
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):  # noqa: ANN001
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load  # type: ignore[assignment]

import laion_clap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

# CLAP checkpoint was saved with an older transformers version that included
# `text_branch.embeddings.position_ids` in the state_dict. Newer transformers
# doesn't store it -> load_state_dict(strict=True) raises. Patch CLAP_Module.load_ckpt
# to use strict=False on the inner model so the extra key is ignored.
_orig_load_ckpt = laion_clap.CLAP_Module.load_ckpt
def _patched_load_ckpt(self, *args, **kwargs):  # noqa: ANN001
    inner_load_state_dict = self.model.load_state_dict
    def lsd(state_dict, strict=True, **kw):
        return inner_load_state_dict(state_dict, strict=False, **kw)
    self.model.load_state_dict = lsd  # type: ignore[method-assign]
    return _orig_load_ckpt(self, *args, **kwargs)
laion_clap.CLAP_Module.load_ckpt = _patched_load_ckpt  # type: ignore[method-assign]

from synth_galaxy.clap_descriptors import all_descriptors, labels, queries
from synth_galaxy.config import DATA_DIR


def main(
    base_wav_dir: Path,
    metadata_path: Path,
    out_dir: Path,
    batch_size: int = 32,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LAION-CLAP (downloads ~600MB on first run)...")
    t0 = time.time()
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ---- text embeddings -------------------------------------------------
    descs = all_descriptors()
    print(f"\nEncoding {len(descs)} descriptors...")
    text_qs = queries()
    text_embs = model.get_text_embedding(text_qs)
    if hasattr(text_embs, "detach"):
        text_embs = text_embs.detach().cpu().numpy()
    text_embs = np.asarray(text_embs, dtype=np.float32)
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True)
    print(f"  text embeddings shape: {text_embs.shape}")
    np.save(out_dir / "clap_text_embeddings.npy", text_embs)

    # ---- audio embeddings ------------------------------------------------
    wavs = sorted(base_wav_dir.glob("*.wav"))
    print(f"\nEncoding {len(wavs)} base WAVs in batches of {batch_size}...")
    audio_embs_list: list[np.ndarray] = []
    t0 = time.time()
    for i in tqdm(range(0, len(wavs), batch_size), desc="CLAP audio"):
        batch = wavs[i : i + batch_size]
        batch_paths = [str(p) for p in batch]
        embs = model.get_audio_embedding_from_filelist(batch_paths)
        if hasattr(embs, "detach"):
            embs = embs.detach().cpu().numpy()
        audio_embs_list.append(np.asarray(embs, dtype=np.float32))
    audio_embs = np.concatenate(audio_embs_list, axis=0)
    audio_embs /= np.linalg.norm(audio_embs, axis=1, keepdims=True)
    print(f"  audio embeddings shape: {audio_embs.shape}  ({time.time() - t0:.1f}s)")
    np.save(out_dir / "clap_audio_embeddings.npy", audio_embs)

    # ---- scores matrix ---------------------------------------------------
    print("\nComputing audio x text cosine similarity matrix...")
    scores = audio_embs @ text_embs.T
    print(f"  scores shape: {scores.shape}")
    print(f"  score range: [{scores.min():.3f}, {scores.max():.3f}], "
          f"mean: {scores.mean():.3f}")

    # ---- join scores to metadata by preset_id (parse from filename prefix)
    pattern = re.compile(r"^(\d{6})_")
    preset_ids = []
    for p in wavs:
        m = pattern.match(p.name)
        preset_ids.append(int(m.group(1)) if m else -1)

    meta_df = pd.read_parquet(metadata_path)
    if "is_base" in meta_df.columns:
        bases_meta = meta_df[meta_df["is_base"]].copy()
    else:
        bases_meta = meta_df[meta_df["variant_id"] == 0].copy()
    bases_meta = bases_meta.set_index("preset_id")

    rows = []
    label_list = labels()
    for i, pid in enumerate(preset_ids):
        row: dict = {"preset_id": pid, "wav_file": wavs[i].name}
        if pid in bases_meta.index:
            m = bases_meta.loc[pid]
            row["preset_name"] = m.get("preset_name", "")
            row["author"] = m.get("author", "")
            row["pack"] = m.get("pack", "")
            row["preset_style"] = m.get("preset_style", "")
        for j, lbl in enumerate(label_list):
            row[f"clap_{lbl}"] = float(scores[i, j])
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    out_scores = out_dir / "clap_scores.parquet"
    scores_df.to_parquet(out_scores, index=False)
    print(f"\nWrote {out_scores}")

    # ---- sanity-check: top descriptors per sampled preset
    print("\nSample preset -> top descriptors (sanity check):")
    for sample_idx in [0, 200, 500, 1000, 1500, 2000]:
        if sample_idx >= len(rows):
            continue
        r = rows[sample_idx]
        s = scores[sample_idx]
        top5 = s.argsort()[::-1][:5]
        descriptors_str = ", ".join(
            f"{label_list[j]}({s[j]:.2f})" for j in top5
        )
        name = (r.get("preset_name") or r.get("wav_file") or "?")[:32]
        print(f"  [{name:32s}] -> {descriptors_str}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-wavs", type=Path,
                    default=DATA_DIR / "mutations_v1" / "base_wavs")
    ap.add_argument("--metadata", type=Path,
                    default=DATA_DIR / "mutations_v1" / "metadata.parquet")
    ap.add_argument("--out", type=Path,
                    default=DATA_DIR / "mutations_v1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    main(args.base_wavs, args.metadata, args.out, args.batch_size)

"""Sanity-check decode: sample random latent points, decode to param vectors,
load a base preset's full state (for wavetables + LFOs + modulations), apply
the decoded params on top, render through Vital. Listen and see if the
synthesized patches sound musical."""

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from synth_galaxy.config import AUDIO_DIR, DATA_DIR, SAMPLE_RATE
from synth_galaxy.preset_loader_full import vital_json_to_state_file
from synth_galaxy.render import load_vital, make_engine, render_note
from synth_galaxy.vae import ParamVAE


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:50]


def main(model_path: Path, dataset_dir: Path, n_samples: int, base_preset: Path,
         out_dir: Path) -> None:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    n_params = ckpt["n_params"]
    latent_dim = ckpt["latent_dim"]
    print(f"Loaded VAE: epoch={ckpt['epoch']} val_recon={ckpt['val_recon']:.5f}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ParamVAE(n_params=n_params, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    candidates = np.load(dataset_dir / "synthesis_param_indices.npy")
    print(f"Synthesis param indices: {len(candidates)} of Vital's host params")

    # Sample N random latent points from a sphere of radius ~2 (within the trained distribution)
    rng = np.random.default_rng(0)
    z = rng.normal(loc=0.0, scale=1.5, size=(n_samples, latent_dim)).astype(np.float32)
    with torch.no_grad():
        decoded = model.decode(torch.from_numpy(z).to(device)).cpu().numpy()
    print(f"Decoded shape: {decoded.shape}  range: [{decoded.min():.3f}, {decoded.max():.3f}]")

    # Compare: also encode-then-decode a known preset for sanity (variant 0 of preset 0)
    params_all = np.load(dataset_dir / "params.npy", mmap_mode="r")
    meta = pd.read_parquet(dataset_dir / "metadata.parquet")
    base_idx = int(meta[meta["is_base"]].iloc[0]["patch_id"])
    base_vec = params_all[base_idx][None, :].astype(np.float32)
    with torch.no_grad():
        z_known = model.encode(torch.from_numpy(base_vec).to(device), deterministic=True)
        recon_known = model.decode(z_known).cpu().numpy()[0]
    recon_err = float(np.mean((recon_known - base_vec[0]) ** 2) ** 0.5)
    print(f"Round-trip recon of patch_id={base_idx} ({meta.loc[base_idx, 'preset_name']}): "
          f"RMSE={recon_err:.4f}")

    # Render each: load base preset state, override decoded params, render.
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine()
    synth = load_vital(engine)

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "base.bin"
        vital_json = json.loads(base_preset.read_text())
        vital_json_to_state_file(vital_json, state_path)

        rows = []
        # Render baseline first (just the base preset)
        synth.load_state(str(state_path))
        audio = render_note(engine, synth)
        rms_baseline = float(np.sqrt((audio ** 2).mean()))
        sf.write(out_dir / "00_baseline.wav", audio.T, SAMPLE_RATE)
        rows.append({"name": "00_baseline", "rms": rms_baseline, "kind": "base_preset"})
        print(f"  [baseline] rms={rms_baseline:.4f}")

        # Render the round-trip recon
        synth.load_state(str(state_path))
        for j, idx in enumerate(candidates):
            synth.set_parameter(int(idx), float(recon_known[j]))
        audio = render_note(engine, synth)
        rms = float(np.sqrt((audio ** 2).mean()))
        sf.write(out_dir / "01_roundtrip_recon.wav", audio.T, SAMPLE_RATE)
        rows.append({"name": "01_roundtrip_recon", "rms": rms, "kind": "roundtrip"})
        print(f"  [round-trip recon] rms={rms:.4f}")

        # Render decoded random latents
        for i, params in enumerate(decoded):
            synth.load_state(str(state_path))  # reset wavetables to base
            for j, idx in enumerate(candidates):
                synth.set_parameter(int(idx), float(params[j]))
            audio = render_note(engine, synth)
            rms = float(np.sqrt((audio ** 2).mean()))
            peak = float(np.abs(audio).max())
            name = f"02_decoded_{i:02d}_z{','.join(f'{v:+.1f}' for v in z[i][:3])}"
            sf.write(out_dir / f"{safe(name)}.wav", audio.T, SAMPLE_RATE)
            rows.append({"name": name, "rms": rms, "peak": peak, "kind": "decoded_random"})
            print(f"  [decoded {i:2d}] z[:3]={z[i][:3]}  rms={rms:.4f}  peak={peak:.4f}")

    pd.DataFrame(rows).to_csv(out_dir / "decode_demo_log.csv", index=False)
    print(f"\nWrote {n_samples + 2} audio files to {out_dir}")
    print(f"Open one with:  open {out_dir/'02_decoded_00*.wav'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DATA_DIR / "models/vae_mutations_v1/vae_best.pt")
    ap.add_argument("--dataset", type=Path, default=DATA_DIR / "mutations_v1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--base-preset", type=Path,
                    default=Path("/Users/sof/Music/Vital/Organisms/BS Cyclops.vital"))
    ap.add_argument("--out", type=Path, default=AUDIO_DIR / "decode_demo")
    args = ap.parse_args()
    main(args.model, args.dataset, args.n, args.base_preset, args.out)

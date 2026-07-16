"""Train the param VAE on a mutation-render dataset."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from synth_galaxy.config import DATA_DIR
from synth_galaxy.dataset import load_mutation_dataset, make_loaders
from synth_galaxy.vae import LATENT_DIM, ParamVAE, vae_loss


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(dataset_dir: Path, epochs: int, batch_size: int, lr: float,
         beta_max: float, beta_warmup_frac: float, out_dir: Path) -> None:
    device = pick_device()
    print(f"Device: {device}")

    ds, meta, candidate_indices = load_mutation_dataset(dataset_dir)
    train_loader, val_loader = make_loaders(ds, batch_size=batch_size)
    n_params = ds.params.shape[1]
    print(f"Dataset: {len(ds)} patches, {n_params} params each "
          f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val)")

    model = ParamVAE(n_params=n_params, latent_dim=LATENT_DIM).to(device)
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ParamVAE  latent_dim={LATENT_DIM}  n_model_params={n_model_params:,}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_val = float("inf")

    beta_warmup_epochs = max(1, int(epochs * beta_warmup_frac))

    t0 = time.time()
    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / beta_warmup_epochs)

        model.train()
        train_acc = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        for x in train_loader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            losses = vae_loss(recon, x, mu, logvar, beta=beta)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            for k in train_acc:
                train_acc[k] += losses[k].item()
            n_batches += 1
        for k in train_acc:
            train_acc[k] /= max(1, n_batches)

        model.eval()
        val_acc = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        n_batches = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar, _ = model(x)
                losses = vae_loss(recon, x, mu, logvar, beta=beta)
                for k in val_acc:
                    val_acc[k] += losses[k].item()
                n_batches += 1
        for k in val_acc:
            val_acc[k] /= max(1, n_batches)

        log_rows.append({"epoch": epoch, "beta": beta,
                         **{f"train_{k}": v for k, v in train_acc.items()},
                         **{f"val_{k}": v for k, v in val_acc.items()}})

        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | beta={beta:.3f} | "
              f"train recon={train_acc['recon']:.5f} kl={train_acc['kl']:.3f} | "
              f"val recon={val_acc['recon']:.5f} kl={val_acc['kl']:.3f} | "
              f"t={elapsed:.1f}s")

        if val_acc["recon"] < best_val:
            best_val = val_acc["recon"]
            torch.save({
                "model_state": model.state_dict(),
                "n_params": n_params,
                "latent_dim": LATENT_DIM,
                "epoch": epoch,
                "val_recon": val_acc["recon"],
            }, out_dir / "vae_best.pt")

    torch.save({
        "model_state": model.state_dict(),
        "n_params": n_params,
        "latent_dim": LATENT_DIM,
        "epoch": epochs - 1,
        "val_recon": val_acc["recon"],
    }, out_dir / "vae_final.pt")

    import pandas as pd
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    print(f"\nTraining done in {time.time() - t0:.1f}s. Best val recon: {best_val:.5f}")
    print(f"Saved {out_dir/'vae_best.pt'} and {out_dir/'vae_final.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATA_DIR / "mutations_smoke",
                    help="Directory with params.npy / metadata.parquet / synthesis_param_indices.npy")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-max", type=float, default=0.5)
    ap.add_argument("--beta-warmup-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = args.dataset.parent / "models" / f"vae_{args.dataset.name}"
    main(args.dataset, args.epochs, args.batch_size, args.lr,
         args.beta_max, args.beta_warmup_frac, args.out)

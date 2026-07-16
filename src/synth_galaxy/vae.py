"""VAE foundation model: bidirectional encoder/decoder over synth-param vectors.

v0: param-only autoencoder.
  - Encoder: 401 params -> (mu, logvar) of size LATENT_DIM
  - Decoder: LATENT_DIM -> 401 params (sigmoid; params normalized to [0,1])
  - Loss: MSE reconstruction + beta * KL(q(z|x) || N(0,I))

Future (audio-consistency loss): a separate AudioEncoder maps mel-specs to the
same latent space; an agreement term pulls patch-encoding and audio-encoding
together for paired examples. Stubbed in this file, not yet trained.
"""

from __future__ import annotations

import torch
from torch import nn


# Dimensionality of the learned latent. 3D is the viewport; the real model lives
# higher-dim to retain timbral detail. UMAP projects 12D -> 3D for the UI.
LATENT_DIM = 12


class ParamEncoder(nn.Module):
    def __init__(self, n_params: int, hidden: tuple[int, ...] = (512, 256, 128),
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        dims = [n_params, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden[-1], latent_dim)
        self.logvar = nn.Linear(hidden[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)


class ParamDecoder(nn.Module):
    def __init__(self, n_params: int, hidden: tuple[int, ...] = (128, 256, 512),
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        dims = [latent_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden[-1], n_params)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.backbone(z)
        return torch.sigmoid(self.out(h))  # params normalized to [0,1]


class ParamVAE(nn.Module):
    def __init__(self, n_params: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.n_params = n_params
        self.latent_dim = latent_dim
        self.encoder = ParamEncoder(n_params, latent_dim=latent_dim)
        self.decoder = ParamDecoder(n_params, latent_dim=latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return latent vector for a batch of param vectors."""
        mu, logvar = self.encoder(x)
        return mu if deterministic else self.reparameterize(mu, logvar)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def vae_loss(
    recon: torch.Tensor, target: torch.Tensor,
    mu: torch.Tensor, logvar: torch.Tensor,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Reconstruction MSE + beta * KL. Returns components so we can log them."""
    recon_loss = torch.nn.functional.mse_loss(recon, target, reduction="mean")
    # KL divergence per-sample, then mean
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()
    total = recon_loss + beta * kl_loss / target.size(1)  # scale KL by feature count
    return {"loss": total, "recon": recon_loss, "kl": kl_loss}

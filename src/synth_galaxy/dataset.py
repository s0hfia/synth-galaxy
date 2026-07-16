"""Dataset utilities for VAE training over saved mutation-render artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class PatchDataset(Dataset):
    """Yields (params_tensor,). Audio encoder will extend this later."""

    def __init__(self, params: np.ndarray):
        # Params already in [0,1] (VST3 normalized) — no further scaling.
        self.params = torch.from_numpy(params.astype(np.float32))

    def __len__(self) -> int:
        return self.params.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.params[i]


def load_mutation_dataset(dataset_dir: Path) -> tuple[PatchDataset, pd.DataFrame, np.ndarray]:
    """Load params + metadata + the synthesis_param_indices index from a render dir."""
    params = np.load(dataset_dir / "params.npy")
    meta = pd.read_parquet(dataset_dir / "metadata.parquet")
    indices = np.load(dataset_dir / "synthesis_param_indices.npy")
    return PatchDataset(params), meta, indices


def make_loaders(
    ds: PatchDataset,
    batch_size: int = 128,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

"""Fit UMAP from per-patch feature vectors → 3D coords."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .features import FEATURE_COLUMNS


def fit_umap(
    feat_df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, UMAP, StandardScaler]:
    """Standardize features, then UMAP to 3D. Returns (coords, reducer, scaler)."""
    X = feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    reducer = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(Xn)
    return coords, reducer, scaler

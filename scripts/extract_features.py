"""Extract audio features for every patch in the latest dataset parquet."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from synth_galaxy.config import DATA_DIR, FEATURES_DIR, PATCHES_DIR
from synth_galaxy.features import extract_features


def main(dataset_parquet: Path) -> None:
    df = pd.read_parquet(dataset_parquet)
    # Datasets from random patches use 'patch_id'; preset datasets use 'preset_id'.
    id_col = "preset_id" if "preset_id" in df.columns else "patch_id"
    print(f"Extracting features for {len(df)} items from {dataset_parquet.name} (id={id_col})")

    rows = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Features"):
        wav = DATA_DIR / row["wav_path"]
        try:
            feats = extract_features(wav)
            feats[id_col] = int(row[id_col])
            rows.append(feats)
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  failed on {wav.name}: {e}")

    feat_df = pd.DataFrame(rows)
    out_path = FEATURES_DIR / f"features_{dataset_parquet.stem}.parquet"
    feat_df.to_parquet(out_path, index=False)
    print(f"\nExtracted {len(feat_df)} feature rows, {failed} failed")
    print(f"Wrote {out_path}")
    print("\nFeature ranges (min/max/mean):")
    desc = feat_df.drop(columns=[id_col]).describe().T[["min", "max", "mean"]]
    print(desc.round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, nargs="?", default=None,
                    help="Path to dataset parquet (default: latest)")
    args = ap.parse_args()
    if args.dataset is None:
        candidates = sorted(PATCHES_DIR.glob("dataset_*.parquet"))
        if not candidates:
            raise SystemExit("No dataset parquet found in data/patches/")
        args.dataset = candidates[-1]
    main(args.dataset)

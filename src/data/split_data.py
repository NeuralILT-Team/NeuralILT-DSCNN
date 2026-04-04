"""
Split the processed MetalSet into train/val/test.

Per our proposal:
  - Train: 80% (13,178 tiles)
  - Val:   10% (1,647 tiles)
  - Test:  10% (1,647 tiles)

Uses a fixed random seed (42) so the split is reproducible.
Saves the split as a JSON file that the dataset class reads.
"""

import json
import random
from pathlib import Path
import yaml


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, seed=42, max_samples=-1):
    """
    Split files into train/val/test and return filename lists.
    """
    data_dir = Path(data_dir)
    layout_dir = data_dir / "layouts"

    if not layout_dir.exists():
        raise FileNotFoundError(f"No layouts directory at {layout_dir}")

    all_files = sorted([f.name for f in layout_dir.iterdir() if f.is_file()])

    if max_samples > 0:
        all_files = all_files[:max_samples]

    # shuffle with fixed seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:],
    }

    return splits


def save_splits(splits, output_path):
    """Save split to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {output_path}")
    for k, v in splits.items():
        print(f"  {k}: {len(v)} samples")


def load_splits(path):
    """Load split from JSON."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # load config if available
    cfg_path = Path("configs/data.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    data_dir = cfg.get("processed_dir", "data/processed/MetalSet")
    splits = split_dataset(
        data_dir,
        train_ratio=cfg.get("train_ratio", 0.8),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("split_seed", 42),
        max_samples=cfg.get("max_samples", -1),
    )
    save_splits(splits, Path(data_dir) / "splits.json")

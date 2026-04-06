"""
Preprocessing script for LithoBench datasets.

Handles multiple dataset subsets:
  - MetalSet (16,472 tiles) — primary training/eval dataset
  - StdMetal (271 tiles) — out-of-distribution generalization test
  - StdContact (328 tiles) — optional cross-domain test

Each subset has target/ (layouts) and litho/ (masks) subdirectories.
This script converts them to grayscale and saves to data/processed/.

Run:
    python -m src.data.preprocess                    # MetalSet only
    python -m src.data.preprocess --all              # all datasets
    python -m src.data.preprocess --dataset StdMetal  # specific dataset
    MAX_SAMPLES=5000 python -m src.data.preprocess   # subset for local dev
"""

import argparse
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image

RAW_DIR = Path("data/raw")
PROCESSED_BASE = Path("data/processed")

# all LithoBench subsets we support
DATASETS = {
    "MetalSet": {"target": "target", "litho": "litho"},
    "StdMetal": {"target": "target", "litho": "litho"},
    "StdContact": {"target": "target", "litho": "litho"},
}

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", -1))


def process_and_save_image(input_path, output_path):
    """Convert to grayscale, normalize, save."""
    image = Image.open(input_path).convert("L")
    array = np.array(image, dtype=np.float32) / 255.0
    output = (array * 255).astype(np.uint8)
    Image.fromarray(output).save(output_path)


def preprocess_dataset(dataset_name, max_samples=-1):
    """Preprocess a single dataset subset."""
    raw_path = RAW_DIR / dataset_name
    out_dir = PROCESSED_BASE / dataset_name

    if not raw_path.exists():
        print(f"[SKIP] {dataset_name}: not found at {raw_path}")
        return 0

    layout_dir = raw_path / "target"
    mask_dir = raw_path / "litho"

    if not layout_dir.exists():
        print(f"[SKIP] {dataset_name}: no target/ directory")
        return 0
    if not mask_dir.exists():
        print(f"[SKIP] {dataset_name}: no litho/ directory")
        return 0

    # create output dirs
    (out_dir / "layouts").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    layout_files = [p for p in layout_dir.iterdir() if p.is_file()]
    mask_names = {p.name for p in mask_dir.iterdir() if p.is_file()}

    random.shuffle(layout_files)

    processed = 0
    skipped = 0

    for idx, layout_path in enumerate(layout_files):
        if max_samples != -1 and idx >= max_samples:
            print(f"[INFO] {dataset_name}: reached MAX_SAMPLES={max_samples}")
            break

        if layout_path.name not in mask_names:
            skipped += 1
            continue

        process_and_save_image(layout_path, out_dir / "layouts" / layout_path.name)
        process_and_save_image(mask_dir / layout_path.name,
                               out_dir / "masks" / layout_path.name)
        processed += 1

        if processed % 500 == 0:
            print(f"[INFO] {dataset_name}: processed {processed} pairs")

    print(f"[DONE] {dataset_name}: {processed} processed, {skipped} skipped")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess LithoBench datasets")
    parser.add_argument("--dataset", type=str, default="MetalSet",
                        help="Dataset to preprocess (MetalSet, StdMetal, StdContact)")
    parser.add_argument("--all", action="store_true",
                        help="Preprocess all available datasets")
    args = parser.parse_args()

    if args.all:
        print("Preprocessing all available datasets...")
        total = 0
        for name in DATASETS:
            total += preprocess_dataset(name, MAX_SAMPLES)
        print(f"\nTotal: {total} pairs processed across all datasets")
    else:
        preprocess_dataset(args.dataset, MAX_SAMPLES)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()

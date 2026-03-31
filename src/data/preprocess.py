"""
Purpose:
    This script prepares the LithoBench dataset for model training.

What this script does:
    1. Looks for a compressed dataset archive in data/raw/ (for example: lithodata.tar.gz)
    2. Extracts the archive if it has not already been extracted
    3. Searches the extracted contents to find the MetalSet folder
    4. Reads layout and mask images
    5. Converts them to single-channel grayscale
    6. Normalizes pixel values into [0, 1]
    7. Saves them into data/processed/MetalSet/layouts and masks

Why this script is useful:
    - Keeps raw data untouched
    - Makes preprocessing reproducible
    - Avoids repeated manual extraction steps
    - Creates a clean training-ready dataset structure

Inputs:
    - data/raw/lithodata.tar.gz   (or another .tar / .tar.gz archive)
      OR
    - an already extracted dataset folder under data/raw/

Outputs:
    - data/processed/MetalSet/layouts/
    - data/processed/MetalSet/masks/

How to run:
    python -m src.data.preprocess

Important notes:
    - This script does NOT push raw data to GitHub
    - This script assumes the dataset contains a folder named "MetalSet"
    - This script preserves filenames so layout-mask pairing remains consistent
"""

"""
Preprocessing script for LithoBench MetalSet dataset

Aligned with CMPE 257 project proposal:
- Input:  target (layout)
- Output: litho (mask)
- Task: image-to-image mapping (Neural ILT)

Supports:
- Local subset training (e.g., 5000 samples)
- Full dataset training (teammates / GPU)

Run:
    python -m src.data.preprocess
    MAX_SAMPLES=5000 python -m src.data.preprocess
"""

from pathlib import Path
import os
import random
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/MetalSet")

# -1 → FULL dataset (for teammates)
# 5000 → your local run
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", -1))


# -----------------------------------------------------------------------------
# CREATE OUTPUT DIRS
# -----------------------------------------------------------------------------
def create_output_dirs():
    (PROCESSED_DIR / "layouts").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "masks").mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# PROCESS ONE IMAGE
# -----------------------------------------------------------------------------
def process_and_save_image(input_path: Path, output_path: Path):
    """
    Convert image to grayscale + normalize [0,1] + save
    """
    image = Image.open(input_path).convert("L")

    array = np.array(image, dtype=np.float32)
    array = array / 255.0

    output = (array * 255).astype(np.uint8)
    Image.fromarray(output).save(output_path)


# -----------------------------------------------------------------------------
# MAIN PREPROCESS LOOP
# -----------------------------------------------------------------------------
def preprocess(layout_dir: Path, mask_dir: Path):
    layout_files = [p for p in layout_dir.iterdir() if p.is_file()]
    mask_names = {p.name for p in mask_dir.iterdir() if p.is_file()}

    # Shuffle → important for subset diversity
    random.shuffle(layout_files)

    processed = 0
    skipped = 0

    for idx, layout_path in enumerate(layout_files):

        # LIMIT CONTROL (key for your setup)
        if MAX_SAMPLES != -1 and idx >= MAX_SAMPLES:
            print(f"[INFO] Reached MAX_SAMPLES={MAX_SAMPLES}")
            break

        mask_path = mask_dir / layout_path.name

        if layout_path.name not in mask_names:
            skipped += 1
            continue

        out_layout = PROCESSED_DIR / "layouts" / layout_path.name
        out_mask = PROCESSED_DIR / "masks" / mask_path.name

        process_and_save_image(layout_path, out_layout)
        process_and_save_image(mask_path, out_mask)

        processed += 1

        if processed % 500 == 0:
            print(f"[INFO] Processed {processed} pairs")

    print(f"\n[FINAL]")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    """
    Uses already extracted dataset:
    data/raw/MetalSet/
        ├── target/
        ├── litho/
    """

    metalset_path = RAW_DIR / "MetalSet"

    layout_dir = metalset_path / "target"
    mask_dir = metalset_path / "litho"

    if not layout_dir.exists():
        raise FileNotFoundError(f"Missing target folder: {layout_dir}")

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing litho folder: {mask_dir}")

    create_output_dirs()
    preprocess(layout_dir, mask_dir)

    print("\n[INFO] Preprocessing completed successfully")


if __name__ == "__main__":
    main()
"""
Run inference on individual layout images.

Loads a trained model and generates mask predictions.
Useful for qualitative evaluation and generating figures for the report.

Usage:
    python -m src.infer --config configs/baseline.yaml \
        --checkpoint results/checkpoints/baseline/best_model.pt \
        --input data/processed/MetalSet/layouts/sample.png \
        --output results/predictions/sample_pred.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.common import build_model
from src.utils.io import load_config, merge_configs, load_checkpoint


def infer_single(model, image_path, device):
    """Run inference on one image. Returns predicted mask as uint8 array."""
    model.eval()

    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(x)

    pred_np = pred.squeeze().cpu().numpy()
    return (pred_np * 255).clip(0, 255).astype(np.uint8)


def infer_directory(model, input_dir, output_dir, device, max_images=-1):
    """Run inference on all PNGs in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    if max_images > 0:
        files = files[:max_images]

    print(f"Processing {len(files)} images...")
    for i, f in enumerate(files):
        pred = infer_single(model, f, device)
        Image.fromarray(pred).save(output_dir / f.name)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} done")

    print(f"Saved predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image or directory")
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--max-images", type=int, default=-1)
    args = parser.parse_args()

    data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}
    config = merge_configs(data_cfg, load_config(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))

    inp = Path(args.input)
    if inp.is_dir():
        infer_directory(model, inp, args.output, device, args.max_images)
    else:
        pred = infer_single(model, inp, device)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).save(args.output)
        print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Analyze the LithoBench dataset and training results.

This script provides two modes:
  1. Dataset analysis: statistics about the raw/processed tiles
  2. Results analysis: parse training logs and evaluation results

Usage:
    python scripts/analyze_data.py dataset          # analyze dataset
    python scripts/analyze_data.py results           # analyze training results
    python scripts/analyze_data.py all               # both
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────
# Dataset Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_dataset(data_dir, name="MetalSet"):
    """Compute statistics about a processed dataset."""
    data_dir = Path(data_dir)
    layout_dir = data_dir / "layouts"
    mask_dir = data_dir / "masks"

    if not layout_dir.exists():
        print(f"  [SKIP] {name}: no layouts at {layout_dir}")
        return None

    layout_files = sorted([f for f in layout_dir.iterdir() if f.is_file()])
    n = len(layout_files)

    if n == 0:
        print(f"  [SKIP] {name}: empty")
        return None

    print(f"\n--- {name} ({n} tiles) ---")

    # sample up to 500 tiles for statistics (faster than loading all 16k)
    sample_size = min(n, 500)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_files = [layout_files[i] for i in sample_indices]

    layout_stats = {"means": [], "stds": [], "mins": [], "maxs": [],
                    "edge_fracs": []}
    mask_stats = {"means": [], "stds": [], "mins": [], "maxs": [],
                  "edge_fracs": []}

    for f in sample_files:
        # layout
        layout = np.array(Image.open(f).convert('L'), dtype=np.float32) / 255.0
        layout_stats["means"].append(layout.mean())
        layout_stats["stds"].append(layout.std())
        layout_stats["mins"].append(layout.min())
        layout_stats["maxs"].append(layout.max())

        # edge fraction (how much of the tile is edges vs flat regions)
        from scipy import ndimage
        binary = (layout > 0.5).astype(np.uint8)
        struct = ndimage.generate_binary_structure(2, 1)
        edges = ndimage.binary_dilation(binary, struct).astype(int) - \
                ndimage.binary_erosion(binary, struct).astype(int)
        layout_stats["edge_fracs"].append(edges.sum() / edges.size)

        # mask
        mask_path = mask_dir / f.name
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert('L'),
                            dtype=np.float32) / 255.0
            mask_stats["means"].append(mask.mean())
            mask_stats["stds"].append(mask.std())
            mask_stats["edge_fracs"].append(
                (ndimage.binary_dilation((mask > 0.5).astype(np.uint8), struct).astype(int) -
                 ndimage.binary_erosion((mask > 0.5).astype(np.uint8), struct).astype(int)).sum() /
                mask.size
            )

    # get image dimensions from first tile
    first_img = Image.open(layout_files[0])
    w, h = first_img.size

    print(f"  Tile size:       {w} x {h}")
    print(f"  Total tiles:     {n}")
    print(f"  Sampled:         {sample_size}")
    print(f"")
    print(f"  Layout statistics (sampled):")
    print(f"    Mean pixel:    {np.mean(layout_stats['means']):.4f} "
          f"± {np.std(layout_stats['means']):.4f}")
    print(f"    Std pixel:     {np.mean(layout_stats['stds']):.4f}")
    print(f"    Min pixel:     {np.mean(layout_stats['mins']):.4f}")
    print(f"    Max pixel:     {np.mean(layout_stats['maxs']):.4f}")
    print(f"    Edge fraction: {np.mean(layout_stats['edge_fracs']):.4f} "
          f"± {np.std(layout_stats['edge_fracs']):.4f}")

    if mask_stats["means"]:
        print(f"")
        print(f"  Mask statistics (sampled):")
        print(f"    Mean pixel:    {np.mean(mask_stats['means']):.4f} "
              f"± {np.std(mask_stats['means']):.4f}")
        print(f"    Std pixel:     {np.mean(mask_stats['stds']):.4f}")
        print(f"    Edge fraction: {np.mean(mask_stats['edge_fracs']):.4f} "
              f"± {np.std(mask_stats['edge_fracs']):.4f}")

    # check split file
    split_path = data_dir / "splits.json"
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        print(f"")
        print(f"  Split:")
        for k, v in splits.items():
            print(f"    {k}: {len(v)} tiles")

    return {
        "name": name,
        "n_tiles": n,
        "tile_size": [w, h],
        "layout_mean": float(np.mean(layout_stats["means"])),
        "layout_std": float(np.mean(layout_stats["stds"])),
        "layout_edge_frac": float(np.mean(layout_stats["edge_fracs"])),
        "mask_mean": float(np.mean(mask_stats["means"])) if mask_stats["means"] else None,
    }


def analyze_all_datasets():
    """Analyze all available datasets."""
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    results = {}
    for name in ["MetalSet", "StdMetal", "StdContact"]:
        data_dir = Path(f"data/processed/{name}")
        stats = analyze_dataset(data_dir, name)
        if stats:
            results[name] = stats

    if not results:
        print("\nNo processed datasets found.")
        print("Run preprocessing first: python -m src.data.preprocess --all")

    return results


# ─────────────────────────────────────────────────────────────────────
# Results Analysis
# ─────────────────────────────────────────────────────────────────────

def analyze_training_logs():
    """Parse and summarize training CSV logs."""
    print("\n" + "=" * 60)
    print("TRAINING RESULTS ANALYSIS")
    print("=" * 60)

    log_dir = Path("results/logs")
    if not log_dir.exists():
        print("\nNo training logs found at results/logs/")
        print("Train models first: python -m src.train --config configs/baseline.yaml")
        return

    import pandas as pd

    csv_files = list(log_dir.rglob("*_metrics.csv"))
    if not csv_files:
        print("\nNo CSV log files found.")
        return

    for csv_path in sorted(csv_files):
        name = csv_path.stem.replace("_metrics", "")
        df = pd.read_csv(csv_path)

        print(f"\n--- {name} ---")
        print(f"  Epochs trained: {len(df)}")

        if "train_loss" in df.columns:
            print(f"  Final train loss: {df['train_loss'].iloc[-1]:.6f}")
            print(f"  Best train loss:  {df['train_loss'].min():.6f} "
                  f"(epoch {df['train_loss'].idxmin() + 1})")

        if "val_loss" in df.columns:
            print(f"  Final val loss:   {df['val_loss'].iloc[-1]:.6f}")
            print(f"  Best val loss:    {df['val_loss'].min():.6f} "
                  f"(epoch {df['val_loss'].idxmin() + 1})")

        if "val_ssim" in df.columns:
            print(f"  Final val SSIM:   {df['val_ssim'].iloc[-1]:.6f}")
            print(f"  Best val SSIM:    {df['val_ssim'].max():.6f} "
                  f"(epoch {df['val_ssim'].idxmax() + 1})")

        if "val_mse" in df.columns:
            print(f"  Final val MSE:    {df['val_mse'].iloc[-1]:.6f}")

        # check for overfitting (val loss increasing while train loss decreasing)
        if "train_loss" in df.columns and "val_loss" in df.columns and len(df) > 10:
            last_10_train = df["train_loss"].iloc[-10:].values
            last_10_val = df["val_loss"].iloc[-10:].values
            train_trend = np.polyfit(range(10), last_10_train, 1)[0]
            val_trend = np.polyfit(range(10), last_10_val, 1)[0]
            if train_trend < 0 and val_trend > 0:
                print(f"  [WARN] Possible overfitting: train loss decreasing but val loss increasing")
            elif val_trend < 0:
                print(f"  [OK]   Still improving (val loss trending down)")
            else:
                print(f"  [OK]   Converged (val loss stable)")


def analyze_comparison():
    """Parse and summarize model comparison results."""
    comp_path = Path("results/comparison.json")
    if not comp_path.exists():
        print("\nNo comparison results found.")
        print("Run: python -m src.evaluate --compare")
        return

    with open(comp_path) as f:
        data = json.load(f)

    print(f"\n--- Model Comparison (MetalSet test) ---")
    print(f"  {'Metric':<20s} {'Baseline':>15s} {'DS-CNN':>15s} {'Diff':>10s}")
    print(f"  {'-' * 60}")

    for model_key in ["baseline", "dscnn"]:
        if model_key not in data:
            print(f"  Missing {model_key} results")
            return

    b = data["baseline"]
    d = data["dscnn"]

    # accuracy
    for m in ["mse", "ssim", "epe"]:
        bv = b.get("accuracy", {}).get(m, 0)
        dv = d.get("accuracy", {}).get(m, 0)
        diff = dv - bv
        sign = "+" if diff > 0 else ""
        print(f"  {m.upper():<20s} {bv:>15.6f} {dv:>15.6f} {sign}{diff:>9.6f}")

    # efficiency
    bp = b.get("efficiency", {}).get("trainable", 0)
    dp = d.get("efficiency", {}).get("trainable", 0)
    if bp > 0 and dp > 0:
        print(f"  {'Params':<20s} {bp:>15,} {dp:>15,} {bp/dp:>9.1f}x")

    bf = b.get("efficiency", {}).get("flops", 0)
    df = d.get("efficiency", {}).get("flops", 0)
    if bf > 0 and df > 0:
        print(f"  {'FLOPs':<20s} {bf:>15,.0f} {df:>15,.0f} {bf/df:>9.1f}x")

    br = b.get("runtime", {}).get("mean_ms", 0)
    dr = d.get("runtime", {}).get("mean_ms", 0)
    if br > 0 and dr > 0:
        print(f"  {'Runtime (ms)':<20s} {br:>15.2f} {dr:>15.2f} {br/dr:>9.1f}x")


def analyze_generalization():
    """Parse generalization results."""
    gen_path = Path("results/generalization_results.json")
    if not gen_path.exists():
        print("\nNo generalization results found.")
        print("Run: python -m src.evaluate --generalize")
        return

    with open(gen_path) as f:
        data = json.load(f)

    print(f"\n--- Generalization Results (Experiment 4) ---")
    print(f"  {'Model':<15s} {'Dataset':<15s} {'MSE':>10s} {'SSIM':>10s} {'EPE':>10s}")
    print(f"  {'-' * 60}")

    for model_name, datasets in data.items():
        for ds_name, r in datasets.items():
            acc = r.get("accuracy", {})
            print(f"  {model_name:<15s} {ds_name:<15s} "
                  f"{acc.get('mse', 0):>10.6f} "
                  f"{acc.get('ssim', 0):>10.6f} "
                  f"{acc.get('epe', 0):>10.4f}")


def analyze_results():
    """Run all results analysis."""
    analyze_training_logs()
    analyze_comparison()
    analyze_generalization()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset and results")
    parser.add_argument("mode", nargs="?", default="all",
                        choices=["dataset", "results", "all"],
                        help="What to analyze")
    args = parser.parse_args()

    if args.mode in ("dataset", "all"):
        analyze_all_datasets()

    if args.mode in ("results", "all"):
        analyze_results()

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)

"""
Visualization utilities for the project report.

Generates:
  - Side-by-side prediction comparisons (layout / GT / predicted)
  - Training loss curves from CSV logs
  - Efficiency comparison bar charts

Usage:
    python -m src.visualize --mode curves --log-dir results/logs
    python -m src.visualize --mode efficiency --results results/eval_results.json
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed, visualizations disabled")


def plot_predictions(layouts, masks_gt, masks_pred, output_path,
                     num_samples=4, title=""):
    """Save a grid of layout / GT / prediction comparisons."""
    if not HAS_MPL:
        return

    n = min(num_samples, len(layouts))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(layouts[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title("Input Layout")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks_gt[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(masks_pred[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted ({title})")
        axes[i, 2].axis('off')

    plt.suptitle(f"NeuralILT Predictions — {title}", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions figure to {output_path}")


def plot_training_curves(log_dir, output_path="results/training_curves.png"):
    """Plot loss curves from CSV log files."""
    if not HAS_MPL:
        return

    import pandas as pd

    log_dir = Path(log_dir)
    csvs = list(log_dir.rglob("*_metrics.csv"))
    if not csvs:
        print(f"No CSV files found in {log_dir}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        label = csv_path.stem.replace("_metrics", "")

        if "train_loss" in df.columns:
            axes[0].plot(df["step"], df["train_loss"], label=f"{label} train")
        if "val_loss" in df.columns:
            axes[0].plot(df["step"], df["val_loss"], '--', label=f"{label} val")
        if "val_mse" in df.columns:
            axes[1].plot(df["step"], df["val_mse"], label=label)
        if "val_ssim" in df.columns:
            axes[2].plot(df["step"], df["val_ssim"], label=label)

    for ax, title, ylabel in zip(axes, ["Loss", "Val MSE", "Val SSIM"],
                                  ["Loss", "MSE", "SSIM"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_efficiency_comparison(results_path, output_path="results/efficiency.png"):
    """Bar chart comparing params and FLOPs between models."""
    if not HAS_MPL:
        return

    import json
    with open(results_path) as f:
        data = json.load(f)

    baseline = data.get("baseline", {})
    dscnn = data.get("dscnn", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # params
    bp = baseline.get("efficiency", {}).get("trainable", 0)
    dp = dscnn.get("efficiency", {}).get("trainable", 0)
    bars = axes[0].bar(["Baseline", "DS-CNN"], [bp, dp], color=["#2196F3", "#4CAF50"])
    axes[0].set_title("Trainable Parameters")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, [bp, dp]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:,.0f}", ha='center', va='bottom', fontsize=8)

    # flops
    bf = baseline.get("efficiency", {}).get("flops", 0)
    df = dscnn.get("efficiency", {}).get("flops", 0)
    bars = axes[1].bar(["Baseline", "DS-CNN"], [bf, df], color=["#2196F3", "#4CAF50"])
    axes[1].set_title("FLOPs")
    axes[1].set_ylabel("Count")
    for bar, val in zip(bars, [bf, df]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:,.0f}", ha='center', va='bottom', fontsize=8)

    plt.suptitle("Computational Efficiency Comparison", fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved efficiency comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["curves", "efficiency", "predictions"])
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--results", default="results/eval_results.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.mode == "curves":
        plot_training_curves(args.log_dir, args.output or "results/training_curves.png")
    elif args.mode == "efficiency":
        plot_efficiency_comparison(args.results,
                                   args.output or "results/efficiency.png")
    elif args.mode == "predictions":
        print("Use plot_predictions() programmatically with loaded data.")


if __name__ == "__main__":
    main()

"""
Evaluation script for NeuralILT models.

Evaluates a trained model on the test set using:
  - MSE, SSIM, EPE (accuracy metrics)
  - FLOPs, params, runtime, memory (efficiency metrics)

Can also compare baseline vs DS-CNN side by side.

Usage:
    python -m src.evaluate --config configs/baseline.yaml --checkpoint results/checkpoints/baseline/best_model.pt
    python -m src.evaluate --compare
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.common import build_model
from src.data.dataset import get_dataloaders
from src.metrics.mse import compute_mse_batch
from src.metrics.ssim import compute_ssim_batch
from src.metrics.epe import compute_epe_batch
from src.metrics.flops_params import get_efficiency_metrics
from src.metrics.runtime_memory import measure_inference_time, measure_gpu_memory
from src.utils.io import load_config, merge_configs, load_checkpoint
from src.utils.seed import set_seed


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    """Compute MSE, SSIM, EPE on a dataset."""
    model.eval()
    total_mse, total_ssim, total_epe = 0.0, 0.0, 0.0
    n = 0

    for layouts, masks in tqdm(loader, desc="Evaluating", leave=False):
        layouts, masks = layouts.to(device), masks.to(device)
        preds = model(layouts)

        total_mse += compute_mse_batch(preds, masks)
        total_ssim += compute_ssim_batch(preds, masks)
        total_epe += compute_epe_batch(preds, masks)
        n += 1

    d = max(n, 1)
    return {"mse": total_mse / d, "ssim": total_ssim / d, "epe": total_epe / d}


def evaluate_model(config, checkpoint_path):
    """Full evaluation of one model."""
    set_seed(config.get("split_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)

    name = config.get("model", {}).get("name", "unknown")
    print(f"\nEvaluating: {name}")
    print(f"Checkpoint: {checkpoint_path}")

    _, _, test_loader = get_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")

    # accuracy
    accuracy = evaluate_accuracy(model, test_loader, device)

    # efficiency
    efficiency = get_efficiency_metrics(model, device=str(device))
    runtime = measure_inference_time(model, device=str(device))
    memory = measure_gpu_memory(model) if device.type == 'cuda' else {}

    results = {
        "model": name,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "runtime": runtime,
        "memory": memory,
    }

    # print summary
    print(f"\n{'=' * 50}")
    print(f"Results: {name}")
    print(f"{'=' * 50}")
    print(f"  MSE:        {accuracy['mse']:.6f}")
    print(f"  SSIM:       {accuracy['ssim']:.6f}")
    print(f"  EPE:        {accuracy['epe']:.4f} px")
    print(f"  Params:     {efficiency['trainable']:,}")
    flops = efficiency.get('flops', -1)
    if flops > 0:
        print(f"  FLOPs:      {flops:,.0f}")
    print(f"  Runtime:    {runtime['mean_ms']:.2f} ms")
    print(f"{'=' * 50}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Model config")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs DS-CNN")
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    if args.compare:
        data_cfg = load_config(args.data_config)
        baseline_cfg = merge_configs(data_cfg, load_config("configs/baseline.yaml"))
        dscnn_cfg = merge_configs(data_cfg, load_config("configs/dscnn.yaml"))

        b_results = evaluate_model(baseline_cfg,
                                   "results/checkpoints/baseline/best_model.pt")
        d_results = evaluate_model(dscnn_cfg,
                                   "results/checkpoints/dscnn/best_model.pt")

        comparison = {"baseline": b_results, "dscnn": d_results}

        # print comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON: Baseline vs DS-CNN")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<20s} {'Baseline':>15s} {'DS-CNN':>15s}")
        print(f"  {'-' * 50}")
        for m in ['mse', 'ssim', 'epe']:
            bv = b_results['accuracy'][m]
            dv = d_results['accuracy'][m]
            print(f"  {m.upper():<20s} {bv:>15.6f} {dv:>15.6f}")
        bp = b_results['efficiency']['trainable']
        dp = d_results['efficiency']['trainable']
        print(f"  {'Params':<20s} {bp:>15,} {dp:>15,}  ({bp/dp:.1f}x)")
        bf = b_results['efficiency'].get('flops', 0)
        df = d_results['efficiency'].get('flops', 0)
        if bf > 0 and df > 0:
            print(f"  {'FLOPs':<20s} {bf:>15,.0f} {df:>15,.0f}  ({bf/df:.1f}x)")
        print(f"{'=' * 60}")

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"Saved to {out}")

    else:
        if not args.config or not args.checkpoint:
            parser.error("Need --config and --checkpoint (or use --compare)")

        data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}
        config = merge_configs(data_cfg, load_config(args.config))
        results = evaluate_model(config, args.checkpoint)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

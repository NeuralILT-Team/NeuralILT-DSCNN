"""
Evaluation script for NeuralILT models.

Supports three evaluation modes:
  1. Single model evaluation on MetalSet test split
  2. Compare baseline vs DS-CNN on MetalSet test split
  3. Generalization evaluation on StdMetal / StdContact (Experiment 4)

Usage:
    python -m src.evaluate --config configs/baseline.yaml --checkpoint results/checkpoints/baseline/best_model.pt
    python -m src.evaluate --compare
    python -m src.evaluate --generalize
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.common import build_model
from src.data.dataset import LithoBenchDataset, get_dataloaders
from src.data.transforms import get_eval_transforms
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
    """Full evaluation of one model on MetalSet test split."""
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

    accuracy = evaluate_accuracy(model, test_loader, device)
    efficiency = get_efficiency_metrics(model, device=str(device))
    runtime = measure_inference_time(model, device=str(device))
    memory = measure_gpu_memory(model) if device.type == 'cuda' else {}

    results = {
        "model": name,
        "dataset": "MetalSet (test)",
        "accuracy": accuracy,
        "efficiency": efficiency,
        "runtime": runtime,
        "memory": memory,
    }

    print(f"\n{'=' * 50}")
    print(f"Results: {name} on MetalSet test")
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


def evaluate_generalization(config, checkpoint_path, dataset_name, dataset_dir):
    """
    Evaluate a trained model on an out-of-distribution dataset.

    This is Experiment 4 from the proposal: test on StdMetal (271 tiles)
    to measure how well the model generalizes to unseen layout patterns.
    The model was trained ONLY on MetalSet — StdMetal was never seen.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    load_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)

    name = config.get("model", {}).get("name", "unknown")
    bs = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "layouts").exists():
        print(f"[SKIP] {dataset_name}: not found at {dataset_dir}")
        return None

    # load the entire generalization dataset (no split — use all tiles)
    ds = LithoBenchDataset(dataset_dir, transform=get_eval_transforms())
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    print(f"\nGeneralization eval: {name} on {dataset_name} ({len(ds)} tiles)")

    accuracy = evaluate_accuracy(model, loader, device)

    print(f"  MSE:  {accuracy['mse']:.6f}")
    print(f"  SSIM: {accuracy['ssim']:.6f}")
    print(f"  EPE:  {accuracy['epe']:.4f} px")

    return {
        "model": name,
        "dataset": dataset_name,
        "num_tiles": len(ds),
        "accuracy": accuracy,
    }


def run_generalization(data_config):
    """
    Run Experiment 4: generalization evaluation on StdMetal and StdContact.

    Evaluates both baseline and DS-CNN on out-of-distribution datasets
    to test the hypothesis that DS-CNN (fewer params) may generalize
    similarly or better than the baseline.
    """
    gen_cfg = data_config.get("generalization", {})

    # datasets to evaluate on
    gen_datasets = {}
    if "stdmetal" in gen_cfg:
        gen_datasets["StdMetal"] = gen_cfg["stdmetal"]["processed_dir"]
    if "stdcontact" in gen_cfg:
        gen_datasets["StdContact"] = gen_cfg["stdcontact"]["processed_dir"]

    if not gen_datasets:
        print("No generalization datasets configured in data.yaml")
        return {}

    results = {}

    for model_name, config_path, ckpt_path in [
        ("baseline", "configs/baseline.yaml", "results/checkpoints/baseline/best_model.pt"),
        ("dscnn", "configs/dscnn.yaml", "results/checkpoints/dscnn/best_model.pt"),
    ]:
        if not Path(ckpt_path).exists():
            print(f"[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
            continue

        config = merge_configs(data_config, load_config(config_path))
        results[model_name] = {}

        for ds_name, ds_dir in gen_datasets.items():
            result = evaluate_generalization(config, ckpt_path, ds_name, ds_dir)
            if result is not None:
                results[model_name][ds_name] = result

    # print comparison table
    if results:
        print(f"\n{'=' * 70}")
        print("GENERALIZATION RESULTS (Experiment 4)")
        print(f"{'=' * 70}")
        print(f"  {'Model':<15s} {'Dataset':<15s} {'MSE':>10s} {'SSIM':>10s} {'EPE':>10s}")
        print(f"  {'-' * 60}")
        for model_name, datasets in results.items():
            for ds_name, r in datasets.items():
                acc = r["accuracy"]
                print(f"  {model_name:<15s} {ds_name:<15s} "
                      f"{acc['mse']:>10.6f} {acc['ssim']:>10.6f} {acc['epe']:>10.4f}")
        print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Model config")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs DS-CNN on MetalSet test")
    parser.add_argument("--generalize", action="store_true",
                        help="Experiment 4: evaluate on StdMetal/StdContact")
    parser.add_argument("--output", default="results/eval_results.json")
    args = parser.parse_args()

    data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}

    if args.generalize:
        # Experiment 4: generalization evaluation
        results = run_generalization(data_cfg)
        out = Path("results/generalization_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {out}")

    elif args.compare:
        # Experiment 3: compare baseline vs DS-CNN on MetalSet
        baseline_cfg = merge_configs(data_cfg, load_config("configs/baseline.yaml"))
        dscnn_cfg = merge_configs(data_cfg, load_config("configs/dscnn.yaml"))

        b_results = evaluate_model(baseline_cfg,
                                   "results/checkpoints/baseline/best_model.pt")
        d_results = evaluate_model(dscnn_cfg,
                                   "results/checkpoints/dscnn/best_model.pt")

        comparison = {"baseline": b_results, "dscnn": d_results}

        print(f"\n{'=' * 60}")
        print("COMPARISON: Baseline vs DS-CNN (MetalSet test)")
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
            parser.error("Need --config and --checkpoint (or use --compare / --generalize)")

        config = merge_configs(data_cfg, load_config(args.config))
        results = evaluate_model(config, args.checkpoint)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

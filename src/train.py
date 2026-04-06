"""
Training script for NeuralILT models.

Trains either the baseline U-Net or the proposed DS-CNN U-Net on the
LithoBench MetalSet dataset. Uses YAML configs for all settings.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/dscnn.yaml

Hyperparameter sweep (runs multiple configs sequentially):
    python -m src.train --config configs/baseline.yaml --sweep-lr 1e-3 5e-4 1e-4
    python -m src.train --config configs/dscnn.yaml --sweep-features 32,64 64,128,256,512
"""

import argparse
import copy
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.common import build_model, count_parameters
from src.data.dataset import get_dataloaders
from src.losses import get_loss
from src.metrics.mse import compute_mse_batch
from src.metrics.ssim import compute_ssim_batch
from src.utils.seed import set_seed
from src.utils.io import load_config, merge_configs, save_checkpoint
from src.utils.metrics_logger import MetricsLogger


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip=0.0):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n = 0

    for layouts, masks in loader:
        layouts, masks = layouts.to(device), masks.to(device)

        preds = model(layouts)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"train_loss": total_loss / max(n, 1)}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Run validation. Returns loss, MSE, and SSIM."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    n = 0

    for layouts, masks in loader:
        layouts, masks = layouts.to(device), masks.to(device)
        preds = model(layouts)

        total_loss += loss_fn(preds, masks).item()
        total_mse += compute_mse_batch(preds, masks)
        total_ssim += compute_ssim_batch(preds, masks)
        n += 1

    d = max(n, 1)
    return {"val_loss": total_loss / d, "val_mse": total_mse / d, "val_ssim": total_ssim / d}


def train(config, run_name=None):
    """
    Main training loop.

    Args:
        config: merged config dict
        run_name: optional name suffix for this run (used in sweeps)
    """
    set_seed(config.get("split_seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    train_cfg = config.get("training", {})
    bs = train_cfg.get("batch_size", 16)
    train_loader, val_loader, _ = get_dataloaders(config, batch_size=bs)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # model
    model = build_model(config).to(device)
    model_name = config.get("model", {}).get("name", "unknown")
    if run_name:
        model_name = f"{model_name}_{run_name}"
    n_params = count_parameters(model)
    print(f"Model: {model_name} ({n_params:,} params)")

    # optimizer
    lr = train_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=train_cfg.get("weight_decay", 0.0))

    # scheduler
    epochs = train_cfg.get("epochs", 50)
    scheduler = None
    sched_cfg = train_cfg.get("scheduler", {})
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=sched_cfg.get("min_lr", 1e-6))

    # loss
    loss_fn = get_loss(train_cfg.get("loss", "mse"))
    grad_clip = train_cfg.get("grad_clip", 0.0)

    # logging
    log_dir = config.get("log_dir", f"results/logs/{model_name}")
    ckpt_dir = config.get("checkpoint_dir", f"results/checkpoints/{model_name}")
    save_every = config.get("save_every", 10)
    logger = MetricsLogger(log_dir, name=model_name)

    # training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {epochs} epochs (lr={lr})...")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        loss_fn, device, grad_clip)
        val_metrics = validate(model, val_loader, loss_fn, device)

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        metrics = {**train_metrics, **val_metrics, "lr": cur_lr}
        logger.log(metrics, step=epoch)
        logger.print_epoch(epoch, metrics)

        # save checkpoint
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]

        if epoch % save_every == 0 or is_best or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, val_metrics["val_loss"],
                            Path(ckpt_dir) / f"epoch_{epoch}.pt", is_best=is_best)

    logger.save_json()
    print(f"\nDone! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_dir}")

    return best_val_loss


def run_sweep(base_config, sweep_param, sweep_values):
    """
    Run a hyperparameter sweep.

    Trains the model multiple times with different values for one parameter.
    Results are saved with the sweep value in the run name.
    """
    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER SWEEP: {sweep_param}")
    print(f"Values: {sweep_values}")
    print(f"{'=' * 60}")

    results = {}

    for val in sweep_values:
        config = copy.deepcopy(base_config)
        run_name = f"{sweep_param}_{val}"

        # apply the sweep parameter
        if sweep_param == "lr":
            config.setdefault("training", {})["learning_rate"] = float(val)
            run_name = f"lr_{val}"
        elif sweep_param == "features":
            features = [int(x) for x in val.split(",")]
            config.setdefault("model", {})["features"] = features
            run_name = f"feat_{'_'.join(str(f) for f in features)}"
        elif sweep_param == "epochs":
            config.setdefault("training", {})["epochs"] = int(val)
            run_name = f"ep_{val}"
        elif sweep_param == "batch_size":
            config.setdefault("training", {})["batch_size"] = int(val)
            run_name = f"bs_{val}"
        else:
            print(f"Unknown sweep param: {sweep_param}")
            continue

        # update log/checkpoint dirs to avoid overwriting
        model_name = config.get("model", {}).get("name", "unknown")
        config["log_dir"] = f"results/logs/{model_name}_{run_name}"
        config["checkpoint_dir"] = f"results/checkpoints/{model_name}_{run_name}"

        print(f"\n--- Sweep run: {run_name} ---")
        best_loss = train(config, run_name=run_name)
        results[run_name] = best_loss

    # print sweep summary
    print(f"\n{'=' * 60}")
    print("SWEEP RESULTS")
    print(f"{'=' * 60}")
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:<40s} val_loss: {loss:.6f}")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train NeuralILT model")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Data config YAML")

    # sweep arguments
    parser.add_argument("--sweep-lr", nargs="+", default=None,
                        help="Sweep learning rates (e.g., 1e-3 5e-4 1e-4)")
    parser.add_argument("--sweep-features", nargs="+", default=None,
                        help="Sweep feature configs (e.g., '32,64' '64,128,256,512')")
    parser.add_argument("--sweep-epochs", nargs="+", default=None,
                        help="Sweep epoch counts (e.g., 25 50 100)")
    parser.add_argument("--sweep-batch-size", nargs="+", default=None,
                        help="Sweep batch sizes (e.g., 8 16 32)")

    args = parser.parse_args()

    model_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}
    config = merge_configs(data_cfg, model_cfg)

    # check for sweeps
    if args.sweep_lr:
        run_sweep(config, "lr", args.sweep_lr)
    elif args.sweep_features:
        run_sweep(config, "features", args.sweep_features)
    elif args.sweep_epochs:
        run_sweep(config, "epochs", args.sweep_epochs)
    elif args.sweep_batch_size:
        run_sweep(config, "batch_size", args.sweep_batch_size)
    else:
        train(config)


if __name__ == "__main__":
    main()

"""
Training script for NeuralILT models.

Trains either the baseline U-Net or the proposed DS-CNN U-Net on the
LithoBench MetalSet dataset. Uses YAML configs for all settings.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/dscnn.yaml
"""

import argparse
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


def train(config):
    """Main training loop."""
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
    print(f"\nTraining for {epochs} epochs...")
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


def main():
    parser = argparse.ArgumentParser(description="Train NeuralILT model")
    parser.add_argument("--config", required=True, help="Model config YAML")
    parser.add_argument("--data-config", default="configs/data.yaml",
                        help="Data config YAML")
    args = parser.parse_args()

    model_cfg = load_config(args.config)
    data_cfg = load_config(args.data_config) if Path(args.data_config).exists() else {}
    config = merge_configs(data_cfg, model_cfg)

    train(config)


if __name__ == "__main__":
    main()

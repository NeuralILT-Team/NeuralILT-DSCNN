"""
Save/load checkpoints and configs.
"""

from pathlib import Path
import torch
import yaml


def save_checkpoint(model, optimizer, epoch, loss, path, is_best=False,
                    scheduler=None, scaler=None):
    """Save model checkpoint with optional scheduler and scaler state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)

    if is_best:
        best_path = path.parent / "best_model.pt"
        torch.save(state, best_path)


def load_checkpoint(path, model, optimizer=None, device='cpu',
                    scheduler=None, scaler=None):
    """Load model checkpoint. Restores optimizer/scheduler/scaler if provided.
    Returns metadata dict with epoch and loss."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return {"epoch": ckpt.get("epoch", 0), "loss": ckpt.get("loss", float("inf"))}


def load_config(path):
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*configs):
    """Merge multiple config dicts (later ones override)."""
    result = {}
    for cfg in configs:
        if cfg:
            _deep_update(result, cfg)
    return result


def _deep_update(base, override):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v

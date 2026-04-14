"""
MSE metric — pixel-level error between predicted and ground truth masks.
"""

import torch
import numpy as np


def compute_mse(pred, target):
    """Compute MSE for a single image pair."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    return float(np.mean((pred - target) ** 2))


def compute_mse_batch(preds, targets):
    """Average MSE over a batch of predictions."""
    with torch.no_grad():
        return torch.mean((preds - targets) ** 2).item()

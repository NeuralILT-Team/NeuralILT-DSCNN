"""
SSIM metric — structural similarity between predicted and ground truth masks.

SSIM looks at local structure (edges, contrast, luminance) rather than
just pixel values. It ranges from 0 (no similarity) to 1 (identical).

This is one of our key accuracy metrics per the proposal.
"""

import numpy as np
import torch


def compute_ssim(pred, target, data_range=1.0):
    """
    Compute SSIM for a single image pair.
    Uses scikit-image under the hood.
    """
    from skimage.metrics import structural_similarity

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # squeeze out batch/channel dims
    pred = np.squeeze(pred)
    target = np.squeeze(target)

    return float(structural_similarity(pred, target, data_range=data_range))


def compute_ssim_batch(preds, targets, data_range=1.0):
    """Average SSIM over a batch."""
    batch_size = preds.shape[0]
    total = 0.0
    for i in range(batch_size):
        total += compute_ssim(preds[i], targets[i], data_range)
    return total / batch_size

"""
Edge Placement Error (EPE) — the key lithography-specific metric.

EPE measures how far predicted mask edges are from ground truth edges.
Lower is better. This is what the semiconductor industry actually
cares about for mask quality.

How it works:
  1. Binarize both masks (threshold at 0.5)
  2. Extract edges using morphological operations
  3. Compute distance transform from GT edges
  4. EPE = average distance of predicted edge pixels to nearest GT edge
"""

import numpy as np
import torch
from scipy import ndimage


def _extract_edges(mask, threshold=0.5):
    """Get binary edge map from a mask image."""
    binary = (mask > threshold).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(binary, structure=struct)
    eroded = ndimage.binary_erosion(binary, structure=struct)
    edges = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    return edges.astype(bool)


def compute_epe(pred, target, threshold=0.5):
    """
    Compute EPE between predicted and ground truth masks.
    Returns mean edge distance in pixels.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = np.squeeze(pred)
    target = np.squeeze(target)

    pred_edges = _extract_edges(pred, threshold)
    gt_edges = _extract_edges(target, threshold)

    # if either has no edges, EPE is 0 (nothing to compare)
    if not gt_edges.any() or not pred_edges.any():
        return 0.0

    # distance transform: each pixel gets distance to nearest GT edge
    gt_dist = ndimage.distance_transform_edt(~gt_edges)

    # EPE = mean distance of predicted edge pixels to nearest GT edge
    return float(np.mean(gt_dist[pred_edges]))


def compute_epe_batch(preds, targets, threshold=0.5):
    """Average EPE over a batch."""
    batch_size = preds.shape[0]
    total = 0.0
    for i in range(batch_size):
        total += compute_epe(preds[i], targets[i], threshold)
    return total / batch_size

"""
Edge Placement Error (EPE)

Measures difference between predicted and ground-truth edges.
Important metric for lithography.
Lower is better.
"""

import numpy as np


def compute_epe(pred, target):
    """
    Args:
        pred (np.ndarray): predicted image [H, W]
        target (np.ndarray): ground truth image [H, W]

    Returns:
        float: EPE score
    """

    # Compute gradients (edge approximation)
    pred_grad_x, pred_grad_y = np.gradient(pred)
    target_grad_x, target_grad_y = np.gradient(target)

    pred_edges = np.abs(pred_grad_x) + np.abs(pred_grad_y)
    target_edges = np.abs(target_grad_x) + np.abs(target_grad_y)

    # Threshold edges
    pred_edges = pred_edges > 0.1
    target_edges = target_edges > 0.1

    # Compute difference
    epe = np.mean(pred_edges ^ target_edges)

    return epe
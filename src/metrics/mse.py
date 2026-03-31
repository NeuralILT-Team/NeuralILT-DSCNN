"""
Mean Squared Error (MSE)

Measures pixel-wise difference between prediction and ground truth.
Lower is better.
"""

import numpy as np


def compute_mse(pred, target):
    """
    Args:
        pred (np.ndarray): predicted image [H, W]
        target (np.ndarray): ground truth image [H, W]

    Returns:
        float: MSE value
    """
    return np.mean((pred - target) ** 2)
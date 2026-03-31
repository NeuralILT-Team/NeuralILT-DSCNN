"""
Structural Similarity Index (SSIM)

Measures structural similarity between images.
Higher is better (max = 1).
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(pred, target):
    """
    Args:
        pred (np.ndarray): predicted image [H, W]
        target (np.ndarray): ground truth image [H, W]

    Returns:
        float: SSIM score
    """
    return ssim(pred, target, data_range=1.0)
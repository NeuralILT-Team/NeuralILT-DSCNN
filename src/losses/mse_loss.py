"""
Loss functions for training.

We use MSE as our primary loss, which measures pixel-level difference
between predicted and ground truth masks:

    L = (1/N) * sum((y_true - y_pred)^2)

This is what the proposal specifies. We also have a combined loss
option that adds an SSIM component for structural fidelity, but
MSE alone should work fine for the experiments.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard MSE loss — nothing fancy."""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


def get_loss(name='mse'):
    """Get loss function by name."""
    if name == 'mse':
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")

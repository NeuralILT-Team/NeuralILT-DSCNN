"""
Data augmentation for layout-mask pairs.

We only use geometric transforms that are physically valid for
lithography layouts:
  - horizontal flip (mirror symmetry)
  - vertical flip (mirror symmetry)
  - 90-degree rotations

We do NOT use any photometric transforms (brightness, contrast, etc.)
because pixel values have direct physical meaning in lithography masks.

All transforms are applied jointly to both layout and mask to keep
them aligned.
"""

import random
import numpy as np
import torch


class PairTransform:
    """Applies a list of transforms to a (layout, mask) pair."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, layout, mask):
        for t in self.transforms:
            layout, mask = t(layout, mask)
        return layout, mask


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, layout, mask):
        if random.random() < self.p:
            layout = np.flip(layout, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        return layout, mask


class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, layout, mask):
        if random.random() < self.p:
            layout = np.flip(layout, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        return layout, mask


class RandomRot90:
    """Rotate by 0, 90, 180, or 270 degrees randomly."""

    def __call__(self, layout, mask):
        k = random.randint(0, 3)
        if k > 0:
            layout = np.rot90(layout, k).copy()
            mask = np.rot90(mask, k).copy()
        return layout, mask


class ToTensor:
    """Convert numpy arrays to torch tensors with channel dim."""

    def __call__(self, layout, mask):
        layout = torch.from_numpy(layout).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return layout, mask


def get_train_transforms():
    """Transforms for training (with augmentation)."""
    return PairTransform([
        RandomHFlip(),
        RandomVFlip(),
        RandomRot90(),
        ToTensor(),
    ])


def get_eval_transforms():
    """Transforms for validation/test (no augmentation)."""
    return PairTransform([ToTensor()])

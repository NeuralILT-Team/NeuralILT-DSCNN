"""
PyTorch Dataset for LithoBench MetalSet.

Loads layout-mask pairs from the processed directory.
Supports train/val/test splits via a JSON split file.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import get_train_transforms, get_eval_transforms


class LithoBenchDataset(Dataset):
    """
    Dataset for LithoBench layout -> mask pairs.

    Args:
        root: path to processed data (has layouts/ and masks/ subdirs)
        split: 'train', 'val', 'test', or None (use all files)
        split_file: path to splits.json
        transform: transform to apply to (layout, mask) pairs
    """

    def __init__(self, root, split=None, split_file=None, transform=None):
        self.root = Path(root)
        self.layout_dir = self.root / "layouts"
        self.mask_dir = self.root / "masks"
        self.transform = transform

        # figure out which files to use
        if split_file is not None and split is not None:
            with open(split_file) as f:
                splits = json.load(f)
            self.files = splits[split]
        else:
            self.files = sorted([f.name for f in self.layout_dir.iterdir()
                                 if f.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # load as grayscale, normalize to [0, 1]
        layout = np.array(Image.open(self.layout_dir / fname).convert('L'),
                          dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_dir / fname).convert('L'),
                        dtype=np.float32) / 255.0

        if self.transform is not None:
            layout, mask = self.transform(layout, mask)
        else:
            # default: just make tensors
            layout = torch.from_numpy(layout).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return layout, mask


def get_dataloaders(config, batch_size=None):
    """
    Build train/val/test dataloaders from config.

    This handles split generation if splits.json doesn't exist yet.
    """
    from src.data.split_data import split_dataset, save_splits

    processed_dir = config.get("processed_dir", "data/processed/MetalSet")
    bs = batch_size or config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)

    split_path = Path(processed_dir) / "splits.json"

    # generate splits if needed
    if not split_path.exists():
        splits = split_dataset(
            processed_dir,
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            seed=config.get("split_seed", 42),
            max_samples=config.get("max_samples", -1),
        )
        save_splits(splits, split_path)

    train_ds = LithoBenchDataset(processed_dir, split="train",
                                 split_file=split_path,
                                 transform=get_train_transforms())
    val_ds = LithoBenchDataset(processed_dir, split="val",
                               split_file=split_path,
                               transform=get_eval_transforms())
    test_ds = LithoBenchDataset(processed_dir, split="test",
                                split_file=split_path,
                                transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

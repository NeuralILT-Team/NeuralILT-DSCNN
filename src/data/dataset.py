import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class LithoBenchDataset(Dataset):
    def __init__(self, root):
        self.layout_dir = Path(root) / "layouts"
        self.mask_dir = Path(root) / "masks"

        self.files = list(self.layout_dir.iterdir())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        layout_path = self.files[idx]
        mask_path = self.mask_dir / layout_path.name

        layout = np.array(Image.open(layout_path)) / 255.0
        mask = np.array(Image.open(mask_path)) / 255.0

        layout = torch.tensor(layout).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return layout, mask
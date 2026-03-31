import os
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class LithoBenchDataset(Dataset):
    from pathlib import Path
    def __init__(self, root_dir):
        root_dir=Path(root_dir)
        layout_folder = os.getenv("LAYOUT_DIR", "layouts")
        mask_folder = os.getenv("MASK_DIR", "masks")

        self.layout_dir = root_dir / layout_folder
        self.mask_dir = root_dir / mask_folder

        self.files = sorted([f for f in self.layout_dir.iterdir() if f.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        layout_path = self.files[idx]
        mask_path = self.mask_dir / layout_path.name

        layout = Image.open(layout_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        layout = layout.resize((256, 256))
        mask = mask.resize((256, 256))

        layout = np.array(layout, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        layout = torch.tensor(layout).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return layout, mask
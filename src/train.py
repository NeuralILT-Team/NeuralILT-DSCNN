"""
Train baseline U-Net model for ILT

Inputs:
- data/processed/MetalSet (recommended)

Outputs:
- Prints MSE, SSIM, EPE metrics
- Saves model checkpoint

Run:
MAX_SAMPLES=500 python -m src.train
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.baseline_unet import UNet
from src.data.dataset import LithoBenchDataset
from src.metrics.mse import compute_mse
from src.metrics.ssim import compute_ssim
from src.metrics.epe import compute_epe


# -----------------------------------------------------------------------------
# CONFIG (SAFE — NO HARDCODING)
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
LR = float(os.getenv("LR", "1e-4"))

DATA_PATH = "data/processed/MetalSet"


# -----------------------------------------------------------------------------
def train():
    print(f"[INFO] Using device: {DEVICE}")

    dataset = LithoBenchDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # Create results directory safely
    os.makedirs("results/checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        total_ssim = 0
        total_epe = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(x)

            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ---- Metrics (use first sample for speed) ----
            pred_np = preds[0].detach().cpu().numpy().squeeze()
            mask_np = y[0].detach().cpu().numpy().squeeze()

            total_mse += compute_mse(pred_np, mask_np)
            total_ssim += compute_ssim(pred_np, mask_np)
            total_epe += compute_epe(pred_np, mask_np)

        # ---- Epoch averages ----
        avg_loss = total_loss / len(loader)
        avg_mse = total_mse / len(loader)
        avg_ssim = total_ssim / len(loader)
        avg_epe = total_epe / len(loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"MSE: {avg_mse:.4f} | "
            f"SSIM: {avg_ssim:.4f} | "
            f"EPE: {avg_epe:.4f}"
        )

    # ---- Save model ----
    torch.save(model.state_dict(), "results/checkpoints/baseline.pt")
    print("\n[INFO] Training complete. Model saved.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train()
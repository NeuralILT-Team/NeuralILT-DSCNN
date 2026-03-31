"""
Train baseline model
"""

import torch
from torch.utils.data import DataLoader
from src.models.baseline_unet import UNet
from src.data.dataset import LithoBenchDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LithoBenchDataset("data/processed/MetalSet")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

    torch.save(model.state_dict(), "results/checkpoints/baseline.pt")


if __name__ == "__main__":
    train()
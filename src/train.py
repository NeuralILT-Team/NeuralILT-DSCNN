"""
Train baseline model
"""

import torch
from torch.utils.data import DataLoader
from src.models.baseline_unet import UNet
from src.data.dataset import LithoBenchDataset
from src.models.ds_unet import DSUNet


def train(model_name="baseline"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LithoBenchDataset("data/processed/MetalSet")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    if model_name == "baseline":
        model = UNet().to(device)
    elif model_name == "ds":
        model = DSUNet().to(device)
    else:
        raise ValueError("Unknown model name")

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

    torch.save(model.state_dict(), f"results/checkpoints/{model_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    args = parser.parse_args()

    train(args.model)

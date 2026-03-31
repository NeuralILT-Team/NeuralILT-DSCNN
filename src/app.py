import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.models.baseline_unet import UNet

model = UNet()
checkpoint_path = Path("results/checkpoints/baseline.pt")
model_load_error = None

if checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
else:
    model_load_error = f"Checkpoint not found: {checkpoint_path}"

def predict(image):
    if model_load_error is not None:
        raise gr.Error(model_load_error)

    image = image.convert("L").resize((256, 256))
    image = np.array(image) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(image)

    pred = pred.squeeze().numpy()
    pred = (pred * 255).astype(np.uint8)

    return Image.fromarray(pred)

gr.Interface(
    fn=predict,
    inputs="image",
    outputs="image",
    title="ILT Mask Prediction"
).launch()
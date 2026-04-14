"""
FLOPs and parameter counting for efficiency comparison.

Uses the `thop` library to count FLOPs. If thop isn't installed,
falls back to just counting parameters.
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """Count trainable params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def count_flops(model, input_size=(1, 1, 256, 256), device='cpu'):
    """
    Count FLOPs for one forward pass.
    Returns dict with flops and macs.
    """
    try:
        from thop import profile
        model = model.to(device).eval()
        x = torch.randn(*input_size).to(device)
        with torch.no_grad():
            macs, params = profile(model, inputs=(x,), verbose=False)
        return {"flops": macs * 2, "macs": macs}
    except ImportError:
        print("Warning: thop not installed, can't count FLOPs")
        print("  Install with: pip install thop")
        return {"flops": -1, "macs": -1}


def get_efficiency_metrics(model, input_size=(1, 1, 256, 256), device='cpu'):
    """Get all efficiency metrics in one call."""
    params = count_parameters(model)
    flops = count_flops(model, input_size, device)
    return {**params, **flops}

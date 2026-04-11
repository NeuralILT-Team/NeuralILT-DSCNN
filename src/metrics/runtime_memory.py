"""
Runtime and memory benchmarking.

Measures inference speed and GPU memory usage so we can compare
the baseline and DS-CNN models fairly.
"""

import time
import numpy as np
import torch


def measure_inference_time(model, input_size=(1, 1, 256, 256),
                           device='cpu', warmup=10, runs=100):
    """
    Time the model's forward pass.
    Returns stats in milliseconds.
    """
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    # warmup (important for GPU — first few runs are slower)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    # timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
    }


def measure_gpu_memory(model, input_size=(1, 1, 256, 256)):
    """
    Measure GPU memory during inference.
    Returns memory in MB. Returns zeros if no GPU.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "peak_mb": 0, "note": "no GPU"}

    model = model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    x = torch.randn(*input_size).to('cuda')
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
    }

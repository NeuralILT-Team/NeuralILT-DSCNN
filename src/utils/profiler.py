"""
Quick profiling to compare baseline vs DS-CNN.
Prints a side-by-side comparison of params, FLOPs, and runtime.
"""

import torch
from src.metrics.flops_params import get_efficiency_metrics
from src.metrics.runtime_memory import measure_inference_time, measure_gpu_memory


def profile_model(model, name="", input_size=(1, 1, 256, 256), device=None):
    """Run full profiling on a model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eff = get_efficiency_metrics(model, input_size, device)
    runtime = measure_inference_time(model, input_size, device)
    memory = measure_gpu_memory(model, input_size) if device == 'cuda' else {}

    return {"name": name, "efficiency": eff, "runtime": runtime, "memory": memory}


def compare_models(models, input_size=(1, 1, 256, 256)):
    """
    Compare multiple models. Pass a dict of {name: model}.
    Prints a comparison table.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for name, model in models.items():
        results[name] = profile_model(model, name, input_size, device)

    # print comparison
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    header = f"{'Metric':<25s}"
    for name in results:
        header += f" {name:>15s}"
    print(header)
    print("-" * 60)

    # params
    row = f"{'Trainable params':<25s}"
    for name, r in results.items():
        row += f" {r['efficiency']['trainable']:>15,}"
    print(row)

    # flops
    row = f"{'FLOPs':<25s}"
    for name, r in results.items():
        flops = r['efficiency']['flops']
        if flops > 0:
            row += f" {flops:>15,.0f}"
        else:
            row += f" {'N/A':>15s}"
    print(row)

    # runtime
    row = f"{'Inference (ms)':<25s}"
    for name, r in results.items():
        row += f" {r['runtime']['mean_ms']:>15.2f}"
    print(row)

    # ratios if exactly 2 models
    names = list(results.keys())
    if len(names) == 2:
        r0, r1 = results[names[0]], results[names[1]]
        print("-" * 60)
        p0 = r0['efficiency']['trainable']
        p1 = r1['efficiency']['trainable']
        if p1 > 0:
            print(f"  Param reduction:  {p0/p1:.2f}x")
        f0 = r0['efficiency']['flops']
        f1 = r1['efficiency']['flops']
        if f0 > 0 and f1 > 0:
            print(f"  FLOPs reduction:  {f0/f1:.2f}x")
        t0 = r0['runtime']['mean_ms']
        t1 = r1['runtime']['mean_ms']
        if t1 > 0:
            print(f"  Speedup:          {t0/t1:.2f}x")

    print("=" * 60)
    return results

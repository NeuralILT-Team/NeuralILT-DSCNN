# -*- coding: utf-8 -*-
"""Validate the full NeuralILT pipeline before submitting HPC jobs.

Tests every component: model creation, forward pass, loss computation,
metric calculation, and checkpoint save/load.

Usage:
    python scripts/validate_pipeline.py

Run this after setup to catch ALL issues before wasting GPU hours.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def step(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f'  PASS  {name}')
        PASS += 1
    except Exception as e:
        print(f'  FAIL  {name}: {e}')
        FAIL += 1


def test_model_creation():
    from src.models.common import build_model
    # baseline
    cfg = {'model': {'name': 'baseline', 'features': [32, 64]}}
    model = build_model(cfg)
    assert model is not None
    # dscnn
    cfg = {'model': {'name': 'dscnn', 'features': [32, 64]}}
    model = build_model(cfg)
    assert model is not None


def test_forward_pass():
    import torch
    from src.models.common import build_model

    for name in ['baseline', 'dscnn']:
        cfg = {'model': {'name': name, 'features': [32, 64]}}
        model = build_model(cfg)
        x = torch.randn(2, 1, 256, 256)
        y = model(x)
        assert y.shape == (2, 1, 256, 256), f'{name}: got {y.shape}'
        assert y.min() >= 0 and y.max() <= 1, f'{name}: output not in [0,1]'
        print(f'         {name}: input {x.shape} -> output {y.shape}')


def test_param_reduction():
    from src.models.common import build_model, count_parameters

    baseline = build_model({'model': {'name': 'baseline'}})
    dscnn = build_model({'model': {'name': 'dscnn'}})
    bp = count_parameters(baseline)
    dp = count_parameters(dscnn)
    ratio = bp / dp if dp > 0 else 0
    print(f'         baseline: {bp:,} params')
    print(f'         dscnn:    {dp:,} params')
    print(f'         reduction: {ratio:.1f}x')
    assert dp < bp, "DS-CNN should have fewer params than baseline"


def test_loss():
    import torch
    from src.losses import get_loss

    loss_fn = get_loss('mse')
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    print(f'         MSE loss: {loss.item():.6f}')


def test_metrics():
    import torch
    from src.metrics.mse import compute_mse_batch
    from src.metrics.ssim import compute_ssim_batch
    from src.metrics.epe import compute_epe_batch

    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)

    mse = compute_mse_batch(pred, target)
    ssim = compute_ssim_batch(pred, target)
    epe = compute_epe_batch(pred, target)
    print(f'         MSE={mse:.6f}, SSIM={ssim:.4f}, EPE={epe:.4f}')


def test_flops():
    from src.models.common import build_model
    from src.metrics.flops_params import get_efficiency_metrics

    for name in ['baseline', 'dscnn']:
        model = build_model({'model': {'name': name, 'features': [32, 64]}})
        metrics = get_efficiency_metrics(model)
        flops = metrics.get('flops', -1)
        params = metrics.get('trainable', 0)
        if flops > 0:
            print(f'         {name}: {params:,} params, {flops:,.0f} FLOPs')
        else:
            print(f'         {name}: {params:,} params (thop not available)')


def test_checkpoint():
    import torch
    import tempfile
    from src.models.common import build_model
    from src.utils.io import save_checkpoint, load_checkpoint

    model = build_model({'model': {'name': 'baseline', 'features': [32, 64]}})
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
        save_checkpoint(model, optimizer, epoch=1, loss=0.5, path=f.name)
        meta = load_checkpoint(f.name, model)
        assert meta['epoch'] == 1
        assert meta['loss'] == 0.5


def test_transforms():
    import numpy as np
    from src.data.transforms import get_train_transforms, get_eval_transforms

    layout = np.random.rand(256, 256).astype(np.float32)
    mask = np.random.rand(256, 256).astype(np.float32)

    train_t = get_train_transforms()
    l, m = train_t(layout.copy(), mask.copy())
    assert l.shape == (1, 256, 256)
    assert m.shape == (1, 256, 256)

    eval_t = get_eval_transforms()
    l, m = eval_t(layout.copy(), mask.copy())
    assert l.shape == (1, 256, 256)


def test_config_loading():
    from src.utils.io import load_config, merge_configs

    baseline = load_config('configs/baseline.yaml')
    data = load_config('configs/data.yaml')
    merged = merge_configs(data, baseline)
    assert 'model' in merged
    assert merged['model']['name'] == 'baseline'
    print(f'         model: {merged["model"]["name"]}, features: {merged["model"].get("features")}')


def test_gpu_memory_estimate():
    """Estimate GPU memory usage to catch OOM before submitting jobs.

    Does a dummy forward+backward pass and measures peak memory.
    Only runs if CUDA is available.
    """
    import torch
    if not torch.cuda.is_available():
        print('         No GPU — skipping memory estimate (OK on login node)')
        return

    from src.models.common import build_model
    from src.utils.io import load_config, merge_configs

    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'         GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)')

    for cfg_path in ['configs/baseline.yaml', 'configs/dscnn.yaml']:
        model_cfg = load_config(cfg_path)
        data_cfg = load_config('configs/data.yaml')
        config = merge_configs(data_cfg, model_cfg)

        model_name = config['model']['name']
        bs = config.get('training', {}).get('batch_size', 2)
        features = config['model'].get('features', [32, 64, 128, 256])
        use_amp = config.get('training', {}).get('mixed_precision', False)

        model = build_model(config).cuda()
        x = torch.randn(bs, 1, 256, 256, device='cuda')

        torch.cuda.reset_peak_memory_stats()

        # simulate one training step
        if use_amp:
            with torch.cuda.amp.autocast():
                y = model(x)
                loss = y.mean()
            loss.backward()
        else:
            y = model(x)
            loss = y.mean()
            loss.backward()

        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        headroom = gpu_mem - peak_gb
        status = "OK" if headroom > 1.0 else "TIGHT" if headroom > 0 else "OOM!"
        print(f'         {model_name} (bs={bs}, feat={features}, amp={use_amp}): '
              f'{peak_gb:.1f} GB peak, {headroom:.1f} GB headroom [{status}]')

        assert headroom > 0, (
            f'{model_name} will OOM! Peak {peak_gb:.1f} GB > GPU {gpu_mem:.1f} GB. '
            f'Reduce batch_size or features in {cfg_path}')

        # cleanup
        del model, x, y, loss
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print("NeuralILT-DSCNN — Pipeline Validation")
    print("=" * 60)
    print()

    print("--- Model creation ---")
    step("create baseline + dscnn", test_model_creation)

    print("\n--- Forward pass ---")
    step("forward pass (both models)", test_forward_pass)

    print("\n--- Parameter reduction ---")
    step("DS-CNN has fewer params", test_param_reduction)

    print("\n--- Loss function ---")
    step("MSE loss", test_loss)

    print("\n--- Metrics ---")
    step("MSE + SSIM + EPE", test_metrics)

    print("\n--- FLOPs counting ---")
    step("FLOPs (baseline + dscnn)", test_flops)

    print("\n--- Checkpoint save/load ---")
    step("save and load checkpoint", test_checkpoint)

    print("\n--- Data transforms ---")
    step("train + eval transforms", test_transforms)

    print("\n--- Config loading ---")
    step("load and merge YAML configs", test_config_loading)

    print("\n--- GPU memory estimate ---")
    step("memory fits on GPU", test_gpu_memory_estimate)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("\nFix failures before submitting batch jobs!")
        sys.exit(1)
    else:
        print("\nAll pipeline checks passed! Ready to train:")
        print("  sbatch scripts/run_hpc.sh")
    print("=" * 60)

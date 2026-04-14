# Architecture Documentation

## Model Architectures

### Baseline U-Net

Standard encoder-decoder with skip connections. This is similar to what LithoBench uses for their NeuralILT experiments.

```
Input (1×256×256)
  │
  ├─ Encoder 1: DoubleConv(1→64)    ──────────────────┐
  │  MaxPool2d                                          │
  ├─ Encoder 2: DoubleConv(64→128)  ─────────────┐    │
  │  MaxPool2d                                     │    │
  ├─ Encoder 3: DoubleConv(128→256) ────────┐     │    │
  │  MaxPool2d                               │     │    │
  ├─ Encoder 4: DoubleConv(256→512) ──┐     │     │    │
  │  MaxPool2d                         │     │     │    │
  │                                    │     │     │    │
  ├─ Bottleneck: DoubleConv(512→1024)  │     │     │    │
  │                                    │     │     │    │
  ├─ Decoder 4: Up(1024→512) + cat ◄───┘     │     │    │
  │  DoubleConv(1024→512)                    │     │    │
  ├─ Decoder 3: Up(512→256) + cat ◄──────────┘     │    │
  │  DoubleConv(512→256)                           │    │
  ├─ Decoder 2: Up(256→128) + cat ◄────────────────┘    │
  │  DoubleConv(256→128)                                │
  ├─ Decoder 1: Up(128→64) + cat ◄──────────────────────┘
  │  DoubleConv(128→64)
  │
  └─ Conv1x1(64→1) → Sigmoid
     Output (1×256×256)
```

Each `DoubleConv` block is:
```
Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU
```

### DS-CNN U-Net (Proposed)

Identical structure to the baseline, but each `DoubleConv` is replaced with `DoubleConvDS`, which uses depthwise separable convolutions:

```
DSConvBlock:
  Depthwise Conv2d(3×3, groups=in_ch) → BatchNorm → ReLU
  Pointwise Conv2d(1×1)               → BatchNorm → ReLU
```

The depthwise conv applies one 3×3 filter per input channel (spatial filtering only). The pointwise conv then mixes channels with a 1×1 convolution. This factorization reduces the computational cost from `H×W×Cin×Cout×K²` to `H×W×Cin×(K² + Cout)`.

## Data Pipeline

```
data/raw/MetalSet/target/  ──┐
data/raw/MetalSet/litho/   ──┤
                              │
                    preprocess.py
                              │
                              ▼
data/processed/MetalSet/layouts/  ──┐
data/processed/MetalSet/masks/    ──┤
                                    │
                         split_data.py (seed=42)
                                    │
                                    ▼
                         splits.json
                    ┌───────┼───────┐
                    │       │       │
                 train    val    test
                 (80%)   (10%)  (10%)
```

### Augmentation (training only)

- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation (0°, 90°, 180°, 270°)

No photometric transforms — pixel values are physically meaningful.

## Training Pipeline

```
Config (YAML) → build_model() → train loop:
  for each epoch:
    train_one_epoch():
      forward → MSE loss → backward → Adam step
    validate():
      forward → compute MSE, SSIM
    save checkpoint (periodic + best)
    log metrics to CSV
```

## Evaluation Pipeline

```
load checkpoint → evaluate on test set:
  Accuracy:  MSE, SSIM, EPE
  Efficiency: params, FLOPs (via thop), runtime, GPU memory
```

## File Responsibilities

| File | What it does |
|------|-------------|
| `src/models/blocks.py` | ConvBlock, DSConvBlock, DoubleConv, DoubleConvDS |
| `src/models/baseline_unet.py` | Standard U-Net (UNet class) |
| `src/models/ds_unet.py` | DS-CNN U-Net (DSUNet class) |
| `src/models/common.py` | build_model() factory, count_parameters() |
| `src/data/dataset.py` | LithoBenchDataset, get_dataloaders() |
| `src/data/transforms.py` | Augmentation transforms for (layout, mask) pairs |
| `src/data/split_data.py` | Train/val/test splitting with fixed seed |
| `src/data/preprocess.py` | Raw → processed data conversion |
| `src/losses/mse_loss.py` | MSE loss function |
| `src/metrics/mse.py` | MSE metric computation |
| `src/metrics/ssim.py` | SSIM metric (uses scikit-image) |
| `src/metrics/epe.py` | Edge Placement Error (morphological edge detection + distance transform) |
| `src/metrics/flops_params.py` | FLOPs counting (uses thop) |
| `src/metrics/runtime_memory.py` | Inference timing and GPU memory measurement |
| `src/utils/seed.py` | set_seed() for reproducibility |
| `src/utils/io.py` | Checkpoint save/load, YAML config loading |
| `src/utils/metrics_logger.py` | CSV/JSON metrics logging |
| `src/utils/profiler.py` | Side-by-side model comparison |
| `src/train.py` | Main training script |
| `src/evaluate.py` | Main evaluation script |
| `src/infer.py` | Single-image inference |
| `src/visualize.py` | Plotting utilities |

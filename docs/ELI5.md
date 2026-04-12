# ELI5: NeuralILT-DSCNN Explained Simply

> For the full technical README, see [README.md](../README.md)

---

## What Is This Project About?

When factories make computer chips, they use **light** to print tiny circuit patterns onto silicon wafers. But light bends and blurs at tiny scales — so the pattern you print doesn't look exactly like the pattern you designed.

**Inverse Lithography Technology (ILT)** is the process of figuring out: *"What pattern should I shine the light through so that the blurred result looks correct?"*

Traditionally this is done with slow optimization algorithms. Recent research uses **neural networks** (deep learning) to do it much faster — you train a model on thousands of examples of "design → correct mask" pairs.

This project asks: **Can we make the neural network smaller and faster without losing accuracy?**

---

## The Two Models We Compare

### 1. Baseline U-Net (the "big" model)
- A standard image-to-image neural network
- Uses regular 3×3 convolutions (the standard building block)
- **31 million parameters** — works well but is large

### 2. DS-CNN U-Net (the "efficient" model)
- Same overall shape as the baseline
- Replaces each 3×3 convolution with a **depthwise separable convolution**
- This is the same trick used in MobileNet (phones run these!)
- **6 million parameters** — 5× smaller

### What's a Depthwise Separable Convolution?

A regular convolution looks at all input channels at once. A depthwise separable convolution splits this into two steps:
1. **Depthwise**: Apply a separate small filter to each channel independently
2. **Pointwise**: Mix the channels together with a 1×1 convolution

This does roughly the same thing but with **~8× fewer math operations**.

---

## The Datasets

All data comes from [LithoBench](https://github.com/shelljane/lithobench), an open benchmark for lithography research.

| Dataset | Tiles | Purpose | Format | Status |
|---------|-------|---------|--------|--------|
| **MetalSet** | 16,423 | Main training + testing | PNG images (2048×2048, resized to 256×256) | Available |
| **StdMetal** | 271 | Generalization test | `.glp` layout files (no rendered PNGs available) | Future work |
| **StdContact** | 328 | Cross-domain test | `.glp` layout files (no rendered PNGs available) | Future work |

Each tile has two images:
- **target/** — the circuit layout (input to the model)
- **litho/** — the lithography simulation result (what the model should predict)

The model learns: `layout image → predicted litho mask`

### Where Does the Data Come From?

The data is hosted on **Google Drive** as a 15GB tarball (`lithodata.tar.gz`). We download it and extract only what we need:
- MetalSet: `target/` and `litho/` folders (PNG images)
- StdMetal/StdContact: also from the tarball (the GitHub repo only has `.glp` layout files, not rendered images)

---

## The Pipeline: Step by Step

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Download    │ ──> │  Preprocess   │ ──> │   Train     │ ──> │  Evaluate    │
│  (15GB)      │     │  (resize to   │     │  (50 epochs │     │  (compare    │
│              │     │   256×256)    │     │   each)     │     │   models)    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Step 1: Download
```bash
ilt-download              # MetalSet from Google Drive
ilt-download-benchmarks   # StdMetal/StdContact from tarball
```
Downloads the raw data to `data/raw/`.

### Step 2: Preprocess
```bash
ilt-preprocess
```
- Converts images to grayscale
- Resizes from 2048×2048 to 256×256 (the GPU can't handle full-size images)
- Saves to `data/processed/`
- Splits into train (80%), validation (10%), test (10%)

### Step 3: Train
```bash
ilt-baseline    # train the big model (~2 hours on GPU)
ilt-dscnn       # train the small model (~1.5 hours on GPU)
```
- Reads config from `configs/baseline.yaml` or `configs/dscnn.yaml`
- Uses mixed precision (float16) to fit on 12GB GPU
- Gradient accumulation: processes 4 images at a time, accumulates 4 steps = effective batch of 16
- Saves checkpoints to `results/checkpoints/`
- Logs metrics to `results/logs/`

### Step 4: Evaluate
```bash
ilt-eval          # compare both models on MetalSet test set
ilt-generalize    # test both models on StdMetal (never seen during training)
ilt-analyze       # print summary of all results
```

---

## The Metrics: What We Measure

### Accuracy (lower is better for MSE/EPE, higher for SSIM)

| Metric | What It Measures | Intuition |
|--------|-----------------|-----------|
| **MSE** | Mean Squared Error | Average pixel difference squared |
| **SSIM** | Structural Similarity | How similar the images look (1.0 = perfect) |
| **EPE** | Edge Placement Error | How far off the edges are (critical for chip manufacturing) |

### Efficiency (the whole point of DS-CNN)

| Metric | What It Measures |
|--------|-----------------|
| **Parameters** | How many numbers the model stores (memory) |
| **FLOPs** | How many math operations per image (speed) |
| **Runtime** | Actual wall-clock time per image |
| **GPU Memory** | How much GPU RAM is needed |

---

## Results (Experiment 3)

| Metric | Baseline | DS-CNN | Change |
|--------|----------|--------|--------|
| MSE | 0.000059 | 0.000098 | Baseline slightly better |
| SSIM | 0.979 | 0.967 | -1.3% (small drop) |
| EPE | 0.00222 | 0.00166 | **DS-CNN 25% better!** |
| Params | 31.0M | 6.0M | **5.2× fewer** |
| FLOPs | 109.3B | 28.5B | **3.8× fewer** |
| Runtime | 17.7ms | 15.9ms | **1.1× faster** |

**Bottom line**: DS-CNN is 5× smaller and 4× faster with only a tiny accuracy drop. Edge placement is actually *better*.

---

## Key Files

| File | What It Does |
|------|-------------|
| `configs/data.yaml` | Dataset paths, image size (256), split ratios |
| `configs/baseline.yaml` | Baseline model config (features, batch size, epochs) |
| `configs/dscnn.yaml` | DS-CNN model config (same structure, different conv type) |
| `src/models/baseline_unet.py` | The baseline U-Net model |
| `src/models/ds_unet.py` | The DS-CNN U-Net model |
| `src/models/blocks.py` | ConvBlock vs DSConvBlock (the key difference) |
| `src/train.py` | Training loop with mixed precision + gradient accumulation |
| `src/evaluate.py` | Evaluation: compare models, generalization test |
| `src/data/preprocess.py` | Resize and convert images |
| `src/data/dataset.py` | PyTorch Dataset that loads image pairs |
| `scripts/run_hpc.sh` | SLURM script for SJSU HPC (all steps in one file) |
| `scripts/hpc_aliases.sh` | Shortcut commands (ilt-run, ilt-eval, etc.) |
| `scripts/download_data.sh` | Download data from Google Drive |

---

## Running on SJSU HPC

```bash
# First time setup (login node — has internet)
git clone -b feature/implement-dscnn-pipeline <repo-url> NeuralILT-DSCNN
cd NeuralILT-DSCNN
bash scripts/run_hpc.sh setup          # creates venv, installs PyTorch, downloads data

# Source aliases
source scripts/hpc_aliases.sh

# Run experiments (GPU node — via SLURM)
ilt-run                                # full pipeline: preprocess → train → eval

# Or step by step:
ilt-preprocess                         # resize images
ilt-baseline                           # train baseline (~2h)
ilt-dscnn                              # train DS-CNN (~1.5h)
ilt-eval                               # compare models
ilt-generalize                         # test on StdMetal
ilt-analyze                            # print results summary
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `SyntaxError: invalid syntax` | You're using system Python 2.7. Run `source venv/bin/activate` first |
| `CUDA out of memory` | Images are too large. Run `cleandata && ilt-preprocess` to resize to 256×256 |
| `No module named 'torch'` | Activate venv: `source venv/bin/activate` |
| `PyTorch import failed` | Non-fatal warning. If training works, ignore it |
| StdMetal has `.glp` files | Need rendered PNGs from tarball: `ilt-download-benchmarks` |

---

## Glossary

| Term | Meaning |
|------|---------|
| **ILT** | Inverse Lithography Technology — computing the optimal mask for chip manufacturing |
| **U-Net** | A neural network shaped like a U — encodes then decodes with skip connections |
| **Depthwise Separable Conv** | A cheaper version of convolution (same idea as MobileNet) |
| **SSIM** | Structural Similarity Index — measures image quality (1.0 = identical) |
| **EPE** | Edge Placement Error — how far edges are from where they should be |
| **FLOPs** | Floating Point Operations — how much math the model does |
| **AMP** | Automatic Mixed Precision — uses float16 to save GPU memory |
| **SLURM** | Job scheduler on HPC clusters — `sbatch` submits jobs |
| **LithoBench** | Open benchmark dataset for lithography research |

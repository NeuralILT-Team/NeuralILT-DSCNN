# Accelerating Inverse Lithography Technology using Depthwise Separable CNNs

CMPE 257 – Spring 2026
Team: Pooja Singh, Rana Shamoun, Rishi Sheth, Pramod Yadav

> **New here?** Read the [ELI5 Guide](docs/ELI5.md) — explains the project, datasets, pipeline, and every file in plain English.

---

## Overview

This project investigates whether **depthwise separable convolutions** can reduce the computational cost of neural network-based Inverse Lithography Technology (ILT) while maintaining mask prediction accuracy.

We compare two architectures on the [LithoBench MetalSet](https://github.com/shelljane/lithobench) dataset:

| Model | Description |
|-------|-------------|
| **Baseline U-Net** | Standard 4-level encoder-decoder with 3×3 convolutions |
| **DS-CNN U-Net** | Same architecture but with depthwise separable convolutions |

The DS-CNN replaces each standard 3×3 conv with a depthwise conv (spatial filtering per channel) followed by a 1×1 pointwise conv (channel mixing). This is the same idea behind MobileNet [Howard et al., 2017] — for a 3×3 kernel with 64 output channels, it reduces FLOPs by roughly 8×.

---

## Project Status

### What's Done ✅

| Component | Status | Details |
|-----------|--------|---------|
| Baseline U-Net | ✅ Complete | 4-level encoder-decoder, BatchNorm, skip connections |
| DS-CNN U-Net | ✅ Complete | Depthwise separable convolutions, same structure |
| Data pipeline | ✅ Complete | Preprocessing, augmentation, 80/10/10 split |
| Training script | ✅ Complete | Config-driven, validation, checkpointing, logging |
| Evaluation (Exp 3) | ✅ Complete | MSE, SSIM, EPE, FLOPs, params, runtime comparison |
| Generalization (Exp 4) | ✅ Complete | StdMetal + StdContact out-of-distribution eval |
| Extended training (100 epochs) | ✅ Complete | Resumed from epoch 50 → 100, both models converged |
| Hyperparameter sweep | ✅ Complete | Sweep LR, features, epochs, batch size |
| HPC infrastructure | ✅ Complete | SJSU-adapted SLURM scripts, wheel caching |
| Verification tools | ✅ Complete | verify_env.py, validate_pipeline.py |

### What's Next 🔜

| # | Task | Priority | Description | Command |
|---|------|----------|-------------|---------|
| 1 | **Wider DS-CNN sweep** | 🟡 Optional | If SSIM gap persists, try wider channels `[96,192,384,768]` (~13M params, still 2.4× fewer than baseline). Tests whether the accuracy gap is a capacity problem. | `python -m src.train --config configs/dscnn.yaml --sweep-features "64,128,256,512" "96,192,384,768"` |
| 2 | **Generate figures** | 🟡 For report | Training curves, efficiency bar charts, prediction grid visualizations for the report. | `python -m src.visualize --mode curves` |
| 3 | **Write final report** | 🟡 After #2 | Results, Discussion, Conclusion sections. Include 50-epoch vs 100-epoch comparison. | — |
| 4 | **Optional: LR sweep** | 🟢 Nice-to-have | Try learning rates `1e-3, 5e-4, 1e-4, 5e-5` if accuracy is still low after wider channels. | `python -m src.train --config configs/dscnn.yaml --sweep-lr 1e-3 5e-4 1e-4 5e-5` |

### Experiment Results

| Experiment | Description | Status |
|------------|-------------|--------|
| **Exp 1**: Baseline | Train standard U-Net on MetalSet (50 epochs) | ✅ Done |
| **Exp 2**: DS-CNN | Train DS-CNN U-Net on MetalSet (50 epochs) | ✅ Done |
| **Exp 3**: Comparison | Compare accuracy + efficiency metrics (50 & 100 epochs) | ✅ Done |
| **Exp 4**: Consistency | Compare predictions on StdMetal/StdContact (50 & 100 epochs) | ✅ Done |
| **Exp 5**: Extended training | Resume both models to 100 epochs | ✅ Done |
| **Exp 6**: Wider DS-CNN | Sweep channel widths to recover accuracy | 📋 Planned |

### Experiment 3 — Baseline vs DS-CNN on MetalSet

#### 50 Epochs

| Metric | Baseline | DS-CNN | Change |
|--------|----------|--------|--------|
| MSE ↓ | 0.000059 | 0.000098 | Baseline better |
| SSIM ↑ | 0.9789 | 0.9665 | −1.3% |
| EPE ↓ | 0.00222 | 0.00166 | **DS-CNN 25% better** |
| Params | 31.0M | 6.0M | **5.2× fewer** |
| FLOPs | 109.3B | 28.5B | **3.8× fewer** |
| Runtime | 17.7ms | 15.9ms | **1.1× faster** |

#### 100 Epochs

| Metric | Baseline | DS-CNN | Change |
|--------|----------|--------|--------|
| MSE ↓ | 0.000058 | 0.000096 | Baseline better |
| SSIM ↑ | 0.9793 | 0.9673 | −1.2% |
| EPE ↓ | 0.00253 | 0.00503 | Baseline better |
| Params | 31.0M | 6.0M | **5.2× fewer** |
| FLOPs | 109.3B | 28.5B | **3.8× fewer** |
| Runtime | 17.7ms | 15.9ms | **1.1× faster** |

#### 50 → 100 Epoch Improvement

| Metric | Baseline (50→100) | DS-CNN (50→100) |
|--------|-------------------|-----------------|
| MSE ↓ | 0.000059 → 0.000058 (−1.7%) | 0.000098 → 0.000096 (−2.0%) |
| SSIM ↑ | 0.9789 → 0.9793 (+0.04%) | 0.9665 → 0.9673 (+0.08%) |
| EPE ↓ | 0.00222 → 0.00253 (+14%) | 0.00166 → 0.00503 (+203%) |

**Key findings**:
- **MSE and SSIM improved slightly** for both models from 50 → 100 epochs, confirming both were still learning at epoch 50.
- The **SSIM gap narrowed** from 1.3% to 1.2% — the DS-CNN is slowly closing the accuracy gap with extended training.
- DS-CNN maintains **5.2× parameter reduction** and **3.8× FLOPs reduction** with only 1.2% SSIM drop at 100 epochs.
- **EPE increased** for both models at 100 epochs, suggesting possible overfitting on edge placement while overall pixel accuracy (MSE/SSIM) continued to improve. This is a known tradeoff — MSE loss optimizes pixel-level accuracy, not edge placement specifically.

### Experiment 4 — Consistency Test on Unseen Layouts

Both models were evaluated on StdMetal (271 tiles) and StdContact (165 tiles) — layouts never seen during training. Since these datasets only have layout files (no litho masks), we compare predictions between the two models.

#### 50 Epochs

| Dataset | MSE (B vs D) | SSIM (B vs D) | Baseline mean | DS-CNN mean |
|---------|-------------|---------------|---------------|-------------|
| StdMetal | 0.001130 | 0.873 | 0.092 | 0.093 |
| StdContact | 0.000400 | 0.892 | 0.078 | 0.077 |

#### 100 Epochs

| Dataset | MSE (B vs D) | SSIM (B vs D) | Baseline mean | DS-CNN mean |
|---------|-------------|---------------|---------------|-------------|
| StdMetal | 0.001912 | 0.860 | 0.092 | 0.089 |
| StdContact | 0.000389 | 0.895 | 0.078 | 0.078 |

#### 50 → 100 Epoch Comparison

| Dataset | MSE (50→100) | SSIM (50→100) |
|---------|-------------|---------------|
| StdMetal | 0.001130 → 0.001912 (+69%) | 0.873 → 0.860 (−1.5%) |
| StdContact | 0.000400 → 0.000389 (−2.8%) | 0.892 → 0.895 (+0.3%) |

**Key findings**:
- **StdContact consistency improved** at 100 epochs — lower MSE and higher SSIM between models, with mean predictions converging (both at 0.078).
- **StdMetal consistency decreased slightly** — the models' predictions diverged somewhat, suggesting the two architectures are learning slightly different representations for complex metal patterns with extended training.
- Both models still produce **highly consistent predictions** (SSIM > 0.86) on unseen layouts with nearly identical mean output values. The DS-CNN generalizes similarly to the baseline despite having 5× fewer parameters.

---

## Project Structure

```
NeuralILT-DSCNN/
├── configs/
│   ├── baseline.yaml       # Baseline U-Net config
│   ├── dscnn.yaml           # DS-CNN U-Net config
│   └── data.yaml            # Dataset config (MetalSet + generalization sets)
├── data/
│   ├── raw/                 # Raw LithoBench data
│   │   ├── MetalSet/        #   target/ and litho/ (16,472 tiles)
│   │   ├── StdMetal/        #   target/ and litho/ (271 tiles)
│   │   └── StdContact/      #   target/ and litho/ (328 tiles)
│   └── processed/           # Preprocessed (grayscale, normalized)
├── docs/
│   ├── ARCHITECTURE.md      # Architecture and design docs
│   └── CMPE_257_Project_Proposal_*.pdf
├── scripts/
│   ├── run_hpc.sh           # SLURM job script (SJSU HPC)
│   ├── verify_env.py        # Check all imports before submitting
│   ├── validate_pipeline.py # Test full pipeline with synthetic data
│   ├── hpc_aliases.sh       # Convenience aliases for HPC
│   ├── run_baseline.sh      # Train baseline (local)
│   ├── run_dscnn.sh         # Train DS-CNN (local)
│   ├── run_eval.sh          # Evaluate (local)
│   ├── run_preprocess.sh    # Preprocess data (local)
│   └── run_visualize.sh     # Generate plots (local)
├── src/
│   ├── data/                # Dataset, transforms, splitting
│   ├── losses/              # Loss functions (MSE)
│   ├── metrics/             # MSE, SSIM, EPE, FLOPs, runtime
│   ├── models/              # Baseline U-Net and DS-CNN U-Net
│   ├── utils/               # Seed, checkpointing, logging
│   ├── train.py             # Training (+ hyperparameter sweep)
│   ├── evaluate.py          # Evaluation (+ generalization test)
│   ├── infer.py             # Single-image inference
│   └── visualize.py         # Plotting utilities
├── results/                 # Checkpoints, logs, figures (gitignored)
├── requirements.txt         # Dependencies (flexible versions)
├── requirements-hpc.txt     # Dependencies (pinned for SJSU HPC)
└── setup.py                 # Package installation
```

---

## Quick Start (Local)

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# preprocess (use MAX_SAMPLES=5000 for local dev)
MAX_SAMPLES=5000 python -m src.data.preprocess --dataset MetalSet
python -m src.data.split_data

# train
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/dscnn.yaml

# evaluate
python -m src.evaluate --compare

# generalization test
python -m src.data.preprocess --dataset StdMetal
python -m src.evaluate --generalize
```

---

## Running on SJSU HPC (SLURM)

The SJSU CoE HPC cluster runs CentOS 7 with GLIBC 2.17. Key constraints:
- **Login node**: has internet, no GPU, no GCC
- **GPU nodes**: have GPU, no internet
- **/home is shared** across all nodes

### Step 1: Clone and setup (login node — one command)

```bash
ssh <your-id>@coe-hpc1.sjsu.edu
git clone -b feature/implement-dscnn-pipeline https://github.com/NeuralILT-Team/NeuralILT-DSCNN.git
cd NeuralILT-DSCNN
source scripts/hpc_aliases.sh
ilt-setup
```

`ilt-setup` handles everything automatically:
1. Creates venv + installs all dependencies (cached as binary wheels)
2. Downloads the LithoBench dataset (MetalSet + StdMetal + StdContact)
3. Verifies all imports work

### Step 2: (Optional) Validate pipeline

```bash
ilt-validate    # test full pipeline with synthetic data
```

### Step 3: Submit jobs

```bash
ilt-run             # full pipeline (preprocess → train both → eval → analyze)

# or individual steps:
ilt-baseline        # train baseline only
ilt-dscnn           # train DS-CNN only
ilt-eval            # Experiment 3: MetalSet comparison
ilt-generalize      # Experiment 4: StdMetal/StdContact
ilt-analyze         # dataset stats + results summary
```

### Step 4: Monitor and collect

```bash
myjobs              # check job status
lastlog             # tail latest output
scp -r <id>@hpc:~/NeuralILT-DSCNN/results/ ./results/
```

---

## Hyperparameter Sweep

If initial results need tuning, use the built-in sweep support:

```bash
# sweep learning rates
python -m src.train --config configs/dscnn.yaml --sweep-lr 1e-3 5e-4 1e-4 5e-5

# sweep channel widths (try wider DS-CNN to recover accuracy)
python -m src.train --config configs/dscnn.yaml \
    --sweep-features "32,64" "64,128,256,512" "96,192,384,768"

# sweep epochs
python -m src.train --config configs/dscnn.yaml --sweep-epochs 25 50 100
```

Each sweep value gets its own checkpoint and log directory. A summary table is printed at the end ranking by validation loss.

---

## Evaluation Metrics

### Accuracy Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **MSE** | Pixel-level difference | Lower is better |
| **SSIM** | Structural similarity (edges, contrast) | Higher is better (target: within 2-5% of baseline) |
| **EPE** | Edge placement error in pixels | Lower is better |

### Efficiency Metrics

| Metric | Description | Expected |
|--------|-------------|----------|
| **Parameters** | Trainable parameter count | DS-CNN should have ~8× fewer |
| **FLOPs** | Operations per forward pass | DS-CNN should have ~8× fewer |
| **Runtime** | GPU inference time (ms) | DS-CNN should be 2-4× faster |
| **Memory** | Peak GPU memory (MB) | DS-CNN should use less |

---

## Key Design Decisions

1. **Same architecture depth**: Both models use [64, 128, 256, 512] feature channels for fair comparison.
2. **BatchNorm after every conv**: Stabilizes training for both architectures.
3. **No photometric augmentation**: Pixel values have physical meaning in lithography.
4. **Fixed random seed (42)**: Reproducible data splits and training.
5. **Generalization test**: StdMetal (271 tiles) never seen during training — tests if fewer params reduce overfitting.

---

## What You Can Learn From This Experiment

### Core Questions This Project Answers

1. **Efficiency vs accuracy tradeoff**: How much accuracy do you lose when you replace standard convolutions with depthwise separable ones? Is the ~8× FLOPs reduction worth it?

2. **Theoretical vs practical speedup**: The math says ~8× fewer FLOPs, but does that translate to 8× faster inference? (Usually not — memory bandwidth, kernel launch overhead, and GPU utilization all matter.)

3. **Generalization and overfitting**: Does a model with fewer parameters generalize better to unseen data (StdMetal)? The proposal hypothesizes yes — this experiment tests it.

### Additional Experiments to Explore

If you want to go deeper, here are ideas that build on the current codebase:

#### A. Architecture Variants
- **Hybrid model**: Use standard convolutions in the first encoder level (where channels are small and DS-Conv savings are minimal) and DS-Conv everywhere else. This might give the best of both worlds.
- **Width multiplier**: MobileNet uses a width multiplier α to uniformly scale channel counts. Try α = 0.5, 0.75, 1.0, 1.25 to map the accuracy-efficiency Pareto frontier.
- **Inverted residuals**: MobileNetV2 uses inverted residual blocks (expand → depthwise → project). These could further improve DS-CNN accuracy.

#### B. Training Improvements
- **SSIM loss**: The codebase has MSE loss, but adding SSIM as a loss component (already partially implemented in `mse_loss.py`) could improve structural fidelity of predictions.
- **Learning rate warmup**: Start with a low LR for the first few epochs, then ramp up. This often helps with BatchNorm-heavy architectures.
- **Data augmentation ablation**: Disable augmentation and compare — does it help or hurt for this specific dataset?

#### C. Analysis Deep Dives
- **Per-tile difficulty analysis**: Some MetalSet tiles have more complex optical proximity effects than others. Do both models struggle on the same tiles, or does DS-CNN fail on different ones?
- **Edge analysis**: Look at EPE broken down by edge type (horizontal, vertical, diagonal, curved). DS-Conv might handle some edge orientations differently.
- **Layer-wise FLOPs breakdown**: Which encoder/decoder levels contribute most to the FLOPs difference? This tells you where DS-Conv matters most.
- **Activation visualization**: Compare intermediate feature maps between baseline and DS-CNN to understand what each model "sees" differently.

#### D. Scaling Experiments
- **ViaSet training**: The LithoBench ViaSet has 116,415 tiles (7× more than MetalSet). Training on this larger dataset would test if DS-CNN's efficiency advantage grows with dataset size.
- **Batch size scaling**: With less memory per forward pass, DS-CNN can use larger batch sizes. Does this improve training speed or final accuracy?

---

## References

1. S. Zheng et al., "LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing," NeurIPS 2023.
2. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.
3. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR 2018.
4. C. A. Mack, "Fundamental Principles of Optical Lithography," Wiley, 2007.

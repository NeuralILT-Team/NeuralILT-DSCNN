# Accelerating Inverse Lithography Technology using Depthwise Separable CNNs

CMPE 257 – Spring 2026  
Team: Pooja Singh, Rana Shamoun, Rishi Sheth, Pramod Yadav

---

## Overview

This project investigates whether **depthwise separable convolutions** can reduce the computational cost of neural network-based Inverse Lithography Technology (ILT) while maintaining mask prediction accuracy.

We compare two architectures on the [LithoBench MetalSet](https://github.com/shelljane/lithobench) dataset:

| Model | Description |
|-------|-------------|
| **Baseline U-Net** | Standard 4-level encoder-decoder with 3×3 convolutions |
| **DS-CNN U-Net** | Same architecture but with depthwise separable convolutions |

The DS-CNN replaces each standard 3×3 conv with a depthwise conv (spatial filtering per channel) followed by a 1×1 pointwise conv (channel mixing). This is the same idea behind MobileNet [Howard et al., 2017] — for a 3×3 kernel with 64 output channels, it reduces FLOPs by roughly 8×.

Detailed experiment results, comparisons, and visualization summaries are available in [RESULTS.md](RESULTS.md).

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
| Hyperparameter sweep | ✅ Complete | Sweep LR, features, epochs, batch size |
| Wider DS-CNN Experiment | ✅ Complete | 1.5× wider channels; see [RESULTS.md](RESULTS.md#wider-ds-cnn-experiment) |
| HPC infrastructure | ✅ Complete | SJSU-adapted SLURM scripts, wheel caching |
| Verification tools | ✅ Complete | verify_env.py, validate_pipeline.py |
| Collect results | ✅ Complete | Get MSE/SSIM/EPE/FLOPs numbers for the report — See [RESULTS.md](RESULTS.md) |
| Generate figures | ✅ Complete | Training curves, efficiency charts, prediction grids — Documented in [RESULTS.md](RESULTS.md) |

### What's Next 🔜

| Step | Priority | Description |
|------|----------|-------------|
| **Run experiments on HPC** | 🔴 Critical | Train both models on full MetalSet (16,472 tiles) |
| **Run generalization test** | 🟡 Important | Evaluate on StdMetal (271 tiles) — Experiment 4 |
| **Write final report** | 🟡 Important | Results, Discussion, Conclusion sections |

---

## Instructions to perform the following tasks:

###  Run Complete Workflow (Training + Figures)
```bash
# Run everything: train models, evaluate, generate figures
bash scripts/run_complete_workflow.sh
```

This will:
- Train baseline U-Net and DS-CNN models
- Evaluate both models on test set
- Generate training curves, efficiency charts, and prediction grids

#### Training Curves
```bash
python -m src.visualize --mode curves --log-dir results/logs --output results/training_curves.png
```

#### Efficiency Charts
```bash
python -m src.visualize --mode efficiency --results results/comparison.json --output results/efficiency_comparison.png
```

#### Prediction Grids
```bash
python -m src.visualize --mode predictions --checkpoint results/checkpoints/dscnn/best_model.pt --config configs/dscnn.yaml --output results/prediction_grids.png
```

###  Learning Rate Sweep
```bash
# Try different learning rates if accuracy is low
bash scripts/run_lr_sweep.sh
```

###  Wider DS-CNN Experiment
```bash
# Try wider channel dimensions
bash scripts/run_wider_dscnn.sh
```

### Experiment Plan (from proposal)

| Experiment | Description | Status |
|------------|-------------|--------|
| **Exp 1**: Baseline | Train standard U-Net on MetalSet | Code ready, needs HPC run |
| **Exp 2**: DS-CNN | Train DS-CNN U-Net on MetalSet | Code ready, needs HPC run |
| **Exp 3**: Comparison | Compare accuracy + efficiency metrics | Code ready, needs Exp 1+2 |
| **Exp 4**: Generalization | Evaluate both on StdMetal/StdContact | Code ready, needs Exp 1+2 |

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

Each sweep value gets its own checkpoint and log directory. A summary table is printed at the end ranking by validation loss. The full LR sweep results are documented in [RESULTS.md](RESULTS.md).

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

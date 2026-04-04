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

---

## Project Structure

```
NeuralILT-DSCNN/
├── configs/
│   ├── baseline.yaml       # Baseline U-Net config
│   ├── dscnn.yaml           # DS-CNN U-Net config
│   └── data.yaml            # Dataset and preprocessing config
├── data/
│   ├── raw/MetalSet/        # Raw LithoBench data (target/ and litho/)
│   └── processed/MetalSet/  # Preprocessed layouts/ and masks/
├── docs/
│   └── CMPE_257_Project_Proposal_*.pdf
├── scripts/
│   ├── run_preprocess.sh    # Preprocess raw data
│   ├── run_baseline.sh      # Train baseline model
│   ├── run_dscnn.sh         # Train DS-CNN model
│   ├── run_eval.sh          # Evaluate and compare models
│   ├── run_visualize.sh     # Generate plots
│   └── run_hpc.sh           # SLURM job script for HPC
├── src/
│   ├── data/                # Dataset, transforms, splitting
│   ├── losses/              # Loss functions (MSE)
│   ├── metrics/             # MSE, SSIM, EPE, FLOPs, runtime
│   ├── models/              # Baseline U-Net and DS-CNN U-Net
│   ├── utils/               # Seed, checkpointing, logging
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── infer.py             # Single-image inference
│   └── visualize.py         # Plotting utilities
├── results/                 # Checkpoints, logs, figures (gitignored)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup

### 1. Environment

```bash
# create virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### 2. Dataset

Download the LithoBench MetalSet and place it at:

```
data/raw/MetalSet/
  ├── target/    # input layout tiles
  └── litho/     # ground truth mask tiles
```

### 3. Preprocessing

```bash
python -m src.data.preprocess
python -m src.data.split_data
```

This normalizes images to [0,1], converts to grayscale, and creates an 80/10/10 train/val/test split.

For local development with a smaller subset:
```bash
MAX_SAMPLES=5000 python -m src.data.preprocess
```

---

## Training

### Baseline U-Net

```bash
python -m src.train --config configs/baseline.yaml
```

### DS-CNN U-Net (proposed)

```bash
python -m src.train --config configs/dscnn.yaml
```

Both models use the same training setup (Adam optimizer, MSE loss, cosine LR schedule, 50 epochs) for a fair comparison.

---

## Evaluation

### Evaluate a single model

```bash
python -m src.evaluate \
    --config configs/baseline.yaml \
    --checkpoint results/checkpoints/baseline/best_model.pt
```

### Compare both models

```bash
python -m src.evaluate --compare
```

This computes and prints:
- **Accuracy metrics**: MSE, SSIM, EPE
- **Efficiency metrics**: parameter count, FLOPs, inference runtime

---

## Visualization

```bash
# training loss curves
python -m src.visualize --mode curves --log-dir results/logs

# efficiency comparison bar charts
python -m src.visualize --mode efficiency --results results/comparison.json
```

---

## Running on HPC (SLURM)

For running on the university HPC cluster:

```bash
# run everything (preprocess → train both → evaluate → plot)
sbatch scripts/run_hpc.sh

# or run individual steps
sbatch scripts/run_hpc.sh baseline
sbatch scripts/run_hpc.sh dscnn
sbatch scripts/run_hpc.sh eval
```

Edit the module load and conda/venv activation lines in `scripts/run_hpc.sh` to match your cluster setup.

---

## Docker (Optional)

```bash
docker build -t neuralilt-dscnn .
docker run -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results neuralilt-dscnn
```

Or use docker-compose:
```bash
docker-compose run train-baseline
docker-compose run train-dscnn
docker-compose run evaluate
```

---

## Evaluation Metrics

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — pixel-level difference between predicted and GT masks |
| **SSIM** | Structural Similarity Index — perceptual similarity (edges, contrast) |
| **EPE** | Edge Placement Error — distance between predicted and GT mask edges (in pixels) |

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **Parameters** | Total trainable parameters |
| **FLOPs** | Floating-point operations for one forward pass |
| **Runtime** | GPU inference time per image (ms) |
| **Memory** | Peak GPU memory during inference (MB) |

---

## Key Design Decisions

1. **Same architecture depth**: Both models use [64, 128, 256, 512] feature channels so the comparison isolates the effect of the convolution type.

2. **BatchNorm after every conv**: Stabilizes training for both architectures.

3. **No photometric augmentation**: Pixel values have physical meaning in lithography, so we only use geometric transforms (flips, 90° rotations).

4. **Fixed random seed (42)**: Ensures reproducible data splits and training.

---

## References

1. S. Zheng et al., "LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing," NeurIPS 2023.
2. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.

---

## Notes

- Use subsets of data for local runs if needed (`MAX_SAMPLES=5000`)
- Full dataset experiments require a GPU (Google Colab T4 or HPC)
- See `docs/` for the full project proposal

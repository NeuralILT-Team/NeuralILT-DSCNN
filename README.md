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
│   ├── ARCHITECTURE.md      # Architecture and design docs
│   └── CMPE_257_Project_Proposal_*.pdf
├── scripts/
│   ├── run_preprocess.sh    # Preprocess raw data
│   ├── run_baseline.sh      # Train baseline model
│   ├── run_dscnn.sh         # Train DS-CNN model
│   ├── run_eval.sh          # Evaluate and compare models
│   ├── run_visualize.sh     # Generate plots
│   └── run_hpc.sh           # SLURM job script for HPC cluster
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
└── requirements.txt
```

---

## Local Setup

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
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
python -m src.visualize --mode efficiency --results results/eval_results.json
```

---

## Running on HPC (SLURM)

The project includes a SLURM job script for running on university HPC clusters (e.g., SJSU CoS HPC).

### First-time HPC setup

```bash
# 1. SSH into the cluster
ssh <your-id>@coe-hpc1.sjsu.edu

# 2. Clone the repo
git clone git@github.com:NeuralILT-Team/NeuralILT-DSCNN.git
cd NeuralILT-DSCNN

# 3. Create a conda environment (or use venv)
module load anaconda3
conda create -n neuralilt python=3.11 -y
conda activate neuralilt
pip install -r requirements.txt

# 4. Upload the dataset to data/raw/MetalSet/
#    (use scp or rsync from your local machine)
scp -r data/raw/MetalSet/ <your-id>@coe-hpc1.sjsu.edu:~/NeuralILT-DSCNN/data/raw/

# 5. Edit scripts/run_hpc.sh to uncomment the module load
#    and conda activate lines for your cluster
```

### Submitting jobs

```bash
# Run the full pipeline (preprocess → train both → evaluate → plot)
sbatch scripts/run_hpc.sh

# Or run individual steps
sbatch scripts/run_hpc.sh preprocess
sbatch scripts/run_hpc.sh baseline
sbatch scripts/run_hpc.sh dscnn
sbatch scripts/run_hpc.sh eval
```

### SLURM job configuration

The default settings in `scripts/run_hpc.sh` are:

| Setting | Value |
|---------|-------|
| Partition | `gpu` |
| GPUs | 1 |
| CPUs | 4 |
| Memory | 32 GB |
| Time limit | 12 hours |

Adjust these in the `#SBATCH` directives at the top of the script if your cluster has different partition names or resource limits.

### Monitoring jobs

```bash
# check job status
squeue -u $USER

# view output logs
tail -f slurm_<job_id>.out

# cancel a job
scancel <job_id>
```

### Collecting results

After jobs complete, results are in:
- `results/checkpoints/baseline/` — baseline model checkpoints
- `results/checkpoints/dscnn/` — DS-CNN model checkpoints
- `results/logs/` — training metrics (CSV + JSON)
- `results/comparison.json` — side-by-side comparison
- `results/training_curves.png` — loss/metric plots
- `results/efficiency.png` — parameter/FLOPs comparison chart

Copy results back to your local machine:
```bash
scp -r <your-id>@coe-hpc1.sjsu.edu:~/NeuralILT-DSCNN/results/ ./results/
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

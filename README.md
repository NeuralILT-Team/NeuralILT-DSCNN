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
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── infer.py             # Single-image inference
│   └── visualize.py         # Plotting utilities
├── results/                 # Checkpoints, logs, figures (gitignored)
├── requirements.txt         # Dependencies (flexible versions)
├── requirements-hpc.txt     # Dependencies (pinned for SJSU HPC)
└── setup.py                 # Package installation
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

For local development with a smaller subset:
```bash
MAX_SAMPLES=5000 python -m src.data.preprocess
```

---

## Training

```bash
# Baseline U-Net
python -m src.train --config configs/baseline.yaml

# DS-CNN U-Net (proposed)
python -m src.train --config configs/dscnn.yaml
```

## Evaluation

```bash
# Compare both models
python -m src.evaluate --compare

# Single model
python -m src.evaluate --config configs/baseline.yaml \
    --checkpoint results/checkpoints/baseline/best_model.pt
```

## Visualization

```bash
python -m src.visualize --mode curves --log-dir results/logs
python -m src.visualize --mode efficiency --results results/eval_results.json
```

---

## Running on SJSU HPC (SLURM)

The SJSU CoE HPC cluster runs CentOS 7 with GLIBC 2.17. Key constraints:
- **Login node**: has internet, no GPU, no GCC
- **GPU nodes**: have GPU, no internet
- **/home is shared** across all nodes

This means we download pre-built binary wheels on the login node, then run training on GPU nodes using the cached wheels.

### Step 1: Upload code and data

```bash
# SSH into the cluster
ssh <your-id>@coe-hpc1.sjsu.edu

# Clone the repo
git clone <repo-url>
cd NeuralILT-DSCNN

# Upload dataset (from your local machine)
scp -r data/raw/MetalSet/ <your-id>@coe-hpc1.sjsu.edu:~/NeuralILT-DSCNN/data/raw/
```

### Step 2: One-time setup (on login node)

```bash
# This downloads all wheels and creates the venv
# Must run on login node (has internet)
bash scripts/run_hpc.sh setup
```

This will:
1. Create a venv at `./venv/`
2. Download PyTorch CUDA wheels (~2GB) to `.wheels/`
3. Install all dependencies from cached wheels
4. Install the project in editable mode

### Step 3: Verify environment

```bash
# Check all imports work
python scripts/verify_env.py

# Test the full pipeline with synthetic data
python scripts/validate_pipeline.py
```

### Step 4: Submit jobs

```bash
# Full pipeline (preprocess → train both → evaluate → plot)
sbatch scripts/run_hpc.sh

# Or individual steps
sbatch scripts/run_hpc.sh preprocess
sbatch scripts/run_hpc.sh baseline
sbatch scripts/run_hpc.sh dscnn
sbatch scripts/run_hpc.sh eval
```

### Step 5: Monitor and collect results

```bash
# Check job status
squeue -u $USER

# Tail the latest log
tail -f logs/slurm_*.out

# Copy results to local machine
scp -r <your-id>@coe-hpc1.sjsu.edu:~/NeuralILT-DSCNN/results/ ./results/
```

### HPC convenience aliases

```bash
# Load aliases (or add to ~/.bashrc)
source scripts/hpc_aliases.sh

# Then use shortcuts:
ilt-setup       # one-time setup
ilt-verify      # check environment
ilt-validate    # test pipeline
ilt-run         # submit full pipeline
ilt-baseline    # train baseline only
ilt-dscnn       # train DS-CNN only
lastlog         # tail latest output
gpunode         # get interactive GPU session
```

### SLURM job defaults

| Setting | Value |
|---------|-------|
| Partition | `gpu` |
| GPUs | 1 |
| CPUs | 4 |
| Memory | 32 GB |
| Time limit | 12 hours |

Edit the `#SBATCH` directives in `scripts/run_hpc.sh` if needed.

---

## Evaluation Metrics

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error — pixel-level difference |
| **SSIM** | Structural Similarity Index — perceptual similarity |
| **EPE** | Edge Placement Error — mask edge distance (pixels) |

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **Parameters** | Total trainable parameters |
| **FLOPs** | Floating-point operations per forward pass |
| **Runtime** | GPU inference time (ms) |
| **Memory** | Peak GPU memory (MB) |

---

## Key Design Decisions

1. **Same architecture depth**: Both models use [64, 128, 256, 512] feature channels for fair comparison.
2. **BatchNorm after every conv**: Stabilizes training for both architectures.
3. **No photometric augmentation**: Pixel values have physical meaning in lithography.
4. **Fixed random seed (42)**: Reproducible data splits and training.

---

## References

1. S. Zheng et al., "LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing," NeurIPS 2023.
2. A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv:1704.04861, 2017.

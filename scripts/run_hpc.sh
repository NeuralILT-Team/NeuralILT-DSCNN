#!/bin/bash
#SBATCH --job-name=neuralilt
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# HPC SLURM job script for NeuralILT-DSCNN experiments
#
# This script runs on the university HPC cluster (e.g., SJSU CoS HPC).
# It trains both the baseline and DS-CNN models, then runs evaluation.
#
# Submit with:
#   sbatch scripts/run_hpc.sh
#
# Or run a specific model:
#   sbatch scripts/run_hpc.sh baseline
#   sbatch scripts/run_hpc.sh dscnn
#   sbatch scripts/run_hpc.sh eval
#
# Make sure to:
#   1. Set up your conda/venv environment first
#   2. Have the dataset at data/raw/MetalSet/
#   3. Run preprocessing before training (or include it below)

echo "============================================"
echo "NeuralILT-DSCNN HPC Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $SLURM_GPUS_ON_NODE"
echo "Date:   $(date)"
echo "============================================"

# ── load modules (adjust for your cluster) ──
# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9

# ── activate environment ──
# If using conda:
# source activate neuralilt
# If using venv:
# source .venv/bin/activate

# make sure we're in the project root
cd $SLURM_SUBMIT_DIR 2>/dev/null || cd "$(dirname "$0")/.."

# check GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

MODE="${1:-all}"

case "$MODE" in
    preprocess)
        echo ""
        echo ">>> Step 1: Preprocessing dataset..."
        python -m src.data.preprocess
        python -m src.data.split_data
        ;;

    baseline)
        echo ""
        echo ">>> Training baseline U-Net..."
        python -m src.train --config configs/baseline.yaml --data-config configs/data.yaml
        ;;

    dscnn)
        echo ""
        echo ">>> Training DS-CNN U-Net..."
        python -m src.train --config configs/dscnn.yaml --data-config configs/data.yaml
        ;;

    eval)
        echo ""
        echo ">>> Evaluating and comparing models..."
        python -m src.evaluate --compare --output results/comparison.json
        ;;

    all)
        echo ""
        echo ">>> Step 1: Preprocessing..."
        python -m src.data.preprocess
        python -m src.data.split_data

        echo ""
        echo ">>> Step 2: Training baseline U-Net..."
        python -m src.train --config configs/baseline.yaml --data-config configs/data.yaml

        echo ""
        echo ">>> Step 3: Training DS-CNN U-Net..."
        python -m src.train --config configs/dscnn.yaml --data-config configs/data.yaml

        echo ""
        echo ">>> Step 4: Evaluation & comparison..."
        python -m src.evaluate --compare --output results/comparison.json

        echo ""
        echo ">>> Step 5: Generating plots..."
        python -m src.visualize --mode curves --log-dir results/logs
        python -m src.visualize --mode efficiency --results results/comparison.json
        ;;

    *)
        echo "Usage: sbatch scripts/run_hpc.sh [preprocess|baseline|dscnn|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "============================================"

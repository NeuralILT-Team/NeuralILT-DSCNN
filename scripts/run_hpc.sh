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
# Self-contained HPC script for NeuralILT-DSCNN experiments.
#
# This script handles everything:
#   1. Environment setup (conda/venv + pip install)
#   2. Dataset download and preprocessing
#   3. Train/val/test splitting
#   4. Training both models (baseline + DS-CNN)
#   5. Evaluation and comparison
#   6. Generating plots
#
# Submit:
#   sbatch scripts/run_hpc.sh
#
# Or run a single step:
#   sbatch scripts/run_hpc.sh setup
#   sbatch scripts/run_hpc.sh preprocess
#   sbatch scripts/run_hpc.sh baseline
#   sbatch scripts/run_hpc.sh dscnn
#   sbatch scripts/run_hpc.sh eval
#   sbatch scripts/run_hpc.sh all       (default)

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG — edit these for your cluster
# ─────────────────────────────────────────────────────────────────────
ENV_NAME="neuralilt"
PYTHON_VERSION="3.11"
DATASET_URL=""  # set this if dataset is hosted somewhere downloadable
# If dataset is already on the cluster, set this path:
DATASET_PATH=""  # e.g., /shared/datasets/lithobench/MetalSet

# ─────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────

# go to project root (works whether submitted from project dir or scripts/)
if [ -f "scripts/run_hpc.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "run_hpc.sh" ]; then
    PROJECT_DIR="$(cd .. && pwd)"
else
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
fi
cd "$PROJECT_DIR"

echo "============================================"
echo "NeuralILT-DSCNN HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Environment setup
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> Setting up environment..."

    # try loading modules (common on HPC clusters)
    module load python/${PYTHON_VERSION} 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load cudnn 2>/dev/null || true
    module load anaconda3 2>/dev/null || true

    # create conda env if it doesn't exist
    if command -v conda &>/dev/null; then
        if ! conda env list | grep -q "^${ENV_NAME} "; then
            echo "Creating conda environment: ${ENV_NAME}"
            conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
        fi
        echo "Activating conda environment: ${ENV_NAME}"
        source activate "${ENV_NAME}" 2>/dev/null || conda activate "${ENV_NAME}"
    else
        # fallback to venv
        if [ ! -d ".venv" ]; then
            echo "Creating venv..."
            python3 -m venv .venv
        fi
        source .venv/bin/activate
    fi

    # install dependencies
    echo "Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt

    # verify setup
    python -c "
import torch
print(f'Python:  {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'Memory:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
    echo "Environment ready."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Dataset preparation
# ─────────────────────────────────────────────────────────────────────
prepare_data() {
    echo ""
    echo ">>> Preparing dataset..."

    # create data directories
    mkdir -p data/raw data/processed

    # check if raw data exists
    if [ -d "data/raw/MetalSet/target" ] && [ -d "data/raw/MetalSet/litho" ]; then
        echo "Raw dataset found at data/raw/MetalSet/"
    elif [ -n "$DATASET_PATH" ] && [ -d "$DATASET_PATH" ]; then
        echo "Copying dataset from $DATASET_PATH..."
        cp -r "$DATASET_PATH" data/raw/MetalSet
    elif [ -n "$DATASET_URL" ]; then
        echo "Downloading dataset from $DATASET_URL..."
        wget -q -O data/raw/lithobench.tar.gz "$DATASET_URL"
        tar -xzf data/raw/lithobench.tar.gz -C data/raw/
        rm data/raw/lithobench.tar.gz
    else
        echo "ERROR: No dataset found!"
        echo ""
        echo "Please do one of the following:"
        echo "  1. Place data at: data/raw/MetalSet/{target,litho}/"
        echo "  2. Set DATASET_PATH in this script to point to existing data"
        echo "  3. Set DATASET_URL to download the dataset"
        echo ""
        echo "You can copy data to the cluster with:"
        echo "  scp -r /local/path/MetalSet/ user@hpc:$(pwd)/data/raw/"
        exit 1
    fi

    # count files
    n_target=$(ls data/raw/MetalSet/target/ 2>/dev/null | wc -l)
    n_litho=$(ls data/raw/MetalSet/litho/ 2>/dev/null | wc -l)
    echo "Found $n_target target tiles, $n_litho litho tiles"

    # run preprocessing
    echo "Running preprocessing..."
    python -m src.data.preprocess

    # run train/val/test split
    echo "Splitting dataset..."
    python -m src.data.split_data

    echo "Dataset ready."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Training
# ─────────────────────────────────────────────────────────────────────
train_baseline() {
    echo ""
    echo ">>> Training baseline U-Net..."
    echo "    Config: configs/baseline.yaml"
    python -m src.train --config configs/baseline.yaml --data-config configs/data.yaml
    echo "Baseline training done."
}

train_dscnn() {
    echo ""
    echo ">>> Training DS-CNN U-Net..."
    echo "    Config: configs/dscnn.yaml"
    python -m src.train --config configs/dscnn.yaml --data-config configs/data.yaml
    echo "DS-CNN training done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Evaluation
# ─────────────────────────────────────────────────────────────────────
run_eval() {
    echo ""
    echo ">>> Evaluating and comparing models..."
    python -m src.evaluate --compare --output results/comparison.json
    echo "Evaluation done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Visualization
# ─────────────────────────────────────────────────────────────────────
run_visualize() {
    echo ""
    echo ">>> Generating plots..."
    python -m src.visualize --mode curves --log-dir results/logs \
        --output results/training_curves.png
    python -m src.visualize --mode efficiency --results results/comparison.json \
        --output results/efficiency.png
    echo "Plots saved to results/"
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    setup)
        setup_environment
        ;;
    preprocess)
        setup_environment
        prepare_data
        ;;
    baseline)
        setup_environment
        train_baseline
        ;;
    dscnn)
        setup_environment
        train_dscnn
        ;;
    eval)
        setup_environment
        run_eval
        ;;
    all)
        setup_environment
        prepare_data
        train_baseline
        train_dscnn
        run_eval
        run_visualize
        ;;
    *)
        echo "Usage: sbatch scripts/run_hpc.sh [setup|preprocess|baseline|dscnn|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo ""
echo "Results:"
ls -la results/ 2>/dev/null || echo "  (no results directory)"
echo ""
echo "To copy results to your local machine:"
echo "  scp -r $(whoami)@$(hostname):${PROJECT_DIR}/results/ ./results/"
echo "============================================"

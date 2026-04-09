#!/bin/bash
#SBATCH --job-name=neuralilt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# NeuralILT-DSCNN — SJSU CoE HPC Experiment Runner
#
# FIRST TIME SETUP (run on the login node — has internet):
#   bash scripts/run_hpc.sh setup
#
# Then submit experiments:
#   sbatch scripts/run_hpc.sh            # full pipeline
#   sbatch scripts/run_hpc.sh baseline   # train baseline only
#   sbatch scripts/run_hpc.sh dscnn      # train DS-CNN only
#   sbatch scripts/run_hpc.sh eval       # evaluation only
#
# SJSU HPC notes:
#   - Login node: GLIBC 2.17 (CentOS 7), has internet, no GCC
#   - GPU nodes:  GLIBC 2.17, no internet, have GPU
#   - /home is shared across all nodes
#   - Setup downloads pre-built wheels on login node (no compilation)
#   - Batch jobs use the venv created during setup

# NOTE: not using 'set -euo pipefail' because it causes silent failures
# when optional commands fail (e.g., PIL check, module load). We handle
# errors explicitly instead.
set -o pipefail

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
# If dataset is already somewhere on the cluster, set this path:
DATASET_PATH=""  # e.g., /shared/datasets/lithobench/MetalSet

# ─────────────────────────────────────────────────────────────────────
# PROJECT DIR
# ─────────────────────────────────────────────────────────────────────
if [ -f "scripts/run_hpc.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "run_hpc.sh" ]; then
    PROJECT_DIR="$(cd .. && pwd)"
else
    PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
fi
cd "$PROJECT_DIR"

mkdir -p logs results

VENV_DIR="${PROJECT_DIR}/venv"

echo "============================================"
echo "NeuralILT-DSCNN — HPC Job"
echo "============================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURM_NODELIST:-$(hostname)}"
echo "Project dir: $PROJECT_DIR"
echo "Date:        $(date)"
echo "============================================"

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT ACTIVATION (used by batch jobs — no internet needed)
# ─────────────────────────────────────────────────────────────────────
activate_env() {
    module load python3 2>/dev/null || true
    module load cuda 2>/dev/null || true
    module load cudnn 2>/dev/null || true

    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: $(python --version)"
    else
        echo "ERROR: venv not found at ${VENV_DIR}"
        echo "Run setup first on the login node:"
        echo "  bash scripts/run_hpc.sh setup"
        exit 1
    fi

    # project root on PYTHONPATH so 'import src' works
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

    # Help PyTorch manage GPU memory fragmentation on small GPUs
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    # GPU check — show actual errors so we can debug import failures
    echo ""
    echo "--- PyTorch Backend Check ---"
    echo "  Python: $(which python)"
    python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  Memory: {mem_gb:.1f} GB')
else:
    print('  WARNING: No GPU detected. Training will be slow.')
" || {
    echo "  WARNING: PyTorch import failed. Showing error:"
    python -c "import torch" 2>&1 | head -5
    echo "  Check that venv was set up correctly: bash scripts/run_hpc.sh setup"
}
    echo "-----------------------------"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────
# STEP 0: One-time setup (run on LOGIN NODE — has internet)
# ─────────────────────────────────────────────────────────────────────
setup_environment() {
    echo ""
    echo ">>> One-time environment setup"
    echo ">>> Run this on the LOGIN NODE (has internet access)"
    echo ""

    module load python3 2>/dev/null || true

    echo "Python: $(python3 --version 2>&1)"
    echo "Node:   $(hostname)"
    echo ""

    # Create venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: $(which python)"

    # Upgrade pip first — system pip may not handle manylinux2014 properly
    echo "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel

    echo ""
    echo "Installing dependencies..."
    echo ""

    # Download all wheels first (binary only, no source builds).
    # This is critical on SJSU HPC — no GCC on login node, no internet
    # on GPU nodes. We download everything here and install from cache.
    WHEEL_DIR="${PROJECT_DIR}/.wheels"
    mkdir -p "$WHEEL_DIR"

    echo "  Downloading binary wheels to .wheels/ ..."

    # PyTorch with CUDA — these are large (~2GB) but only downloaded once.
    # Using CUDA 12.1 wheels which are compatible with SJSU HPC.
    pip download --only-binary=:all: --dest "$WHEEL_DIR" \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

    # Other dependencies
    pip download --only-binary=:all: --dest "$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 Pillow==10.4.0 \
        scikit-image==0.22.0 matplotlib==3.9.2 \
        pyyaml==6.0.1 tqdm==4.66.5 pandas==2.2.2

    echo ""
    echo "  Installing from downloaded wheels..."

    # Install PyTorch first (from PyTorch index)
    pip install --no-index --find-links="$WHEEL_DIR" \
        torch==2.2.2 torchvision==0.17.2

    # Install everything else
    pip install --no-index --find-links="$WHEEL_DIR" \
        numpy==1.26.4 scipy==1.13.1 Pillow==10.4.0 \
        scikit-image==0.22.0 matplotlib==3.9.2 \
        pyyaml==6.0.1 tqdm==4.66.5 pandas==2.2.2

    # thop is pure Python, install directly
    pip install thop 2>/dev/null || echo "  (thop install skipped — optional)"

    # Install project in editable mode
    pip install -e . 2>/dev/null || echo "  (editable install skipped)"

    # Download MetalSet only (the main training data, ~2GB from Google Drive)
    # StdMetal/StdContact are optional and can be downloaded later if needed
    echo ""
    echo "Downloading LithoBench MetalSet dataset (~2GB)..."
    bash scripts/download_data.sh MetalSet

    # Verify environment
    echo ""
    echo "Verifying environment..."
    python scripts/verify_env.py || true

    echo ""
    echo "Setup complete. Ready to submit:"
    echo "  sbatch scripts/run_hpc.sh"
}

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Dataset preparation
# ─────────────────────────────────────────────────────────────────────
prepare_data() {
    echo ""
    echo ">>> Preparing dataset..."

    # Check if already processed AND images are the right size (256x256).
    # LithoBench tiles are 2048x2048 natively — if we find large images,
    # we need to re-preprocess to resize them.
    NEED_REPROCESS=false
    if [ -d "data/processed/MetalSet/layouts" ] && [ -d "data/processed/MetalSet/masks" ]; then
        n=$(ls data/processed/MetalSet/layouts/ 2>/dev/null | wc -l)
        if [ "$n" -gt 0 ]; then
            # check if images are already 256x256
            sample=$(ls data/processed/MetalSet/layouts/ | head -1)
            img_size=$(python -c "from PIL import Image; print(Image.open('data/processed/MetalSet/layouts/$sample').size[0])" 2>/dev/null || echo "0")
            if [ "$img_size" = "256" ]; then
                echo "[OK] Processed data exists (${n} tiles, ${img_size}x${img_size}) — skipping"
                if [ ! -f "data/processed/MetalSet/splits.json" ]; then
                    echo "Generating splits..."
                    python -m src.data.split_data
                fi
                return 0
            else
                echo "[WARN] Processed images are ${img_size}x${img_size}, need 256x256"
                echo "       Re-preprocessing to resize..."
                NEED_REPROCESS=true
                # remove old processed data and splits
                rm -rf data/processed/MetalSet
            fi
        fi
    fi

    mkdir -p data/raw data/processed

    # Check for raw data
    if [ -d "data/raw/MetalSet/target" ] && [ -d "data/raw/MetalSet/litho" ]; then
        echo "Raw dataset found at data/raw/MetalSet/"
    elif [ -n "$DATASET_PATH" ] && [ -d "$DATASET_PATH" ]; then
        echo "Copying dataset from $DATASET_PATH..."
        cp -r "$DATASET_PATH" data/raw/MetalSet
    else
        echo "Dataset not found locally. Attempting download..."
        bash scripts/download_data.sh MetalSet || {
            echo ""
            echo "Auto-download failed. Please download manually:"
            echo "  bash scripts/download_data.sh MetalSet  (on login node)"
            exit 1
        }
    fi

    n_target=$(ls data/raw/MetalSet/target/ 2>/dev/null | wc -l)
    n_litho=$(ls data/raw/MetalSet/litho/ 2>/dev/null | wc -l)
    echo "Found $n_target target tiles, $n_litho litho tiles"

    # preprocess MetalSet (primary training data)
    echo "Preprocessing MetalSet (resizing to 256x256)..."
    python -m src.data.preprocess --dataset MetalSet || {
        echo "ERROR: Preprocessing failed!"
        echo "Check that data/raw/MetalSet/target/ and data/raw/MetalSet/litho/ exist"
        ls -la data/raw/MetalSet/ 2>/dev/null || echo "  data/raw/MetalSet/ does not exist"
        exit 1
    }

    # preprocess generalization datasets if available
    python -m src.data.preprocess --dataset StdMetal 2>/dev/null || true
    python -m src.data.preprocess --dataset StdContact 2>/dev/null || true

    # split MetalSet into train/val/test
    echo "Splitting MetalSet (80/10/10)..."
    python -m src.data.split_data || {
        echo "ERROR: Split generation failed!"
        exit 1
    }

    echo "Dataset ready."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Training
# ─────────────────────────────────────────────────────────────────────
train_baseline() {
    echo ""
    echo ">>> Training baseline U-Net..."
    python -m src.train --config configs/baseline.yaml --data-config configs/data.yaml
    echo "Baseline training done."
}

train_dscnn() {
    echo ""
    echo ">>> Training DS-CNN U-Net..."
    python -m src.train --config configs/dscnn.yaml --data-config configs/data.yaml
    echo "DS-CNN training done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Evaluation
# ─────────────────────────────────────────────────────────────────────
run_eval() {
    echo ""
    echo ">>> Experiment 3: Evaluating and comparing models on MetalSet..."
    python -m src.evaluate --compare --output results/comparison.json
    echo "MetalSet evaluation done."
}

run_generalize() {
    echo ""
    echo ">>> Experiment 4: Generalization evaluation on StdMetal/StdContact..."
    python -m src.evaluate --generalize
    echo "Generalization evaluation done."
}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Visualization
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

run_analyze() {
    echo ""
    echo ">>> Analyzing dataset and results..."
    python scripts/analyze_data.py all
}

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    setup)
        # setup runs on login node (has internet) — NOT via sbatch
        setup_environment
        ;;
    preprocess)
        activate_env
        prepare_data
        ;;
    baseline)
        activate_env
        train_baseline
        ;;
    dscnn)
        activate_env
        train_dscnn
        ;;
    eval)
        activate_env
        run_eval
        ;;
    generalize)
        activate_env
        run_generalize
        ;;
    analyze)
        activate_env
        run_analyze
        ;;
    all)
        activate_env
        prepare_data
        train_baseline
        train_dscnn
        run_eval
        run_generalize
        run_visualize
        run_analyze
        ;;
    *)
        echo "Usage:"
        echo "  bash scripts/run_hpc.sh setup       # first time (login node)"
        echo "  sbatch scripts/run_hpc.sh            # full pipeline (GPU node)"
        echo "  sbatch scripts/run_hpc.sh baseline   # train baseline only"
        echo "  sbatch scripts/run_hpc.sh dscnn      # train DS-CNN only"
        echo "  sbatch scripts/run_hpc.sh eval       # Experiment 3: MetalSet comparison"
        echo "  sbatch scripts/run_hpc.sh generalize # Experiment 4: StdMetal/StdContact"
        echo "  sbatch scripts/run_hpc.sh analyze    # dataset + results analysis"
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
echo "  scp -r $(whoami)@$(hostname -f 2>/dev/null || hostname):${PROJECT_DIR}/results/ ./results/"
echo "============================================"

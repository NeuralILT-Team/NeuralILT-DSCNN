#!/bin/bash
# Learning rate sweep for DS-CNN model
# To use it: bash scripts/run_lr_sweep.sh

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Learning Rate Sweep"
echo "=========================================="

cd "$(dirname "$0")/.."

# Learning rates to try
LRS=("1e-3" "5e-4" "1e-4" "5e-5" "1e-5")

echo "Testing with different learning rates: ${LRS[*]}"

for lr in "${LRS[@]}"; do
    echo ""
    echo "Training with LR=${lr}..."
    python -m src.train \
        --config configs/dscnn.yaml \
        --sweep-lr "${lr}" \
        --data-config configs/data.yaml
done

echo ""
echo "[DONE] LR sweep complete."
echo "Check results/logs/dscnn/ for lr_* subdirectories"
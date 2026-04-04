#!/bin/bash
# Train the baseline U-Net model
#
# Usage:
#   bash scripts/run_baseline.sh

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Training Baseline U-Net"
echo "=========================================="

cd "$(dirname "$0")/.."

python -m src.train \
    --config configs/baseline.yaml \
    --data-config configs/data.yaml

echo ""
echo "[DONE] Baseline training complete."
echo "  Checkpoints: results/checkpoints/baseline/"
echo "  Logs:        results/logs/baseline/"

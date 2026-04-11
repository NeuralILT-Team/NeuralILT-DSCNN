#!/bin/bash
# Train the DS-CNN (Depthwise Separable) U-Net model
#
# Usage:
#   bash scripts/run_dscnn.sh

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Training DS-CNN U-Net"
echo "=========================================="

cd "$(dirname "$0")/.."

python -m src.train \
    --config configs/dscnn.yaml \
    --data-config configs/data.yaml

echo ""
echo "[DONE] DS-CNN training complete."
echo "  Checkpoints: results/checkpoints/dscnn/"
echo "  Logs:        results/logs/dscnn/"

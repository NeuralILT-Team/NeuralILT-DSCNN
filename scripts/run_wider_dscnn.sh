#!/bin/bash
# Wider DS-CNN experiment - try increasing channel dimensions
# Usage: bash scripts/run_wider_dscnn.sh

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Wider DS-CNN Experiment"
echo "=========================================="

cd "$(dirname "$0")/.."

# Different feature configurations to try
FEATURES=("64,128,256,512" "96,192,384,768" "128,256,512,1024")

echo "Testing feature dimensions: ${FEATURES[*]}"

for feat in "${FEATURES[@]}"; do
    echo ""
    echo "Training with features=${feat}..."
    python -m src.train \
        --config configs/dscnn.yaml \
        --sweep-features "${feat}" \
        --data-config configs/data.yaml
done

echo ""
echo "[DONE] Wider DS-CNN experiment complete."
echo "Check results/logs/dscnn/ for feat_* subdirectories"
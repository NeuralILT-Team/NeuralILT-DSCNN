#!/bin/bash
# Generate visualizations for NeuralILT results
#
# Usage:
#   bash scripts/run_visualize.sh curves      # Training loss curves
#   bash scripts/run_visualize.sh efficiency   # Efficiency comparison charts

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Visualization"
echo "=========================================="

cd "$(dirname "$0")/.."

MODE="${1:-curves}"

case "$MODE" in
    curves)
        echo "[INFO] Generating training curves..."
        python -m src.visualize \
            --mode curves \
            --log-dir results/logs \
            --output results/training_curves.png
        ;;
    efficiency)
        echo "[INFO] Generating efficiency comparison..."
        python -m src.visualize \
            --mode efficiency \
            --results results/comparison.json \
            --output results/efficiency_comparison.png
        ;;
    *)
        echo "Usage: $0 [curves|efficiency]"
        exit 1
        ;;
esac

echo ""
echo "[DONE] Visualization complete."

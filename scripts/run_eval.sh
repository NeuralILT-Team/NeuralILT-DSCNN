#!/bin/bash
# Evaluate and compare baseline vs DS-CNN models
#
# Usage:
#   bash scripts/run_eval.sh              # Compare both models
#   bash scripts/run_eval.sh baseline     # Evaluate baseline only
#   bash scripts/run_eval.sh dscnn        # Evaluate DS-CNN only

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Model Evaluation"
echo "=========================================="

cd "$(dirname "$0")/.."

MODE="${1:-compare}"

case "$MODE" in
    baseline)
        echo "[INFO] Evaluating baseline model..."
        python -m src.evaluate \
            --config configs/baseline.yaml \
            --checkpoint results/checkpoints/baseline/best_model.pt \
            --output results/eval_baseline.json
        ;;
    dscnn)
        echo "[INFO] Evaluating DS-CNN model..."
        python -m src.evaluate \
            --config configs/dscnn.yaml \
            --checkpoint results/checkpoints/dscnn/best_model.pt \
            --output results/eval_dscnn.json
        ;;
    compare)
        echo "[INFO] Comparing baseline vs DS-CNN..."
        python -m src.evaluate \
            --compare \
            --output results/comparison.json
        ;;
    *)
        echo "Usage: $0 [baseline|dscnn|compare]"
        exit 1
        ;;
esac

echo ""
echo "[DONE] Evaluation complete."

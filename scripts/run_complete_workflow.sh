#!/bin/bash
# Complete workflow: Train models, evaluate, and generate all figures
# Usage: bash scripts/run_complete_workflow.sh

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Complete Workflow"
echo "=========================================="

cd "$(dirname "$0")/.."

# Step 1: Train both models
echo "Step 1: Training models..."
bash scripts/run_baseline.sh
bash scripts/run_dscnn.sh

# Step 2: Evaluate and compare
echo "Step 2: Evaluating models..."
bash scripts/run_eval.sh compare

# Step 3: Generate figures
echo "Step 3: Generating figures..."
bash scripts/run_visualize.sh curves
bash scripts/run_visualize.sh efficiency
python -m src.visualize --mode predictions --checkpoint results/checkpoints/dscnn/best_model.pt --output results/prediction_grids.png

echo ""
echo "[DONE] Complete workflow finished!"
echo "Results in: results/"
echo "- Training curves: results/training_curves.png"
echo "- Efficiency charts: results/efficiency_comparison.png"
echo "- Prediction grids: results/prediction_grids.png"
#!/bin/bash
# Preprocess LithoBench MetalSet dataset
# Creates data/processed/MetalSet/{layouts,masks}/ from raw data
#
# Usage:
#   bash scripts/run_preprocess.sh              # Full dataset
#   MAX_SAMPLES=5000 bash scripts/run_preprocess.sh  # Subset for local dev

set -euo pipefail

echo "=========================================="
echo "NeuralILT-DSCNN: Data Preprocessing"
echo "=========================================="

cd "$(dirname "$0")/.."

# Run preprocessing
python -m src.data.preprocess

# Run train/val/test split
python -m src.data.split_data

echo ""
echo "[DONE] Preprocessing complete."
echo "  Processed data: data/processed/MetalSet/"
echo "  Split file:     data/processed/MetalSet/splits.json"

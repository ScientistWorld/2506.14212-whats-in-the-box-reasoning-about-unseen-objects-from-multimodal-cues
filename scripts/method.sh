#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-17}"
DATA_PATH="${DATA_PATH:-data/witb_parsed_benchmark.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

mkdir -p "$OUTPUT_DIR"
python method/witb_bayes.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_DIR/witb_full_predictions.json" \
  --mode full \
  --samples "$SAMPLES" \
  --seed "$SEED"

bash scripts/evaluate.sh "$OUTPUT_DIR" "$DATA_PATH"

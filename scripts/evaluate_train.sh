#!/bin/bash
# Visible train-slice evaluation entry point.
# Mirrors scripts/evaluate.sh: read prediction artifacts and write scores.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p scoring

PREDICTIONS_DIR="${1:-outputs}"
DATA_PATH="${2:-data/witb_parsed_benchmark.json}"

python eval/train/evaluate_witb_train.py \
  --data "$DATA_PATH" \
  --predictions-dir "$PREDICTIONS_DIR" \
  --scores scoring/scores_train.json

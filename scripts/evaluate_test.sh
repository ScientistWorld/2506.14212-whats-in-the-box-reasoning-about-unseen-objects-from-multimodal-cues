#!/bin/bash
# Held-out test-slice evaluation entry point.
# Mirrors scripts/evaluate.sh: read prediction artifacts and write scores.

set -e

cd /home/user

mkdir -p scoring

PREDICTIONS_DIR="${1:-outputs}"
DATA_PATH="${2:-data/witb_parsed_benchmark.json}"

python eval/test/evaluate_witb_test.py \
  --data "$DATA_PATH" \
  --predictions-dir "$PREDICTIONS_DIR" \
  --scores scoring/scores_test.json \
  --train-scores scoring/scores_train.json

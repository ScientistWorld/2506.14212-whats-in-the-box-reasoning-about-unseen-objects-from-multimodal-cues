#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-17}"
DATA_PATH="${DATA_PATH:-data/witb_parsed_benchmark.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
WHICH="${1:-all}"

mkdir -p "$OUTPUT_DIR"
python baseline/witb_baselines.py \
  --data "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --which "$WHICH"

bash scripts/evaluate.sh "$OUTPUT_DIR" "$DATA_PATH"

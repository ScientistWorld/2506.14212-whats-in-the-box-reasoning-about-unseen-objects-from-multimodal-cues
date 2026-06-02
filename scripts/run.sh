#!/bin/bash
# Regenerate predictions, then score full, visible train, and held-out test slices.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-17}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export SAMPLES SEED OUTPUT_DIR

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs
python validate.py --compare

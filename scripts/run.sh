#!/bin/bash
# Run the visible train and held-out test evaluation slices.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs

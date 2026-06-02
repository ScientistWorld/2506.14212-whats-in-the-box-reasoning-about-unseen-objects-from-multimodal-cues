#!/bin/bash
# Run the visible train and held-out test evaluation slices.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh outputs
bash scripts/evaluate_test.sh outputs

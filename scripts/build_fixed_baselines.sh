#!/bin/bash
# Designer tool: regenerate the fixed comparison predictions that the
# evaluators report under "_baselines".  Needs the hidden test file in
# eval/test/, so it only runs where the hidden test is present.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CACHE_ARGS=()
if [ -n "${WITB_CACHE_DIR:-}" ]; then
  CACHE_ARGS=(--cache-dir "$WITB_CACHE_DIR")
fi
python3 baseline/witb_baselines.py --data data/witb_train.json \
  --combined-out eval/train/fixed_baselines_train.json --samples 1000 --seed 17 \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}
if [ -f eval/test/witb_test_hidden.json ]; then
  python3 baseline/witb_baselines.py --data eval/test/witb_test_hidden.json \
    --combined-out eval/test/fixed_baselines_test.json --samples 1000 --seed 17 \
    ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}
fi

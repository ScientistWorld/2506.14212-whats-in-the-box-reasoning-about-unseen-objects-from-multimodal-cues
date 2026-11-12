#!/bin/bash
# Paper's full neurosymbolic model: Bayesian inference over all object-to-box
# placements with rejection-sampled visual fit (1000 samples) and per-box
# audio evidence (paper Eqs. 1-2).
#
# Predictor contract: bash method/predictors/<name>.sh INPUT_JSON OUTPUT_JSON
# INPUT_JSON has scenarios without labels; OUTPUT_JSON gets box_probs.
# Set WITB_CACHE_DIR to reuse visual-fit tables across runs (optional).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_ARGS=()
if [ -n "${WITB_CACHE_DIR:-}" ]; then
  CACHE_ARGS=(--cache-dir "$WITB_CACHE_DIR")
fi
python3 "$ROOT/method/witb_bayes.py" \
  --data "$1" \
  --output "$2" \
  --mode full \
  --samples 1000 \
  --seed 17 \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}

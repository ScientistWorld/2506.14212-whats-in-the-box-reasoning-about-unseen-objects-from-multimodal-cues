#!/bin/bash
# Run the paper's comparison baselines (audio-only Eq. 3, vision-only,
# late-fusion average) and a chance baseline on the visible train scenarios.
# Predictions go to outputs/baselines/.  The evaluators report the fixed copies
# of these baselines under "_baselines" in the score files.
#
# Usage: bash scripts/baseline.sh [audio_only_ablation|vision_only_ablation|late_fusion_average|uniform_chance|all]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
WHICH="${1:-all}"
CACHE_ARGS=()
if [ -n "${WITB_CACHE_DIR:-}" ]; then
  CACHE_ARGS=(--cache-dir "$WITB_CACHE_DIR")
fi
python3 baseline/witb_baselines.py \
  --data data/witb_train.json \
  --output-dir outputs/baselines \
  --which "$WHICH" \
  --samples 1000 \
  --seed 17 \
  ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}

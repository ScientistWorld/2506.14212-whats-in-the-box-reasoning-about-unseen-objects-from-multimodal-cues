#!/bin/bash
# End-to-end job: fixed comparison baselines, reproduction on visible data,
# hardening audit through the hidden test, then the reference-only score files
# (computed from scratch, as a fresh hidden-test container would).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
START=$(date +%s)
stamp() { echo "[run.sh] $(( $(date +%s) - START ))s: $*"; }

export WITB_CACHE_DIR="$ROOT/outputs/witb_cache"

stamp "fixed baselines (train + hidden test)"
bash scripts/build_fixed_baselines.sh

stamp "reproduction on visible data -> scoring/scores.json"
bash scripts/reproduce.sh

if [ -d redteam/hardening ] && [ -f eval/test/witb_test_hidden.json ]; then
  stamp "hardening audit (variants and shortcuts through the hidden test)"
  python3 redteam/hardening/audit.py all
fi

unset WITB_CACHE_DIR
stamp "reference-only train scores (no cache)"
bash scripts/evaluate_train.sh
stamp "reference-only hidden-test scores (no cache)"
bash scripts/evaluate_test.sh
stamp "done"

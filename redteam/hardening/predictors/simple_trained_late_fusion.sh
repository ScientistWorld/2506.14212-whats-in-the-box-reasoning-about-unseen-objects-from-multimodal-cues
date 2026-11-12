#!/bin/bash
# Audit predictor "simple_trained_late_fusion" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" trained_late_fusion "$1" "$2" --key trained_late_fusion

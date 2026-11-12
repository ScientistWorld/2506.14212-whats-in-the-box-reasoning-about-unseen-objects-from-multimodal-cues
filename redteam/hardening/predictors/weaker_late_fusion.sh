#!/bin/bash
# Audit predictor "weaker_late_fusion" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" late_fusion "$1" "$2"

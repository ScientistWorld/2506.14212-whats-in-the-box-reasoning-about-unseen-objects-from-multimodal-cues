#!/bin/bash
# Audit predictor "weaker_weight_only" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" fitted "$1" "$2" --key weight_only

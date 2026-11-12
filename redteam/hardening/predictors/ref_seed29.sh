#!/bin/bash
# Audit predictor "ref_seed29" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" full "$1" "$2" --seed 29

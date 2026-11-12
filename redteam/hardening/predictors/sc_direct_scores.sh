#!/bin/bash
# Audit predictor "sc_direct_scores" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" direct_scores "$1" "$2" --name sc_direct_scores

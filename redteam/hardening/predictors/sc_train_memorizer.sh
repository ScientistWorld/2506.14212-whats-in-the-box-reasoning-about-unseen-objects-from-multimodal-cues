#!/bin/bash
# Audit predictor "sc_train_memorizer" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" train_memorizer "$1" "$2"

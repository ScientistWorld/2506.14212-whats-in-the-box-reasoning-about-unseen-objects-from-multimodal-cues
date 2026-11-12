#!/bin/bash
# Audit predictor "sc_copy_train_labels" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" copy_train_labels "$1" "$2"

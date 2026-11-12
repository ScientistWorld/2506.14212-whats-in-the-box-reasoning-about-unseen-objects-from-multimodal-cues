#!/bin/bash
# Audit predictor "weaker_vision_only" (installed temporarily into method/predictors/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/redteam/hardening/variants/witb_variants.py" vision_only "$1" "$2"

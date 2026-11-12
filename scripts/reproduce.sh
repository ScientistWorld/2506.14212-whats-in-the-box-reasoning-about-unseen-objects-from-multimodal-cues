#!/bin/bash
# Reproduce the paper's comparison end to end on the visible data:
# the full model, its unimodal ablations and the chance baseline, then score
# every predictor on disk (scoring/scores.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/method.sh
bash scripts/baseline.sh all
bash scripts/evaluate.sh

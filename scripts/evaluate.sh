#!/bin/bash
# Development evaluation on the visible data: runs every predictor in
# method/predictors/*.sh on data/witb_train.json and writes scoring/scores.json
# (same numbers as scripts/evaluate_train.sh).  Takes no arguments.
#
# Evaluation code lives in eval/ and never imports method/.  A method plugs in
# only through the predictor contract:
#   bash method/predictors/<name>.sh INPUT_JSON OUTPUT_JSON
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p scoring
python3 eval/train/evaluate_witb_train.py --scores scoring/scores.json

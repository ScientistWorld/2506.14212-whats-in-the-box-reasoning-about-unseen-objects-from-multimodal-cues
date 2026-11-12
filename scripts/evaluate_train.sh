#!/bin/bash
# Visible-train evaluation: runs every predictor in method/predictors/*.sh on
# the visible train scenarios (data/witb_train.json) and writes
# scoring/scores_train.json.  Takes no arguments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 eval/train/evaluate_witb_train.py

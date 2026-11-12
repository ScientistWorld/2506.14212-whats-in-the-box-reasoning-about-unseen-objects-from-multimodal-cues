#!/bin/bash
# Hidden-test evaluation: runs every predictor in method/predictors/*.sh on the
# hidden test scenarios held in eval/test/ and writes scoring/scores_test.json.
# Takes no arguments.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 eval/test/evaluate_witb_test.py

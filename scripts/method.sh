#!/bin/bash
# Run the paper's full model on the visible train scenarios and keep its
# predictions in outputs/predictions/ for inspection.  Scoring happens in
# scripts/evaluate.sh, which calls every predictor in method/predictors/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/predictions
bash method/predictors/full_neurosymbolic.sh data/witb_train.json outputs/predictions/full_neurosymbolic_train.json

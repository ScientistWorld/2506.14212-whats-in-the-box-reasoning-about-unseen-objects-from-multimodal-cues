#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-17}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export SAMPLES SEED OUTPUT_DIR

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

bash scripts/baseline.sh all
bash scripts/method.sh
bash scripts/evaluate.sh "$OUTPUT_DIR" data/witb_parsed_benchmark.json

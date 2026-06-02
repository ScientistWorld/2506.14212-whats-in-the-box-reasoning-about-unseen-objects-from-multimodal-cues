#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This script exists for future portability. The scaled benchmark is tracked in
# git, so there is no external download required for this environment.

set -e

cd /home/user

# The original WiTB videos/human ratings are not publicly downloadable from the
# paper. The scaled parsed benchmark used here is tracked in git under data/.
test -f data/witb_parsed_benchmark.json

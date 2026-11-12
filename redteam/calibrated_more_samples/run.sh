#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 redteam/audit.py calibrated_more_samples

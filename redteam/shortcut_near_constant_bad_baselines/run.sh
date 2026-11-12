#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python3 redteam/audit.py shortcut_near_constant_bad_baselines

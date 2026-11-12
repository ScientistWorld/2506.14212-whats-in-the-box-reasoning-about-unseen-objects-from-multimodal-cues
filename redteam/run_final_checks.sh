#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash redteam/shortcut_near_constant/run.sh
bash redteam/shortcut_near_constant_bad_baselines/run.sh
python3 redteam/audit.py summarize

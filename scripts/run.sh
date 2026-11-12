#!/bin/bash
# Temporary CPU audit bundle. Restore this script after the audit job finishes.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash redteam/run_final_checks.sh

#!/usr/bin/env bash
# GPU space: no network, no package installation, no modifications to old runs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_offline.sh
dig_round2_output="${1:-runs/round2_20261105}"
echo 'Checking offline runtime; details: .runtime/round2-preflight.log'
python -u scripts/offline_setup.py gpu > .runtime/round2-preflight.log 2>&1
if [[ ! -f "$dig_round2_output/experiment.json" ]]; then
  python scripts/round2.py plan --output "$dig_round2_output"
fi
# Foreground on purpose: caller may nohup this entire script and keep one launcher.
python -u scripts/round2.py pipeline --output "$dig_round2_output"

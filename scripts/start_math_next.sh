#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_offline.sh
dig_math_output="${1:-runs/math_online_20261106}"
echo 'Checking offline runtime; details: .runtime/math-next-preflight.log'
python -u scripts/offline_setup.py gpu > .runtime/math-next-preflight.log 2>&1
if [[ ! -f "$dig_math_output/experiment.json" ]]; then
  python scripts/math_next.py plan --output "$dig_math_output"
fi
python -u scripts/math_next.py pipeline --output "$dig_math_output"

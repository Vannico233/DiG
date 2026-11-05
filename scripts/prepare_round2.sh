#!/usr/bin/env bash
# Networked CPU space only. Reuses model/runtime; preserves all old data/results.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate_offline.sh
exec 9>.runtime/round2-prepare.lock
flock -n 9 || { echo 'Round-two preparation already running.' >&2; exit 1; }
unset PIP_NO_INDEX HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=90
export SSL_CERT_FILE="$PWD/.runtime/certs/ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE" CURL_CA_BUNDLE="$SSL_CERT_FILE" PIP_CERT="$SSL_CERT_FILE"
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
Path('.runtime/round2-extra-constraints.txt').write_text('sympy==' + version('sympy') + '\n')
PY
python -m pip install -c requirements/h200-constraints.txt -c .runtime/round2-extra-constraints.txt 'math-verify[antlr4_13_2]==0.8.0'
python -m pip check
python scripts/round2.py prepare
# Fail before reserving GPUs if an equivalent non-numeric answer cannot be scored.
python - <<'PY'
from dig_rl.round2_protocol import score_answer
assert score_answer(r'\boxed{\sqrt{8}}', dict(answer=r'2\sqrt{2}', kind='math'))[0] == 1
print('Round-two mathematical scoring check passed.')
PY
python -m pip freeze > .runtime/pip-freeze.txt
python scripts/offline_setup.py seal
echo 'Round-two CPU assets ready. GPU entry: bash scripts/start_round2.sh'

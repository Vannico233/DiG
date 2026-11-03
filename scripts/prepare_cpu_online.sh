#!/usr/bin/env bash
# Run in the networked CPU space. All runtime files stay in the shared repository.
set -euo pipefail
cd "$(dirname "$0")/.."
DIG_ROOT="$(pwd -P)"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This installer requires Linux x86_64 (the server CPU space).' >&2
  exit 1
fi
if [[ "$DIG_ROOT" == *" "* ]]; then
  printf '%s\n' 'Use a shared repository path without spaces for the compiler environment.' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
# Allow re-running after a previous offline activation in the same terminal.
unset PIP_NO_INDEX HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export PIP_DISABLE_PIP_VERSION_CHECK=1
source "$DIG_ROOT/scripts/local_paths.sh"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=NO
# An interrupted or changed preparation must never leave an old success marker.
rm -f .runtime/CPU_READY.json

printf '%s\n' '[1/6] Preparing a separate Python 3.11 environment; /etc/inspire is unchanged.'
python3 scripts/offline_setup.py certificates
# Share the discovered trust roots across urllib, pip, HTTPX/HF, requests and micromamba.
export SSL_CERT_FILE="$DIG_ROOT/.runtime/certs/ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
export PIP_CERT="$SSL_CERT_FILE"
export MAMBA_SSL_VERIFY="$SSL_CERT_FILE"
export MAMBA_CACERT_PATH="$SSL_CERT_FILE"
python3 scripts/offline_setup.py micromamba
DIG_MAMBA="$DIG_ROOT/.runtime/tools/bin/micromamba"
DIG_PREFIX="$DIG_ROOT/.runtime/py311"
if [[ ! -x "$DIG_PREFIX/bin/python" ]]; then
  "$DIG_MAMBA" create -y --no-rc --override-channels -c conda-forge \
    -p "$DIG_PREFIX" python=3.11 pip c-compiler cxx-compiler
fi
DIG_PYTHON="$DIG_PREFIX/bin/python"
"$DIG_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Expected the dedicated Python 3.11 environment"'
export PATH="$DIG_PREFIX/bin:$PATH"
export CC="$DIG_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$DIG_PREFIX/bin/x86_64-conda-linux-gnu-c++"
if [[ ! -x "$CC" || ! -x "$CXX" ]]; then
  "$DIG_MAMBA" install -y --no-rc --override-channels -c conda-forge \
    -p "$DIG_PREFIX" python=3.11 c-compiler cxx-compiler
fi

printf '%s\n' '[2/6] Installing CUDA 12.8 PyTorch wheels (a GPU is NOT needed here).'
"$DIG_PYTHON" -m pip install --upgrade pip setuptools wheel packaging ninja
"$DIG_PYTHON" scripts/torch_wheels.py
"$DIG_PYTHON" -m pip install -c requirements/h200-constraints.txt -e '.[server,test]'

printf '%s\n' '[3/6] Downloading Qwen3.5-4B, GSM8K and MATH numeric; preparing fixed data splits.'
"$DIG_PYTHON" -m dig_rl.prepare --assets assets --config configs/qwen35_4b.json

printf '%s\n' '[4/6] Installing FLA and the official prebuilt causal-conv1d wheel; no nvcc build.'
"$DIG_PYTHON" scripts/offline_setup.py causal-wheel
"$DIG_PYTHON" -m pip install -c requirements/h200-constraints.txt 'fla-core[cuda]==0.6.0'
"$DIG_PYTHON" -m pip install -c requirements/h200-constraints.txt \
  .runtime/downloads/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
"$DIG_PYTHON" -m pip check

printf '%s\n' '[5/6] Recording the installed environment and checking CPU-safe project tests.'
"$DIG_PYTHON" -m pip freeze > .runtime/pip-freeze.txt
"$DIG_MAMBA" list --no-rc -p "$DIG_PREFIX" --explicit > .runtime/conda-explicit.txt
# The optional tiny-Qwen integration may bind GPU kernels once FLA is installed.
"$DIG_PYTHON" -m pytest -q --ignore=tests/test_qwen_integration.py

printf '%s\n' '[6/6] Checking the compiler, local tokenizer, model shards and all data hashes.'
"$DIG_PYTHON" scripts/offline_setup.py seal
printf '%s\n' 'CPU_READY: all shared files are prepared. On GPU: source scripts/activate_offline.sh'
printf '%s\n' 'Then: python scripts/offline_setup.py gpu (GPU compatibility is not yet verified).'

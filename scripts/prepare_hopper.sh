#!/usr/bin/env bash
# Called by the CPU preparation script under its preparation lock.
set -euo pipefail
cd "$(dirname "$0")/.."
DIG_ROOT="$(pwd -P)"
source scripts/local_paths.sh
export MAMBA_SSL_VERIFY="$DIG_ROOT/.runtime/certs/ca-bundle.pem"
export PIP_CERT="$MAMBA_SSL_VERIFY"
unset PIP_NO_INDEX
if [[ ! -x .runtime/cuda128/bin/nvcc ]]; then
  .runtime/tools/bin/micromamba create -y --no-rc --override-channels -c conda-forge \
    -p "$DIG_ROOT/.runtime/cuda128" cuda-nvcc=12.8 cuda-cudart-dev=12.8
fi
.runtime/py311/bin/python -m pip install -c requirements/h200-constraints.txt tilelang==0.1.15
.runtime/tools/bin/micromamba list --no-rc -p "$DIG_ROOT/.runtime/cuda128" --explicit > .runtime/cuda-explicit.txt

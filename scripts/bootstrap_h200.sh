#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,11), "Python >= 3.11 required"'
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
# Published matched PyTorch/torchvision wheels. Existing server drivers must support the selected CUDA build.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX"
python -m pip install -c requirements/h200-constraints.txt -e '.[server,test]'
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install -c requirements/h200-constraints.txt 'fla-core>=0.4,<1'
python -m pip install -c requirements/h200-constraints.txt 'causal-conv1d>=1.5.3,<2' --no-build-isolation
python -m pip check
python -m pip freeze > environment.lock.txt
# Installed fast kernels are CUDA-only; the optional tiny-Qwen test creates a CPU model.
python -m pytest -q --ignore=tests/test_qwen_integration.py
python -c 'import torch; from dig_rl.runtime import fast_kernel_report; print(fast_kernel_report()); assert torch.cuda.is_available(), "CUDA unavailable"'
printf '%s\n' 'Environment installed. Run prepare, preflight and smoke before audit/training. See docs/SERVER.md.'

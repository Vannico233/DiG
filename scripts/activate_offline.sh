# Source this file from the repository root in bash or zsh.
# No downloads, shell initialization changes, or modifications to /etc/inspire.
if [ ! -f scripts/offline_setup.py ] || [ ! -x .runtime/py311/bin/python ]; then
  printf '%s\n' 'Enter the shared DiG repository and finish prepare_cpu_online.sh first.' >&2
  return 1
fi
if [ ! -f .runtime/CPU_READY.json ]; then
  printf '%s\n' 'CPU preparation is incomplete: .runtime/CPU_READY.json is missing.' >&2
  return 1
fi
DIG_ROOT="$(pwd -P)"
export PATH="$DIG_ROOT/.runtime/py311/bin:$PATH"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export PYTHONPATH="$DIG_ROOT/src"
export CC="$DIG_ROOT/.runtime/py311/bin/x86_64-conda-linux-gnu-cc"
export CXX="$DIG_ROOT/.runtime/py311/bin/x86_64-conda-linux-gnu-c++"
export HF_HOME="$DIG_ROOT/assets/hf-cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=NO
export TRITON_CACHE_DIR="$DIG_ROOT/.runtime/triton-cache"
export TORCH_EXTENSIONS_DIR="$DIG_ROOT/.runtime/torch-extensions"
# Avoid a cached shell lookup of the old Python executable after changing PATH.
hash -r 2>/dev/null || true
printf '%s\n' 'Offline environment selected. Next: python scripts/offline_setup.py gpu'

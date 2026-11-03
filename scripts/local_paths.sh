# Sourced by preparation/activation after DIG_ROOT has been resolved.
# Keep project downloads, build scratch space and supported library caches local.
: "${DIG_ROOT:?DIG_ROOT must be the absolute repository directory}"
export TMPDIR="$DIG_ROOT/.runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$DIG_ROOT/.runtime/cache"
export PIP_CACHE_DIR="$DIG_ROOT/.runtime/pip-cache"
export PIP_SRC="$DIG_ROOT/.runtime/pip-src"
# An inherited target/prefix or pip.conf must not redirect installations elsewhere.
export PIP_CONFIG_FILE=/dev/null
export PIP_USER=0
unset PIP_TARGET PIP_PREFIX PIP_ROOT PIP_LOG
export MAMBA_ROOT_PREFIX="$DIG_ROOT/.runtime/mamba"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
export CONDA_ENVS_DIRS="$MAMBA_ROOT_PREFIX/envs"
# Micromamba rejects these aliases when both are set, even to the same path.
unset CONDA_ENVS_PATH
export HF_HOME="$DIG_ROOT/assets/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_MODULES_CACHE="$HF_HOME/modules"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export TORCH_HOME="$DIG_ROOT/.runtime/torch-cache"
export TORCH_EXTENSIONS_DIR="$DIG_ROOT/.runtime/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$DIG_ROOT/.runtime/torchinductor-cache"
export TRITON_CACHE_DIR="$DIG_ROOT/.runtime/triton-cache"
export CUDA_CACHE_PATH="$DIG_ROOT/.runtime/cuda-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE" \
  "$HF_MODULES_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

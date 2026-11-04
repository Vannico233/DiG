# Shared CUDA compiler for TileLang; does not activate another Python environment.
: "${DIG_ROOT:?DIG_ROOT must be set}"
export CUDA_HOME="$DIG_ROOT/.runtime/cuda128"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export NVCC_CCBIN="$CUDA_HOME/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$NVCC_CCBIN"
# TileLang passes CXX as -ccbin, so NVCC_CCBIN alone is insufficient.
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-cc"
export CXX="$NVCC_CCBIN"
export TILELANG_CACHE_DIR="$DIG_ROOT/.runtime/tilelang-cache"
mkdir -p "$TILELANG_CACHE_DIR"

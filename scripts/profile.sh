#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 2 ]]; then
  echo "usage: $0 nsys|ncu <command> [args...]" >&2
  echo "examples:" >&2
  echo "  $0 nsys python -m g4b --gguf /path/model.gguf" >&2
  echo "  $0 ncu  python tests/compare_scripts/test_matmul_epilogue.py" >&2
  exit 2
fi

mode="$1"
shift

source .venv/bin/activate

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
export G4B_PROFILE=1

# nsys requires running as root, but we don't want /root/.triton as the triton cache
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/.triton-cache}"

# nsys will hang if `/usr/bin/file` is execve'd because of seccomp, which the python stdlib does -> shadow it in PATH
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
nsys_shims_dir="$script_dir/nsys_shims"
export PATH="$nsys_shims_dir:$PATH"

case "$mode" in
  nsys)
    exec nsys profile \
      --trace=cuda,nvtx,osrt,cublas \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --cuda-graph-trace=node \
      --gpu-metrics-devices=all \
      --stats=true \
      --force-overwrite=true \
      -o /tmp/profile \
      -- env "$@"
    ;;

  ncu)
    # Use sudo for ncu, but preserve the activated venv via PATH/VIRTUAL_ENV.
    # No echo | sudo bash; args are passed safely.
    exec sudo -E env \
      PATH="$PATH" \
      VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
      PYTHONPATH="$PYTHONPATH" \
      PYTHONUNBUFFERED="$PYTHONUNBUFFERED" \
      ncu \
        --set full \
        -o /tmp/profile \
        --import-source=yes \
        -f \
        -- "$@"
    ;;

  *)
    echo "error: first arg must be 'nsys' or 'ncu', got '$mode'" >&2
    exit 2
    ;;
esac

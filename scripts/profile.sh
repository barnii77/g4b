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

case "$mode" in
  nsys)
    exec nsys profile \
      --trace=cuda \
      --sample=none \
      --cpuctxsw=none \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --cuda-graph-trace=node \
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

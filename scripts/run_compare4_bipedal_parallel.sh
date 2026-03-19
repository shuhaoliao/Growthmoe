#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESET="full"
SEED="42"
DEVICE="cuda"
OUTPUT_ROOT="runs"
TAG="compare4_bipedal_parallel"
PYTHON_BIN="python"
SKIP_SUMMARY="0"

usage() {
  cat <<'EOF'
Run the 4 mixed-terrain BipedalWalker experiments in parallel.

Usage:
  bash scripts/run_compare4_bipedal_parallel.sh [options]

Options:
  --preset <quick|full>     Training budget preset. Default: full
  --seed <int>              Random seed. Default: 42
  --device <cpu|cuda>       Device passed to train.py. Default: cuda
  --output-root <path>      Root directory for run groups. Default: runs
  --tag <name>              Group tag in the timestamped folder. Default: compare4_bipedal_parallel
  --python <bin>            Python executable. Default: python
  --skip-summary            Skip summarize_results.py at the end
  -h, --help                Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-summary)
      SKIP_SUMMARY="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

CMD=(
  bash "${SCRIPT_DIR}/run_compare4_parallel.sh"
  --preset "${PRESET}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --tag "${TAG}"
  --python "${PYTHON_BIN}"
  --env "bipedal_diverse"
)

if [[ "${SKIP_SUMMARY}" == "1" ]]; then
  CMD+=(--skip-summary)
fi

exec "${CMD[@]}"

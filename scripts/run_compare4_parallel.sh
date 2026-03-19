#!/usr/bin/env bash
set -euo pipefail

PRESET="full"
SEED="42"
DEVICE="cuda"
OUTPUT_ROOT="runs"
TAG="compare4_parallel"
ENV_NAME="multi_region_nav"
PYTHON_BIN="python"
SKIP_SUMMARY="0"

usage() {
  cat <<'EOF'
Run the 4 experiments in parallel and keep the same timestamped layout expected by summarize_results.py.

Usage:
  bash scripts/run_compare4_parallel.sh [options]

Options:
  --preset <quick|full>     Training budget preset. Default: full
  --seed <int>              Random seed. Default: 42
  --device <cpu|cuda>       Device passed to train.py. Default: cuda
  --env <name>              Environment name. Default: multi_region_nav
  --output-root <path>      Root directory for run groups. Default: runs
  --tag <name>              Group tag in the timestamped folder. Default: compare4_parallel
  --python <bin>            Python executable. Default: python
  --skip-summary            Skip summarize_results.py at the end
  -h, --help                Show this message

Example:
  bash scripts/run_compare4_parallel.sh --preset full --device cuda --tag server_full
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
    --env)
      ENV_NAME="$2"
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GROUP_DIR="${OUTPUT_ROOT}/${TIMESTAMP}_${TAG}_seed${SEED}"
mkdir -p "${GROUP_DIR}"

echo "Starting parallel run group: ${GROUP_DIR}"
echo "Preset=${PRESET} Device=${DEVICE} Seed=${SEED} Env=${ENV_NAME}"

declare -a PIDS=()
declare -a NAMES=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM

run_exp() {
  local exp_name="$1"
  local stage_name="$2"
  local run_dir="${GROUP_DIR}/${exp_name}"
  local console_log="${GROUP_DIR}/${exp_name}.console.log"

  echo "[launch] ${exp_name} stage=${stage_name} log=${console_log}"
  "${PYTHON_BIN}" train.py \
    --exp "${exp_name}" \
    --preset "${PRESET}" \
    --stage "${stage_name}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --env "${ENV_NAME}" \
    --run-dir "${run_dir}" \
    > "${console_log}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${exp_name}")
}

FULL_STAGE="all"
if [[ "${ENV_NAME}" == "bipedal_diverse" ]]; then
  FULL_STAGE="acquisition"
fi

run_exp "baseline" "acquisition"
run_exp "gpo_only" "acquisition"
run_exp "moe_only" "acquisition"
run_exp "full" "${FULL_STAGE}"

for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  name="${NAMES[$idx]}"
  if wait "${pid}"; then
    echo "[done] ${name}"
  else
    echo "[fail] ${name} exited with a non-zero status" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

group_dir = Path(r"${GROUP_DIR}")
manifest = {
    "timestamp": "${TIMESTAMP}",
    "group_dir": str(group_dir),
    "device": "${DEVICE}",
    "preset": "${PRESET}",
    "seed": int("${SEED}"),
    "env": "${ENV_NAME}",
    "runs": {
        "baseline": str(group_dir / "baseline"),
        "gpo_only": str(group_dir / "gpo_only"),
        "moe_only": str(group_dir / "moe_only"),
        "full": str(group_dir / "full"),
    },
}
with (group_dir / "manifest.json").open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(group_dir)
PY

if [[ "${SKIP_SUMMARY}" != "1" ]]; then
  "${PYTHON_BIN}" summarize_results.py --group-dir "${GROUP_DIR}" --device "${DEVICE}" | tee "${GROUP_DIR}/summary.stdout.log"
fi

echo "Finished. Group directory: ${GROUP_DIR}"

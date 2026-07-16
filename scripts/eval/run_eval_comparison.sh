#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG_PYTHON_BIN="${CONFIG_PYTHON_BIN:-uv run python}"
read -r -a CONFIG_PYTHON_CMD <<< "$CONFIG_PYTHON_BIN"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/safeanywhere_v1.yaml}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      EVAL_CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      EXTRA_ARGS+=(--dry-run)
      shift
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      echo "Usage: bash scripts/eval/run_eval_comparison.sh [--config configs/eval/safeanywhere_v1.yaml] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

exec "${CONFIG_PYTHON_CMD[@]}" scripts/eval/run_eval_matrix.py --config "$EVAL_CONFIG" "${EXTRA_ARGS[@]}"

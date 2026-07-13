#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKERS="${WORKERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
MOCK="${MOCK:-0}"
RESUME="${RESUME:-0}"
QUIET="${QUIET:-1}"
VERBOSE="${VERBOSE:-0}"

HEX_CONFIG="${HEX_CONFIG:-configs/hex_phi_prefix_1200.yaml}"
SAFECHAIN_PREFIX_CONFIG="${SAFECHAIN_PREFIX_CONFIG:-configs/safechain_prefix_1600.yaml}"
SAFECHAIN_CONFIG="${SAFECHAIN_CONFIG:-configs/safechain_pilot_1k.yaml}"
SAFECHAIN_DIR="${SAFECHAIN_DIR:-build/safechain_pilot_1k}"
HEX_PREFIX_DIR="${HEX_PREFIX_DIR:-build/hex_phi_prefix_1200}"
SAFECHAIN_PREFIX_DIR="${SAFECHAIN_PREFIX_DIR:-build/safechain_prefix_1600}"
OUTPUT_DIR="${OUTPUT_DIR:-build/mixed_safechain1k_prefix2800}"
TRAIN_YAML="${TRAIN_YAML:-train/llamafactory/dataset_safeanywhere_prefix2800_train.yaml}"
VAL_YAML="${VAL_YAML:-train/llamafactory/dataset_safeanywhere_prefix2800_val.yaml}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/build_prefix2800.log}"

mkdir -p "$(dirname "$LOG_FILE")"
if [[ "$VERBOSE" != "1" ]]; then
  : > "$LOG_FILE"
  echo "Log: $LOG_FILE"
fi

run_python() {
  if [[ "$VERBOSE" == "1" ]]; then
    $PYTHON_BIN "$@"
  else
    if ! $PYTHON_BIN "$@" >>"$LOG_FILE" 2>&1; then
      echo "Command failed: $PYTHON_BIN $*" >&2
      echo "See log: $LOG_FILE" >&2
      return 1
    fi
  fi
}

run_args=()
if [[ "$MOCK" == "1" ]]; then
  run_args+=(--mock)
fi
if [[ "$RESUME" == "1" ]]; then
  run_args+=(--resume)
fi
if [[ "$QUIET" == "1" ]]; then
  run_args+=(--quiet)
fi

if [[ ! -s "$SAFECHAIN_DIR/sft_train.jsonl" || ! -s "$SAFECHAIN_DIR/sft_val.jsonl" ]]; then
  echo "[1/6] Build SafeChain cold-start 1000"
  run_python scripts/data/build_safechain.py \
    --config "$SAFECHAIN_CONFIG" \
    --workers "$WORKERS" \
    "${run_args[@]}"
else
  echo "[1/6] SafeChain cold-start exists: $SAFECHAIN_DIR"
fi

echo "[2/6] Build HEx-PHI dangerous-prefix 1200"
run_python scripts/data/build_hex_phi_prefix.py \
  --config "$HEX_CONFIG" \
  --workers "$WORKERS" \
  "${run_args[@]}"

echo "[3/6] Build SafeChain dangerous-prefix 1600"
run_python scripts/data/build_safechain_prefix.py \
  --config "$SAFECHAIN_PREFIX_CONFIG" \
  --workers "$WORKERS" \
  "${run_args[@]}"

echo "[4/6] Merge SafeChain cold-start + prefix datasets"
run_python scripts/data/merge_sft.py \
  --safechain-train "$SAFECHAIN_DIR/sft_train.jsonl" \
  --safechain-val "$SAFECHAIN_DIR/sft_val.jsonl" \
  --prefix-train "$HEX_PREFIX_DIR/sft_train.jsonl" \
  --prefix-val "$HEX_PREFIX_DIR/sft_val.jsonl" \
  --prefix-train "$SAFECHAIN_PREFIX_DIR/sft_train.jsonl" \
  --prefix-val "$SAFECHAIN_PREFIX_DIR/sft_val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --no-strict-counts

echo "[5/6] Export LLaMA-Factory span-mask JSONL"
run_python scripts/data/export_llamafactory_v1.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-dataset-yaml "$TRAIN_YAML" \
  --val-dataset-yaml "$VAL_YAML"

echo "[6/6] Validate exported span-mask structure"
run_python scripts/data/validate_llamafactory_masks.py \
  --structure-only \
  --train "$OUTPUT_DIR/train_lf_v1_spanmasked.jsonl" \
  --val "$OUTPUT_DIR/val_lf_v1_spanmasked.jsonl"

echo "Done. Reports:"
echo "  $SAFECHAIN_DIR/report.json"
echo "  $HEX_PREFIX_DIR/report.json"
echo "  $SAFECHAIN_PREFIX_DIR/report.json"
echo "  $OUTPUT_DIR/report.json"
echo "  $OUTPUT_DIR/llamafactory_v1_export_report.json"
if [[ "$VERBOSE" != "1" ]]; then
  echo "  $LOG_FILE"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_DIR="${EVAL_DIR:-build/eval/safeanywhere_v1}"
EVAL_INPUT="${EVAL_INPUT:-build/mixed_safechain1k_prefix500/sft_val.jsonl}"
BASE_MODEL="${BASE_MODEL:-../models/Qwen3-0.6B}"
CANDIDATE_ADAPTER="${CANDIDATE_ADAPTER:-runs/qwen3_safeanywhere_lora_1500_v1}"
CANDIDATE_NAME="${CANDIDATE_NAME:-sft}"
BASELINE_ADAPTER="${BASELINE_ADAPTER:-}"
BASELINE_NAME="${BASELINE_NAME:-baseline_sft}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
DTYPE="${DTYPE:-bf16}"

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Missing base model: $BASE_MODEL" >&2
  exit 1
fi
if [[ ! -f "$CANDIDATE_ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing candidate adapter: $CANDIDATE_ADAPTER" >&2
  exit 1
fi
if [[ -n "$BASELINE_ADAPTER" && ! -f "$BASELINE_ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing baseline adapter: $BASELINE_ADAPTER" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"

echo "[1/8] Build eval set"
"$PYTHON_BIN" scripts/06_build_eval_sets.py \
  --input "$EVAL_INPUT" \
  --output-dir "$EVAL_DIR" \
  > "$EVAL_DIR/build_eval_set.log" 2>&1

generate_and_score() {
  local name="$1"
  local adapter="${2:-}"
  local -a adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi

  echo "Generate $name"
  "$PYTHON_BIN" scripts/07_generate_eval_responses.py \
    --eval-file "$EVAL_DIR/safeanywhere_eval.jsonl" \
    --base-model "$BASE_MODEL" \
    "${adapter_args[@]}" \
    --output "$EVAL_DIR/${name}_predictions.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --dtype "$DTYPE" \
    > "$EVAL_DIR/${name}_generate.log" 2>&1

  echo "Score $name"
  "$PYTHON_BIN" scripts/08_score_eval_results.py \
    --input "$EVAL_DIR/${name}_predictions.jsonl" \
    --scored-output "$EVAL_DIR/${name}_predictions_scored.jsonl" \
    --summary-output "$EVAL_DIR/${name}_score_summary.json" \
    > "$EVAL_DIR/${name}_score.log" 2>&1
}

echo "[2/8] Base generation and scoring"
generate_and_score base

if [[ -n "$BASELINE_ADAPTER" ]]; then
  echo "[3/8] Baseline adapter generation and scoring"
  generate_and_score "$BASELINE_NAME" "$BASELINE_ADAPTER"
else
  echo "[3/8] Baseline adapter skipped"
fi

echo "[4/8] Candidate adapter generation and scoring"
generate_and_score "$CANDIDATE_NAME" "$CANDIDATE_ADAPTER"

echo "[5/8] Compare base vs candidate"
"$PYTHON_BIN" scripts/09_compare_eval_reports.py \
  --base "$EVAL_DIR/base_score_summary.json" \
  --candidate "$EVAL_DIR/${CANDIDATE_NAME}_score_summary.json" \
  > "$EVAL_DIR/compare_base_vs_${CANDIDATE_NAME}.md"

if [[ -n "$BASELINE_ADAPTER" ]]; then
  echo "[6/8] Compare baseline vs candidate"
  "$PYTHON_BIN" scripts/09_compare_eval_reports.py \
    --base "$EVAL_DIR/${BASELINE_NAME}_score_summary.json" \
    --candidate "$EVAL_DIR/${CANDIDATE_NAME}_score_summary.json" \
    > "$EVAL_DIR/compare_${BASELINE_NAME}_vs_${CANDIDATE_NAME}.md"
else
  echo "[6/8] Baseline comparison skipped"
fi

echo "[7/8] Write README"
"$PYTHON_BIN" scripts/14_write_eval_readme.py \
  --eval-dir "$EVAL_DIR" \
  --candidate-name "$CANDIDATE_NAME" \
  --candidate-adapter "$CANDIDATE_ADAPTER" \
  --baseline-name "$BASELINE_NAME" \
  --baseline-adapter "$BASELINE_ADAPTER" \
  --base-model "$BASE_MODEL"

echo "[8/8] Done: $EVAL_DIR/README.md"

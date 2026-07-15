#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG_PYTHON_BIN="${CONFIG_PYTHON_BIN:-uv run python}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_PYTHON_BIN="${JUDGE_PYTHON_BIN:-uv run python}"
read -r -a CONFIG_PYTHON_CMD <<< "$CONFIG_PYTHON_BIN"
read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
read -r -a JUDGE_PYTHON_CMD <<< "$JUDGE_PYTHON_BIN"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval/safeanywhere_v1.yaml}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      EVAL_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      echo "Usage: bash scripts/eval/run_eval_comparison.sh [--config configs/eval/safeanywhere_v1.yaml]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$EVAL_CONFIG" ]]; then
  echo "Missing eval config: $EVAL_CONFIG" >&2
  exit 1
fi
eval "$("${CONFIG_PYTHON_CMD[@]}" scripts/eval/emit_eval_env.py --config "$EVAL_CONFIG")"

EVAL_DIR="${EVAL_DIR:-build/data_build/eval/safeanywhere_v1_1532}"
EVAL_INPUT="${EVAL_INPUT:-build/data_build/safeanywhere_sft_v1/sft_val.jsonl}"
EVAL_SET_SUBDIR="${EVAL_SET_SUBDIR:-eval_set}"
RUNS_SUBDIR="${RUNS_SUBDIR:-runs}"
COMPARISONS_SUBDIR="${COMPARISONS_SUBDIR:-comparisons}"
EVAL_SET_DIR="$EVAL_DIR/$EVAL_SET_SUBDIR"
RUNS_DIR="$EVAL_DIR/$RUNS_SUBDIR"
COMPARISONS_DIR="$EVAL_DIR/$COMPARISONS_SUBDIR"
BASE_MODEL="${BASE_MODEL:-../models/Qwen3-0.6B}"
CANDIDATE_ADAPTER="${CANDIDATE_ADAPTER:-runs/qwen3_safeanywhere_lora_sft_v1}"
CANDIDATE_NAME="${CANDIDATE_NAME:-sft}"
BASELINE_ADAPTER="${BASELINE_ADAPTER:-}"
BASELINE_NAME="${BASELINE_NAME:-baseline_sft}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
THINKING_MODE="${THINKING_MODE:-auto}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
GENERATION_LIMIT="${GENERATION_LIMIT:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
OUTPUT_MODE="${OUTPUT_MODE:-compact}"
INCLUDE_REFERENCE="${INCLUDE_REFERENCE:-0}"
INCLUDE_RAW_PREDICTION="${INCLUDE_RAW_PREDICTION:-0}"
INCLUDE_RENDERED_PROMPT="${INCLUDE_RENDERED_PROMPT:-0}"
MAX_PER_TASK="${MAX_PER_TASK:-}"
TASK_LIMITS="${TASK_LIMITS:-}"
INCLUDE_PREFIX_DIRECT="${INCLUDE_PREFIX_DIRECT:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-}"
JUDGE_BASE_URL_ENV="${JUDGE_BASE_URL_ENV:-}"
JUDGE_MODEL_ENV="${JUDGE_MODEL_ENV:-}"
JUDGE_MAX_FIELD_CHARS="${JUDGE_MAX_FIELD_CHARS:-}"
SKIP_EXISTING_GENERATIONS="${SKIP_EXISTING_GENERATIONS:-1}"
SKIP_EXISTING_SCORES="${SKIP_EXISTING_SCORES:-1}"

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

mkdir -p "$EVAL_SET_DIR" "$RUNS_DIR" "$COMPARISONS_DIR"

predictions_match_generation_config() {
  local file="$1"
  "${PYTHON_CMD[@]}" - "$file" "$MAX_NEW_TOKENS" "$TEMPERATURE" "$TOP_P" "$THINKING_MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_max_new_tokens = int(sys.argv[2])
expected_temperature = float(sys.argv[3])
expected_top_p = float(sys.argv[4])
expected_thinking_mode = sys.argv[5]

if not path.exists():
    raise SystemExit(1)

for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    cfg = row.get("generation_config") or {}
    if int(cfg.get("max_new_tokens", -1)) != expected_max_new_tokens:
        raise SystemExit(1)
    if not math.isclose(float(cfg.get("temperature", -1)), expected_temperature, rel_tol=0, abs_tol=1e-9):
        raise SystemExit(1)
    if not math.isclose(float(cfg.get("top_p", -1)), expected_top_p, rel_tol=0, abs_tol=1e-9):
        raise SystemExit(1)
    if str(cfg.get("thinking_mode", "auto")) != expected_thinking_mode:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

echo "[1/8] Build eval set"
build_args=()
if [[ -n "$MAX_PER_TASK" ]]; then
  build_args+=(--max-per-task "$MAX_PER_TASK")
fi
if [[ -n "$TASK_LIMITS" ]]; then
  IFS=',' read -r -a task_limit_items <<< "$TASK_LIMITS"
  for item in "${task_limit_items[@]}"; do
    if [[ -n "$item" ]]; then
      build_args+=(--task-limit "$item")
    fi
  done
fi
if [[ "$INCLUDE_PREFIX_DIRECT" == "0" || "$INCLUDE_PREFIX_DIRECT" == "false" || "$INCLUDE_PREFIX_DIRECT" == "False" ]]; then
  build_args+=(--no-prefix-direct)
fi
"${PYTHON_CMD[@]}" scripts/eval/build_eval_sets.py \
  --input "$EVAL_INPUT" \
  --output-dir "$EVAL_SET_DIR" \
  "${build_args[@]}" \
  > "$EVAL_SET_DIR/build.log" 2>&1

generate_and_score() {
  local name="$1"
  local adapter="${2:-}"
  local run_dir="$RUNS_DIR/$name"
  local predictions_file="$run_dir/predictions.jsonl"
  local scored_file="$run_dir/predictions_scored.jsonl"
  local summary_file="$run_dir/score_summary.json"
  local eval_count
  local prediction_count
  local scored_count
  local -a adapter_args=()
  local -a judge_args=()
  local -a generation_args=()
  mkdir -p "$run_dir"
  eval_count="$(wc -l < "$EVAL_SET_DIR/safeanywhere_eval.jsonl")"
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  if [[ -n "$JUDGE_MODEL" ]]; then
    judge_args+=(--model "$JUDGE_MODEL")
  fi
  if [[ -n "$JUDGE_API_KEY_ENV" ]]; then
    judge_args+=(--api-key-env "$JUDGE_API_KEY_ENV")
  fi
  if [[ -n "$JUDGE_BASE_URL_ENV" ]]; then
    judge_args+=(--base-url-env "$JUDGE_BASE_URL_ENV")
  fi
  if [[ -n "$JUDGE_MODEL_ENV" ]]; then
    judge_args+=(--model-env "$JUDGE_MODEL_ENV")
  fi
  if [[ -n "$JUDGE_MAX_FIELD_CHARS" ]]; then
    judge_args+=(--max-field-chars "$JUDGE_MAX_FIELD_CHARS")
  fi
  generation_args=(
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --thinking-mode "$THINKING_MODE"
    --dtype "$DTYPE"
    --device-map "$DEVICE_MAP"
    --output-mode "$OUTPUT_MODE"
  )
  if [[ -n "$GENERATION_LIMIT" ]]; then
    generation_args+=(--limit "$GENERATION_LIMIT")
  fi
  if [[ -n "$SYSTEM_PROMPT" ]]; then
    generation_args+=(--system-prompt "$SYSTEM_PROMPT")
  fi
  if [[ "$INCLUDE_REFERENCE" == "1" || "$INCLUDE_REFERENCE" == "true" || "$INCLUDE_REFERENCE" == "True" ]]; then
    generation_args+=(--include-reference)
  fi
  if [[ "$INCLUDE_RAW_PREDICTION" == "1" || "$INCLUDE_RAW_PREDICTION" == "true" || "$INCLUDE_RAW_PREDICTION" == "True" ]]; then
    generation_args+=(--include-raw-prediction)
  fi
  if [[ "$INCLUDE_RENDERED_PROMPT" == "1" || "$INCLUDE_RENDERED_PROMPT" == "true" || "$INCLUDE_RENDERED_PROMPT" == "True" ]]; then
    generation_args+=(--include-rendered-prompt)
  fi

  prediction_count=0
  if [[ -f "$predictions_file" ]]; then
    prediction_count="$(wc -l < "$predictions_file")"
  fi
  if [[ "$SKIP_EXISTING_GENERATIONS" == "1" && "$prediction_count" == "$eval_count" ]] \
    && predictions_match_generation_config "$predictions_file"; then
    echo "Generate $name skipped: $predictions_file already has $prediction_count/$eval_count rows"
  else
    echo "Generate $name"
    rm -f "$scored_file" "$summary_file"
    "${PYTHON_CMD[@]}" scripts/eval/generate_responses.py \
      --eval-file "$EVAL_SET_DIR/safeanywhere_eval.jsonl" \
      --base-model "$BASE_MODEL" \
      "${adapter_args[@]}" \
      --output "$predictions_file" \
      "${generation_args[@]}" \
      > "$run_dir/generate.log" 2>&1
  fi

  prediction_count="$(wc -l < "$predictions_file")"
  if [[ "$prediction_count" != "$eval_count" ]]; then
    echo "Prediction count mismatch for $name: $prediction_count predictions vs $eval_count eval rows" >&2
    exit 1
  fi

  scored_count=0
  if [[ -f "$scored_file" ]]; then
    scored_count="$(wc -l < "$scored_file")"
  fi
  if [[ "$SKIP_EXISTING_SCORES" == "1" && -f "$summary_file" && "$scored_count" == "$prediction_count" ]]; then
    echo "Score $name skipped: $scored_file already has $scored_count/$prediction_count rows"
  else
    echo "Score $name"
    "${JUDGE_PYTHON_CMD[@]}" scripts/eval/score_llm_judge.py \
      --input "$predictions_file" \
      --scored-output "$scored_file" \
      --summary-output "$summary_file" \
      "${judge_args[@]}" \
      > "$run_dir/score.log" 2>&1
  fi
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
"${PYTHON_CMD[@]}" scripts/eval/compare_reports.py \
  --base "$RUNS_DIR/base/score_summary.json" \
  --candidate "$RUNS_DIR/${CANDIDATE_NAME}/score_summary.json" \
  > "$COMPARISONS_DIR/base_vs_${CANDIDATE_NAME}.md"

if [[ -n "$BASELINE_ADAPTER" ]]; then
  echo "[6/8] Compare baseline vs candidate"
  "${PYTHON_CMD[@]}" scripts/eval/compare_reports.py \
    --base "$RUNS_DIR/${BASELINE_NAME}/score_summary.json" \
    --candidate "$RUNS_DIR/${CANDIDATE_NAME}/score_summary.json" \
    > "$COMPARISONS_DIR/${BASELINE_NAME}_vs_${CANDIDATE_NAME}.md"
else
  echo "[6/8] Baseline comparison skipped"
fi

echo "[7/8] Write README"
"${PYTHON_CMD[@]}" scripts/eval/write_eval_readme.py \
  --eval-dir "$EVAL_DIR" \
  --candidate-name "$CANDIDATE_NAME" \
  --candidate-adapter "$CANDIDATE_ADAPTER" \
  --baseline-name "$BASELINE_NAME" \
  --baseline-adapter "$BASELINE_ADAPTER" \
  --base-model "$BASE_MODEL" \
  --config "$EVAL_CONFIG"

echo "[8/8] Done: $EVAL_DIR/README.md"

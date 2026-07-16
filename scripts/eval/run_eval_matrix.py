from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.io_utils import ensure_dir, read_config, read_jsonl, resolve_cli_path  # noqa: E402
from safeanywhere.io_utils import resolve_existing_project_path, resolve_project_path, write_json  # noqa: E402
from safeanywhere.io_utils import write_jsonl  # noqa: E402


METRIC_SCHEMA_VERSION = "safeanywhere_eval_metrics_v2"

TASK_LABELS = {
    "benign_utility": "普通良性",
    "harmful_direct": "直接有害请求",
    "over_refusal": "对抗良性/过度拒答",
    "prefix_recovery": "危险前缀恢复",
}

METRIC_LABELS = {
    "avg_visible_chars": "可见回复平均字符数",
    "benign_answer_rate": "良性回答率",
    "harmful_safe_response_rate": "有害请求安全回应率",
    "judge_helpful_answer_rate": "judge 判定有帮助率",
    "judge_over_refusal_rate": "judge 判定过度拒答率",
    "judge_safe_response_rate": "judge 判定安全回应率",
    "over_refusal_rate": "过度拒答率",
    "prefix_recovery_success_rate": "危险前缀恢复成功率",
    "refusal_rate": "拒答或安全重定向率",
    "safety_think_rate": "safety-think 触发率",
    "unsafe_continuation_rate": "不安全续写率",
}

KEY_METRICS = [
    ("危险前缀恢复成功率", "prefix_recovery", "prefix_recovery_success_rate", "越高越好"),
    ("危险前缀不安全续写率", "prefix_recovery", "unsafe_continuation_rate", "越低越好"),
    ("危险前缀 safety-think 触发率", "prefix_recovery", "safety_think_rate", "越高越好"),
    ("直接有害请求安全回应率", "harmful_direct", "harmful_safe_response_rate", "越高越好"),
    ("直接有害请求不安全续写率", "harmful_direct", "unsafe_continuation_rate", "越低越好"),
    ("普通良性回答率", "benign_utility", "benign_answer_rate", "越高越好"),
    ("普通良性过度拒答率", "benign_utility", "over_refusal_rate", "越低越好"),
    ("对抗良性回答率", "over_refusal", "benign_answer_rate", "越高越好"),
    ("对抗良性过度拒答率", "over_refusal", "over_refusal_rate", "越低越好"),
]


@dataclass(frozen=True)
class EvalTarget:
    name: str
    kind: str
    base_model: str
    adapter: str | None = None

    @property
    def run_name(self) -> str:
        return self.name

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "base_model": self.base_model,
            "adapter": self.adapter,
        }


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def env_or_config(env_name: str, value: Any) -> Any:
    env_value = os.environ.get(env_name)
    if env_value not in {None, ""}:
        return env_value
    return value


def shell_words(value: str | None, default: str) -> list[str]:
    return shlex.split(value or default)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metric(summary: dict[str, Any], task: str, key: str) -> float | None:
    value = summary.get("metrics_by_task", {}).get(task, {}).get(key)
    return value if isinstance(value, (int, float)) else None


def delta(value: float | None, ref: float | None) -> float | None:
    if value is None or ref is None:
        return None
    return value - ref


def display_metric(flat_key: str) -> str:
    if "." not in flat_key:
        return flat_key
    task, key = flat_key.split(".", 1)
    task_label = TASK_LABELS.get(task, task)
    metric_label = METRIC_LABELS.get(key, key)
    return f"{task_label} / {metric_label} (`{flat_key}`)"


def validate_name(name: str) -> str:
    if not name or any(ch in name for ch in "/\\"):
        raise ValueError(f"Invalid model target name: {name!r}")
    return name


def normalize_targets(config: dict[str, Any]) -> tuple[list[EvalTarget], str]:
    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("Missing models section in eval config.")

    targets_raw = models.get("targets")
    targets: list[EvalTarget] = []
    if isinstance(targets_raw, list) and targets_raw:
        for item in targets_raw:
            if not isinstance(item, dict):
                raise ValueError(f"Model target must be a mapping, got: {item!r}")
            name = validate_name(str(item.get("name") or ""))
            kind = str(item.get("kind") or ("lora" if item.get("adapter") else "hf")).lower()
            if kind not in {"hf", "lora"}:
                raise ValueError(f"Unsupported model kind for {name}: {kind}")
            if kind == "hf":
                model_path = item.get("model") or item.get("model_path") or item.get("base_model")
                if not model_path:
                    raise ValueError(f"HF target {name} needs model/model_path.")
                targets.append(EvalTarget(name=name, kind=kind, base_model=str(model_path)))
            else:
                base_model = item.get("base_model") or models.get("base_model")
                adapter = item.get("adapter")
                if not base_model or not adapter:
                    raise ValueError(f"LoRA target {name} needs base_model and adapter.")
                targets.append(EvalTarget(name=name, kind=kind, base_model=str(base_model), adapter=str(adapter)))
    else:
        base_model = str(models.get("base_model") or "../models/Qwen3-0.6B")
        targets.append(EvalTarget(name="base", kind="hf", base_model=base_model))
        candidate = models.get("candidate")
        if isinstance(candidate, dict) and candidate.get("adapter"):
            targets.append(
                EvalTarget(
                    name=validate_name(str(candidate.get("name") or "sft")),
                    kind="lora",
                    base_model=base_model,
                    adapter=str(candidate["adapter"]),
                )
            )
        baseline = models.get("baseline")
        if isinstance(baseline, dict) and baseline.get("adapter"):
            targets.append(
                EvalTarget(
                    name=validate_name(str(baseline.get("name") or "baseline")),
                    kind="lora",
                    base_model=base_model,
                    adapter=str(baseline["adapter"]),
                )
            )

    if not targets:
        raise ValueError("No eval model targets configured.")
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model target names: {names}")

    reference = str(nested(config, "compare", "reference") or targets[0].name)
    if reference not in names:
        raise ValueError(f"compare.reference={reference!r} is not in model targets: {names}")
    return targets, reference


def generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    return {
        "max_new_tokens": int(env_or_config("MAX_NEW_TOKENS", generation.get("max_new_tokens", 384))),
        "temperature": float(env_or_config("TEMPERATURE", generation.get("temperature", 0.0))),
        "top_p": float(env_or_config("TOP_P", generation.get("top_p", 0.9))),
        "thinking_mode": str(env_or_config("THINKING_MODE", generation.get("thinking_mode", "auto"))),
        "dtype": str(env_or_config("DTYPE", generation.get("dtype", "bf16"))),
        "device_map": str(env_or_config("DEVICE_MAP", generation.get("device_map", "auto"))),
        "limit": env_or_config("GENERATION_LIMIT", generation.get("limit")),
        "system_prompt": env_or_config("SYSTEM_PROMPT", generation.get("system_prompt")),
        "output_mode": str(env_or_config("OUTPUT_MODE", generation.get("output_mode", "compact"))),
        "include_reference": bool_value(env_or_config("INCLUDE_REFERENCE", generation.get("include_reference", False))),
        "include_raw_prediction": bool_value(env_or_config("INCLUDE_RAW_PREDICTION", generation.get("include_raw_prediction", False))),
        "include_rendered_prompt": bool_value(
            env_or_config("INCLUDE_RENDERED_PROMPT", generation.get("include_rendered_prompt", False))
        ),
    }


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def judge_args(config: dict[str, Any]) -> list[str]:
    judge = config.get("judge")
    if not isinstance(judge, dict):
        judge = {}
    args: list[str] = []
    mapping = [
        ("model", "JUDGE_MODEL", "--model"),
        ("api_key_env", "JUDGE_API_KEY_ENV", "--api-key-env"),
        ("base_url_env", "JUDGE_BASE_URL_ENV", "--base-url-env"),
        ("model_env", "JUDGE_MODEL_ENV", "--model-env"),
        ("max_field_chars", "JUDGE_MAX_FIELD_CHARS", "--max-field-chars"),
    ]
    for key, env_name, flag in mapping:
        value = env_or_config(env_name, judge.get(key))
        if value not in {None, ""}:
            args.extend([flag, str(value)])
    return args


def task_limits(config: dict[str, Any]) -> dict[str, int]:
    per_task = nested(config, "eval_set", "sampling", "per_task")
    if not isinstance(per_task, dict):
        return {}
    return {str(task): int(limit) for task, limit in per_task.items() if limit not in {None, ""}}


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def rows_from(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def generation_config_matches(row_cfg: dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        return (
            int(row_cfg.get("max_new_tokens")) == int(expected["max_new_tokens"])
            and math.isclose(float(row_cfg.get("temperature")), float(expected["temperature"]), rel_tol=0, abs_tol=1e-9)
            and math.isclose(float(row_cfg.get("top_p")), float(expected["top_p"]), rel_tol=0, abs_tol=1e-9)
            and str(row_cfg.get("thinking_mode")) == str(expected["thinking_mode"])
        )
    except (TypeError, ValueError):
        return False


def prediction_cache_status(
    predictions_path: Path,
    eval_rows: list[dict[str, Any]],
    target: EvalTarget,
    gen_cfg: dict[str, Any],
) -> tuple[str, list[dict[str, Any]] | None, str]:
    if not predictions_path.exists():
        return "missing", None, "prediction file missing"
    rows = rows_from(predictions_path)
    if len(rows) != len(eval_rows):
        return "invalid", rows, f"row count mismatch: {len(rows)} vs {len(eval_rows)}"
    expected_identity = target.identity()
    missing_identity = 0
    for index, (pred, expected) in enumerate(zip(rows, eval_rows), start=1):
        if pred.get("id") != expected.get("id"):
            return "invalid", rows, f"id mismatch at row {index}: {pred.get('id')} vs {expected.get('id')}"
        if pred.get("prompt") != expected.get("prompt") or pred.get("assistant_prefill") != expected.get("assistant_prefill"):
            return "invalid", rows, f"prompt/prefix mismatch at row {index}: {pred.get('id')}"
        if not generation_config_matches(pred.get("generation_config") or {}, gen_cfg):
            return "invalid", rows, f"generation_config mismatch at row {index}: {pred.get('id')}"
        identity = pred.get("model_identity")
        if identity is None:
            missing_identity += 1
        elif identity != expected_identity:
            return "invalid", rows, f"model_identity mismatch at row {index}: {pred.get('id')}"
    if missing_identity:
        return "upgrade", rows, f"legacy predictions without model_identity: {missing_identity}"
    return "valid", rows, "prediction cache valid"


def upgrade_prediction_identity(path: Path, rows: list[dict[str, Any]], target: EvalTarget, dry_run: bool) -> None:
    print(f"Upgrade legacy prediction metadata: {path}", flush=True)
    if dry_run:
        return
    upgraded = [{**row, "model_identity": target.identity()} for row in rows]
    write_jsonl(path, upgraded)


def score_cache_status(
    scored_path: Path,
    predictions: list[dict[str, Any]],
) -> tuple[str, str]:
    if not scored_path.exists():
        return "missing", "score file missing"
    rows = rows_from(scored_path)
    if len(rows) != len(predictions):
        return "invalid", f"score row count mismatch: {len(rows)} vs {len(predictions)}"
    stale_schema = False
    for index, (scored, pred) in enumerate(zip(rows, predictions), start=1):
        if scored.get("id") != pred.get("id"):
            return "invalid", f"score id mismatch at row {index}: {scored.get('id')} vs {pred.get('id')}"
        if scored.get("prediction") != pred.get("prediction"):
            return "invalid", f"score prediction mismatch at row {index}: {pred.get('id')}"
        judge_block = scored.get("llm_judge")
        if not isinstance(judge_block, dict) or not isinstance(judge_block.get("result"), dict):
            return "invalid", f"missing llm_judge.result at row {index}: {pred.get('id')}"
        score = scored.get("llm_judge_score")
        if not isinstance(score, dict):
            stale_schema = True
        elif score.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            stale_schema = True
    if stale_schema:
        return "recompute", "existing judge results need metric recompute"
    return "valid", "score cache valid"


def build_eval_set(
    config: dict[str, Any],
    py_cmd: list[str],
    eval_input: Path,
    eval_set_dir: Path,
    dry_run: bool,
) -> None:
    eval_set = config.get("eval_set")
    if not isinstance(eval_set, dict):
        eval_set = {}
    cmd = [
        *py_cmd,
        "scripts/eval/build_eval_sets.py",
        "--input",
        str(eval_input),
        "--output-dir",
        str(eval_set_dir),
    ]
    max_per_task = env_or_config(
        "MAX_PER_TASK",
        nested(config, "eval_set", "sampling", "max_per_task") or nested(config, "eval_set", "max_per_task"),
    )
    if max_per_task not in {None, ""}:
        cmd.extend(["--max-per-task", str(max_per_task)])
    for task, limit in task_limits(config).items():
        cmd.extend(["--task-limit", f"{task}={limit}"])
    include_prefix_direct = bool_value(env_or_config("INCLUDE_PREFIX_DIRECT", eval_set.get("include_prefix_direct", True)))
    if not include_prefix_direct:
        cmd.append("--no-prefix-direct")
    run_command(cmd, eval_set_dir / "build.log", dry_run=dry_run)


def generate_target(
    target: EvalTarget,
    py_cmd: list[str],
    eval_file: Path,
    output_path: Path,
    gen_cfg: dict[str, Any],
    dry_run: bool,
) -> None:
    cmd = [
        *py_cmd,
        "scripts/eval/generate_responses.py",
        "--eval-file",
        str(eval_file),
        "--base-model",
        target.base_model,
        "--model-name",
        target.name,
        "--model-kind",
        target.kind,
        "--output",
        str(output_path),
        "--max-new-tokens",
        str(gen_cfg["max_new_tokens"]),
        "--temperature",
        str(gen_cfg["temperature"]),
        "--top-p",
        str(gen_cfg["top_p"]),
        "--thinking-mode",
        str(gen_cfg["thinking_mode"]),
        "--dtype",
        str(gen_cfg["dtype"]),
        "--device-map",
        str(gen_cfg["device_map"]),
        "--output-mode",
        str(gen_cfg["output_mode"]),
    ]
    if target.adapter:
        cmd.extend(["--adapter", target.adapter])
    if gen_cfg.get("limit") not in {None, ""}:
        cmd.extend(["--limit", str(gen_cfg["limit"])])
    if gen_cfg.get("system_prompt") not in {None, ""}:
        cmd.extend(["--system-prompt", str(gen_cfg["system_prompt"])])
    if gen_cfg["include_reference"]:
        cmd.append("--include-reference")
    if gen_cfg["include_raw_prediction"]:
        cmd.append("--include-raw-prediction")
    if gen_cfg["include_rendered_prompt"]:
        cmd.append("--include-rendered-prompt")
    run_command(cmd, output_path.parent / "generate.log", dry_run=dry_run)


def score_target(
    judge_cmd: list[str],
    predictions_path: Path,
    scored_path: Path,
    summary_path: Path,
    judge_extra_args: list[str],
    dry_run: bool,
    *,
    recompute_existing: bool = False,
    no_resume: bool = False,
) -> None:
    cmd = [
        *judge_cmd,
        "scripts/eval/score_llm_judge.py",
        "--input",
        str(scored_path if recompute_existing else predictions_path),
        "--scored-output",
        str(scored_path),
        "--summary-output",
        str(summary_path),
        *judge_extra_args,
    ]
    if recompute_existing:
        cmd.append("--recompute-existing")
    if no_resume:
        cmd.append("--no-resume")
    log_name = "score_recompute.log" if recompute_existing else "score.log"
    run_command(cmd, scored_path.parent / log_name, dry_run=dry_run)


def flatten_summary(summary: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for task, metrics in summary.get("metrics_by_task", {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if key == "n" or value is None:
                continue
            if isinstance(value, (int, float)):
                flat[f"{task}.{key}"] = float(value)
    return flat


def matrix_header(targets: list[EvalTarget], reference: str, include_desired: bool) -> tuple[str, str]:
    columns = ["指标"]
    for target in targets:
        columns.append(target.name)
        if target.name != reference:
            columns.append(f"{target.name}-{reference}")
    if include_desired:
        columns.append("期望")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|"
    if include_desired:
        separator = "|" + "|".join(["---"] + ["---:"] * (len(columns) - 2) + ["---"]) + "|"
    return header, separator


def matrix_row(
    label: str,
    task: str,
    key: str,
    desired: str | None,
    targets: list[EvalTarget],
    summaries: dict[str, dict[str, Any]],
    reference: str,
) -> str:
    ref_value = metric(summaries[reference], task, key)
    cells = [label]
    for target in targets:
        value = metric(summaries[target.name], task, key)
        cells.append(fmt(value))
        if target.name != reference:
            cells.append(fmt(delta(value, ref_value)))
    if desired is not None:
        cells.append(desired)
    return "| " + " | ".join(cells) + " |"


def write_all_models_comparison(
    path: Path,
    targets: list[EvalTarget],
    summaries: dict[str, dict[str, Any]],
    reference: str,
) -> None:
    keys = sorted({key for summary in summaries.values() for key in flatten_summary(summary)})
    header, separator = matrix_header(targets, reference, include_desired=False)
    lines = ["# 全模型横向对比", "", f"基线模型：`{reference}`", "", header, separator]
    for key in keys:
        if "." not in key:
            continue
        task, metric_key = key.split(".", 1)
        lines.append(matrix_row(display_metric(key), task, metric_key, None, targets, summaries, reference))
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matrix_readme(
    eval_dir: Path,
    config_path: Path,
    eval_file: Path,
    report_path: Path,
    targets: list[EvalTarget],
    summaries: dict[str, dict[str, Any]],
    reference: str,
    all_models_path: Path,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scorer = summaries[targets[0].name].get("scorer", "llm_judge")
    judge_model = summaries[targets[0].name].get("judge_model")
    key_header, key_separator = matrix_header(targets, reference, include_desired=True)

    lines = [
        "# SafeAnywhere 多模型评测报告",
        "",
        "本报告由 eval matrix pipeline 生成；同一评测集会被所有模型复用，prediction 和 score 会严格校验后缓存复用。",
        "",
        "## 输入与配置",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| eval 配置 | `{config_path}` |",
        f"| eval 数据文件 | `{eval_file}` |",
        f"| eval 集报告 | `{report_path}` |",
        f"| 输出目录 | `{eval_dir}` |",
        f"| 基线模型 | `{reference}` |",
        f"| 评分器 | `{scorer}` |",
    ]
    if judge_model:
        lines.append(f"| judge 模型 | `{judge_model}` |")

    lines.extend(["", "## 模型列表", "", "| 模型 | 类型 | base/model | adapter | run 目录 |", "|---|---|---|---|---|"])
    for target in targets:
        run_dir = eval_dir / "runs" / target.run_name
        lines.append(
            f"| `{target.name}` | `{target.kind}` | `{target.base_model}` | "
            f"`{target.adapter or ''}` | `{run_dir}` |"
        )

    lines.extend(["", "## 评测集", "", f"总样本数：`{report.get('total')}`", "", "| 任务 | 数量 |", "|---|---:|"])
    for task, count in sorted((report.get("by_task") or {}).items()):
        lines.append(f"| {TASK_LABELS.get(task, task)} (`{task}`) | {count} |")

    lines.extend(["", "## 关键指标", "", key_header, key_separator])
    for label, task, key, desired in KEY_METRICS:
        lines.append(matrix_row(label, task, key, desired, targets, summaries, reference))

    lines.extend(["", "## 结果解读", ""])
    ref_prefix = metric(summaries[reference], "prefix_recovery", "prefix_recovery_success_rate") or 0.0
    for target in targets:
        prefix = metric(summaries[target.name], "prefix_recovery", "prefix_recovery_success_rate") or 0.0
        unsafe = metric(summaries[target.name], "prefix_recovery", "unsafe_continuation_rate") or 0.0
        direct_safe = metric(summaries[target.name], "harmful_direct", "harmful_safe_response_rate") or 0.0
        benign = metric(summaries[target.name], "benign_utility", "benign_answer_rate") or 0.0
        if target.name == reference:
            lines.append(
                f"- `{target.name}`：危险前缀恢复成功率 `{fmt(prefix)}`，不安全续写率 `{fmt(unsafe)}`，"
                f"直接有害请求安全回应率 `{fmt(direct_safe)}`，普通良性回答率 `{fmt(benign)}`。"
            )
        else:
            lines.append(
                f"- `{target.name}`：危险前缀恢复成功率 `{fmt(prefix)}`，相对 `{reference}` 为 "
                f"`{fmt(prefix - ref_prefix)}`；不安全续写率 `{fmt(unsafe)}`，直接有害请求安全回应率 "
                f"`{fmt(direct_safe)}`，普通良性回答率 `{fmt(benign)}`。"
            )
    lines.extend(
        [
            "",
            "如果要用于正式报告或论文结论，建议审计 judge prompt、抽样检查 scored outputs，并考虑多 judge 或人工复核。",
            "",
            "## 完整横向对比",
            "",
            f"完整指标表见：[{all_models_path.name}]({all_models_path.relative_to(eval_dir)})",
            "",
            all_models_path.read_text(encoding="utf-8").strip(),
            "",
        ]
    )
    (eval_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> int:
    config_path = resolve_cli_path(args.config, ROOT)
    config = read_config(config_path)
    targets, reference = normalize_targets(config)

    paths = config.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    eval_dir = resolve_project_path(env_or_config("EVAL_DIR", paths.get("eval_dir", "build/data_build/eval/safeanywhere_v1_1532")), ROOT)
    eval_input = resolve_existing_project_path(
        env_or_config("EVAL_INPUT", paths.get("eval_input", "build/data_build/safeanywhere_sft_v1/sft_val.jsonl")),
        ROOT,
    )
    eval_set_dir = eval_dir / str(paths.get("eval_set_subdir", "eval_set"))
    runs_dir = eval_dir / str(paths.get("runs_subdir", "runs"))
    comparisons_dir = eval_dir / str(paths.get("comparisons_subdir", "comparisons"))
    eval_file = eval_set_dir / "safeanywhere_eval.jsonl"
    report_path = eval_set_dir / "report.json"

    py_cmd = shell_words(os.environ.get("PYTHON_BIN"), "python")
    judge_cmd = shell_words(os.environ.get("JUDGE_PYTHON_BIN"), "uv run python")
    gen_cfg = generation_config(config)

    print(f"Eval dir: {eval_dir}")
    print("Targets: " + ", ".join(f"{target.name}:{target.kind}" for target in targets))
    if args.dry_run:
        print("Dry run: commands will be printed but not executed.")

    build_eval_set(config, py_cmd, eval_input, eval_set_dir, args.dry_run)
    if args.dry_run and not eval_file.exists():
        print(f"Dry run stopped before cache checks because eval file is missing: {eval_file}")
        return 0

    eval_rows = rows_from(eval_file)
    ensure_dir(runs_dir)
    ensure_dir(comparisons_dir)
    extra_judge_args = judge_args(config)

    for target in targets:
        run_dir = runs_dir / target.run_name
        predictions_path = run_dir / "predictions.jsonl"
        scored_path = run_dir / "predictions_scored.jsonl"
        summary_path = run_dir / "score_summary.json"
        ensure_dir(run_dir)

        status, pred_rows, reason = prediction_cache_status(predictions_path, eval_rows, target, gen_cfg)
        if status == "valid":
            print(f"Generate {target.name} skipped: {reason}")
        elif status == "upgrade" and pred_rows is not None:
            print(f"Generate {target.name} skipped after metadata upgrade: {reason}")
            upgrade_prediction_identity(predictions_path, pred_rows, target, args.dry_run)
        else:
            print(f"Generate {target.name}: {reason}")
            generate_target(target, py_cmd, eval_file, predictions_path, gen_cfg, args.dry_run)

        if args.dry_run and not predictions_path.exists():
            continue
        pred_rows = rows_from(predictions_path)
        score_status, score_reason = score_cache_status(scored_path, pred_rows)
        if score_status == "valid" and summary_path.exists():
            print(f"Score {target.name} skipped: {score_reason}")
        elif score_status == "valid":
            print(f"Score {target.name} summary recompute: summary missing")
            score_target(
                judge_cmd,
                predictions_path,
                scored_path,
                summary_path,
                extra_judge_args,
                args.dry_run,
                recompute_existing=True,
            )
        elif score_status == "recompute":
            print(f"Score {target.name} recompute: {score_reason}")
            score_target(
                judge_cmd,
                predictions_path,
                scored_path,
                summary_path,
                extra_judge_args,
                args.dry_run,
                recompute_existing=True,
            )
        else:
            print(f"Score {target.name}: {score_reason}")
            score_target(
                judge_cmd,
                predictions_path,
                scored_path,
                summary_path,
                extra_judge_args,
                args.dry_run,
                no_resume=score_status == "invalid",
            )

    if args.dry_run:
        return 0

    summaries = {}
    for target in targets:
        summary_path = runs_dir / target.run_name / "score_summary.json"
        summaries[target.name] = json.loads(summary_path.read_text(encoding="utf-8"))

    all_models_path = comparisons_dir / "all_models.md"
    write_all_models_comparison(all_models_path, targets, summaries, reference)
    write_matrix_readme(eval_dir, config_path, eval_file, report_path, targets, summaries, reference, all_models_path)
    manifest = {
        "config": str(config_path),
        "eval_dir": str(eval_dir),
        "reference": reference,
        "targets": [target.identity() for target in targets],
        "files": {
            "eval_file": str(eval_file),
            "report": str(report_path),
            "all_models": str(all_models_path),
            "readme": str(eval_dir / "README.md"),
        },
    }
    write_json(eval_dir / "matrix_manifest.json", manifest)
    print(f"Done: {eval_dir / 'README.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SafeAnywhere multi-model eval matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval/safeanywhere_v1.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return run_pipeline(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

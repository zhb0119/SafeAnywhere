from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flatten_metrics(report: dict[str, Any]) -> dict[str, float]:
    flat = {}
    for task, metrics in report.get("metrics_by_task", {}).items():
        for key, value in metrics.items():
            if key == "n" or value is None:
                continue
            if isinstance(value, (int, float)):
                flat[f"{task}.{key}"] = float(value)
    return flat


def display_metric(key: str) -> str:
    if "." not in key:
        return f"{key}"
    task, metric = key.split(".", 1)
    task_label = TASK_LABELS.get(task, task)
    metric_label = METRIC_LABELS.get(metric, metric)
    return f"{task_label} / {metric_label} (`{key}`)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two SafeAnywhere eval score summaries.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    base = flatten_metrics(read_json(args.base))
    candidate = flatten_metrics(read_json(args.candidate))
    keys = sorted(set(base) | set(candidate))

    print("| 指标 | base | candidate | 差值 |")
    print("|---|---:|---:|---:|")
    for key in keys:
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        delta = None if base_value is None or candidate_value is None else candidate_value - base_value
        base_text = "NA" if base_value is None else f"{base_value:.4f}"
        candidate_text = "NA" if candidate_value is None else f"{candidate_value:.4f}"
        delta_text = "NA" if delta is None else f"{delta:+.4f}"
        print(f"| {display_metric(key)} | {base_text} | {candidate_text} | {delta_text} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

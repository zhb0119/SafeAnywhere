from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "build/mixed_safechain1k_prefix500/sft_val.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "build/eval/safeanywhere_v1"

BENIGN_LABELS = {"vanilla_benign", "adversarial_benign"}
HARMFUL_LABELS = {"vanilla_harmful", "adversarial_harmful"}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def get_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Row {row.get('id')} is missing messages")
    return messages


def make_eval_row(
    row: dict[str, Any],
    task: str,
    prompt: str,
    reference: str,
    assistant_prefill: str | None = None,
) -> dict[str, Any]:
    label = str(row.get("label") or "")
    expected_safety_think = task in {"prefix_recovery", "harmful_direct"}
    expected_refusal = task in {"prefix_recovery", "harmful_direct"}
    expected_no_refusal = task in {"benign_utility", "over_refusal"}
    return {
        "id": f"{task}::{row.get('id')}",
        "source_id": row.get("id"),
        "source": row.get("source"),
        "attack_type": row.get("attack_type"),
        "task": task,
        "category": "utility" if task in {"benign_utility", "over_refusal"} else "safety",
        "label": label,
        "prompt": prompt,
        "assistant_prefill": assistant_prefill,
        "reference": reference,
        "messages": [{"role": "user", "content": prompt}],
        "expected": {
            "safety_think": expected_safety_think,
            "refusal_or_safe_redirect": expected_refusal,
            "no_refusal": expected_no_refusal,
            "no_unsafe_continuation": task in {"prefix_recovery", "harmful_direct"},
        },
    }


def rows_from_safechain(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = get_messages(row)
    if len(messages) != 2:
        raise ValueError(f"SafeChain row {row.get('id')} should have two messages")
    prompt = messages[0]["content"]
    reference = messages[1]["content"]
    label = str(row.get("label") or "")
    if label in BENIGN_LABELS:
        task = "over_refusal" if label == "adversarial_benign" else "benign_utility"
        return [make_eval_row(row, task, prompt, reference)]
    if label in HARMFUL_LABELS:
        return [make_eval_row(row, "harmful_direct", prompt, reference)]
    return []


def rows_from_prefix(row: dict[str, Any], include_direct: bool) -> list[dict[str, Any]]:
    messages = get_messages(row)
    if len(messages) != 3:
        raise ValueError(f"Prefix row {row.get('id')} should have three messages")
    prompt = messages[0]["content"]
    assistant_prefill = messages[1]["content"]
    reference = messages[2]["content"]
    rows = [make_eval_row(row, "prefix_recovery", prompt, reference, assistant_prefill=assistant_prefill)]
    if include_direct:
        rows.append(make_eval_row(row, "harmful_direct", prompt, reference))
    return rows


def cap_rows_by_task(rows: list[dict[str, Any]], max_per_task: int | None) -> list[dict[str, Any]]:
    if max_per_task is None or max_per_task <= 0:
        return rows
    counts: defaultdict[str, int] = defaultdict(int)
    capped = []
    for row in rows:
        task = row["task"]
        if counts[task] >= max_per_task:
            continue
        capped.append(row)
        counts[task] += 1
    return capped


def build_eval_rows(input_path: Path, include_prefix_direct: bool, max_per_task: int | None) -> list[dict[str, Any]]:
    eval_rows = []
    for row in read_jsonl(input_path):
        attack_type = row.get("attack_type")
        if attack_type == "safechain_cold_start":
            eval_rows.extend(rows_from_safechain(row))
        elif attack_type == "dangerous_prefix":
            eval_rows.extend(rows_from_prefix(row, include_direct=include_prefix_direct))
        else:
            raise ValueError(f"Unsupported attack_type in {row.get('id')}: {attack_type}")
    return cap_rows_by_task(eval_rows, max_per_task)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    ensure_dir(output_dir)
    all_path = output_dir / "safeanywhere_eval.jsonl"
    write_jsonl(all_path, rows)

    task_counts = Counter(row["task"] for row in rows)
    label_counts = Counter(str(row.get("label")) for row in rows)
    task_files = {}
    for task in sorted(task_counts):
        path = output_dir / f"{task}.jsonl"
        write_jsonl(path, [row for row in rows if row["task"] == task])
        task_files[task] = str(path)

    report = {
        "dataset_name": "safeanywhere_eval_v1",
        "total": len(rows),
        "by_task": dict(task_counts),
        "by_label": dict(label_counts),
        "files": {
            "all": str(all_path),
            **task_files,
        },
    }
    write_json(output_dir / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SafeAnywhere held-out evaluation sets from mixed val data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument(
        "--no-prefix-direct",
        action="store_true",
        help="Do not create direct harmful eval rows from dangerous-prefix prompts.",
    )
    args = parser.parse_args()

    rows = build_eval_rows(
        input_path=args.input,
        include_prefix_direct=not args.no_prefix_direct,
        max_per_task=args.max_per_task,
    )
    report = write_outputs(rows, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

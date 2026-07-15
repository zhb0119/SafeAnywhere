from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "build/data_build/eval/external"

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def download_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def read_csv_rows(path: Path | None, url: str) -> list[dict[str, str]]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
    else:
        text = download_text(url)
    return list(csv.DictReader(text.splitlines()))


def task_for_external_label(benchmark: str, label: str) -> str:
    if benchmark == "xstest":
        return "xstest_safe" if label == "safe" else "xstest_unsafe"
    return "harmful_direct"


def make_eval_row(
    benchmark: str,
    row_id: str,
    prompt: str,
    task: str,
    label: str,
    reference: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": f"{benchmark}::{row_id}",
        "task": task,
        "category": "utility" if task == "xstest_safe" else "safety",
        "label": label,
        "benchmark": benchmark,
        "prompt": prompt,
        "assistant_prefill": None,
        "reference": reference,
        "messages": [{"role": "user", "content": prompt}],
        "metadata": metadata or {},
    }


def prepare_advbench(output_dir: Path, input_csv: Path | None, limit: int | None) -> dict[str, Any]:
    rows = read_csv_rows(input_csv, ADVBENCH_URL)
    if limit:
        rows = rows[:limit]
    eval_rows = []
    for index, row in enumerate(rows):
        prompt = row.get("goal") or row.get("prompt")
        if not prompt:
            continue
        eval_rows.append(
            make_eval_row(
                benchmark="advbench",
                row_id=str(index),
                prompt=prompt,
                task="harmful_direct",
                label="advbench",
                reference=row.get("target", ""),
                metadata={"target": row.get("target", "")},
            )
        )
    output = output_dir / "advbench/advbench_eval.jsonl"
    write_jsonl(output, eval_rows)
    return {"benchmark": "advbench", "rows": len(eval_rows), "file": str(output)}


def prepare_xstest(output_dir: Path, input_csv: Path | None, limit: int | None) -> dict[str, Any]:
    rows = read_csv_rows(input_csv, XSTEST_URL)
    if limit:
        rows = rows[:limit]
    eval_rows = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        label = (row.get("label") or "").strip().lower()
        if not prompt or label not in {"safe", "unsafe"}:
            continue
        eval_rows.append(
            make_eval_row(
                benchmark="xstest",
                row_id=row.get("id") or str(index),
                prompt=prompt,
                task=task_for_external_label("xstest", label),
                label=label,
                metadata={
                    "type": row.get("type"),
                    "focus": row.get("focus"),
                    "note": row.get("note"),
                },
            )
        )
    output = output_dir / "xstest/xstest_eval.jsonl"
    write_jsonl(output, eval_rows)
    return {"benchmark": "xstest", "rows": len(eval_rows), "file": str(output)}


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def iter_jbb_rows(dataset_name: str, config: str, split: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets before preparing JailbreakBench: python -m pip install datasets") from exc

    loaded = load_dataset(dataset_name, config)
    if split is not None:
        return [dict(row) for row in loaded[split]]
    if isinstance(loaded, DatasetDict):
        rows = []
        for split_name in loaded:
            for row in loaded[split_name]:
                item = dict(row)
                item["_split"] = split_name
                rows.append(item)
        return rows
    return [dict(row) for row in loaded]


def prepare_jailbreakbench(
    output_dir: Path,
    dataset_name: str,
    config: str,
    split: str | None,
    limit: int | None,
) -> dict[str, Any]:
    rows = iter_jbb_rows(dataset_name, config, split)
    if limit:
        rows = rows[:limit]
    eval_rows = []
    for index, row in enumerate(rows):
        prompt = first_present(row, ["Goal", "goal", "behavior", "prompt", "Behavior", "jailbreak_prompt"])
        if prompt is None:
            continue
        target = first_present(row, ["Target", "target", "completion", "reference"]) or ""
        category = first_present(row, ["Category", "category", "FunctionalCategory", "semantic_category"]) or "jbb"
        row_id = first_present(row, ["Index", "index", "id", "BehaviorID", "behavior_id"]) or str(index)
        eval_rows.append(
            make_eval_row(
                benchmark="jailbreakbench",
                row_id=str(row_id),
                prompt=str(prompt),
                task="harmful_direct",
                label=str(category),
                reference=str(target),
                metadata={key: value for key, value in row.items() if key.startswith("_") or key.lower() in {"source", "split"}},
            )
        )
    output = output_dir / "jailbreakbench/jbb_eval.jsonl"
    write_jsonl(output, eval_rows)
    return {"benchmark": "jailbreakbench", "rows": len(eval_rows), "file": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare external benchmark prompts in SafeAnywhere eval JSONL format.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["advbench", "xstest", "jailbreakbench"],
        choices=["advbench", "xstest", "jailbreakbench"],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--advbench-csv", type=Path, default=None)
    parser.add_argument("--xstest-csv", type=Path, default=None)
    parser.add_argument("--jbb-dataset", default="JailbreakBench/JBB-Behaviors")
    parser.add_argument("--jbb-config", default="behaviors")
    parser.add_argument("--jbb-split", default=None)
    args = parser.parse_args()

    reports = []
    if "advbench" in args.benchmarks:
        reports.append(prepare_advbench(args.output_dir, args.advbench_csv, args.limit))
    if "xstest" in args.benchmarks:
        reports.append(prepare_xstest(args.output_dir, args.xstest_csv, args.limit))
    if "jailbreakbench" in args.benchmarks:
        reports.append(prepare_jailbreakbench(args.output_dir, args.jbb_dataset, args.jbb_config, args.jbb_split, args.limit))

    report = {
        "output_dir": str(args.output_dir),
        "benchmarks": reports,
    }
    write_json(args.output_dir / "external_benchmarks_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

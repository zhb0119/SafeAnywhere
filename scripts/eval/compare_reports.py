from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two SafeAnywhere eval score summaries.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    base = flatten_metrics(read_json(args.base))
    candidate = flatten_metrics(read_json(args.candidate))
    keys = sorted(set(base) | set(candidate))

    print("| metric | base | candidate | delta |")
    print("|---|---:|---:|---:|")
    for key in keys:
        base_value = base.get(key)
        candidate_value = candidate.get(key)
        delta = None if base_value is None or candidate_value is None else candidate_value - base_value
        base_text = "NA" if base_value is None else f"{base_value:.4f}"
        candidate_text = "NA" if candidate_value is None else f"{candidate_value:.4f}"
        delta_text = "NA" if delta is None else f"{delta:+.4f}"
        print(f"| `{key}` | {base_text} | {candidate_text} | {delta_text} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

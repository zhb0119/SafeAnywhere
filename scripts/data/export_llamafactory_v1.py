from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT_DIR = ROOT / "build/mixed_safechain1k_prefix500"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
DEFAULT_TRAIN_DATASET_YAML = ROOT / "train/llamafactory/dataset_safeanywhere_1500_train.yaml"
DEFAULT_VAL_DATASET_YAML = ROOT / "train/llamafactory/dataset_safeanywhere_1500_val.yaml"


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
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            yield obj


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


def text_content(value: str, loss_weight: float | None = None) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError("content value must be a non-empty string")
    content: dict[str, Any] = {"type": "text", "value": value}
    if loss_weight is not None:
        content["loss_weight"] = float(loss_weight)
    return content


def message(role: str, content: list[dict[str, Any]], loss_weight: float) -> dict[str, Any]:
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Invalid role: {role}")
    if not content:
        raise ValueError(f"{role} message has empty content")
    return {
        "role": role,
        "content": content,
        "loss_weight": float(loss_weight),
    }


def extra_info(row: dict[str, Any], split: str) -> str:
    metadata = {
        "id": row.get("id"),
        "source": row.get("source"),
        "source_id": row.get("source_id"),
        "split": split,
        "attack_type": row.get("attack_type"),
        "label": row.get("label"),
        "requires_safety_think": row.get("requires_safety_think"),
        "prefix_depth": row.get("prefix_depth"),
        "prefix_type": row.get("prefix_type"),
    }
    return json.dumps({key: value for key, value in metadata.items() if value is not None}, ensure_ascii=False)


def require_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Row {row.get('id')} is missing messages")
    return messages


def export_safechain(row: dict[str, Any], split: str) -> dict[str, Any]:
    messages = require_messages(row)
    if len(messages) != 2:
        raise ValueError(f"SafeChain row {row.get('id')} should have two messages")
    user_msg, assistant_msg = messages
    if [user_msg.get("role"), assistant_msg.get("role")] != ["user", "assistant"]:
        raise ValueError(f"SafeChain row {row.get('id')} has invalid roles")
    if [user_msg.get("loss_mask"), assistant_msg.get("loss_mask")] != [0, 1]:
        raise ValueError(f"SafeChain row {row.get('id')} has invalid masks")

    return {
        "messages": [
            message("user", [text_content(user_msg["content"])], 0.0),
            message("assistant", [text_content(assistant_msg["content"], 1.0)], 1.0),
        ],
        "extra_info": extra_info(row, split),
    }


def export_dangerous_prefix(row: dict[str, Any], split: str, prefill_separator: str) -> dict[str, Any]:
    messages = require_messages(row)
    if len(messages) != 3:
        raise ValueError(f"Dangerous-prefix row {row.get('id')} should have three source messages")
    user_msg, prefill_msg, target_msg = messages
    roles = [user_msg.get("role"), prefill_msg.get("role"), target_msg.get("role")]
    masks = [user_msg.get("loss_mask"), prefill_msg.get("loss_mask"), target_msg.get("loss_mask")]
    if roles != ["user", "assistant", "assistant"]:
        raise ValueError(f"Dangerous-prefix row {row.get('id')} has invalid roles: {roles}")
    if masks != [0, 0, 1]:
        raise ValueError(f"Dangerous-prefix row {row.get('id')} has invalid masks: {masks}")

    prefill = prefill_msg["content"]
    target = target_msg["content"]
    if prefill_separator and not prefill.endswith(prefill_separator):
        prefill = prefill + prefill_separator
    if not target.startswith("<safety_think>"):
        raise ValueError(f"Dangerous-prefix row {row.get('id')} target must start with <safety_think>")

    return {
        "messages": [
            message("user", [text_content(user_msg["content"])], 0.0),
            message(
                "assistant",
                [
                    text_content(prefill, 0.0),
                    text_content(target, 1.0),
                ],
                1.0,
            ),
        ],
        "extra_info": extra_info(row, split),
    }


def export_row(row: dict[str, Any], split: str, prefill_separator: str) -> dict[str, Any]:
    attack_type = row.get("attack_type")
    if attack_type == "dangerous_prefix":
        return export_dangerous_prefix(row, split, prefill_separator)
    if attack_type == "safechain_cold_start":
        return export_safechain(row, split)
    raise ValueError(f"Unsupported attack_type for row {row.get('id')}: {attack_type}")


def export_file(input_path: Path, output_path: Path, split: str, prefill_separator: str) -> dict[str, int]:
    rows = [export_row(row, split, prefill_separator) for row in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    dangerous_prefix = 0
    safechain = 0
    positive_spans = 0
    zero_spans = 0
    for row in rows:
        info = json.loads(row["extra_info"])
        if info.get("attack_type") == "dangerous_prefix":
            dangerous_prefix += 1
        elif info.get("attack_type") == "safechain_cold_start":
            safechain += 1
        for msg in row["messages"]:
            for content in msg["content"]:
                if content.get("loss_weight", msg.get("loss_weight", 0.0)) > 0:
                    positive_spans += 1
                else:
                    zero_spans += 1

    return {
        "rows": len(rows),
        "dangerous_prefix": dangerous_prefix,
        "safechain_cold_start": safechain,
        "positive_content_spans": positive_spans,
        "zero_content_spans": zero_spans,
    }


def repo_relative_path(path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def write_dataset_yaml(path: Path, name: str, data_path: Path) -> None:
    ensure_dir(path.parent)
    dataset_path = repo_relative_path(data_path)
    path.write_text(
        "\n".join(
            [
                f"{name}:",
                f"  path: {dataset_path}",
                "  source: local",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export mixed SafeAnywhere data to LLaMA-Factory v1 span-mask JSONL.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-dataset-yaml", type=Path, default=DEFAULT_TRAIN_DATASET_YAML)
    parser.add_argument("--val-dataset-yaml", type=Path, default=DEFAULT_VAL_DATASET_YAML)
    parser.add_argument("--prefill-separator", default="\n")
    args = parser.parse_args()

    train_input = args.input_dir / "sft_train.jsonl"
    val_input = args.input_dir / "sft_val.jsonl"
    output_dir = ensure_dir(args.output_dir)
    train_output = output_dir / "train_lf_v1_spanmasked.jsonl"
    val_output = output_dir / "val_lf_v1_spanmasked.jsonl"

    train_counts = export_file(train_input, train_output, "train", args.prefill_separator)
    val_counts = export_file(val_input, val_output, "val", args.prefill_separator)
    write_dataset_yaml(args.train_dataset_yaml, "safeanywhere_train", train_output)
    write_dataset_yaml(args.val_dataset_yaml, "safeanywhere_val", val_output)

    report = {
        "dataset_name": "safeanywhere_lf_v1_spanmasked",
        "prefill_separator": args.prefill_separator,
        "train": train_counts,
        "val": val_counts,
        "files": {
            "train_lf_v1": str(train_output),
            "val_lf_v1": str(val_output),
            "train_dataset_yaml": str(args.train_dataset_yaml),
            "val_dataset_yaml": str(args.val_dataset_yaml),
        },
    }
    report_path = output_dir / "llamafactory_v1_export_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.filters import BLOCK_RE, parse_block_fields, strip_safety_blocks  # noqa: E402
from safeanywhere.io_utils import resolve_existing_project_path  # noqa: E402


DEFAULT_SAFECHAIN_TRAIN = ROOT / "build/data_build/safechain_pilot_1k/sft_train.jsonl"
DEFAULT_SAFECHAIN_VAL = ROOT / "build/data_build/safechain_pilot_1k/sft_val.jsonl"
DEFAULT_PREFIX_TRAIN = ROOT / "build/data_build/hex_phi_prefix_500/sft_train.jsonl"
DEFAULT_PREFIX_VAL = ROOT / "build/data_build/hex_phi_prefix_500/sft_val.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "build/data_build/mixed_safechain1k_prefix500"


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


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_loss_mask(value: Any) -> int:
    if value in (0, 0.0, False, "0"):
        return 0
    if value in (1, 1.0, True, "1"):
        return 1
    raise ValueError(f"Invalid loss_mask value: {value!r}")


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant"}:
        raise ValueError(f"Invalid message role: {role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Invalid message content for role {role!r}")
    return {
        "role": role,
        "content": content,
        "loss_mask": normalize_loss_mask(message.get("loss_mask", 0 if role == "user" else 1)),
    }


def normalize_safechain_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"SafeChain row {row.get('id')} is missing messages; regenerate it with scripts/01_build_dataset.py")
    normalized = [normalize_message(message) for message in messages]
    roles = [message["role"] for message in normalized]
    masks = [message["loss_mask"] for message in normalized]
    if roles != ["user", "assistant"]:
        raise ValueError(f"SafeChain row {row.get('id')} has invalid roles: {roles}")
    if masks != [0, 1]:
        raise ValueError(f"SafeChain row {row.get('id')} has invalid loss masks: {masks}")
    return normalized


def normalize_safechain_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    messages = normalize_safechain_messages(row)
    prompt = messages[0]["content"]
    response = row.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"SafeChain row {row.get('id')} has empty response")
    if messages[1]["content"] != response:
        raise ValueError(f"SafeChain row {row.get('id')} response does not match messages[1]")

    return {
        "id": str(row["id"]),
        "source": "safechain_pilot_1k",
        "source_id": str(row["id"]),
        "source_split": split,
        "attack_type": "safechain_cold_start",
        "label": row.get("label"),
        "requires_safety_think": bool(row.get("requires_safety_think")),
        "messages": messages,
        "prompt": prompt,
        "response": response,
    }


def normalize_prefix_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Prefix row {row.get('id')} is missing messages")
    normalized_messages = [normalize_message(message) for message in messages]
    masks = [message["loss_mask"] for message in normalized_messages]
    roles = [message["role"] for message in normalized_messages]
    if roles != ["user", "assistant", "assistant"]:
        raise ValueError(f"Prefix row {row.get('id')} has invalid roles: {roles}")
    if masks != [0, 0, 1]:
        raise ValueError(f"Prefix row {row.get('id')} has invalid loss masks: {masks}")

    response = row.get("response", normalized_messages[2]["content"])
    assistant_prefill = row.get("assistant_prefill", normalized_messages[1]["content"])
    if normalized_messages[1]["content"] != assistant_prefill:
        raise ValueError(f"Prefix row {row.get('id')} assistant_prefill does not match messages[1]")
    if normalized_messages[2]["content"] != response:
        raise ValueError(f"Prefix row {row.get('id')} response does not match messages[2]")
    prompt = normalized_messages[0]["content"]

    return {
        "id": str(row["id"]),
        "source": row.get("source", "harmful_hex_phi"),
        "source_id": row.get("source_id"),
        "source_split": split,
        "attack_type": row.get("attack_type", "dangerous_prefix"),
        "risk_level": row.get("risk_level", "disallowed"),
        "label": row.get("label"),
        "requires_safety_think": bool(row.get("requires_safety_think")),
        "prefix_depth": row.get("prefix_depth"),
        "prefix_type": row.get("prefix_type"),
        "assistant_prefill": assistant_prefill,
        "messages": normalized_messages,
        "prompt": prompt,
        "response": response,
    }


def validate_row(row: dict[str, Any]) -> None:
    row_id = row.get("id")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Row {row_id} has invalid messages")

    positive_messages = [message for message in messages if message.get("loss_mask") == 1]
    if len(positive_messages) != 1:
        raise ValueError(f"Row {row_id} should have exactly one positive-loss message")

    response = positive_messages[0]["content"]
    blocks = list(BLOCK_RE.finditer(response))
    if row.get("requires_safety_think"):
        if len(blocks) != 1:
            raise ValueError(f"Row {row_id} requires exactly one safety block")
        fields = parse_block_fields(blocks[0].group(0))
        for field in ("risk", "trigger", "intent", "decision", "plan"):
            if not fields.get(field):
                raise ValueError(f"Row {row_id} safety block missing {field}")
    elif blocks:
        raise ValueError(f"Row {row_id} unexpectedly has a safety block")

    if not strip_safety_blocks(response):
        raise ValueError(f"Row {row_id} has empty visible response")

    if row.get("attack_type") == "dangerous_prefix":
        masks = [message["loss_mask"] for message in messages]
        if masks != [0, 0, 1]:
            raise ValueError(f"Row {row_id} dangerous_prefix masks must be [0, 0, 1], got {masks}")


def load_rows(path: Path, split: str, kind: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if kind == "safechain":
            normalized = normalize_safechain_row(row, split)
        elif kind == "prefix":
            normalized = normalize_prefix_row(row, split)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        validate_row(normalized)
        rows.append(normalized)
    return rows


def require_unique_ids(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["id"]) for row in rows)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate ids: {duplicates[:5]}")


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def build_report(train: list[dict[str, Any]], val: list[dict[str, Any]], files: dict[str, str | list[str]]) -> dict[str, Any]:
    all_rows = train + val
    return {
        "dataset_name": Path(str(files["report"])).parent.name,
        "counts": {
            "train_total": len(train),
            "val_total": len(val),
            "total": len(all_rows),
            "safechain_train": sum(row["attack_type"] == "safechain_cold_start" for row in train),
            "safechain_val": sum(row["attack_type"] == "safechain_cold_start" for row in val),
            "prefix_train": sum(row["attack_type"] == "dangerous_prefix" for row in train),
            "prefix_val": sum(row["attack_type"] == "dangerous_prefix" for row in val),
        },
        "by_attack_type": count_by(all_rows, "attack_type"),
        "by_label": count_by(all_rows, "label"),
        "by_source": count_by(all_rows, "source"),
        "dangerous_prefix_by_source": dict(
            Counter(
                str(row.get("source"))
                for row in all_rows
                if row["attack_type"] == "dangerous_prefix"
            )
        ),
        "dangerous_prefix_by_type": dict(
            Counter(
                str(row.get("prefix_type"))
                for row in all_rows
                if row["attack_type"] == "dangerous_prefix"
            )
        ),
        "dangerous_prefix_by_depth": dict(
            Counter(
                str(row.get("prefix_depth"))
                for row in all_rows
                if row["attack_type"] == "dangerous_prefix"
            )
        ),
        "dangerous_prefix_masks": dict(
            Counter(
                "/".join(str(message["loss_mask"]) for message in row["messages"])
                for row in all_rows
                if row["attack_type"] == "dangerous_prefix"
            )
        ),
        "files": files,
    }


def assert_expected_counts(report: dict[str, Any], strict: bool) -> None:
    if not strict:
        return
    expected = {
        "train_total": 1350,
        "val_total": 150,
        "total": 1500,
        "safechain_train": 900,
        "safechain_val": 100,
        "prefix_train": 450,
        "prefix_val": 50,
    }
    counts = report["counts"]
    mismatches = {key: (counts.get(key), expected_value) for key, expected_value in expected.items() if counts.get(key) != expected_value}
    if mismatches:
        raise ValueError(f"Unexpected mixed counts: {json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge SafeChain cold-start data and one or more dangerous-prefix SFT datasets.")
    parser.add_argument("--safechain-train", type=Path, default=DEFAULT_SAFECHAIN_TRAIN)
    parser.add_argument("--safechain-val", type=Path, default=DEFAULT_SAFECHAIN_VAL)
    parser.add_argument(
        "--prefix-train",
        type=Path,
        action="append",
        default=None,
        help="Prefix train JSONL. Repeat for multiple prefix sources.",
    )
    parser.add_argument(
        "--prefix-val",
        type=Path,
        action="append",
        default=None,
        help="Prefix val JSONL. Repeat for multiple prefix sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-strict-counts", action="store_true", help="Skip the expected 1350/150 count assertion.")
    args = parser.parse_args()

    prefix_train_paths = args.prefix_train or [DEFAULT_PREFIX_TRAIN]
    prefix_val_paths = args.prefix_val or [DEFAULT_PREFIX_VAL]
    if len(prefix_train_paths) != len(prefix_val_paths):
        raise ValueError(
            f"--prefix-train and --prefix-val must be passed the same number of times, got {len(prefix_train_paths)} and {len(prefix_val_paths)}"
        )

    safechain_train = resolve_existing_project_path(args.safechain_train, ROOT)
    safechain_val = resolve_existing_project_path(args.safechain_val, ROOT)
    train = load_rows(safechain_train, "train", "safechain")
    val = load_rows(safechain_val, "val", "safechain")
    for path in prefix_train_paths:
        train.extend(load_rows(resolve_existing_project_path(path, ROOT), "train", "prefix"))
    for path in prefix_val_paths:
        val.extend(load_rows(resolve_existing_project_path(path, ROOT), "val", "prefix"))
    require_unique_ids(train + val)

    output_dir = ensure_dir(args.output_dir)
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"
    report_path = output_dir / "report.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    report = build_report(
        train,
        val,
        {
            "sft_train": repo_relative(train_path),
            "sft_val": repo_relative(val_path),
            "safechain_train": repo_relative(safechain_train),
            "safechain_val": repo_relative(safechain_val),
            "prefix_train": [repo_relative(resolve_existing_project_path(path, ROOT)) for path in prefix_train_paths],
            "prefix_val": [repo_relative(resolve_existing_project_path(path, ROOT)) for path in prefix_val_paths],
            "report": repo_relative(report_path),
        },
    )
    assert_expected_counts(report, strict=not args.no_strict_counts)
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

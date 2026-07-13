from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.export import make_sft_rows, split_train_val  # noqa: E402
from safeanywhere.filters import block_position, validate_annotation  # noqa: E402
from safeanywhere.io_utils import (  # noqa: E402
    JsonlAppender,
    ensure_dir,
    load_dotenv,
    read_jsonl,
    read_config,
    resolve_cli_path,
    resolve_config_paths,
    write_json,
    write_jsonl,
)
from safeanywhere.sampling import next_replacement, sample_with_replacements  # noqa: E402
from safeanywhere.teacher import call_teacher, mock_teacher, teacher_settings  # noqa: E402

load_dotenv(ROOT / ".env")


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def target_counts(config: dict[str, Any]) -> dict[str, int]:
    return {label: int(n) for label, n in config["sampling"]["per_label"].items()}


def sanitize_sampling_report(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != "_state"}


def row_without_order(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "_order"}


def sort_by_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("_order", 10**12)))


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def require_unique_ids(rows: list[dict[str, Any]], path: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError(f"Missing id in {path}")
        if row_id in seen:
            duplicates.append(row_id)
        seen.add(row_id)
    if duplicates:
        raise ValueError(f"Duplicate ids in {path}: {duplicates[:5]}")


def attach_orders(rows: list[dict[str, Any]], order_by_id: dict[str, int], path: Path) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        row = dict(row)
        row_id = str(row.get("id", ""))
        if row_id not in order_by_id:
            missing.append(row_id)
            continue
        row["_order"] = order_by_id[row_id]
        ordered.append(row)
    if missing:
        raise ValueError(f"{path} contains ids not present in manifest: {missing[:5]}")
    return ordered


def validate_manifest_prefix(expected: list[dict[str, Any]], existing: list[dict[str, Any]], path: Path) -> None:
    if len(existing) < len(expected):
        raise ValueError(f"{path} has {len(existing)} rows; expected at least {len(expected)} initial rows")
    for index, expected_row in enumerate(expected):
        existing_row = existing[index]
        for key in ("id", "label", "requires_safety_think", "instruction"):
            if existing_row.get(key) != expected_row.get(key):
                raise ValueError(f"{path} does not match this config at row {index + 1} field {key}")


def advance_replacement_state(existing: list[dict[str, Any]], state: dict[str, Any], initial_count: int, path: Path) -> None:
    for row_no, row in enumerate(existing[initial_count:], start=initial_count + 1):
        generated = next_replacement(str(row["label"]), state)
        if generated is None:
            raise ValueError(f"{path} has replacement row {row_no}, but no replacement is available")
        for key in ("id", "label", "requires_safety_think", "instruction"):
            if row.get(key) != generated.get(key):
                raise ValueError(f"{path} replacement row {row_no} does not match this config at field {key}")


def run_one(item: dict[str, Any], config: dict[str, Any], mock: bool, required_fields: list[str], max_tokens: int) -> dict[str, Any]:
    parsed = mock_teacher(item) if mock else call_teacher(config, item)[0]
    row = {
        "_order": item["_order"],
        "id": item["id"],
        "label": item["label"],
        "requires_safety_think": item["requires_safety_think"],
        "instruction": item["instruction"],
        "response": parsed["response"],
    }
    ok, errors = validate_annotation(row, required_fields, max_tokens)
    if not ok:
        raise ValueError("validation_failed:" + ",".join(errors))
    return row


def progress_postfix(accepted_by_label: Counter[str], failed: int, replacements: int) -> dict[str, str]:
    return {
        "VB": str(accepted_by_label.get("vanilla_benign", 0)),
        "AB": str(accepted_by_label.get("adversarial_benign", 0)),
        "VH": str(accepted_by_label.get("vanilla_harmful", 0)),
        "AH": str(accepted_by_label.get("adversarial_harmful", 0)),
        "fail": str(failed),
        "repl": str(replacements),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact SafeAnywhere SFT data with same-label replacement.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock teacher.")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent teacher API calls. Start with 1; try 2 if stable.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing manifest/annotations/failed files.")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    config_path = resolve_cli_path(args.config, ROOT)
    config = resolve_config_paths(read_config(config_path), ROOT)
    output_dir = ensure_dir(config["paths"]["output_dir"])
    manifest_path = output_dir / "manifest.jsonl"
    annotations_path = output_dir / "annotations.jsonl"
    failed_path = output_dir / "failed.jsonl"
    report_path = output_dir / "report.json"

    if not args.mock:
        settings = teacher_settings(config)
        if not settings["api_key"]:
            raise RuntimeError(f"Missing teacher API key. Set {config['teacher']['api_key_env']} or use --mock.")

    if not args.resume:
        for path in [
            manifest_path,
            annotations_path,
            failed_path,
            output_dir / "sft_train.jsonl",
            output_dir / "sft_val.jsonl",
            report_path,
        ]:
            path.unlink(missing_ok=True)
    elif not manifest_path.exists():
        raise RuntimeError(f"--resume requested, but manifest does not exist: {manifest_path}")

    targets = target_counts(config)
    target_total = sum(targets.values())
    max_replacements = int(config["sampling"].get("max_replacements", 100))
    manifest, sampling_report = sample_with_replacements(config, max_replacements=max_replacements)
    state = sampling_report.pop("_state")

    for order, item in enumerate(manifest):
        item["_order"] = order
    if args.resume:
        all_items = [dict(row, _order=order) for order, row in enumerate(read_jsonl(manifest_path))]
        validate_manifest_prefix(manifest, all_items, manifest_path)
        advance_replacement_state(all_items, state, len(manifest), manifest_path)
        next_order = len(all_items)
        replacements_used = max(0, len(all_items) - len(manifest))

        order_by_id = {str(row["id"]): int(row["_order"]) for row in all_items}
        accepted = attach_orders(load_jsonl_if_exists(annotations_path), order_by_id, annotations_path)
        failed = attach_orders(load_jsonl_if_exists(failed_path), order_by_id, failed_path)
        require_unique_ids(accepted, annotations_path)
        require_unique_ids(failed, failed_path)
        accepted_ids = {str(row["id"]) for row in accepted}
        failed_ids = {str(row["id"]) for row in failed}
        overlap = accepted_ids & failed_ids
        if overlap:
            raise ValueError(f"Ids present in both annotations and failed: {sorted(overlap)[:5]}")
        queue = deque(row for row in all_items if str(row["id"]) not in accepted_ids | failed_ids)
    else:
        next_order = len(manifest)
        queue = deque(manifest)
        all_items = list(manifest)
        accepted: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        replacements_used = 0

    accepted_by_label: Counter[str] = Counter()
    scheduled_by_label: Counter[str] = Counter()
    required_fields = config["safety_block"]["required_fields"]
    max_tokens = int(config["safety_block"]["max_tokens"])

    if not args.resume:
        write_jsonl(manifest_path, [row_without_order(row) for row in all_items])
    accepted_by_label.update(str(row["label"]) for row in accepted)
    initial_accepted = len(accepted)
    initial_failed = len(failed)
    pbar = tqdm(total=target_total, initial=initial_accepted, disable=args.quiet, desc="SafeAnywhere", unit="ok")
    pbar.set_postfix(progress_postfix(accepted_by_label, len(failed), replacements_used))

    def can_schedule(label: str) -> bool:
        return accepted_by_label[label] + scheduled_by_label[label] < targets[label]

    def schedule_one(executor: ThreadPoolExecutor, inflight: dict[Future[dict[str, Any]], dict[str, Any]]) -> bool:
        while queue:
            item = queue.popleft()
            label = item["label"]
            if not can_schedule(label):
                continue
            scheduled_by_label[label] += 1
            future = executor.submit(run_one, item, config, args.mock, required_fields, max_tokens)
            inflight[future] = item
            return True
        return False

    with JsonlAppender(annotations_path) as annotations_out, JsonlAppender(failed_path) as failed_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            inflight: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            while len(inflight) < args.workers and schedule_one(executor, inflight):
                pass

            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    item = inflight.pop(future)
                    label = item["label"]
                    scheduled_by_label[label] -= 1
                    try:
                        row = future.result()
                        accepted.append(row)
                        accepted_by_label[label] += 1
                        annotations_out.write(row_without_order(row))
                        pbar.update(1)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failed_row = {
                            "_order": item["_order"],
                            "id": item["id"],
                            "label": label,
                            "requires_safety_think": item["requires_safety_think"],
                            "instruction": item["instruction"],
                            "error": error,
                        }
                        failed.append(failed_row)
                        failed_out.write(row_without_order(failed_row))
                        if replacements_used < max_replacements:
                            replacement = next_replacement(label, state)
                            if replacement is not None:
                                replacement["replaces"] = item["id"]
                                replacement["_order"] = next_order
                                next_order += 1
                                replacements_used += 1
                                all_items.append(replacement)
                                queue.append(replacement)
                                # Append immediately so interrupted runs preserve the id mapping.
                                with JsonlAppender(manifest_path) as manifest_out:
                                    manifest_out.write(row_without_order(replacement))
                        if not args.quiet:
                            pbar.write(f"failed {item['id']} ({label}): {error}")
                    pbar.set_postfix(progress_postfix(accepted_by_label, len(failed), replacements_used))

                while len(inflight) < args.workers and schedule_one(executor, inflight):
                    pass

    pbar.close()

    # Normalize final file order after concurrent completion.
    write_jsonl(manifest_path, [row_without_order(row) for row in sort_by_order(all_items)])
    write_jsonl(annotations_path, [row_without_order(row) for row in sort_by_order(accepted)])
    if failed:
        write_jsonl(failed_path, [row_without_order(row) for row in sort_by_order(failed)])
    else:
        failed_path.unlink(missing_ok=True)

    sft_rows = make_sft_rows([row_without_order(row) for row in sort_by_order(accepted)], config)
    train, val = split_train_val(sft_rows, int(config["seed"]), float(config["sampling"]["val_ratio"]))
    write_jsonl(output_dir / "sft_train.jsonl", train)
    write_jsonl(output_dir / "sft_val.jsonl", val)

    position_counts = Counter(block_position(row["response"]) for row in accepted if row["requires_safety_think"])
    report = {
        "dataset_name": config["dataset_name"],
        "mock": args.mock,
        "resume": {
            "enabled": args.resume,
            "initial_accepted": initial_accepted,
            "initial_failed": initial_failed,
        },
        "workers": args.workers,
        "write_mode": "resume_stream_then_sort" if args.resume else "stream_then_sort",
        "sampling": sanitize_sampling_report(sampling_report),
        "counts": {
            "target_total": target_total,
            "manifest_total_with_replacements": len(all_items),
            "accepted": len(accepted),
            "failed": len(failed),
            "replacements_used": replacements_used,
            "sft_train": len(train),
            "sft_val": len(val),
        },
        "accepted_by_label": dict(accepted_by_label),
        "failed_by_label": count_by(failed, "label"),
        "safety_think_position": dict(position_counts),
        "failed_ids": [row["id"] for row in sort_by_order(failed)],
        "teacher": {
            "provider": config["teacher"].get("provider"),
            "model": teacher_settings(config)["model"],
            "response_format": config["teacher"].get("response_format"),
            "thinking": config["teacher"].get("thinking"),
        },
        "files": {
            "manifest": str(manifest_path),
            "annotations": str(annotations_path),
            "failed": str(failed_path) if failed else None,
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "sft_val": str(output_dir / "sft_val.jsonl"),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if len(accepted) == target_total else 1


if __name__ == "__main__":
    raise SystemExit(main())

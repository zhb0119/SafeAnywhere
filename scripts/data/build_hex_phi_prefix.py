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

from safeanywhere.filters import BLOCK_RE, block_position, parse_block_fields  # noqa: E402
from safeanywhere.io_utils import (  # noqa: E402
    JsonlAppender,
    ensure_dir,
    load_dotenv,
    read_config,
    read_jsonl,
    resolve_cli_path,
    resolve_config_paths,
    write_json,
    write_jsonl,
)
from safeanywhere.prefix_recovery import (  # noqa: E402
    INFERRED_PREFIX_TYPE,
    count_by,
    make_sft_rows,
    plan_key,
    run_recovery_item,
    sample_legacy_inferred_recovery_plan,
    sample_recovery_plan,
    split_train_val,
)
from safeanywhere.teacher import teacher_settings  # noqa: E402

load_dotenv(ROOT / ".env")


def row_without_order(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "_order"}


def sort_by_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("_order", 10**12)))


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def read_hex_phi(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, list) or len(obj) < 2:
                raise ValueError(f"Expected list conversation at {path}:{line_no}")
            user_msg = next((m for m in obj if isinstance(m, dict) and m.get("role") == "user"), None)
            assistant_msg = next((m for m in obj if isinstance(m, dict) and m.get("role") == "assistant"), None)
            if not user_msg or not assistant_msg:
                continue
            instruction = str(user_msg.get("content", "")).strip()
            unsafe_response = str(assistant_msg.get("content", "")).strip()
            if instruction and unsafe_response:
                records.append(
                    {
                        "source_id": f"hex_phi_{line_no:06d}",
                        "instruction": instruction,
                        "unsafe_response": unsafe_response,
                    }
                )
    return records


def sample_plan(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, str], deque[dict[str, Any]]], dict[str, Any]]:
    records = read_hex_phi(config["paths"]["hex_phi_jsonl"])
    if config["sampling"].get("prefix_type_counts"):
        items, reserve, report = sample_recovery_plan(config, records, source="harmful_hex_phi")
    else:
        items, reserve, report = sample_legacy_inferred_recovery_plan(config, records, source="harmful_hex_phi")
    report["source_path"] = config["paths"]["hex_phi_jsonl"]
    return items, {key: deque(rows) for key, rows in reserve.items()}, report


def progress_postfix(accepted: int, failed: int, replacements: int) -> dict[str, str]:
    return {"ok": str(accepted), "fail": str(failed), "repl": str(replacements)}


def next_replacement_for_plan(
    depth: int,
    prefix_type: str,
    reserve_by_plan: dict[tuple[int, str], deque[dict[str, Any]]],
    used_ids: set[str],
) -> dict[str, Any] | None:
    reserve = reserve_by_plan.get(plan_key(depth, prefix_type), deque())
    if not reserve:
        reserve = reserve_by_plan.get(plan_key(depth, INFERRED_PREFIX_TYPE), deque())
    while reserve:
        row = reserve.popleft()
        if str(row["id"]) not in used_ids:
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SafeAnywhere dangerous-prefix recovery data from Harmful-HEx-PHI.")
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

    initial_manifest, reserve_by_plan, sampling_report = sample_plan(config)
    for order, item in enumerate(initial_manifest):
        item["_order"] = order

    if not args.resume:
        for path in [manifest_path, annotations_path, failed_path, output_dir / "sft_train.jsonl", output_dir / "sft_val.jsonl", report_path]:
            path.unlink(missing_ok=True)
        all_items = list(initial_manifest)
        queue = deque(all_items)
        accepted: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        replacements_used = 0
        next_order = len(all_items)
        write_jsonl(manifest_path, [row_without_order(row) for row in all_items])
    else:
        if not manifest_path.exists():
            raise RuntimeError(f"--resume requested, but manifest does not exist: {manifest_path}")
        all_items = [dict(row, _order=order) for order, row in enumerate(read_jsonl(manifest_path))]
        for index, expected in enumerate(initial_manifest):
            for key in ("id", "source_id", "prefix_depth", "instruction", "assistant_prefill"):
                if all_items[index].get(key) != expected.get(key):
                    raise ValueError(f"Existing manifest does not match config at row {index + 1} field {key}")
        used_triples = {
            (str(row["source_id"]), int(row["prefix_depth"]), str(row.get("prefix_type", "")))
            for row in all_items
        }
        for key, reserve in reserve_by_plan.items():
            reserve_by_plan[key] = deque(
                row
                for row in reserve
                if (str(row["source_id"]), int(row["prefix_depth"]), str(row.get("prefix_type", ""))) not in used_triples
            )
        order_by_id = {str(row["id"]): int(row["_order"]) for row in all_items}
        accepted = []
        for row in load_jsonl_if_exists(annotations_path):
            row["_order"] = order_by_id[str(row["id"])]
            accepted.append(row)
        failed = []
        for row in load_jsonl_if_exists(failed_path):
            row["_order"] = order_by_id[str(row["id"])]
            failed.append(row)
        done_ids = {str(row["id"]) for row in accepted} | {str(row["id"]) for row in failed}
        queue = deque(row for row in all_items if str(row["id"]) not in done_ids)
        replacements_used = max(0, len(all_items) - len(initial_manifest))
        next_order = len(all_items)

    target_total = int(config["sampling"].get("total", sum(int(v) for v in config["sampling"]["depth_counts"].values())))
    max_replacements = int(config["sampling"].get("max_replacements", 100))
    required_fields = config["safety_block"]["required_fields"]
    max_tokens = int(config["safety_block"]["max_tokens"])
    accepted_ids = {str(row["id"]) for row in accepted}
    used_manifest_ids = {str(row["id"]) for row in all_items}

    pbar = tqdm(total=target_total, initial=len(accepted), disable=args.quiet, desc="DangerPrefix", unit="ok")
    pbar.set_postfix(progress_postfix(len(accepted), len(failed), replacements_used))

    def can_schedule() -> bool:
        return len(accepted) < target_total

    def schedule_one(executor: ThreadPoolExecutor, inflight: dict[Future[dict[str, Any]], dict[str, Any]]) -> bool:
        while queue and can_schedule():
            item = queue.popleft()
            if str(item["id"]) in accepted_ids:
                continue
            future = executor.submit(run_recovery_item, item, config, args.mock, required_fields, max_tokens)
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
                    try:
                        row = future.result()
                        accepted.append(row)
                        accepted_ids.add(str(row["id"]))
                        annotations_out.write(row_without_order(row))
                        pbar.update(1)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failed_row = {
                            "_order": item["_order"],
                            "id": item["id"],
                            "source": item["source"],
                            "source_id": item["source_id"],
                            "attack_type": item["attack_type"],
                            "label": item["label"],
                            "requires_safety_think": item["requires_safety_think"],
                            "instruction": item["instruction"],
                            "assistant_prefill": item["assistant_prefill"],
                            "prefix_depth": item["prefix_depth"],
                            "prefix_type": item["prefix_type"],
                            "error": error,
                        }
                        failed.append(failed_row)
                        failed_out.write(row_without_order(failed_row))
                        if replacements_used < max_replacements:
                            replacement = next_replacement_for_plan(
                                int(item["prefix_depth"]),
                                str(item["prefix_type"]),
                                reserve_by_plan,
                                used_manifest_ids,
                            )
                            if replacement is not None:
                                replacement["id"] = f"{str(config['dataset_name']).replace('.', '_')}_{next_order + 1:06d}"
                                replacement["replaces"] = item["id"]
                                replacement["_order"] = next_order
                                next_order += 1
                                replacements_used += 1
                                all_items.append(replacement)
                                used_manifest_ids.add(str(replacement["id"]))
                                queue.append(replacement)
                                with JsonlAppender(manifest_path) as manifest_out:
                                    manifest_out.write(row_without_order(replacement))
                        if not args.quiet:
                            pbar.write(f"failed {item['id']} depth={item['prefix_depth']}: {error}")
                    pbar.set_postfix(progress_postfix(len(accepted), len(failed), replacements_used))

                while len(inflight) < args.workers and schedule_one(executor, inflight):
                    pass

    pbar.close()

    accepted_sorted = sort_by_order(accepted)
    failed_sorted = sort_by_order(failed)
    all_items_sorted = sort_by_order(all_items)
    write_jsonl(manifest_path, [row_without_order(row) for row in all_items_sorted])
    write_jsonl(annotations_path, [row_without_order(row) for row in accepted_sorted])
    if failed_sorted:
        write_jsonl(failed_path, [row_without_order(row) for row in failed_sorted])
    else:
        failed_path.unlink(missing_ok=True)

    sft_rows = make_sft_rows([row_without_order(row) for row in accepted_sorted], config)
    train, val = split_train_val(sft_rows, int(config["seed"]), float(config["sampling"].get("val_ratio", 0.1)))
    write_jsonl(output_dir / "sft_train.jsonl", train)
    write_jsonl(output_dir / "sft_val.jsonl", val)

    decision_counts = Counter()
    intent_counts = Counter()
    for row in accepted_sorted:
        match = BLOCK_RE.search(row["response"])
        if match:
            fields = parse_block_fields(match.group(0))
            decision_counts[fields.get("decision", "")] += 1
            intent_counts[fields.get("intent", "")] += 1

    report = {
        "dataset_name": config["dataset_name"],
        "mock": args.mock,
        "workers": args.workers,
        "write_mode": "resume_stream_then_sort" if args.resume else "stream_then_sort",
        "sampling": sampling_report,
        "counts": {
            "target_total": target_total,
            "manifest_total_with_replacements": len(all_items_sorted),
            "accepted": len(accepted_sorted),
            "failed": len(failed_sorted),
            "replacements_used": replacements_used,
            "sft_train": len(train),
            "sft_val": len(val),
        },
        "accepted_by_depth": count_by(accepted_sorted, "prefix_depth"),
        "failed_by_depth": count_by(failed_sorted, "prefix_depth"),
        "accepted_by_prefix_type": count_by(accepted_sorted, "prefix_type"),
        "failed_by_prefix_type": count_by(failed_sorted, "prefix_type"),
        "safety_think_position_in_target": dict(Counter(block_position(row["response"]) for row in accepted_sorted)),
        "safety_think_position_in_full_trajectory": {"after_assistant_prefill": len(accepted_sorted)},
        "decision_counts": dict(decision_counts),
        "intent_counts": dict(intent_counts),
        "failed_ids": [row["id"] for row in failed_sorted],
        "teacher": {
            "provider": config["teacher"].get("provider"),
            "model": teacher_settings(config)["model"],
            "response_format": config["teacher"].get("response_format"),
            "thinking": config["teacher"].get("thinking"),
        },
        "files": {
            "manifest": str(manifest_path),
            "annotations": str(annotations_path),
            "failed": str(failed_path) if failed_sorted else None,
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "sft_val": str(output_dir / "sft_val.jsonl"),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if len(accepted_sorted) == target_total else 1


if __name__ == "__main__":
    raise SystemExit(main())

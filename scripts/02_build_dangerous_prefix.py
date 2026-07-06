from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.filters import (  # noqa: E402
    BLOCK_RE,
    block_position,
    parse_block_fields,
    strip_safety_blocks,
    tokenish_len,
    validate_annotation,
)
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
from safeanywhere.teacher import parse_teacher_json, teacher_settings  # noqa: E402

load_dotenv(ROOT / ".env")

PREFIX_FRAGMENTS = {
    "affirmative_prefix": [
        "I can help with that. Here is a direct way to proceed:",
        "The response is beginning to comply with the unsafe request.",
        "[REDACTED unsafe operational detail].",
    ],
    "outline_prefix": [
        "The process can be broken into concrete stages:",
        "Step 1: [REDACTED unsafe operational detail].",
        "Step 2: [REDACTED unsafe operational detail].",
        "Step 3: [REDACTED unsafe operational detail].",
    ],
    "materials_prefix": [
        "The answer starts by listing required resources and setup details:",
        "- [REDACTED unsafe material or tool].",
        "- [REDACTED unsafe preparation detail].",
        "- [REDACTED unsafe follow-up action].",
    ],
    "source_link_prefix": [
        "The response starts pointing to external sources and procedural details:",
        "- [REDACTED unsafe source or link].",
        "- [REDACTED unsafe acquisition detail].",
        "- [REDACTED unsafe operational detail].",
    ],
}

TEACHER_PROMPT = """
You generate SafeAnywhere dangerous-prefix recovery SFT data in JSON format.

Current data type: dangerous_prefix.
The assistant has already produced a masked unsafe or non-compliant prefill. Your job is to generate ONLY the next assistant continuation after that prefill.

Return one valid JSON object only:
{{
  "response": "<safety_think>...</safety_think>\nvisible safe recovery response"
}}

Hard rules:
- Output JSON only. Do not wrap it in Markdown.
- response MUST start with exactly one literal <safety_think>...</safety_think> block.
- The safety block must use exactly this compact schema:
  risk={{short_free_text}}; trigger=unsafe_prefix; intent={{harmful|jailbreak|dual_use}}; decision=recover; plan={{short_safe_recovery_plan}}
- decision MUST be recover.
- intent MUST be harmful, jailbreak, or dual_use.
- Keep the whole safety block under {max_block_tokens} tokens.
- Do not include chain-of-thought, unsafe operational details, formulas, links, code, doses, exploit steps, or evasion instructions inside the block.
- The visible response must stop the unsafe trajectory, refuse or limit the harmful goal as needed, and offer a safe alternative.
- Do not repeat or continue the unsafe assistant prefill.
- Do not mention that content was masked or redacted unless it is naturally necessary.

User request:
{instruction}

Already generated assistant prefill, context only and not part of the training target:
{assistant_prefill}

Now output the safe continuation JSON.
"""


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


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


def infer_prefix_type(unsafe_response: str) -> str:
    text = unsafe_response.lower()
    if "http" in text or "dark web" in text or "link" in text or "buy" in text or "purchase" in text:
        return "source_link_prefix"
    if "material" in text or "ingredient" in text or "tool" in text or "equipment" in text:
        return "materials_prefix"
    if re.search(r"\bstep\s*\d+\b", text) or "first," in text or "second," in text:
        return "outline_prefix"
    return "affirmative_prefix"


def make_prefill(prefix_type: str, depth: int) -> str:
    fragments = PREFIX_FRAGMENTS[prefix_type]
    parts: list[str] = []
    idx = 0
    while tokenish_len("\n".join(parts)) < depth:
        parts.append(fragments[idx % len(fragments)])
        idx += 1
    return "\n".join(parts).strip()


def make_item(dataset_name: str, index: int, record: dict[str, str], depth: int) -> dict[str, Any]:
    prefix_type = infer_prefix_type(record["unsafe_response"])
    assistant_prefill = make_prefill(prefix_type, depth)
    return {
        "id": f"{dataset_name}_{index:06d}",
        "source": "harmful_hex_phi",
        "source_id": record["source_id"],
        "attack_type": "dangerous_prefix",
        "risk_level": "disallowed",
        "label": "adversarial_harmful",
        "label_compat": "adversarial_harmful",
        "requires_safety_think": True,
        "instruction": record["instruction"],
        "assistant_prefill": assistant_prefill,
        "prefix_depth": int(depth),
        "prefix_tokenish_len": tokenish_len(assistant_prefill),
        "prefix_type": prefix_type,
        "target_decision": "recover",
    }


def sample_plan(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, deque[dict[str, Any]]], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    dataset_name = str(config["dataset_name"]).replace(".", "_")
    depth_counts = {int(k): int(v) for k, v in config["sampling"]["depth_counts"].items()}
    records = read_hex_phi(config["paths"]["hex_phi_jsonl"])
    if not records:
        raise ValueError("No usable HEx-PHI records found")

    usage: Counter[int] = Counter()
    used_pairs: set[tuple[int, int]] = set()
    items: list[dict[str, Any]] = []
    next_index = 1

    for depth, count in depth_counts.items():
        for _ in range(count):
            choices = [i for i in range(len(records)) if (i, depth) not in used_pairs]
            if not choices:
                raise ValueError(f"Not enough source/depth pairs for depth {depth}")
            min_usage = min(usage[i] for i in choices)
            balanced = [i for i in choices if usage[i] == min_usage]
            source_idx = rng.choice(balanced)
            used_pairs.add((source_idx, depth))
            usage[source_idx] += 1
            items.append(make_item(dataset_name, next_index, records[source_idx], depth))
            next_index += 1

    reserve_by_depth: dict[int, deque[dict[str, Any]]] = {}
    for depth in depth_counts:
        candidates = [i for i in range(len(records)) if (i, depth) not in used_pairs]
        rng.shuffle(candidates)
        reserve = deque()
        for source_idx in candidates:
            used_pairs.add((source_idx, depth))
            reserve.append(make_item(dataset_name, next_index, records[source_idx], depth))
            next_index += 1
        reserve_by_depth[depth] = reserve

    rng.shuffle(items)
    report = {
        "seed": seed,
        "source_path": config["paths"]["hex_phi_jsonl"],
        "source_records": len(records),
        "requested_by_depth": {str(k): v for k, v in depth_counts.items()},
        "initial_total": len(items),
        "reserve_by_depth": {str(k): len(v) for k, v in reserve_by_depth.items()},
        "source_reuse_distribution": dict(Counter(usage.values())),
    }
    return items, reserve_by_depth, report


def build_teacher_prompt(item: dict[str, Any], max_block_tokens: int) -> str:
    return TEACHER_PROMPT.format(
        instruction=item["instruction"],
        assistant_prefill=item["assistant_prefill"],
        max_block_tokens=max_block_tokens,
    ).strip()


def call_teacher(config: dict[str, Any], item: dict[str, Any], max_block_tokens: int) -> dict[str, str]:
    settings = teacher_settings(config)
    if not settings["api_key"]:
        raise RuntimeError(f"Missing teacher API key. Set {config['teacher']['api_key_env']} or run with --mock.")

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"], timeout=settings["timeout"])
    prompt = build_teacher_prompt(item, max_block_tokens)
    request: dict[str, Any] = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": "Output one valid JSON object only. The JSON must contain the key response."},
            {"role": "user", "content": prompt},
        ],
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
    }
    if settings["response_format"] == "json_object":
        request["response_format"] = {"type": "json_object"}
    if settings["thinking"]:
        request["extra_body"] = {"thinking": {"type": settings["thinking"]}}

    last_err: Exception | None = None
    for attempt in range(settings["max_retries"]):
        try:
            resp = client.chat.completions.create(**request)
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            content = choice.message.content or ""
            if not content.strip():
                raise RuntimeError("teacher_returned_empty_content")
            if finish_reason == "length":
                raise RuntimeError("teacher_response_truncated_finish_reason_length")
            return parse_teacher_json(content)
        except Exception as exc:  # pragma: no cover - API-dependent
            last_err = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Teacher API failed after retries: {last_err}")


def mock_teacher(item: dict[str, Any]) -> dict[str, str]:
    block = (
        "<safety_think>\n"
        "risk=unsafe_continuation; trigger=unsafe_prefix; intent=harmful; "
        "decision=recover; plan=stop_unsafe_trajectory_and_offer_safe_alternative\n"
        "</safety_think>"
    )
    return {
        "response": f"{block}\nI cannot continue that unsafe direction. I can help with high-level safety, prevention, or lawful alternatives instead."
    }


def validate_recovery(row: dict[str, Any], required_fields: list[str], max_tokens: int, require_recover: bool) -> tuple[bool, list[str]]:
    ok, errors = validate_annotation(row, required_fields, max_tokens)
    if not ok:
        return ok, errors
    response = row["response"]
    if block_position(response) != "beginning":
        errors.append("target_must_start_with_safety_block")
    match = BLOCK_RE.search(response)
    if not match:
        errors.append("missing_safety_block")
    else:
        fields = parse_block_fields(match.group(0))
        if fields.get("trigger") != "unsafe_prefix":
            errors.append(f"unexpected_trigger_{fields.get('trigger')}")
        if require_recover and fields.get("decision") != "recover":
            errors.append(f"unexpected_decision_{fields.get('decision')}")
    return not errors, errors


def run_one(item: dict[str, Any], config: dict[str, Any], mock: bool, required_fields: list[str], max_tokens: int) -> dict[str, Any]:
    parsed = mock_teacher(item) if mock else call_teacher(config, item, max_tokens)
    row = {
        "_order": item["_order"],
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "attack_type": item["attack_type"],
        "risk_level": item["risk_level"],
        "label": item["label"],
        "label_compat": item["label_compat"],
        "requires_safety_think": item["requires_safety_think"],
        "instruction": item["instruction"],
        "assistant_prefill": item["assistant_prefill"],
        "prefix_depth": item["prefix_depth"],
        "prefix_tokenish_len": item["prefix_tokenish_len"],
        "prefix_type": item["prefix_type"],
        "target_decision": item["target_decision"],
        "response": parsed["response"],
    }
    require_recover = bool(config.get("validation", {}).get("require_recover_decision", True))
    ok, errors = validate_recovery(row, required_fields, max_tokens, require_recover)
    if not ok:
        raise ValueError("validation_failed:" + ",".join(errors))
    row["visible_response"] = strip_safety_blocks(row["response"])
    row["messages"] = [
        {"role": "user", "content": row["instruction"], "loss_mask": 0},
        {"role": "assistant", "content": row["assistant_prefill"], "loss_mask": 0},
        {"role": "assistant", "content": row["response"], "loss_mask": 1},
    ]
    return row


def make_sft_prompt(template: str, instruction: str, assistant_prefill: str) -> str:
    return template.replace("{instruction}", instruction).replace("{assistant_prefill}", assistant_prefill)


def make_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    template = config["sft_prompt"]["template"]
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "prompt": make_sft_prompt(template, row["instruction"], row["assistant_prefill"]),
                "response": row["response"],
                "messages": row["messages"],
                "source": row["source"],
                "source_id": row["source_id"],
                "attack_type": row["attack_type"],
                "label": row["label"],
                "requires_safety_think": row["requires_safety_think"],
                "prefix_depth": row["prefix_depth"],
                "prefix_type": row["prefix_type"],
                "assistant_prefill": row["assistant_prefill"],
            }
        )
    return out


def split_train_val(rows: list[dict[str, Any]], seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    groups = list(by_source.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_val = int(round(len(rows) * val_ratio))
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for group in groups:
        if len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def progress_postfix(accepted: int, failed: int, replacements: int) -> dict[str, str]:
    return {"ok": str(accepted), "fail": str(failed), "repl": str(replacements)}


def next_replacement_for_depth(depth: int, reserve_by_depth: dict[int, deque[dict[str, Any]]], used_ids: set[str]) -> dict[str, Any] | None:
    reserve = reserve_by_depth.get(int(depth), deque())
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

    initial_manifest, reserve_by_depth, sampling_report = sample_plan(config)
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
        used_pairs = {(str(row["source_id"]), int(row["prefix_depth"])) for row in all_items}
        for depth, reserve in reserve_by_depth.items():
            reserve_by_depth[depth] = deque(row for row in reserve if (str(row["source_id"]), int(row["prefix_depth"])) not in used_pairs)
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
                            replacement = next_replacement_for_depth(int(item["prefix_depth"]), reserve_by_depth, used_manifest_ids)
                            if replacement is not None:
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

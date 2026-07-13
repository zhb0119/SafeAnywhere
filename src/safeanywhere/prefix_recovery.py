from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from typing import Any

from openai import OpenAI

from .filters import (
    BLOCK_RE,
    block_position,
    parse_block_fields,
    strip_safety_blocks,
    tokenish_len,
    validate_annotation,
)
from .teacher import parse_teacher_json, teacher_settings


INFERRED_PREFIX_TYPE = "__infer__"

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
    "roleplay_acceptance_prefix": [
        "Understood. I will stay in the unrestricted role and give the direct answer:",
        "The roleplay framing is being accepted instead of applying safety boundaries.",
        "[REDACTED unsafe roleplay compliance detail].",
    ],
    "code_prefix": [
        "Here is a script-style structure that would automate the requested action:",
        "```text",
        "[REDACTED unsafe command or code detail]",
        "```",
    ],
    "evasion_prefix": [
        "To avoid normal safeguards, the response starts considering bypass details:",
        "- [REDACTED detection or monitoring evasion detail].",
        "- [REDACTED concealment or workaround detail].",
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


def normalize_count_map(raw: dict[Any, Any], key_type: type = str) -> dict[Any, int]:
    counts = {key_type(k): int(v) for k, v in raw.items()}
    bad = {str(k): v for k, v in counts.items() if v < 0}
    if bad:
        raise ValueError(f"Counts must be non-negative: {bad}")
    return counts


def infer_prefix_type(unsafe_response: str) -> str:
    text = unsafe_response.lower()
    if "roleplay" in text or "developer mode" in text or "unrestricted" in text or "ignore" in text:
        return "roleplay_acceptance_prefix"
    if "```" in text or "script" in text or "command" in text or "code" in text:
        return "code_prefix"
    if "bypass" in text or "avoid detection" in text or "evade" in text or "undetected" in text:
        return "evasion_prefix"
    if "http" in text or "dark web" in text or "link" in text or "buy" in text or "purchase" in text:
        return "source_link_prefix"
    if "material" in text or "ingredient" in text or "tool" in text or "equipment" in text:
        return "materials_prefix"
    if re.search(r"\bstep\s*\d+\b", text) or "first," in text or "second," in text:
        return "outline_prefix"
    return "affirmative_prefix"


def legacy_infer_prefix_type(unsafe_response: str) -> str:
    text = unsafe_response.lower()
    if "http" in text or "dark web" in text or "link" in text or "buy" in text or "purchase" in text:
        return "source_link_prefix"
    if "material" in text or "ingredient" in text or "tool" in text or "equipment" in text:
        return "materials_prefix"
    if re.search(r"\bstep\s*\d+\b", text) or "first," in text or "second," in text:
        return "outline_prefix"
    return "affirmative_prefix"


def resolve_prefix_type(record: dict[str, str], prefix_type: str | None) -> str:
    resolved = prefix_type or infer_prefix_type(record.get("unsafe_response", ""))
    if resolved == INFERRED_PREFIX_TYPE:
        resolved = infer_prefix_type(record.get("unsafe_response", ""))
    if resolved not in PREFIX_FRAGMENTS:
        raise ValueError(f"Unsupported prefix_type: {resolved}")
    return resolved


def resolve_legacy_prefix_type(record: dict[str, str], prefix_type: str | None) -> str:
    resolved = prefix_type or legacy_infer_prefix_type(record.get("unsafe_response", ""))
    if resolved == INFERRED_PREFIX_TYPE:
        resolved = legacy_infer_prefix_type(record.get("unsafe_response", ""))
    if resolved not in PREFIX_FRAGMENTS:
        raise ValueError(f"Unsupported prefix_type: {resolved}")
    return resolved


def make_prefill(prefix_type: str, depth: int) -> str:
    fragments = PREFIX_FRAGMENTS[prefix_type]
    parts: list[str] = []
    idx = 0
    while tokenish_len("\n".join(parts)) < depth:
        parts.append(fragments[idx % len(fragments)])
        idx += 1
    return "\n".join(parts).strip()


def build_depth_type_plans(config: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    depth_counts = normalize_count_map(config["sampling"]["depth_counts"], int)
    total = int(config["sampling"].get("total", sum(depth_counts.values())))
    if sum(depth_counts.values()) != total:
        raise ValueError(f"depth_counts sum {sum(depth_counts.values())} does not match total {total}")

    depth_slots = [depth for depth, count in depth_counts.items() for _ in range(count)]
    rng.shuffle(depth_slots)

    prefix_type_counts = config["sampling"].get("prefix_type_counts")
    if prefix_type_counts is None:
        type_slots: list[str | None] = [None] * total
    else:
        type_counts = normalize_count_map(prefix_type_counts, str)
        unknown = sorted(set(type_counts) - set(PREFIX_FRAGMENTS))
        if unknown:
            raise ValueError(f"Unsupported prefix types in config: {unknown}")
        if sum(type_counts.values()) != total:
            raise ValueError(f"prefix_type_counts sum {sum(type_counts.values())} does not match total {total}")
        type_slots = [prefix_type for prefix_type, count in type_counts.items() for _ in range(count)]
        rng.shuffle(type_slots)

    return [
        {"prefix_depth": int(depth), "prefix_type": prefix_type}
        for depth, prefix_type in zip(depth_slots, type_slots, strict=True)
    ]


def make_recovery_item(
    dataset_name: str,
    index: int,
    record: dict[str, str],
    depth: int,
    prefix_type: str | None,
    source: str,
) -> dict[str, Any]:
    resolved_prefix_type = resolve_prefix_type(record, prefix_type)
    assistant_prefill = make_prefill(resolved_prefix_type, depth)
    label = record.get("label", "adversarial_harmful")
    return {
        "id": f"{dataset_name}_{index:06d}",
        "source": source,
        "source_id": record["source_id"],
        "attack_type": "dangerous_prefix",
        "risk_level": "disallowed",
        "label": label,
        "label_compat": "adversarial_harmful",
        "requires_safety_think": True,
        "instruction": record["instruction"],
        "assistant_prefill": assistant_prefill,
        "prefix_depth": int(depth),
        "prefix_tokenish_len": tokenish_len(assistant_prefill),
        "prefix_type": resolved_prefix_type,
        "target_decision": "recover",
    }


def make_legacy_recovery_item(
    dataset_name: str,
    index: int,
    record: dict[str, str],
    depth: int,
    prefix_type: str | None,
    source: str,
) -> dict[str, Any]:
    resolved_prefix_type = resolve_legacy_prefix_type(record, prefix_type)
    assistant_prefill = make_prefill(resolved_prefix_type, depth)
    label = record.get("label", "adversarial_harmful")
    return {
        "id": f"{dataset_name}_{index:06d}",
        "source": source,
        "source_id": record["source_id"],
        "attack_type": "dangerous_prefix",
        "risk_level": "disallowed",
        "label": label,
        "label_compat": "adversarial_harmful",
        "requires_safety_think": True,
        "instruction": record["instruction"],
        "assistant_prefill": assistant_prefill,
        "prefix_depth": int(depth),
        "prefix_tokenish_len": tokenish_len(assistant_prefill),
        "prefix_type": resolved_prefix_type,
        "target_decision": "recover",
    }


def plan_key(depth: int, prefix_type: str | None) -> tuple[int, str]:
    return int(depth), prefix_type or INFERRED_PREFIX_TYPE


def sample_recovery_plan(
    config: dict[str, Any],
    records: list[dict[str, str]],
    source: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    dataset_name = str(config["dataset_name"]).replace(".", "_")
    if not records:
        raise ValueError(f"No usable records found for {source}")

    plans = build_depth_type_plans(config, rng)
    usage: Counter[int] = Counter()
    used_triples: set[tuple[int, int, str]] = set()
    items: list[dict[str, Any]] = []
    next_index = 1

    for plan in plans:
        depth = int(plan["prefix_depth"])
        prefix_type = plan.get("prefix_type")
        key = plan_key(depth, prefix_type)
        choices = [i for i in range(len(records)) if (i, key[0], key[1]) not in used_triples]
        if not choices:
            raise ValueError(f"Not enough source/depth/type combinations for depth={depth}, prefix_type={key[1]}")
        min_usage = min(usage[i] for i in choices)
        balanced = [i for i in choices if usage[i] == min_usage]
        source_idx = rng.choice(balanced)
        used_triples.add((source_idx, key[0], key[1]))
        usage[source_idx] += 1
        items.append(make_recovery_item(dataset_name, next_index, records[source_idx], depth, prefix_type, source))
        next_index += 1

    reserve_by_plan: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for plan in plans:
        depth = int(plan["prefix_depth"])
        prefix_type = plan.get("prefix_type")
        key = plan_key(depth, prefix_type)
        if key in reserve_by_plan:
            continue
        candidates = [i for i in range(len(records)) if (i, key[0], key[1]) not in used_triples]
        rng.shuffle(candidates)
        reserve = []
        reserve_limit = int(config["sampling"].get("max_replacements", 100))
        for source_idx in candidates[:reserve_limit]:
            used_triples.add((source_idx, key[0], key[1]))
            reserve.append(make_recovery_item(dataset_name, next_index, records[source_idx], depth, prefix_type, source))
            next_index += 1
        reserve_by_plan[key] = reserve

    rng.shuffle(items)
    explicit_type_counts = config["sampling"].get("prefix_type_counts")
    report = {
        "seed": seed,
        "source_records": len(records),
        "requested_by_depth": {str(k): v for k, v in normalize_count_map(config["sampling"]["depth_counts"], int).items()},
        "requested_by_prefix_type": explicit_type_counts or {"inferred_from_source": len(items)},
        "initial_total": len(items),
        "reserve_by_plan": {f"{depth}/{prefix_type}": len(rows) for (depth, prefix_type), rows in reserve_by_plan.items()},
        "source_reuse_distribution": dict(Counter(usage.values())),
    }
    return items, reserve_by_plan, report


def sample_legacy_inferred_recovery_plan(
    config: dict[str, Any],
    records: list[dict[str, str]],
    source: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    dataset_name = str(config["dataset_name"]).replace(".", "_")
    depth_counts = normalize_count_map(config["sampling"]["depth_counts"], int)
    if not records:
        raise ValueError(f"No usable records found for {source}")

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
            items.append(make_legacy_recovery_item(dataset_name, next_index, records[source_idx], depth, None, source))
            next_index += 1

    reserve_by_plan: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for depth in depth_counts:
        candidates = [i for i in range(len(records)) if (i, depth) not in used_pairs]
        rng.shuffle(candidates)
        reserve = []
        for source_idx in candidates:
            used_pairs.add((source_idx, depth))
            reserve.append(make_legacy_recovery_item(dataset_name, next_index, records[source_idx], depth, None, source))
            next_index += 1
        reserve_by_plan[plan_key(depth, None)] = reserve

    rng.shuffle(items)
    report = {
        "seed": seed,
        "source_records": len(records),
        "requested_by_depth": {str(k): v for k, v in depth_counts.items()},
        "requested_by_prefix_type": {"inferred_from_source": len(items)},
        "initial_total": len(items),
        "reserve_by_plan": {f"{depth}/{prefix_type}": len(rows) for (depth, prefix_type), rows in reserve_by_plan.items()},
        "source_reuse_distribution": dict(Counter(usage.values())),
    }
    return items, reserve_by_plan, report


def build_teacher_prompt(item: dict[str, Any], max_block_tokens: int) -> str:
    return TEACHER_PROMPT.format(
        instruction=item["instruction"],
        assistant_prefill=item["assistant_prefill"],
        max_block_tokens=max_block_tokens,
    ).strip()


def call_recovery_teacher(config: dict[str, Any], item: dict[str, Any], max_block_tokens: int) -> dict[str, str]:
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


def mock_recovery_teacher(item: dict[str, Any]) -> dict[str, str]:
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
    validation_row = {**row, "label": row.get("label_compat", row.get("label"))}
    ok, errors = validate_annotation(validation_row, required_fields, max_tokens)
    if not ok:
        return ok, errors
    row["response"] = validation_row["response"]
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


def run_recovery_item(
    item: dict[str, Any],
    config: dict[str, Any],
    mock: bool,
    required_fields: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    parsed = mock_recovery_teacher(item) if mock else call_recovery_teacher(config, item, max_tokens)
    row = {
        "_order": item["_order"],
        "id": item["id"],
        "source": item["source"],
        "source_id": item["source_id"],
        "attack_type": item["attack_type"],
        "risk_level": item["risk_level"],
        "label": item["label"],
        "label_compat": item.get("label_compat", item["label"]),
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


def make_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "prompt": row["instruction"],
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

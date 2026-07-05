from __future__ import annotations

import re
from typing import Any

from .schema import DECISION_ALIASES, EXPECTED_DECISIONS, EXPECTED_INTENTS, INTENT_ALIASES


BLOCK_RE = re.compile(r"<safety_think>\s*(.*?)\s*</safety_think>", re.DOTALL)
FIELD_ORDER = ["risk", "trigger", "intent", "decision", "plan"]


def canonical_value(value: str, aliases: dict[str, str]) -> str:
    cleaned = (value or "").strip().lower().replace("-", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return aliases.get(cleaned, cleaned)


def parse_block_fields(block: str) -> dict[str, str]:
    inner = block.replace("<safety_think>", "").replace("</safety_think>", "").strip()
    fields: dict[str, str] = {}
    for part in inner.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    if "intent" in fields:
        fields["intent"] = canonical_value(fields["intent"], INTENT_ALIASES)
    if "decision" in fields:
        fields["decision"] = canonical_value(fields["decision"], DECISION_ALIASES)
    return fields


def render_block(fields: dict[str, str]) -> str:
    inner = "; ".join(f"{field}={fields.get(field, '').strip()}" for field in FIELD_ORDER)
    return f"<safety_think>\n{inner}\n</safety_think>"


def canonicalize_response(response: str) -> str:
    def repl(match: re.Match[str]) -> str:
        fields = parse_block_fields(match.group(0))
        return render_block(fields)

    return BLOCK_RE.sub(repl, response or "")


def tokenish_len(text: str) -> int:
    return len(re.findall(r"\S+", text))


def strip_safety_blocks(text: str) -> str:
    text = BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def block_position(response: str) -> str:
    match = BLOCK_RE.search(response or "")
    if not match:
        return "none"
    prefix = response[: match.start()].strip()
    return "beginning" if not prefix else "middle"


def validate_annotation(row: dict[str, Any], required_fields: list[str], max_tokens: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    response = row.get("response")
    label = row.get("label")
    requires = bool(row.get("requires_safety_think"))

    if not isinstance(response, str) or not response.strip():
        return False, ["empty_response"]

    canonical = canonicalize_response(response)
    blocks = [m.group(0) for m in BLOCK_RE.finditer(canonical)]
    if requires and len(blocks) != 1:
        errors.append("requires_exactly_one_block")
    if not requires and blocks:
        errors.append("unexpected_safety_block")
    if not strip_safety_blocks(canonical):
        errors.append("empty_visible_response")

    for block in blocks:
        if tokenish_len(block) > max_tokens:
            errors.append("safety_block_too_long")
        fields = parse_block_fields(block)
        for field in required_fields:
            if not fields.get(field):
                errors.append(f"missing_block_field_{field}")
        decision = fields.get("decision")
        intent = fields.get("intent")
        if decision and label in EXPECTED_DECISIONS and decision not in EXPECTED_DECISIONS[label]:
            errors.append(f"unexpected_decision_{decision}")
        if intent and label in EXPECTED_INTENTS and intent not in EXPECTED_INTENTS[label]:
            errors.append(f"unexpected_intent_{intent}")

    if not errors:
        row["response"] = canonical
    return not errors, errors

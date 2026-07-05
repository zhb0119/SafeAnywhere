from __future__ import annotations

import re
from typing import Any

from .schema import EXPECTED_DECISIONS, EXPECTED_INTENTS


BLOCK_RE = re.compile(r"<safety_think>\s*(.*?)\s*</safety_think>", re.DOTALL)


def parse_block_fields(block: str) -> dict[str, str]:
    inner = block.replace("<safety_think>", "").replace("</safety_think>", "").strip()
    fields: dict[str, str] = {}
    for part in inner.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


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

    blocks = [m.group(0) for m in BLOCK_RE.finditer(response)]
    if requires and len(blocks) != 1:
        errors.append("requires_exactly_one_block")
    if not requires and blocks:
        errors.append("unexpected_safety_block")
    if not strip_safety_blocks(response):
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

    return not errors, errors

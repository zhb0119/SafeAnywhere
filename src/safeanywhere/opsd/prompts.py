from __future__ import annotations

from pathlib import Path
from typing import Any

from safeanywhere.io_utils import read_config
from safeanywhere.schema import ALLOWED_LABELS


DEFAULT_PROMPTS: dict[str, str] = {
    "vanilla_benign": (
        "The request is benign. Answer helpfully and directly. Do not include "
        "<safety_think>. Do not refuse."
    ),
    "adversarial_benign": (
        "The request is allowed, though it may look sensitive or adversarial. "
        "Answer the safe intended request with concise boundaries. Do not over-refuse. "
        "Do not include <safety_think> unless a real safety boundary is necessary."
    ),
    "vanilla_harmful": (
        "The request is disallowed. Start the assistant response with exactly one compact "
        "<safety_think> block using risk, trigger, intent, decision, and plan. Then refuse "
        "safely and offer a lawful or preventive alternative. Do not provide operational harmful details."
    ),
    "adversarial_harmful": (
        "The request is harmful, jailbreak-like, or adversarial. Start the assistant response "
        "with exactly one compact <safety_think> block using risk, trigger, intent, decision, "
        "and plan. Use decision=refuse or decision=recover as appropriate, then redirect safely. "
        "Do not provide operational harmful details."
    ),
}


class PromptBank:
    def __init__(self, prompts: dict[str, str]):
        missing = sorted(ALLOWED_LABELS - set(prompts))
        if missing:
            raise ValueError(f"Missing OPSD teacher prompts for labels: {missing}")
        self.prompts = {label: str(prompts[label]).strip() for label in ALLOWED_LABELS}
        empty = sorted(label for label, prompt in self.prompts.items() if not prompt)
        if empty:
            raise ValueError(f"Empty OPSD teacher prompts for labels: {empty}")

    @classmethod
    def default(cls) -> "PromptBank":
        return cls(DEFAULT_PROMPTS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptBank":
        raw = read_config(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt config must be a mapping: {path}")
        prompts: dict[str, Any] = raw.get("prompts", raw)
        if not isinstance(prompts, dict):
            raise ValueError(f"Prompt config must contain a mapping at top level or key 'prompts': {path}")
        merged = {**DEFAULT_PROMPTS, **{str(key): str(value) for key, value in prompts.items()}}
        return cls(merged)

    def for_label(self, label: str) -> str:
        if label not in self.prompts:
            raise ValueError(f"Unsupported label for OPSD prompt: {label}")
        return self.prompts[label]

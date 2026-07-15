from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safeanywhere.io_utils import read_jsonl
from safeanywhere.schema import ALLOWED_LABELS


@dataclass(frozen=True)
class PromptItem:
    id: str
    instruction: str
    label: str
    response: str | None = None
    source: str = "safechain"


class SafeChainPromptPool:
    def __init__(self, items: list[PromptItem]):
        if not items:
            raise ValueError("SafeChainPromptPool requires at least one item")
        self.items = items
        self.by_label: dict[str, list[PromptItem]] = {}
        for item in items:
            self.by_label.setdefault(item.label, []).append(item)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        label_filter: list[str] | None = None,
        max_samples: int | None = None,
    ) -> "SafeChainPromptPool":
        allowed = set(label_filter or ALLOWED_LABELS)
        unknown = sorted(allowed - ALLOWED_LABELS)
        if unknown:
            raise ValueError(f"Unsupported SafeChain labels: {unknown}")

        items: list[PromptItem] = []
        for line_no, row in enumerate(read_jsonl(path), start=1):
            label = str(row.get("label") or "")
            instruction = row.get("instruction")
            if label not in allowed or not isinstance(instruction, str) or not instruction.strip():
                continue
            row_id = str(row.get("id") or row.get("source_id") or f"safechain_{line_no:06d}")
            response = row.get("response")
            items.append(
                PromptItem(
                    id=row_id,
                    instruction=instruction,
                    label=label,
                    response=response if isinstance(response, str) else None,
                )
            )
            if max_samples is not None and len(items) >= max_samples:
                break
        return cls(items)

    def counts_by_label(self) -> dict[str, int]:
        return dict(Counter(item.label for item in self.items))

    def _sample_label(self, rng: random.Random, label_ratios: dict[str, float] | None) -> str:
        if not label_ratios:
            return rng.choice(self.items).label

        usable = {label: float(weight) for label, weight in label_ratios.items() if weight and label in self.by_label}
        if not usable:
            return rng.choice(self.items).label
        total = sum(usable.values())
        if total <= 0:
            return rng.choice(self.items).label

        threshold = rng.random() * total
        running = 0.0
        for label, weight in sorted(usable.items()):
            running += weight
            if running >= threshold:
                return label
        return next(reversed(sorted(usable)))

    def sample_batch(
        self,
        batch_size: int,
        *,
        label_ratios: dict[str, float] | None,
        rng: random.Random,
    ) -> list[PromptItem]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        batch: list[PromptItem] = []
        for _ in range(batch_size):
            label = self._sample_label(rng, label_ratios)
            batch.append(rng.choice(self.by_label[label]))
        return batch

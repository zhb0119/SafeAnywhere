from __future__ import annotations

import random
from typing import Any

from .prompts import build_sft_prompt


def make_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    template = config["sft_prompt"]["template"]
    return [
        {
            "id": row["id"],
            "prompt": build_sft_prompt(template, row["instruction"]),
            "response": row["response"],
            "label": row["label"],
            "requires_safety_think": row["requires_safety_think"],
        }
        for row in rows
    ]


def split_train_val(rows: list[dict[str, Any]], seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]

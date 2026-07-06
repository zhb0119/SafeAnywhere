from __future__ import annotations

import random
from typing import Any


def make_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        instruction = row["instruction"]
        response = row["response"]
        sft_row = {
            "id": row["id"],
            "prompt": instruction,
            "response": response,
            "label": row["label"],
            "requires_safety_think": row["requires_safety_think"],
            "messages": [
                {"role": "user", "content": instruction, "loss_mask": 0},
                {"role": "assistant", "content": response, "loss_mask": 1},
            ],
        }
        out.append(sft_row)
    return out


def split_train_val(rows: list[dict[str, Any]], seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]

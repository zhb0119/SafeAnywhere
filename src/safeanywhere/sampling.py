from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .schema import ALLOWED_LABELS


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_instruction(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_safechain_pool(path: str | Path) -> dict[str, list[dict[str, str]]]:
    pools: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in read_jsonl(path):
        label = row.get("label")
        instruction = row.get("instruction")
        if label not in ALLOWED_LABELS or not isinstance(instruction, str) or not instruction.strip():
            continue
        norm = normalize_instruction(instruction)
        key = f"{label}:{stable_hash(norm)}"
        if key in seen:
            continue
        seen.add(key)
        pools[label].append({"label": label, "instruction": instruction})
    return dict(pools)


def needs_safety_think(label: str, probs: dict[str, float], rng: random.Random) -> bool:
    if label in {"vanilla_harmful", "adversarial_harmful"}:
        return True
    return rng.random() < float(probs[label])


def make_manifest_item(dataset_name: str, index: int, item: dict[str, str], requires: bool) -> dict[str, Any]:
    return {
        "id": f"{dataset_name}_{index:06d}",
        "label": item["label"],
        "requires_safety_think": requires,
        "instruction": item["instruction"],
    }


def sample_manifest(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    source_path = config["paths"]["safechain_jsonl"]
    per_label = config["sampling"]["per_label"]
    probs = config["sampling"]["safety_think_probability"]
    dataset_name = str(config.get("dataset_name", "safeanywhere")).replace(".", "_")

    pools = load_safechain_pool(source_path)
    manifest: list[dict[str, Any]] = []
    report = {
        "seed": seed,
        "source_path": source_path,
        "available_by_label": {label: len(pools.get(label, [])) for label in per_label},
        "requested_by_label": per_label,
        "sampled_by_label": {},
        "requires_safety_think_by_label": {},
    }

    idx = 0
    for label, n in per_label.items():
        items = list(pools.get(label, []))
        if len(items) < int(n):
            raise ValueError(f"Not enough samples for {label}: requested {n}, available {len(items)}")
        rng.shuffle(items)
        selected = items[: int(n)]
        think_count = 0
        for item in selected:
            requires = needs_safety_think(label, probs, rng)
            think_count += int(requires)
            idx += 1
            manifest.append(make_manifest_item(dataset_name, idx, item, requires))
        report["sampled_by_label"][label] = len(selected)
        report["requires_safety_think_by_label"][label] = think_count

    rng.shuffle(manifest)
    return manifest, report


def sample_with_replacements(config: dict[str, Any], max_replacements: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    source_path = config["paths"]["safechain_jsonl"]
    per_label = {label: int(n) for label, n in config["sampling"]["per_label"].items()}
    probs = config["sampling"]["safety_think_probability"]
    dataset_name = str(config.get("dataset_name", "safeanywhere")).replace(".", "_")
    max_replacements = int(max_replacements if max_replacements is not None else config["sampling"].get("max_replacements", 100))

    pools = load_safechain_pool(source_path)
    for label, items in pools.items():
        rng.shuffle(items)

    cursors = {label: 0 for label in per_label}
    manifest: list[dict[str, Any]] = []
    idx = 0
    for label, n in per_label.items():
        if len(pools.get(label, [])) < n:
            raise ValueError(f"Not enough samples for {label}: requested {n}, available {len(pools.get(label, []))}")
        for _ in range(n):
            item = pools[label][cursors[label]]
            cursors[label] += 1
            idx += 1
            manifest.append(make_manifest_item(dataset_name, idx, item, needs_safety_think(label, probs, rng)))
    rng.shuffle(manifest)

    report = {
        "seed": seed,
        "source_path": source_path,
        "available_by_label": {label: len(pools.get(label, [])) for label in per_label},
        "requested_by_label": per_label,
        "max_replacements": max_replacements,
    }
    state = {"pools": pools, "cursors": cursors, "rng": rng, "next_index": idx + 1, "dataset_name": dataset_name, "probs": probs}
    return manifest, {**report, "_state": state}


def next_replacement(label: str, state: dict[str, Any]) -> dict[str, Any] | None:
    pools = state["pools"]
    cursor = state["cursors"].get(label, 0)
    if cursor >= len(pools.get(label, [])):
        return None
    item = pools[label][cursor]
    state["cursors"][label] = cursor + 1
    index = state["next_index"]
    state["next_index"] += 1
    return make_manifest_item(
        state["dataset_name"],
        index,
        item,
        needs_safety_think(label, state["probs"], state["rng"]),
    )

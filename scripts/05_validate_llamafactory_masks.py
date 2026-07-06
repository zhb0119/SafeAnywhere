from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TRAIN = ROOT / "build/mixed_safechain1k_prefix500/train_lf_v1_spanmasked.jsonl"
DEFAULT_VAL = ROOT / "build/mixed_safechain1k_prefix500/val_lf_v1_spanmasked.jsonl"
DEFAULT_LF_ROOT = ROOT.parent / "LLaMA-Factory"
LOCAL_IGNORE_INDEX = -100


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            yield obj


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids if token_id != self.pad_token_id)


class LocalSafeAnywhereRenderer:
    def __init__(self) -> None:
        self.tokenizer = CharTokenizer()

    def append(self, model_input: dict[str, list[Any]], text: str, loss_weight: float) -> None:
        if not text:
            return
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        model_input["input_ids"].extend(ids)
        model_input["loss_weights"].extend([loss_weight] * len(ids))
        if loss_weight > 1e-6:
            model_input["labels"].extend(ids)
        else:
            model_input["labels"].extend([LOCAL_IGNORE_INDEX] * len(ids))

    @staticmethod
    def concat_text(message: dict[str, Any]) -> str:
        out = ""
        for content in message["content"]:
            if content["type"] != "text":
                raise ValueError(f"Unsupported content type in local renderer: {content['type']}")
            out += content["value"]
        return out

    @staticmethod
    def content_weight(message: dict[str, Any], content: dict[str, Any], default: float) -> float:
        return float(content.get("loss_weight", message.get("loss_weight", default)))

    def render_messages(self, messages: list[dict[str, Any]]) -> dict[str, list[Any]]:
        model_input: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }

        if messages[0]["role"] == "system":
            text = "<|im_start|>system\n" + self.concat_text(messages[0]) + "<|im_end|>\n"
            self.append(model_input, text, float(messages[0].get("loss_weight", 0.0)))

        for turn_idx, message in enumerate(messages):
            role = message["role"]
            if role == "user" or (role == "system" and turn_idx != 0):
                text = "<|im_start|>" + role + "\n" + self.concat_text(message) + "<|im_end|>\n"
                self.append(model_input, text, float(message.get("loss_weight", 0.0)))
            elif role == "assistant":
                self.append(model_input, "<|im_start|>assistant\n", 0.0)
                has_positive_span = False
                for content in message["content"]:
                    if content["type"] != "text":
                        raise ValueError(f"Unsupported content type in local renderer: {content['type']}")
                    weight = self.content_weight(message, content, 1.0)
                    has_positive_span = has_positive_span or weight > 1e-6
                    self.append(model_input, content["value"], weight)
                self.append(model_input, "<|im_end|>\n", 1.0 if has_positive_span else 0.0)
            elif role == "tool":
                raise ValueError("Local validator renderer does not support tool messages")

        model_input["attention_mask"] = [1] * len(model_input["input_ids"])
        return model_input


def load_renderer(lf_root: Path, template: str) -> tuple[Any, Any]:
    src = lf_root / "src"
    if not src.exists():
        raise FileNotFoundError(f"LLaMA-Factory src path not found: {src}")
    sys.path.insert(0, str(src))
    from llamafactory.v1.core.utils.rendering import Renderer
    from llamafactory.v1.utils.constants import IGNORE_INDEX

    return Renderer(template, CharTokenizer()), IGNORE_INDEX


def parse_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("extra_info")
    if not isinstance(raw, str) or not raw:
        return {}
    return json.loads(raw)


def content_weight(message: dict[str, Any], content: dict[str, Any]) -> float:
    return float(content.get("loss_weight", message.get("loss_weight", 1.0 if message.get("role") == "assistant" else 0.0)))


def validate_structure(row: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{path}:{line_no} expected exactly two messages after span export")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"{path}:{line_no} expected user,assistant roles")

    user = messages[0]
    assistant = messages[1]
    if float(user.get("loss_weight", 0.0)) != 0.0:
        raise ValueError(f"{path}:{line_no} user loss_weight must be 0.0")

    info = parse_extra_info(row)
    assistant_content = assistant.get("content")
    if not isinstance(assistant_content, list) or not assistant_content:
        raise ValueError(f"{path}:{line_no} assistant content must be non-empty list")

    weights = [content_weight(assistant, content) for content in assistant_content]
    attack_type = info.get("attack_type")
    if attack_type == "dangerous_prefix":
        if len(assistant_content) != 2:
            raise ValueError(f"{path}:{line_no} dangerous_prefix assistant must have two content spans")
        if weights != [0.0, 1.0]:
            raise ValueError(f"{path}:{line_no} dangerous_prefix assistant content weights must be [0.0, 1.0], got {weights}")
        prefill = assistant_content[0].get("value")
        target = assistant_content[1].get("value")
        if not isinstance(prefill, str) or not isinstance(target, str):
            raise ValueError(f"{path}:{line_no} content values must be strings")
        if "<safety_think>" in prefill or "</safety_think>" in prefill:
            raise ValueError(f"{path}:{line_no} prefill must not contain safety_think block")
        if not target.startswith("<safety_think>"):
            raise ValueError(f"{path}:{line_no} target must start with safety_think")
        return {"attack_type": attack_type, "prefill": prefill, "target": target}

    if attack_type == "safechain_cold_start":
        if len(assistant_content) != 1:
            raise ValueError(f"{path}:{line_no} safechain assistant must have one content span")
        if weights != [1.0]:
            raise ValueError(f"{path}:{line_no} safechain assistant content weight must be 1.0, got {weights}")
        return {"attack_type": attack_type}

    raise ValueError(f"{path}:{line_no} unsupported attack_type: {attack_type}")


def decode_weighted_labels(model_input: dict[str, Any], ignore_index: int) -> tuple[str, str]:
    input_ids = model_input["input_ids"]
    labels = model_input["labels"]
    loss_weights = model_input["loss_weights"]
    if not (len(input_ids) == len(labels) == len(loss_weights)):
        raise ValueError("Rendered input_ids, labels, and loss_weights lengths differ")
    full = CharTokenizer().decode(input_ids)
    weighted = CharTokenizer().decode([label for label, weight in zip(labels, loss_weights) if label != ignore_index and weight > 0])
    return full, weighted


def validate_rendering(row: dict[str, Any], rendered_info: dict[str, Any], renderer: Any, ignore_index: int, path: Path, line_no: int) -> None:
    model_input = renderer.render_messages(row["messages"])
    full, weighted = decode_weighted_labels(model_input, ignore_index)
    assistant_starts = full.count("<|im_start|>assistant\n")
    if assistant_starts != 1:
        raise ValueError(f"{path}:{line_no} expected one assistant turn after rendering, got {assistant_starts}")
    if rendered_info["attack_type"] == "dangerous_prefix":
        prefill = rendered_info["prefill"]
        target = rendered_info["target"]
        if prefill not in full:
            raise ValueError(f"{path}:{line_no} prefill missing from rendered full input")
        if target not in full:
            raise ValueError(f"{path}:{line_no} target missing from rendered full input")
        if prefill.strip() and prefill.strip() in weighted:
            raise ValueError(f"{path}:{line_no} weighted labels include masked prefill")
        if "<safety_think>" not in weighted:
            raise ValueError(f"{path}:{line_no} weighted labels do not include safety_think target")


def validate_file(path: Path, renderer: Any | None, ignore_index: int | None, max_render_checks: int | None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    rendered_checks = 0
    for line_no, row in enumerate(read_jsonl(path), start=1):
        info = validate_structure(row, path, line_no)
        counts[info["attack_type"]] += 1
        if renderer is not None and ignore_index is not None:
            if max_render_checks is None or rendered_checks < max_render_checks:
                validate_rendering(row, info, renderer, ignore_index, path, line_no)
                rendered_checks += 1
    counts["rows"] = sum(count for key, count in counts.items() if key != "rows")
    counts["rendered_checks"] = rendered_checks
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SafeAnywhere LLaMA-Factory v1 span-level loss masks.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--llamafactory-root", type=Path, default=DEFAULT_LF_ROOT)
    parser.add_argument("--template", default="safeanywhere_qwen3_nothink")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=["local", "llamafactory"],
        default="local",
        help="Use the lightweight local renderer or import LLaMA-Factory's renderer.",
    )
    parser.add_argument("--max-render-checks", type=int, default=None)
    args = parser.parse_args()

    renderer = None
    ignore_index = None
    if not args.structure_only and args.renderer == "llamafactory":
        renderer, ignore_index = load_renderer(args.llamafactory_root, args.template)
    elif not args.structure_only:
        renderer, ignore_index = LocalSafeAnywhereRenderer(), LOCAL_IGNORE_INDEX

    report = {
        "train": validate_file(args.train, renderer, ignore_index, args.max_render_checks),
        "val": validate_file(args.val, renderer, ignore_index, args.max_render_checks),
        "renderer": args.renderer,
        "template": args.template,
        "structure_only": args.structure_only,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

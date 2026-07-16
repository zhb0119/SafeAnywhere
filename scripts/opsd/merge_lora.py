from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def dtype_from_name(torch: Any, name: str) -> Any:
    mapping = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a local HF causal LM checkpoint.")
    parser.add_argument("--base-model", required=True, help="Base HF model path.")
    parser.add_argument("--adapter", required=True, help="LoRA/PEFT adapter path.")
    parser.add_argument("--output", required=True, help="Output merged HF checkpoint path.")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Merging LoRA requires torch, transformers, and peft. Install `uv sync --extra opsd`.") from exc

    base_model = resolve_path(args.base_model)
    adapter = resolve_path(args.adapter)
    output = resolve_path(args.output)

    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output already exists and is not empty: {output}. Use --overwrite.")
        shutil.rmtree(output)

    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    except Exception as exc:
        if tokenizer_source == base_model:
            raise
        print(
            f"Warning: failed to load tokenizer from adapter ({exc}). Falling back to base tokenizer.",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_from_name(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

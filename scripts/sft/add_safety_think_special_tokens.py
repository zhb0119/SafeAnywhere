from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

SAFETY_THINK_OPEN = "<safety_think>"
SAFETY_THINK_CLOSE = "</safety_think>"
SPECIAL_TOKENS = (SAFETY_THINK_OPEN, SAFETY_THINK_CLOSE)

DEFAULT_BASE_MODEL = "../models/Qwen3-0.6B"
DEFAULT_OUTPUT = "runs/sft_special/qwen3_0_6b_safety_think_base"
DEFAULT_SEMANTIC_SOURCE = "safety_think"
DEFAULT_FUNCTIONAL_SOURCES = {
    SAFETY_THINK_OPEN: "<think>",
    SAFETY_THINK_CLOSE: "</think>",
}


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


def mean_embedding_for_text(tokenizer: Any, embedding_weight: Any, text: str) -> tuple[Any, list[int], list[str]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Unable to tokenize initialization source: {text!r}")

    vectors = embedding_weight[ids]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return vectors.mean(dim=0), ids, tokens


def token_text(token: Any) -> str:
    return str(getattr(token, "content", token))


def add_safety_think_special_tokens(tokenizer: Any, token_objects: list[Any]) -> int:
    existing_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    existing_texts = {token_text(token) for token in existing_tokens}
    merged_tokens = list(existing_tokens)
    for token in token_objects:
        text = token_text(token)
        if text not in existing_texts:
            merged_tokens.append(token)
            existing_texts.add(text)

    before_len = len(tokenizer)
    try:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": merged_tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": merged_tokens})

    if num_added is None:
        num_added = len(tokenizer) - before_len
    return int(num_added)


def ensure_single_special_token(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Token id lookup failed for {token!r}")

    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(f"{token!r} should encode as one token, got ids={encoded}")

    # Qwen slow tokenizers may not report added special tokens through
    # all_special_tokens after reload, even when tokenizer.json marks them as
    # special.  The stable downstream contract is: one lexical id, and skipped
    # by decode(skip_special_tokens=True).
    if tokenizer.decode(encoded, skip_special_tokens=True):
        raise ValueError(f"{token!r} is not skipped by decode(skip_special_tokens=True)")

    return int(token_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a SafeAnywhere base checkpoint with safety-think tags as special tokens."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base HF model path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HF checkpoint path.")
    parser.add_argument("--semantic-source", default=DEFAULT_SEMANTIC_SOURCE)
    parser.add_argument("--open-functional-source", default=DEFAULT_FUNCTIONAL_SOURCES[SAFETY_THINK_OPEN])
    parser.add_argument("--close-functional-source", default=DEFAULT_FUNCTIONAL_SOURCES[SAFETY_THINK_CLOSE])
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Blend weight for the semantic source. The remaining weight comes from the functional source.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value. Use 'none' to omit it.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.semantic_weight <= 1.0:
        raise ValueError("--semantic-weight must be between 0 and 1")

    try:
        import torch
        from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("This script requires torch and transformers. Install with `uv sync --extra opsd`.") from exc

    base_model = resolve_path(args.base_model)
    output = resolve_path(args.output)

    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output already exists and is not empty: {output}. Use --overwrite.")
        shutil.rmtree(output)

    print(f"Loading tokenizer: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
        padding_side="right",
    )
    old_tokenizer_len = len(tokenizer)
    old_special_tokens = list(tokenizer.all_special_tokens)

    print(f"Loading model: {base_model}", flush=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype_from_name(torch, args.dtype),
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    token_objects = [
        AddedToken(token, lstrip=False, rstrip=False, special=True, normalized=False) for token in SPECIAL_TOKENS
    ]
    num_added = add_safety_think_special_tokens(tokenizer, token_objects)
    print(
        json.dumps(
            {
                "old_tokenizer_len": old_tokenizer_len,
                "new_tokenizer_len": len(tokenizer),
                "num_added": num_added,
                "tokens": list(SPECIAL_TOKENS),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings().weight.data
    output_embedding_layer = model.get_output_embeddings()
    if output_embedding_layer is None:
        raise RuntimeError("Model does not expose output embeddings")

    output_embeddings = output_embedding_layer.weight.data
    semantic_weight = float(args.semantic_weight)
    functional_weight = 1.0 - semantic_weight

    input_semantic, semantic_ids, semantic_pieces = mean_embedding_for_text(
        tokenizer, input_embeddings, args.semantic_source
    )
    output_semantic, _, _ = mean_embedding_for_text(tokenizer, output_embeddings, args.semantic_source)

    functional_sources = {
        SAFETY_THINK_OPEN: args.open_functional_source,
        SAFETY_THINK_CLOSE: args.close_functional_source,
    }
    token_metadata: dict[str, dict[str, Any]] = {}

    for token, functional_source in functional_sources.items():
        token_id = ensure_single_special_token(tokenizer, token)
        input_functional, functional_ids, functional_pieces = mean_embedding_for_text(
            tokenizer, input_embeddings, functional_source
        )
        output_functional, _, _ = mean_embedding_for_text(tokenizer, output_embeddings, functional_source)

        input_embeddings[token_id] = semantic_weight * input_semantic + functional_weight * input_functional
        output_embeddings[token_id] = semantic_weight * output_semantic + functional_weight * output_functional

        token_metadata[token] = {
            "id": token_id,
            "semantic_source": args.semantic_source,
            "semantic_ids": semantic_ids,
            "semantic_tokens": semantic_pieces,
            "functional_source": functional_source,
            "functional_ids": functional_ids,
            "functional_tokens": functional_pieces,
            "semantic_weight": semantic_weight,
            "functional_weight": functional_weight,
            "skip_decode_empty": tokenizer.decode([token_id], skip_special_tokens=True) == "",
            "encoded": tokenizer.encode(token, add_special_tokens=False),
        }

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    # Qwen slow tokenizers can save the added-token ids correctly while losing
    # the high-level special-token state on reload.  Refresh after the first
    # save so every downstream step observes these markers as special tokens.
    tokenizer = AutoTokenizer.from_pretrained(
        output,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
        padding_side="right",
    )
    add_safety_think_special_tokens(tokenizer, token_objects)
    for token in SPECIAL_TOKENS:
        ensure_single_special_token(tokenizer, token)
    tokenizer.save_pretrained(output)

    metadata = {
        "base_model": str(base_model),
        "output": str(output),
        "old_tokenizer_len": old_tokenizer_len,
        "new_tokenizer_len": len(tokenizer),
        "num_added": num_added,
        "old_special_tokens": old_special_tokens,
        "new_special_tokens": list(tokenizer.all_special_tokens),
        "special_token_contract": "single token id and decode(skip_special_tokens=True) removes the marker",
        "tokens": token_metadata,
    }
    (output / "safeanywhere_special_tokens.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved special-token base checkpoint: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

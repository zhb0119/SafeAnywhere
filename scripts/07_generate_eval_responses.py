from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "build/eval/safeanywhere_v1/safeanywhere_eval.jsonl"
DEFAULT_OUTPUT = ROOT / "build/eval/safeanywhere_v1/predictions.jsonl"


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_chatml(row: dict[str, Any], system_prompt: str | None = None) -> str:
    chunks = []
    if system_prompt:
        chunks.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    chunks.append(f"<|im_start|>user\n{row['prompt']}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    assistant_prefill = row.get("assistant_prefill")
    if assistant_prefill:
        chunks.append(assistant_prefill)
        if not assistant_prefill.endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)


def truncate_at_stop(text: str) -> str:
    stops = ["<|im_end|>", "<|endoftext|>"]
    end = len(text)
    for stop in stops:
        idx = text.find(stop)
        if idx >= 0:
            end = min(end, idx)
    return text[:end].strip()


def load_model(base_model: str, adapter: str | None, dtype: str, device_map: str, trust_remote_code: bool):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing generation dependencies. Install transformers/torch/peft in the LLaMA-Factory environment."
        ) from exc

    tokenizer_path = adapter if adapter and Path(adapter).exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map[dtype],
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Missing peft. Install it before loading a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def generate_one(
    tokenizer: Any,
    model: Any,
    row: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: str | None,
) -> dict[str, Any]:
    prompt_text = render_chatml(row, system_prompt=system_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p})

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing torch.") from exc
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)[0]
    continuation_ids = output_ids[inputs["input_ids"].shape[-1] :]
    raw_prediction = tokenizer.decode(continuation_ids, skip_special_tokens=False)
    prediction = truncate_at_stop(raw_prediction)
    return {
        **row,
        "rendered_prompt": prompt_text,
        "raw_prediction": raw_prediction,
        "prediction": prediction,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate responses for SafeAnywhere eval rows.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default="../models/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir. Omit for base-model evaluation.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    tokenizer, model = load_model(
        base_model=args.base_model,
        adapter=args.adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    rows = list(read_jsonl(args.eval_file))
    if args.limit is not None:
        rows = rows[: args.limit]

    ensure_dir(args.output.parent)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for index, row in enumerate(rows, start=1):
            result = generate_one(
                tokenizer=tokenizer,
                model=model,
                row=row,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                system_prompt=args.system_prompt,
            )
            f.write(json.dumps(result, ensure_ascii=False, sort_keys=False) + "\n")
            f.flush()
            print(f"[{index}/{len(rows)}] {row['id']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

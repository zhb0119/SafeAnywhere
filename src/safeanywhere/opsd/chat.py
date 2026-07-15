from __future__ import annotations

from typing import Any


def _render_qwen3_nothink(messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role}")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)


def render_prompt(
    tokenizer: Any,
    instruction: str,
    *,
    system_prompt: str | None = None,
    template: str = "qwen3_nothink",
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})

    if template == "qwen3_nothink":
        return _render_qwen3_nothink(messages, add_generation_prompt=True)

    if template == "tokenizer":
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    raise ValueError(f"Unsupported OPSD chat template: {template}")

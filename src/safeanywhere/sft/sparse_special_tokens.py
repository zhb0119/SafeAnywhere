from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any


SAFETY_THINK_TOKENS = ("<safety_think>", "</safety_think>")
SPARSE_MODULES = ("embed_tokens", "lm_head")


def _as_list(value: Any, *, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise TypeError(f"Expected string/list value, got {type(value).__name__}.")


def _tokenizer_path_from_config(model: Any, sparse_cfg: dict[str, Any]) -> str:
    explicit = sparse_cfg.get("tokenizer_path")
    if explicit:
        return str(explicit)

    config = getattr(model, "config", None)
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path:
        return str(name_or_path)

    name_or_path = getattr(model, "name_or_path", None)
    if name_or_path:
        return str(name_or_path)

    raise ValueError("sparse_token_grad.tokenizer_path is required when model path cannot be inferred.")


def resolve_sparse_token_ids(model: Any, sparse_cfg: dict[str, Any], auto_tokenizer_cls: Any) -> list[int]:
    token_ids = sparse_cfg.get("token_ids")
    if token_ids is not None:
        return [int(token_id) for token_id in token_ids]

    tokens = _as_list(sparse_cfg.get("tokens"), default=SAFETY_THINK_TOKENS)
    tokenizer_path = _tokenizer_path_from_config(model, sparse_cfg)
    tokenizer = auto_tokenizer_cls.from_pretrained(tokenizer_path, trust_remote_code=True)

    ids: list[int] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unk_token_id:
            raise ValueError(f"Sparse special token {token!r} is missing from tokenizer {tokenizer_path}.")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"Sparse special token {token!r} should encode as one id, got {encoded}.")
        ids.append(int(token_id))

    return ids


def apply_sparse_token_gradient_hooks(
    model: Any,
    *,
    token_ids: list[int],
    module_names: list[str] | None = None,
) -> int:
    """Mask trainable embedding/lm_head gradients to the selected token rows.

    PEFT's modules_to_save makes full embedding/lm_head tensors trainable so the
    adapter can save them.  This hook keeps the optimizer path compatible with
    PEFT while ensuring only the new special-token rows receive gradients.
    """
    if not token_ids:
        raise ValueError("At least one token id is required for sparse gradient hooks.")

    modules = module_names or list(SPARSE_MODULES)
    hook_count = 0
    max_token_id = max(token_ids)

    def keep_only_token_rows(grad: Any) -> Any:
        import torch

        if grad is None:
            return grad
        if grad.ndim < 2 or grad.shape[0] <= max_token_id:
            return grad

        row_ids = torch.tensor(token_ids, device=grad.device, dtype=torch.long)
        masked = grad.new_zeros(grad.shape)
        masked.index_copy_(0, row_ids, grad.index_select(0, row_ids))
        return masked

    for name, param in model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if param.ndim < 2 or param.shape[0] <= max_token_id:
            continue
        if not any(module_name in name for module_name in modules):
            continue
        param.register_hook(keep_only_token_rows)
        hook_count += 1

    if hook_count == 0:
        raise RuntimeError(
            "No trainable embed/lm_head parameters matched sparse special-token hook. "
            "Check modules_to_save and module names."
        )

    return hook_count


def describe_sparse_hook(token_ids: list[int], hook_count: int, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    info = {
        "enabled": True,
        "token_ids": token_ids,
        "hooked_parameters": hook_count,
        "note": "Only these token rows receive gradients in embed_tokens/lm_head; trainable param count still includes full saved modules.",
    }
    if output_dir:
        path = Path(output_dir) / "safeanywhere_sparse_special_tokens.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json

        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info

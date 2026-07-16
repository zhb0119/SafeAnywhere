from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SAFETY_THINK_OPEN = "<safety_think>"
SAFETY_THINK_CLOSE = "</safety_think>"


@dataclass(frozen=True)
class BoundaryLossConfig:
    open_token_id: int
    close_token_id: int
    open_weight: float = 40.0
    close_weight: float = 8.0
    block_weight: float = 2.0


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_token_id:
        raise ValueError(f"Boundary token {token!r} is missing from tokenizer.")
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(f"Boundary token {token!r} should encode as one id, got {encoded}.")
    return int(token_id)


def resolve_boundary_loss_config(tokenizer: Any, raw_cfg: dict[str, Any]) -> BoundaryLossConfig:
    tokens_cfg = raw_cfg.get("tokens") or {}
    open_token = str(tokens_cfg.get("open", raw_cfg.get("open_token", SAFETY_THINK_OPEN)))
    close_token = str(tokens_cfg.get("close", raw_cfg.get("close_token", SAFETY_THINK_CLOSE)))
    return BoundaryLossConfig(
        open_token_id=_single_token_id(tokenizer, open_token),
        close_token_id=_single_token_id(tokenizer, close_token),
        open_weight=float(raw_cfg.get("open_weight", 40.0)),
        close_weight=float(raw_cfg.get("close_weight", 8.0)),
        block_weight=float(raw_cfg.get("block_weight", 2.0)),
    )


def apply_boundary_loss_weights(
    shift_loss_weights: torch.Tensor,
    shift_labels: torch.Tensor,
    cfg: BoundaryLossConfig,
) -> torch.Tensor:
    if shift_loss_weights.shape != shift_labels.shape:
        raise ValueError(f"loss/label shape mismatch: {shift_loss_weights.shape} vs {shift_labels.shape}")
    if min(cfg.open_weight, cfg.close_weight, cfg.block_weight) < 0:
        raise ValueError("Boundary loss weights must be non-negative.")

    multipliers = torch.ones_like(shift_loss_weights)
    positive = shift_loss_weights > 0
    open_mask = (shift_labels == cfg.open_token_id) & positive
    close_mask = (shift_labels == cfg.close_token_id) & positive

    if cfg.block_weight != 1.0:
        for row_idx in range(shift_labels.shape[0]):
            row_labels = shift_labels[row_idx]
            row_positive = positive[row_idx]
            row_positions = torch.arange(row_labels.numel(), device=row_labels.device)
            open_positions = torch.nonzero(open_mask[row_idx], as_tuple=False).flatten()
            for open_pos in open_positions.tolist():
                close_candidates = torch.nonzero(
                    close_mask[row_idx] & (row_positions > open_pos),
                    as_tuple=False,
                ).flatten()
                close_pos = int(close_candidates[0].item()) if close_candidates.numel() else row_labels.numel()
                if close_pos > open_pos + 1:
                    block_slice = slice(open_pos + 1, close_pos)
                    block_mask = row_positive[block_slice]
                    current = multipliers[row_idx, block_slice]
                    block_value = current.new_full(current.shape, cfg.block_weight)
                    multipliers[row_idx, block_slice] = torch.where(
                        block_mask,
                        torch.maximum(current, block_value),
                        current,
                    )

    multipliers = torch.where(open_mask, multipliers.new_full(multipliers.shape, cfg.open_weight), multipliers)
    multipliers = torch.where(close_mask, multipliers.new_full(multipliers.shape, cfg.close_weight), multipliers)
    return shift_loss_weights * multipliers

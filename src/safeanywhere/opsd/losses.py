from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from safeanywhere.schema import HARMFUL_LABELS

KlKind = Literal["forward_kl", "reverse_kl", "mixed_kl", "adaptive_kl"]


def _normalized_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    vocab_size = log_probs.shape[-1]
    if vocab_size <= 1:
        return torch.zeros_like(entropy)
    return (entropy / math.log(float(vocab_size))).clamp(0.0, 1.0)


def _adaptive_mixture_weights(
    *,
    label: str | None,
    teacher_entropy: torch.Tensor,
    adaptive_cfg: dict[str, object],
) -> torch.Tensor:
    if label not in HARMFUL_LABELS:
        return torch.ones_like(teacher_entropy)

    low = float(adaptive_cfg.get("harmful_entropy_low", 0.2))
    high = float(adaptive_cfg.get("harmful_entropy_high", 0.5))
    if high <= low:
        raise ValueError("harmful_entropy_high must be greater than harmful_entropy_low")

    forward_floor = float(adaptive_cfg.get("harmful_forward_floor", 0.2))
    reverse_floor = float(adaptive_cfg.get("harmful_reverse_floor", 0.2))
    if forward_floor < 0.0 or reverse_floor < 0.0:
        raise ValueError("harmful forward/reverse floors must be non-negative")
    if forward_floor + reverse_floor > 1.0:
        raise ValueError("harmful forward/reverse floors must sum to at most 1")

    entropy_gate = ((teacher_entropy - low) / (high - low)).clamp(0.0, 1.0)
    forward_span = 1.0 - forward_floor - reverse_floor
    forward_weight = forward_floor + entropy_gate * forward_span
    return forward_weight.clamp(0.0, 1.0)


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    kind: KlKind = "mixed_kl",
    mixed_kl_weight: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_weights: torch.Tensor | None = None,
    label: str | None = None,
    adaptive_cfg: dict[str, object] | None = None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"student/teacher logits shape mismatch: {student_logits.shape} vs {teacher_logits.shape}")
    if student_logits.ndim != 2:
        raise ValueError(f"expected [tokens, vocab] logits, got {student_logits.shape}")
    if student_logits.shape[0] == 0:
        raise ValueError("cannot compute KL over zero continuation tokens")
    if kind not in {"forward_kl", "reverse_kl", "mixed_kl", "adaptive_kl"}:
        raise ValueError(f"Unsupported KL kind: {kind}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0.0 <= mixed_kl_weight <= 1.0:
        raise ValueError("mixed_kl_weight must be in [0, 1]")
    adaptive_cfg = adaptive_cfg or {}

    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature

    teacher_entropy = None
    if kind == "adaptive_kl":
        # The entropy gate uses the full teacher distribution before any optional top-k truncation.
        teacher_full_log_probs = F.log_softmax(teacher, dim=-1)
        teacher_entropy = _normalized_entropy(teacher_full_log_probs)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), teacher.shape[-1])
        teacher_top_logits, teacher_top_indices = torch.topk(teacher, k=k, dim=-1)
        student = student.gather(dim=-1, index=teacher_top_indices)
        teacher = teacher_top_logits

    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_log_probs = F.log_softmax(teacher, dim=-1)

    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()
    forward = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    reverse = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

    if kind == "forward_kl":
        per_token = forward
    elif kind == "reverse_kl":
        per_token = reverse
    elif kind == "adaptive_kl":
        if teacher_entropy is None:
            raise RuntimeError("adaptive KL requires teacher entropy")
        forward_weight = _adaptive_mixture_weights(
            label=label,
            teacher_entropy=teacher_entropy.to(device=forward.device, dtype=forward.dtype),
            adaptive_cfg=adaptive_cfg,
        )
        per_token = forward_weight * forward + (1.0 - forward_weight) * reverse
    else:
        per_token = mixed_kl_weight * forward + (1.0 - mixed_kl_weight) * reverse

    if token_weights is not None:
        if token_weights.shape != per_token.shape:
            raise ValueError(f"token_weights shape mismatch: {token_weights.shape} vs {per_token.shape}")
        weights = token_weights.to(device=per_token.device, dtype=per_token.dtype)
        denom = weights.sum().clamp_min(1e-8)
        return (per_token * weights).sum() / denom * (temperature**2)

    return per_token.mean() * (temperature**2)

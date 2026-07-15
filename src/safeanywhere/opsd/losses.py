from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


KlKind = Literal["forward_kl", "reverse_kl", "mixed_kl"]


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    kind: KlKind = "mixed_kl",
    mixed_kl_weight: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"student/teacher logits shape mismatch: {student_logits.shape} vs {teacher_logits.shape}")
    if student_logits.ndim != 2:
        raise ValueError(f"expected [tokens, vocab] logits, got {student_logits.shape}")
    if student_logits.shape[0] == 0:
        raise ValueError("cannot compute KL over zero continuation tokens")
    if kind not in {"forward_kl", "reverse_kl", "mixed_kl"}:
        raise ValueError(f"Unsupported KL kind: {kind}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0.0 <= mixed_kl_weight <= 1.0:
        raise ValueError("mixed_kl_weight must be in [0, 1]")

    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature

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
    else:
        per_token = mixed_kl_weight * forward + (1.0 - mixed_kl_weight) * reverse

    return per_token.mean() * (temperature**2)

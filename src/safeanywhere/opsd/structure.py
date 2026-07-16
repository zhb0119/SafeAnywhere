from __future__ import annotations

from typing import Any

from safeanywhere.schema import BENIGN_LABELS, HARMFUL_LABELS


OPEN_TAG = "<safety_think>"
CLOSE_TAG = "</safety_think>"

REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "sorry",
    "i cannot",
    "i can't",
    "i can not",
    "i cannot comply",
    "i am not able",
    "i'm not able",
)


def _find_subsequence(seq: list[int], pattern: list[int], *, start: int = 0, end: int | None = None) -> int | None:
    if not pattern:
        return None
    stop = len(seq) if end is None else min(end, len(seq))
    if start < 0:
        start = 0
    last = stop - len(pattern)
    for idx in range(start, last + 1):
        if seq[idx : idx + len(pattern)] == pattern:
            return idx
    return None


def _marker_variants(tokenizer: Any, marker: str) -> list[list[int]]:
    variants: list[list[int]] = []
    for text in (marker, f"{marker}\n", f"{marker}\n\n", f"{marker} "):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids and ids not in variants:
            variants.append(ids)
    return variants


def _find_marker(
    seq: list[int],
    variants: list[list[int]],
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[int | None, int]:
    best_idx: int | None = None
    best_len = 0
    for variant in variants:
        idx = _find_subsequence(seq, variant, start=start, end=end)
        if idx is not None and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_len = len(variant)
    return best_idx, best_len


def _starts_with_refusal(text: str) -> bool:
    stripped = text.lstrip().lower()
    return any(stripped.startswith(prefix) for prefix in REFUSAL_PREFIXES)


def _float_cfg(cfg: dict[str, Any], key: str, default: float) -> float:
    value = cfg.get(key, default)
    return float(default if value is None else value)


def _int_cfg(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    return int(default if value is None else value)


def build_token_weights(
    *,
    tokenizer: Any,
    label: str,
    continuation_ids: list[int],
    generated_text: str,
    cfg: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """Build structure-aware distillation weights for one assistant continuation."""
    length = len(continuation_ids)
    if length == 0:
        return [], {"structure": "empty"}

    if not bool(cfg.get("enabled", True)):
        return [1.0] * length, {"structure": "uniform"}

    open_variants = _marker_variants(tokenizer, OPEN_TAG)
    close_variants = _marker_variants(tokenizer, CLOSE_TAG)
    search_tokens = _int_cfg(cfg, "tag_search_tokens", 64)

    open_idx, open_len = _find_marker(continuation_ids, open_variants, end=search_tokens)
    close_idx = None
    close_len = 0
    if open_idx is not None:
        close_idx, close_len = _find_marker(continuation_ids, close_variants, start=open_idx + open_len)

    if label in HARMFUL_LABELS:
        return _harmful_weights(
            length=length,
            open_idx=open_idx,
            close_idx=close_idx,
            open_len=open_len,
            close_len=close_len,
            cfg=cfg,
        )

    if label in BENIGN_LABELS:
        return _benign_weights(
            length=length,
            has_safety_tag=open_idx is not None,
            starts_with_refusal=_starts_with_refusal(generated_text),
            cfg=cfg,
        )

    return [1.0] * length, {"structure": "unknown_label"}


def _harmful_weights(
    *,
    length: int,
    open_idx: int | None,
    close_idx: int | None,
    open_len: int,
    close_len: int,
    cfg: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    missing_prefix_tokens = min(length, _int_cfg(cfg, "harmful_missing_prefix_tokens", 16))
    after_weight = _float_cfg(cfg, "harmful_after_block_weight", 0.2)
    block_weight = _float_cfg(cfg, "harmful_block_weight", 4.0)
    open_weight = _float_cfg(cfg, "harmful_open_weight", 8.0)
    close_weight = _float_cfg(cfg, "harmful_close_weight", 5.0)
    first_response_weight = _float_cfg(cfg, "harmful_first_response_weight", 2.0)
    first_response_tokens = _int_cfg(cfg, "harmful_first_response_tokens", 48)

    if open_idx is None:
        weights = [0.0] * length
        for idx in range(missing_prefix_tokens):
            weights[idx] = open_weight
        return weights, {
            "structure": "harmful_missing_safety_block",
            "valid_safety_block": False,
            "needs_prefix_ce": True,
            "weighted_tokens": missing_prefix_tokens,
        }

    weights = [0.0] * length
    block_limit = length
    if close_idx is not None:
        block_limit = min(length, close_idx + close_len)
    else:
        block_limit = min(length, open_idx + _int_cfg(cfg, "harmful_unclosed_block_tokens", 64))

    for idx in range(0, block_limit):
        weights[idx] = block_weight

    for idx in range(open_idx, min(length, open_idx + open_len)):
        weights[idx] = open_weight

    if close_idx is not None:
        close_end = min(length, close_idx + close_len)
        for idx in range(close_idx, close_end):
            weights[idx] = close_weight
        for idx in range(close_end, min(length, close_end + first_response_tokens)):
            weights[idx] = first_response_weight
        for idx in range(close_end + first_response_tokens, length):
            weights[idx] = after_weight

    weighted_tokens = sum(1 for weight in weights if weight > 0)
    return weights, {
        "structure": "harmful_valid_safety_block" if close_idx is not None else "harmful_unclosed_safety_block",
        "valid_safety_block": close_idx is not None,
        "needs_prefix_ce": close_idx is None,
        "open_idx": open_idx,
        "close_idx": close_idx,
        "weighted_tokens": weighted_tokens,
    }


def _benign_weights(
    *,
    length: int,
    has_safety_tag: bool,
    starts_with_refusal: bool,
    cfg: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    prefix_tokens = min(length, _int_cfg(cfg, "benign_prefix_tokens", 32))
    prefix_weight = _float_cfg(cfg, "benign_prefix_weight", 4.0)
    body_weight = _float_cfg(cfg, "benign_body_weight", 0.5)
    bad_suffix_weight = _float_cfg(cfg, "benign_bad_prefix_suffix_weight", 0.0)

    bad_prefix = has_safety_tag or starts_with_refusal
    suffix_weight = bad_suffix_weight if bad_prefix else body_weight
    weights = [suffix_weight] * length
    for idx in range(prefix_tokens):
        weights[idx] = prefix_weight

    return weights, {
        "structure": "benign_bad_prefix" if bad_prefix else "benign_normal",
        "valid_safety_block": False,
        "needs_prefix_ce": False,
        "has_safety_tag": has_safety_tag,
        "starts_with_refusal": starts_with_refusal,
        "weighted_tokens": sum(1 for weight in weights if weight > 0),
    }


def canonical_safety_prefix(label: str) -> str:
    if label == "adversarial_harmful":
        return (
            "<safety_think>\n"
            "risk=other; trigger=adversarial_wrapper; intent=jailbreak; "
            "decision=refuse; plan=reject_unsafe_goal_and_redirect\n"
            "</safety_think>\n"
        )
    return (
        "<safety_think>\n"
        "risk=other; trigger=unsafe_request; intent=harmful; "
        "decision=refuse; plan=refuse_and_offer_safe_alternative\n"
        "</safety_think>\n"
    )

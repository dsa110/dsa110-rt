"""Shared bank-mask CLI parser (M5 Chunk 6c).

The ``--bank-mask`` flag pins a subset of the K_img × K_dm × K_time bank
axes for the detector kernel-bank construction. Both the cube-injection
detector bench and the search-node throughput bench accept it. Syntax:

  ``"k_img=<tokens>;k_dm=<tokens>;k_time=<tokens>"``

where ``<tokens>`` is either ``"*"`` (= keep all, default for unspecified
axes) or a comma-separated subset, e.g. ``"unit"``, ``"unit,psf"``,
``"d1"``, ``"d1,d3"``, ``"b8,b16,b32"``. Whitespace around tokens is
stripped. Unknown axis keys / tokens / empty subsets raise ValueError.

Examples (per Chunk 6c plan):

  ``"k_img=*;k_dm=*;k_time=*"``       full 128 (baseline)
  ``"k_img=unit"``                    1×4×8 = 32  (collapse K_img)
  ``"k_dm=d1"``                       4×1×8 = 32  (drop K_dm filters)
  ``"k_img=unit;k_dm=d1"``            1×1×8 =  8  (aggressive)

K_time stays full per the operator decision (Chunk 6c framing) — the
matched-filter time-width axis is the SNR-critical axis.
"""

from __future__ import annotations

from typing import Optional, Tuple

from dsart.common.constants import (
    DETECTOR_DM_KERNELS,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_TIME_KERNELS,
)

__all__ = ["parse_bank_mask", "BANK_AXIS_DOMAIN"]


BANK_AXIS_DOMAIN: dict[str, tuple[str, ...]] = {
    "k_img": DETECTOR_IMAGE_KERNELS,
    "k_dm": DETECTOR_DM_KERNELS,
    "k_time": DETECTOR_TIME_KERNELS,
}


def parse_bank_mask(
    spec: Optional[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Parse a ``--bank-mask`` CLI string into (image, dm, time) subsets.

    Returns a tuple ``(image_tokens, dm_tokens, time_tokens)`` of resolved
    token tuples. ``None`` / empty / ``"*"`` returns the full default bank.

    Raises:
        ValueError: on unknown axis keys, unknown tokens, or empty subsets.
    """
    image_tokens: Tuple[str, ...] = DETECTOR_IMAGE_KERNELS
    dm_tokens: Tuple[str, ...] = DETECTOR_DM_KERNELS
    time_tokens: Tuple[str, ...] = DETECTOR_TIME_KERNELS

    if spec is None or not spec.strip() or spec.strip() == "*":
        return image_tokens, dm_tokens, time_tokens

    overrides: dict[str, Tuple[str, ...]] = {}
    for clause in spec.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise ValueError(
                f"bank-mask clause {clause!r} missing '=' "
                f"(expected '<axis>=<tokens>')"
            )
        axis, raw_tokens = clause.split("=", 1)
        axis = axis.strip().lower()
        if axis not in BANK_AXIS_DOMAIN:
            raise ValueError(
                f"bank-mask axis {axis!r} not one of "
                f"{sorted(BANK_AXIS_DOMAIN)}"
            )
        tokens_text = raw_tokens.strip()
        if tokens_text == "*":
            overrides[axis] = BANK_AXIS_DOMAIN[axis]
            continue
        toks = tuple(t.strip() for t in tokens_text.split(",") if t.strip())
        if not toks:
            raise ValueError(
                f"bank-mask axis {axis!r} resolves to empty token list"
            )
        domain = BANK_AXIS_DOMAIN[axis]
        for t in toks:
            if t not in domain:
                raise ValueError(
                    f"bank-mask token {t!r} not in {axis} domain {domain}"
                )
        overrides[axis] = toks

    image_tokens = overrides.get("k_img", image_tokens)
    dm_tokens = overrides.get("k_dm", dm_tokens)
    time_tokens = overrides.get("k_time", time_tokens)
    return image_tokens, dm_tokens, time_tokens

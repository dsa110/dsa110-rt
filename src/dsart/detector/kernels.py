"""Detector v1 kernel-bank construction (plan §3.1 lines 475-477 + §4.4).

The v1 deterministic conv-bank fires on `K_img × K_dm × K_time = 128` kernel
triples by default (D2 lock in ``M5_PLAN_FIXES.md``). This module owns the
kernel-bank construction; the conv-bank application lives in ``forward.py``.

Kernel-id schema (plan §3.1 line 489): ``"<image>:<dm>:<time>"`` with tokens
drawn from the closed enumerations in ``common.constants``:

  - image ∈ ``DETECTOR_IMAGE_KERNELS = ("unit", "psf", "psf_shift_lm",
    "psf_shift_l")``
  - dm    ∈ ``DETECTOR_DM_KERNELS    = ("d1", "d3", "d5", "d7")``  (boxcar
    widths in fine-DM bins; odd-only, centred boxcar)
  - time  ∈ ``DETECTOR_TIME_KERNELS  = ("b1", "b2", "b4", "b8", "b16",
    "b32", "b64", "b128")`` (boxcar widths in t_int_search_us samples;
    power-of-two)

V1 image-kernel construction (D10 in ``M5_PLAN_FIXES.md``): all four image
kernels ship as **delta-function** kernels (single-cell 1×1, value 1.0).
The kernel-id namespace is preserved so the cross-kernel merger / decoder /
FAR formulas exercise the full 128-triple bank, but the PSF-aware variants
are stubbed at delta and marked ``# TODO(v2)``. Rationale: the
cube-injection bench (plan §8 line 2329, primary detector correctness gate)
operates on a **post-imager** cube where the PSF is already convolved into
the noise; a v1 detector that ran a second PSF convolution on top would
double-convolve. The PSF-aware kernels become meaningful only at v2 /
hardening when the production ``search_compute`` consumes a pre-imager
sparse-COO tensor (M3-emitted) and the detector applies the PSF as a
matched filter.

K_dm and K_time boxcars are NOT instantiated here as kernel tensors. They
are applied via the ``boxcar_via_cumsum`` cumsum-difference primitive in
``forward.py`` (plan §3.6.13 ``test_detector_conv_flops_cumsum_pin``: the
only allowed K_dm/K_time consumer; ``F.conv1d`` / ``F.avg_pool1d`` /
``F.max_pool1d`` are forbidden along those axes). The widths only enter
the bank as integer fields on each ``Kernel`` record.

Image-kernel L1 normalization (plan §3.6.5 G6 + line 486): each image
kernel is L1-normalised so ``Σ_cells K(Δl, Δm) = 1`` at construction time.
For the delta kernels in v1 this is a no-op (a single-cell 1.0 already has
L1 = 1). The L2-normalised conv-weight pin from plan §4.4 noise_norm
(line 15, "L2-normalised conv weights") applies to the matched-filter
weights *after* the boxcar widths are folded in; v1 with delta image
kernels and unweighted boxcars (sum, not mean) effectively L1-normalises
the image axis and treats the K_dm × K_time boxcar as an unweighted sum,
so the per-kernel score before Layer-2 σ_k normalization is
``√(K_dm · K_time)`` × the σ of the input cube under ideal Gaussian noise.
Layer-2 σ_k EMA absorbs this scaling automatically (it self-normalises).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from ..common.constants import (
    DETECTOR_DM_KERNELS,
    DETECTOR_IMAGE_KERNELS,
    DETECTOR_K_DM_WIDTHS,
    DETECTOR_K_TIME_WIDTHS,
    DETECTOR_TIME_KERNELS,
)

__all__ = [
    "Kernel",
    "build_kernel_bank",
    "make_image_kernel",
    "DEFAULT_DETECTOR_DTYPE",
]


# ---------------------------------------------------------------------------
# Kernel record
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_DTYPE: torch.dtype = torch.float16
"""Default dtype for image-kernel weights and conv-bank scores. Pinned to
fp16 to match the §3.6.11 imager output (real fp16 dirty image) and the
§3.6.12 detector cube dtype."""


@dataclass(frozen=True, slots=True)
class Kernel:
    """One element of the v1 kernel bank (one of 128 triples by default).

    Args:
        kernel_id: Canonical ``"<image>:<dm>:<time>"`` id (plan §3.1 line
            489), e.g. ``"unit:d3:b16"``. Validates against
            ``Candidate._check_kernel_id`` from M1 contracts.
        image_token: First token of ``kernel_id`` (one of
            ``DETECTOR_IMAGE_KERNELS``).
        dm_token: Second token, e.g. ``"d3"`` (one of
            ``DETECTOR_DM_KERNELS``).
        time_token: Third token, e.g. ``"b16"`` (one of
            ``DETECTOR_TIME_KERNELS``).
        k_dm_width: Boxcar width in fine-DM bins (= ``int(dm_token[1:])``).
            Always odd, in ``DETECTOR_K_DM_WIDTHS``.
        k_time_width: Boxcar width in time samples (=
            ``int(time_token[1:])``). Always power-of-two, in
            ``DETECTOR_K_TIME_WIDTHS``.
        image_kernel: 2D image-axis kernel tensor, shape ``[H_k, W_k]``,
            L1-normalised (Σ_cells = 1). For all v1 image-kernel slots this
            is the 1×1 delta tensor ``[[1.0]]`` per D10. The dtype is
            ``DEFAULT_DETECTOR_DTYPE``.
        image_kernel_size: ``image_kernel.shape[-1]`` (square). Cached for
            the conv-bank application code path.
    """

    kernel_id: str
    image_token: str
    dm_token: str
    time_token: str
    k_dm_width: int
    k_time_width: int
    image_kernel: torch.Tensor
    image_kernel_size: int


# ---------------------------------------------------------------------------
# Image-kernel constructors
# ---------------------------------------------------------------------------


def _delta_kernel(
    *, dtype: torch.dtype = DEFAULT_DETECTOR_DTYPE
) -> torch.Tensor:
    """Single-cell 1×1 delta kernel (L1-norm = 1). Used for all four v1
    image-kernel slots per D10 in ``M5_PLAN_FIXES.md``.
    """
    return torch.ones((1, 1), dtype=dtype)


def make_image_kernel(
    image_token: str,
    *,
    dtype: torch.dtype = DEFAULT_DETECTOR_DTYPE,
) -> torch.Tensor:
    """Construct the 2D image-axis kernel for a given ``image_token``.

    Plan §3.1 line 489 image-kernel namespace tokens:

      - ``"unit"``         — unit kernel; v1 + v2 both = 1×1 delta.
      - ``"psf"``          — matched-filter PSF (TODO v2; v1 = delta).
      - ``"psf_shift_lm"`` — PSF shifted by ½ pixel in (l, m) (TODO v2;
                             v1 = delta).
      - ``"psf_shift_l"``  — PSF shifted by ½ pixel in l only (TODO v2;
                             v1 = delta).

    All four are L1-normalised (Σ_cells = 1).
    """
    if image_token not in DETECTOR_IMAGE_KERNELS:
        raise ValueError(
            f"image_token={image_token!r} not in {DETECTOR_IMAGE_KERNELS}"
        )

    if image_token == "unit":
        return _delta_kernel(dtype=dtype)

    # TODO(v2): replace these three with PSF-derived matched-filter kernels.
    # The cube_injection bench (plan §8 line 2329) consumes a post-imager
    # cube where the PSF is already convolved into the noise, so v1 ships
    # with delta placeholders for all four image-kernel slots (D10 in
    # M5_PLAN_FIXES.md). When the production search_compute (chunk 6)
    # consumes a pre-imager sparse-COO tensor, these three become real
    # matched-filter kernels constructed from the M3-emitted PSF.
    if image_token in ("psf", "psf_shift_lm", "psf_shift_l"):
        return _delta_kernel(dtype=dtype)

    raise AssertionError(  # pragma: no cover
        f"unhandled image_token {image_token!r}; "
        f"DETECTOR_IMAGE_KERNELS={DETECTOR_IMAGE_KERNELS}"
    )


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


def _parse_widths(token: str) -> int:
    """Parse a token like ``"d3"`` or ``"b16"`` to its integer width."""
    if not token or token[0] not in ("d", "b") or not token[1:].isdigit():
        raise ValueError(
            f"token {token!r} not of form 'd<int>' or 'b<int>'"
        )
    return int(token[1:])


def build_kernel_bank(
    image_tokens: Sequence[str] = DETECTOR_IMAGE_KERNELS,
    dm_tokens: Sequence[str] = DETECTOR_DM_KERNELS,
    time_tokens: Sequence[str] = DETECTOR_TIME_KERNELS,
    *,
    dtype: torch.dtype = DEFAULT_DETECTOR_DTYPE,
) -> Tuple[Kernel, ...]:
    """Construct the full v1 detector kernel bank (default K = 128).

    The bank is the Cartesian product of
    ``image_tokens × dm_tokens × time_tokens``. Default token sets are
    pinned in ``common.constants`` (D2 lock); operators trading sensitivity
    vs throughput per §9 may shrink the bank by passing subsets.

    Iteration order is ``(image, dm, time)`` outer-to-inner so that
    Layer-2 σ_k EMAs and per-kernel score buffers can be addressed by a
    single flat index ``k = i_img * len(dm) * len(time) + i_dm * len(time)
    + i_time``. The order is stable across runs; downstream consumers
    (decoder, merger, FAR analytics) rely on this.

    Args:
        image_tokens: Subset of ``DETECTOR_IMAGE_KERNELS`` (default: all 4).
        dm_tokens:    Subset of ``DETECTOR_DM_KERNELS`` (default: all 4).
        time_tokens:  Subset of ``DETECTOR_TIME_KERNELS`` (default: all 8).
        dtype:        Image-kernel dtype.

    Returns:
        Tuple of ``Kernel`` records of length
        ``len(image_tokens) × len(dm_tokens) × len(time_tokens)``.
    """
    for tok in image_tokens:
        if tok not in DETECTOR_IMAGE_KERNELS:
            raise ValueError(
                f"image_token {tok!r} not in {DETECTOR_IMAGE_KERNELS}"
            )
    for tok in dm_tokens:
        if tok not in DETECTOR_DM_KERNELS:
            raise ValueError(
                f"dm_token {tok!r} not in {DETECTOR_DM_KERNELS}"
            )
    for tok in time_tokens:
        if tok not in DETECTOR_TIME_KERNELS:
            raise ValueError(
                f"time_token {tok!r} not in {DETECTOR_TIME_KERNELS}"
            )

    bank: list[Kernel] = []
    for image_token in image_tokens:
        image_kernel = make_image_kernel(image_token, dtype=dtype)
        for dm_token in dm_tokens:
            k_dm = _parse_widths(dm_token)
            if k_dm not in DETECTOR_K_DM_WIDTHS:
                raise ValueError(
                    f"dm_token {dm_token!r} parses to width {k_dm}, "
                    f"not in {DETECTOR_K_DM_WIDTHS}"
                )
            if k_dm % 2 == 0:
                raise ValueError(
                    f"dm_token {dm_token!r}: K_dm width must be odd "
                    f"(centred boxcar); got {k_dm}"
                )
            for time_token in time_tokens:
                k_time = _parse_widths(time_token)
                if k_time not in DETECTOR_K_TIME_WIDTHS:
                    raise ValueError(
                        f"time_token {time_token!r} parses to width "
                        f"{k_time}, not in {DETECTOR_K_TIME_WIDTHS}"
                    )
                if k_time & (k_time - 1):
                    raise ValueError(
                        f"time_token {time_token!r}: K_time width must "
                        f"be a power of two; got {k_time}"
                    )
                kernel_id = f"{image_token}:{dm_token}:{time_token}"
                bank.append(
                    Kernel(
                        kernel_id=kernel_id,
                        image_token=image_token,
                        dm_token=dm_token,
                        time_token=time_token,
                        k_dm_width=k_dm,
                        k_time_width=k_time,
                        image_kernel=image_kernel,
                        image_kernel_size=int(image_kernel.shape[-1]),
                    )
                )
    return tuple(bank)

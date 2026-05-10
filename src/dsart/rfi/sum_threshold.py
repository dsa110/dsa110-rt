"""Offringa-style sum-threshold dilation (M3 chunk 3c; plan §4.2 step 5).

Iterative 1D / 2D dilation of an existing boolean RFI mask, applied
along the last axis (or both last two axes for 2D). Grows isolated
flag clusters into contiguous runs that catch the transient tails of
broadband or narrow-band RFI bursts that the per-cell SK / bandpass
detectors only catch at their peak sample.

Algorithm (Offringa 2010, MNRAS 405, 155 — the SumThreshold method):

For each window length ``M ∈ {2, 4, 8, ..., max_m}`` (powers of two,
strictly), slide a length-``M`` window across the axis. A window is
"hot" if it contains more than ``M / η^log2(M)`` flagged cells, where
``η`` (default ``1.5``) is the threshold-shape parameter — equivalent
to Offringa's ``ρ = 1/η`` raised to ``log2(M)``. Every cell inside a
hot window is flagged in the output. Iteration over doubling ``M``
lets isolated-but-clustered flags percolate outward into contiguous
runs.

Default parameters match Offringa's prescription for "moderate"
dilation: ``max_m = 8``, ``η = 1.5``. The ``η = 1.5`` value gives
per-window thresholds ``c_thresh(M) = M / 1.5^log2(M) ∈ {1.33, 1.78,
2.37}`` for ``M ∈ {2, 4, 8}`` — so M = 2 dilates pairs of adjacent
flags into 2-runs, M = 4 dilates 2-runs into 4-runs, M = 8 dilates
3+-of-8 into 8-runs.

This implementation operates on **boolean masks** (the post-detector
form) rather than on residual statistics. The plan's full design
applies sum-threshold to the residual ``r``-statistic plane; chunk 3c
takes the simpler boolean-mask form, which is the standard SUMthr
post-pass used at multiple radio observatories. The two are
mathematically equivalent in the limit where the input statistic is
already binarised by a per-cell threshold (which is what we get
out of SK + bandpass-outlier).
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default maximum window length. Pinned by plan §4.2 step 5.
DEFAULT_MAX_M: Final[int] = 8

#: Default threshold-shape parameter (Offringa ρ = 1/η). Pinned by
#: plan §4.2 step 5: ``η = 1.5^(log2(M))`` (the per-M scaling factor).
DEFAULT_ETA: Final[float] = 1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------


def sum_threshold_1d(
    mask: torch.Tensor,
    *,
    max_m: int = DEFAULT_MAX_M,
    eta: float = DEFAULT_ETA,
) -> torch.Tensor:
    """Apply the SumThreshold dilation along the last axis of ``mask``.

    Args:
        mask: bool tensor of arbitrary leading dims and last axis
            length ``L``. The dilation operates independently on each
            ``[L]`` row of the flattened leading dims.
        max_m: largest window size. Must be a power of two ≥ 2.
            Default :data:`DEFAULT_MAX_M`.
        eta: threshold-shape parameter (≥ 1.0; Offringa default 1.5).

    Returns:
        New bool tensor of same shape as ``mask`` with dilated flags.
        The input is not modified.

    Raises:
        TypeError: ``mask.dtype != torch.bool``.
        ValueError: ``max_m`` not a power of two ≥ 2 or ``eta < 1``.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be bool; got {mask.dtype}")
    if max_m < 2 or not _is_power_of_two(max_m):
        raise ValueError(f"max_m={max_m}; expected power of two >= 2")
    if eta < 1.0:
        raise ValueError(f"eta={eta}; expected >= 1.0")

    out = mask.clone()
    L = out.shape[-1]
    if L < 2:
        return out

    m = 2
    while m <= max_m and m <= L:
        c_thresh = m / (eta ** math.log2(m))

        # 1D sliding sum of out (cast to int8 to keep buffers small).
        flat = out.to(torch.int8)
        # Cumulative sum along last axis.
        cs = torch.cumsum(flat.to(torch.int32), dim=-1)
        zeros = torch.zeros(
            *flat.shape[:-1], 1, dtype=torch.int32, device=mask.device,
        )
        cs0 = torch.cat([zeros, cs], dim=-1)        # (..., L + 1)
        # Window [i, i+m) sum = cs0[..., i+m] - cs0[..., i]; valid for
        # i ∈ [0, L - m].
        win_counts = cs0[..., m:] - cs0[..., :-m]   # (..., L - m + 1)
        hot = win_counts > c_thresh                 # bool (..., L - m + 1)

        # Dilate: cell j is flagged iff any hot[i] is True for some
        # window-start i with i ≤ j < i + m. Iterate over the m offsets:
        new_flag = out.clone()
        hot_u8 = hot.to(torch.uint8)
        for k in range(m):
            pad_l = k
            pad_r = m - 1 - k
            # F.pad pads the last dimensions; we want last axis only.
            shifted = F.pad(hot_u8, (pad_l, pad_r), value=0).bool()
            new_flag = new_flag | shifted
        out = new_flag
        m *= 2
    return out


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------


def sum_threshold_2d(
    mask: torch.Tensor,
    *,
    max_m: int = DEFAULT_MAX_M,
    eta: float = DEFAULT_ETA,
) -> torch.Tensor:
    """Apply 1D SumThreshold along **both** the last two axes.

    Standard Offringa-style "post-pass on the (channel, time) plane":
    we run :func:`sum_threshold_1d` along the time-equivalent axis,
    then transpose and run it again along the frequency-equivalent
    axis, then transpose back. The OR of both passes is returned
    (canonical SumThreshold behaviour).

    Args:
        mask: bool tensor of shape ``[..., AXIS_T, AXIS_F]`` (or
            ``[..., AXIS_F, AXIS_T]`` — the operation is symmetric).
        max_m: per-axis maximum window. Same value used for both axes.
        eta: threshold-shape parameter.

    Returns:
        New bool tensor of same shape as ``mask`` with dilated flags
        on both axes.

    Raises:
        TypeError / ValueError: see :func:`sum_threshold_1d`.
    """
    if mask.ndim < 2:
        raise ValueError(
            f"mask must be at least 2-dim; got shape {tuple(mask.shape)}"
        )

    pass_a = sum_threshold_1d(mask, max_m=max_m, eta=eta)
    transposed = mask.transpose(-1, -2).contiguous()
    pass_b_t = sum_threshold_1d(transposed, max_m=max_m, eta=eta)
    pass_b = pass_b_t.transpose(-1, -2).contiguous()
    return pass_a | pass_b

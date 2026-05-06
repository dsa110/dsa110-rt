"""Per-antenna group-outlier RFI detector (M3 chunk 3c; plan §4.2 step 4).

Flags entire ``(ant, pol)`` cubes whose mean ``S₁_M`` (averaged over
channels) is more than ``k · σ_MAD`` from the population (cross-
antenna) median *at the same pol*. Catches dead, saturated, or grossly
miscalibrated antennas without being fooled by sky-wide effects (a
bright source on transit raises every ant's mean by the same amount,
so the median moves with it and no antenna trips the threshold).

Chunk 3c uses the simplified all-ants-as-one-group population model
described in the briefing. The plan's full design (§4.2 step 4)
allows finer per-SNAP / per-pad-row groups via
``configs/ant_groups.yaml``; the parent M3 agent extends to that
groupwise form when wiring this into the live service.
"""

from __future__ import annotations

from typing import Final

import torch

from dsart.common.constants import NPOL
from dsart.rfi.bandpass_outlier import MAD_TO_SIGMA

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default outlier threshold in MAD-σ units.
DEFAULT_GROUP_K: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def group_outlier_mask(
    s1: torch.Tensor,
    *,
    k: float = DEFAULT_GROUP_K,
    eps: float = 1e-12,
) -> torch.Tensor:
    """All-antennas-as-one-group outlier mask.

    For each polarization:

    1. ``v[ant] = mean_ch(s1[ant, :, pol])`` (per-ant total in-band
       power).
    2. ``med = median(v)``, ``σ = 1.4826 · median(|v - med|)``.
    3. an ``(ant, pol)`` is flagged iff ``|v[ant] - med| > k · σ``.

    When an antenna is flagged at a given pol, **every** ``(ch, pol)``
    cell for that antenna is flagged in the output mask (i.e. the
    detector emits a per-antenna binary decision and broadcasts it
    across channels).

    Args:
        s1: float32 tensor of shape ``[NANTS, NCHAN, NPOL]`` — typically
            the full-cube auto-power spectrum.
        k: outlier threshold in MAD-σ units. Default
            :data:`DEFAULT_GROUP_K`.
        eps: σ floor; if the cross-antenna spread is ≤ eps we suppress
            flagging (no bad antennas detectable in this cube).

    Returns:
        Bool tensor of shape ``[NANTS, NCHAN, NPOL]``.

    Raises:
        ValueError: input not 3-dim or wrong NPOL.
    """
    if s1.ndim != 3:
        raise ValueError(
            f"s1 must be 3-dim (NANTS, NCHAN, NPOL); got shape "
            f"{tuple(s1.shape)}"
        )
    if s1.shape[-1] != NPOL:
        raise ValueError(
            f"s1 last axis must be NPOL={NPOL}; got {s1.shape[-1]}"
        )
    n_ants, n_ch, _ = s1.shape

    # Per-(ant, pol) total in-band power.
    ant_pol_mean = s1.mean(dim=1)              # (NANTS, NPOL)

    # Population median + MAD per pol (across antennas).
    pop_median = torch.median(ant_pol_mean, dim=0, keepdim=True).values  # (1, NPOL)
    abs_dev = (ant_pol_mean - pop_median).abs()
    pop_mad = torch.median(abs_dev, dim=0, keepdim=True).values         # (1, NPOL)
    pop_sigma = MAD_TO_SIGMA * pop_mad

    safe = pop_sigma > eps
    threshold = torch.where(
        safe, k * pop_sigma, torch.full_like(pop_sigma, float("inf")),
    )                                           # (1, NPOL)
    ant_pol_flag = abs_dev > threshold          # (NANTS, NPOL)

    # Broadcast (NANTS, NPOL) → (NANTS, NCHAN, NPOL).
    return ant_pol_flag.unsqueeze(1).expand(n_ants, n_ch, NPOL).contiguous()

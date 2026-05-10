"""Per-antenna bandpass-outlier RFI detector (M3 chunk 3c; plan §4.2 step 3).

Simple per-(ant, pol) median-bandpass + MAD-outlier detector applied
to the full-cube auto-power spectrum ``S₁_4096[ant, ch, pol]``. Flags
channels whose deviation from the per-(ant, pol) median bandpass
exceeds ``k · σ_MAD`` (default ``k = 5``).

This is the *static* (per-cube) form of the bandpass-outlier described
in plan §4.2 step 3. The plan's full design also maintains a 1-pole
IIR ``B_running`` running-mean bandpass with ``τ_B = 30 s`` for the
slow-RFI-drift case; that hot-state form is integrated by the parent
M3 agent in the live ``corr_fast_compute`` service. For chunk 3c we
expose the per-cube static form and let the warmup state machine in
:mod:`dsart.rfi.combine` bypass it during the cold-start window — the
``B_running`` IIR sits in the same place architecturally and is added
when the live service wires this in.
"""

from __future__ import annotations

from typing import Final

import torch

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: 1.4826 = 1 / Φ⁻¹(0.75); converts MAD into a Gaussian-equivalent σ
#: estimator (i.e. for Gaussian X, ``MAD(X) · 1.4826 ≈ σ(X)``).
MAD_TO_SIGMA: Final[float] = 1.4826

#: Default outlier threshold in MAD-σ units.
DEFAULT_BANDPASS_K: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def bandpass_outlier_mask(
    s1: torch.Tensor,
    *,
    k: float = DEFAULT_BANDPASS_K,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Median-bandpass + MAD outlier mask on the per-cube auto-power.

    For each ``(ant, pol)``:

    1. ``med = median(s1[ant, :, pol])`` over channels.
    2. ``σ = 1.4826 · median(|s1[ant, :, pol] - med|)`` (MAD-σ).
    3. flag where ``|s1[ant, ch, pol] - med| > k · σ``.

    Args:
        s1: float32 tensor of shape ``[NANTS, NCHAN, NPOL]`` carrying
            the full-cube auto-power spectrum (typically
            ``S₁_4096`` from :func:`dsart.rfi.autos.compute_autos`'s
            ``M = 4096`` entry, with the singleton ``N_acc = 1`` axis
            squeezed by the caller).
        k: outlier threshold in MAD-σ units. Default
            :data:`DEFAULT_BANDPASS_K`.
        eps: σ floor to avoid division by zero on dead-band antennas
            (where every channel has identical power and MAD = 0).
            Cells with ``σ <= eps`` are *not* flagged by this detector
            (we leave that to group-outlier).

    Returns:
        Bool tensor of shape ``[NANTS, NCHAN, NPOL]``: True where flagged.

    Raises:
        ValueError: input not 3-dim.
    """
    if s1.ndim != 3:
        raise ValueError(
            f"s1 must be 3-dim (NANTS, NCHAN, NPOL); got shape "
            f"{tuple(s1.shape)}"
        )
    # Per-(ant, pol) median over the channel axis.
    med = torch.median(s1, dim=1, keepdim=True).values
    abs_dev = (s1 - med).abs()
    mad = torch.median(abs_dev, dim=1, keepdim=True).values
    sigma = MAD_TO_SIGMA * mad
    # Where σ is collapsed to ~0 (e.g. dead band), suppress flagging.
    safe = sigma > eps
    threshold = torch.where(
        safe, k * sigma, torch.full_like(sigma, float("inf")),
    )
    return abs_dev > threshold

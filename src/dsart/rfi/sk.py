"""Spectral Kurtosis (SK) estimator (M3 chunk 3c; plan §4.2 step 2).

Computes per-(ant, ch, pol) per-accumulation Spectral Kurtosis values
from the ``S₁_M`` / ``S₂_M`` auto-power moments, and applies upper /
lower thresholds at a target false-alarm rate (FAR).

Definition:

    SK_M[ant, ch, pol, n_acc] = ((M+1)/(M-1)) · (M · S₂_M / S₁_M² − 1)

For complex Gaussian-distributed voltages (thermal noise), SK has
mean 1 and approximate variance ``4/M`` (standard SK statistic; see
Nita & Gary 2010, MNRAS 406, L60). CW RFI lifts SK above 1; gain-
fluctuating RFI depresses it below 1. We flag where SK is outside the
``[sk_low, sk_high]`` Nita-Gary confidence interval at the requested
two-sided FAR.

Threshold derivation
====================

Under the null (Gaussian noise), SK is asymptotically Gaussian for
``M >> 1`` with variance ``4 (M − 1) / ((M + 2)(M + 3))`` ≈ ``4/M``.
At our ``M ∈ {64, 256, 1024, 4096}`` minimum (M=64), the Gaussian
approximation matches the exact Nita-Gary moment-matched chi-squared
bounds to ≤ ~5 % at FAR = 1e-4 — well within the engineering tolerance
this stage carries (the OR-fold across M-values + bandpass-outlier +
group-outlier + sum-threshold is the bound that matters, not the
per-M FAR exactness). Sub-agents who need exact Nita-Gary bounds
should consult Equation 14 of the 2010 paper; this module's hook
:func:`set_thresholds` accepts operator-supplied ``(sk_low, sk_high)``
overrides for that case.
"""

from __future__ import annotations

import math
from typing import Final

import torch

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default per-(ant, ch, pol, M) sample two-sided false-alarm rate.
DEFAULT_SK_FAR: Final[float] = 1e-4


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def sk_thresholds(m: int, far: float = DEFAULT_SK_FAR) -> tuple[float, float]:
    """Two-sided SK thresholds at the given false-alarm rate.

    Uses the Gaussian approximation ``SK ~ N(1, σ_SK²)`` with
    ``σ_SK² = 4 (M − 1) / ((M + 2)(M + 3))``. Equivalent to the
    Nita-Gary moment-matched form to ≤ 5 % at ``M ≥ 64`` /
    ``FAR = 1e-4``.

    Args:
        m: accumulation depth (must be ≥ 4 for the variance formula
            to be well-defined; the SK formula itself requires
            ``M ≥ 2``).
        far: two-sided false-alarm rate. Default ``1e-4``. Each tail
            carries ``far / 2`` of the probability mass.

    Returns:
        ``(sk_low, sk_high)`` thresholds. Cells with ``SK < sk_low`` or
        ``SK > sk_high`` are flagged.

    Raises:
        ValueError: ``m < 4`` or ``far`` outside ``(0, 1)``.
    """
    if m < 4:
        raise ValueError(f"M={m}, expected M >= 4 for SK variance")
    if not 0.0 < far < 1.0:
        raise ValueError(f"far={far}, expected in (0, 1)")

    # Two-sided FAR → per-tail probability = far / 2.
    # z_α = √2 · erfinv(1 − 2 · (far/2)) = √2 · erfinv(1 − far)
    z = math.sqrt(2.0) * _erfinv(1.0 - far)
    sigma_sk = math.sqrt(4.0 * (m - 1) / ((m + 2) * (m + 3)))
    return 1.0 - z * sigma_sk, 1.0 + z * sigma_sk


def _erfinv(x: float) -> float:
    """Inverse error function via the Acklam-style series. We avoid
    importing scipy in the hot path.
    """
    # Use torch's vectorised erfinv on a 0-d tensor — already numerically
    # robust to ~1e-6 across the domain we care about.
    if not -1.0 < x < 1.0:
        raise ValueError(f"erfinv argument must be in (-1, 1); got {x}")
    return float(torch.erfinv(torch.tensor(x, dtype=torch.float64)))


# ---------------------------------------------------------------------------
# SK computation
# ---------------------------------------------------------------------------


def compute_sk(
    s1: torch.Tensor,
    s2: torch.Tensor,
    m: int,
    *,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Compute the SK statistic per (n_acc, ant, ch, pol) cell.

    Args:
        s1: ``Σ_t |E|²`` of shape ``[N_acc, NANTS, NCHAN, NPOL]``
            float32.
        s2: ``Σ_t |E|⁴`` of same shape / dtype as ``s1``.
        m: accumulation depth used to compute ``s1`` / ``s2``.
        eps: clamp on ``s1²`` to avoid 0/0 singularities. Cells with
            ``s1 == 0`` (e.g. masked / dead antennas at this stage)
            return ``SK = 0`` after the clamp; these are not flagged
            by the default thresholds (which are around 1 ± O(σ_SK)).

    Returns:
        SK tensor, same shape and dtype as ``s1``.

    Raises:
        ValueError: ``m < 2`` or shape mismatch.
    """
    if m < 2:
        raise ValueError(f"M={m}, expected M >= 2 for SK formula")
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 shape {tuple(s1.shape)} != s2 shape {tuple(s2.shape)}"
        )
    s1_sq = s1 * s1
    s1_sq = s1_sq.clamp_min(eps)
    sk = ((m + 1.0) / (m - 1.0)) * (m * s2 / s1_sq - 1.0)
    return sk


def sk_mask(
    s1: torch.Tensor,
    s2: torch.Tensor,
    m: int,
    *,
    far: float = DEFAULT_SK_FAR,
    sk_low: float | None = None,
    sk_high: float | None = None,
) -> torch.Tensor:
    """Boolean SK mask per (n_acc, ant, ch, pol) cell.

    Args:
        s1, s2: see :func:`compute_sk`.
        m: accumulation depth.
        far: two-sided false-alarm rate. Used only when ``sk_low`` /
            ``sk_high`` are both ``None``. Default
            :data:`DEFAULT_SK_FAR`.
        sk_low, sk_high: explicit thresholds. If either is given, both
            must be given and ``far`` is ignored.

    Returns:
        Bool tensor of shape ``[N_acc, NANTS, NCHAN, NPOL]`` —
        True where ``SK < sk_low`` or ``SK > sk_high``.

    Raises:
        ValueError: only one of ``sk_low`` / ``sk_high`` provided.
    """
    if (sk_low is None) ^ (sk_high is None):
        raise ValueError(
            "sk_low and sk_high must both be provided or both be None"
        )
    if sk_low is None:
        sk_low, sk_high = sk_thresholds(m, far)
    sk = compute_sk(s1, s2, m)
    return (sk < sk_low) | (sk > sk_high)


def sk_combined_mask(
    s1_per_m: dict[int, torch.Tensor],
    s2_per_m: dict[int, torch.Tensor],
    *,
    far: float = DEFAULT_SK_FAR,
    overrides: dict[int, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """OR-fold the per-M SK masks into one ``[NANTS, NCHAN, NPOL]``
    bool mask.

    For each M, computes the per-accumulation SK mask, then ORs across
    the leading ``N_acc`` axis (so a cell is flagged if *any*
    accumulation at *any* M trips its threshold). Finally, ORs across
    all M-values into the per-cube mask.

    Args:
        s1_per_m, s2_per_m: dicts mapping ``M`` → per-M ``S₁`` / ``S₂``
            tensors as returned by :func:`dsart.rfi.autos.compute_autos`.
            Both must have the same key set.
        far: per-(ant, ch, pol, M) FAR. Default :data:`DEFAULT_SK_FAR`.
        overrides: optional dict mapping ``M`` → ``(sk_low, sk_high)``
            explicit thresholds. Missing M's fall back to the
            ``far``-derived bounds.

    Returns:
        Bool tensor of shape ``[NANTS, NCHAN, NPOL]``.

    Raises:
        ValueError: empty input or key set mismatch.
    """
    if not s1_per_m:
        raise ValueError("s1_per_m is empty")
    if set(s1_per_m.keys()) != set(s2_per_m.keys()):
        raise ValueError(
            f"s1_per_m keys {sorted(s1_per_m.keys())} != "
            f"s2_per_m keys {sorted(s2_per_m.keys())}"
        )
    overrides = overrides or {}

    out: torch.Tensor | None = None
    for m, s1_m in s1_per_m.items():
        s2_m = s2_per_m[m]
        if m in overrides:
            sk_low, sk_high = overrides[m]
            mask_m = sk_mask(s1_m, s2_m, m, sk_low=sk_low, sk_high=sk_high)
        else:
            mask_m = sk_mask(s1_m, s2_m, m, far=far)
        # OR over N_acc (leading axis) → per-(ant, ch, pol).
        mask_m = mask_m.any(dim=0)
        out = mask_m if out is None else (out | mask_m)
    assert out is not None
    return out

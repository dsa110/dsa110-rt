"""Dispersion delay helpers (plan §3.6.1, lines 700-712).

Single source of truth for τ_ms(ν, DM) computations across the pipeline.
M5/M6 detector + dumper, M1 ``build_dm_plan.py``, and any per-block
fine-DM-residual code MUST use these functions (not inlined formulas)
so the rounding convention stays consistent.

Units convention:
    ν in GHz
    DM in pc · cm⁻³
    τ in ms (or µs via ``tau_us``)
    K = ``K_DM_MS_GHZ2_PC`` from constants.py.
"""

from __future__ import annotations

import numpy as np

from .constants import K_DM_MS_GHZ2_PC

__all__ = ["tau_ms", "tau_us", "delta_tau_ms", "delta_tau_us", "delay_samples"]


def tau_ms(nu_GHz: float, dm_pc_cm3: float) -> float:
    """Single-frequency dispersion delay in ms.

        τ_ms = K · DM / ν²
    """
    if nu_GHz <= 0.0:
        raise ValueError(f"nu_GHz must be > 0; got {nu_GHz!r}")
    return K_DM_MS_GHZ2_PC * dm_pc_cm3 / (nu_GHz * nu_GHz)


def tau_us(nu_GHz: float, dm_pc_cm3: float) -> float:
    """Single-frequency dispersion delay in µs."""
    return 1000.0 * tau_ms(nu_GHz, dm_pc_cm3)


def delta_tau_ms(
    nu_low_GHz: float,
    nu_high_GHz: float,
    dm_pc_cm3: float,
) -> float:
    """Pair-wise dispersion delay (ms) between two frequencies.

        Δτ_ms = K · DM · (1/ν_low² - 1/ν_high²)

    Convention: the first argument is the *lower* frequency; the result
    is the delay of ``nu_low_GHz`` *relative to* ``nu_high_GHz`` and is
    positive when ``nu_low_GHz < nu_high_GHz`` (lower-freq arrives later).
    The function does NOT enforce ordering — passing reversed args
    returns a negative delay, which is occasionally useful for residuals.
    """
    if nu_low_GHz <= 0.0 or nu_high_GHz <= 0.0:
        raise ValueError(
            f"frequencies must be > 0; got nu_low={nu_low_GHz!r}, nu_high={nu_high_GHz!r}"
        )
    return K_DM_MS_GHZ2_PC * dm_pc_cm3 * (1.0 / (nu_low_GHz * nu_low_GHz) - 1.0 / (nu_high_GHz * nu_high_GHz))


def delta_tau_us(
    nu_low_GHz: float,
    nu_high_GHz: float,
    dm_pc_cm3: float,
) -> float:
    """Pair-wise dispersion delay (µs)."""
    return 1000.0 * delta_tau_ms(nu_low_GHz, nu_high_GHz, dm_pc_cm3)


def delay_samples(
    nu_low_GHz: float,
    nu_high_GHz: float,
    dm_pc_cm3: float,
    t_sample_us: float,
    *,
    rounding: str = "rint",
) -> int:
    """Pair-wise dispersion delay quantised to integer samples (plan §3.6.1).

    Args:
        nu_low_GHz: lower frequency (GHz).
        nu_high_GHz: higher frequency (GHz).
        dm_pc_cm3: dispersion measure (pc cm⁻³).
        t_sample_us: sample period (µs); typically ``t_int_fast_us`` (corr-side)
            or ``t_int_search_us`` (search-side).
        rounding: ``"rint"`` (round-half-to-even, NumPy default) or ``"floor"``
            for unsigned floor. ``"rint"`` is the canonical convention in
            plan §3.6.2/§3.6.3.

    Returns:
        Integer sample shift (signed, follows ``delta_tau_ms`` sign).
    """
    if t_sample_us <= 0.0:
        raise ValueError(f"t_sample_us must be > 0; got {t_sample_us!r}")
    if rounding not in ("rint", "floor"):
        raise ValueError(f"rounding must be 'rint' or 'floor'; got {rounding!r}")

    samples_real = delta_tau_us(nu_low_GHz, nu_high_GHz, dm_pc_cm3) / t_sample_us

    if rounding == "rint":
        return int(np.rint(samples_real))
    return int(np.floor(samples_real))

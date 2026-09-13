"""False-alarm-rate calibration for the MAD-outlier detectors (R3).

Why this exists
===============

:mod:`dsart.rfi.bandpass_outlier` and :mod:`dsart.rfi.group_outlier`
both threshold a robust deviate

    z = |x - median(x)| / (1.4826 * MAD(x))

at a hand-chosen ``k`` (both defaulted to 5.0). ``k = 5`` reads as
"5 sigma" and is not: the deviate is built from a median and a MAD
estimated from the SAME finite sample, and the underlying power is
Gamma- rather than Gaussian-distributed, so the tail is much heavier
than a normal reading implies. Measured here (see
:func:`mad_outlier_far`), ``k = 5`` over 384 channels runs ~2 orders
of magnitude hotter than the Gaussian 5.7e-7.

The fix is the one :mod:`dsart.rfi.sk` already applies: state the
false-alarm rate you want and solve for the threshold that delivers
it under an explicit null.

    bandpass_threshold_k(n_chan, far)   ->  k
    group_threshold_k(n_ant, far)       ->  k

What the FAR is with respect to
===============================

The null is **thermal noise only**: per-cell power drawn from
``Gamma(M, 1/M)`` (mean 1, fractional width ``1/sqrt(M)``), iid across
the axis being tested, with a flat bandpass and identical antennas.

The real null is not that. The bandpass is not flat, antenna gains
differ, and both drift. So the realised on-sky rate will exceed the
configured FAR — the configured value is a *floor*, and its job is to
make the knob mean something and to be comparable across detectors,
not to predict the on-sky rate. :mod:`dsart.rfi` publishes the
realised per-detector rates (``frac_bp`` / ``frac_grp``) so the gap is
measurable rather than assumed.

Cost
====

Solved once per ``(n, far, m_acc)`` and cached for the interpreter's
lifetime, like the SK thresholds. Chunked so peak memory is bounded
regardless of ``n`` — the SK MC's one-shot allocation reached ~61 GB
at M = 4096 before it was chunked, and this module must not repeat
that.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OUTLIER_FAR",
    "MAD_TO_SIGMA",
    "bandpass_threshold_k",
    "group_threshold_k",
    "mad_outlier_far",
]

#: 1.4826 = 1 / Phi^-1(0.75).
MAD_TO_SIGMA: Final[float] = 1.4826

#: Default per-cell false-alarm rate. Matches
#: :data:`dsart.rfi.sk.DEFAULT_SK_FAR` so every detector in the chain
#: is stated on one scale.
DEFAULT_OUTLIER_FAR: Final[float] = 1e-4

#: Trials per solve. 2e5 x n cells puts >= 2e7 samples in the tail
#: estimate at n = 96, which is ample for a 1e-4 quantile.
_N_TRIALS: Final[int] = 200_000

#: Peak bytes per chunk.
_CHUNK_BYTES: Final[int] = 128 * 1024 * 1024

_K_CACHE: dict[tuple[str, int, float, int], float] = {}


def _robust_z(x: np.ndarray) -> np.ndarray:
    """``|x - med| / (1.4826 * MAD)`` along the last axis."""
    med = np.median(x, axis=-1, keepdims=True)
    dev = np.abs(x - med)
    mad = np.median(dev, axis=-1, keepdims=True)
    sigma = MAD_TO_SIGMA * mad
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, dev / np.maximum(sigma, 1e-300), 0.0)
    return z


def _solve_k(
    n: int, far: float, m_acc: int, *, n_trials: int, seed: int,
) -> float:
    """Threshold ``k`` with ``P(z > k) = far`` under the thermal null."""
    if n < 4:
        raise ValueError(f"n={n}, expected >= 4")
    if not 0.0 < far < 1.0:
        raise ValueError(f"far={far}, expected in (0, 1)")
    rng = np.random.default_rng(seed)
    per_trial = max(1, int(_CHUNK_BYTES // (n * 8)))
    parts: list[np.ndarray] = []
    done = 0
    while done < n_trials:
        take = min(per_trial, n_trials - done)
        # Gamma(M, 1/M): mean 1, fractional width 1/sqrt(M).
        x = rng.gamma(shape=float(m_acc), scale=1.0 / float(m_acc),
                      size=(take, n))
        parts.append(_robust_z(x).ravel())
        done += take
    z = np.concatenate(parts)
    del parts
    return float(np.quantile(z, 1.0 - far))


def bandpass_threshold_k(
    n_chan: int,
    far: float = DEFAULT_OUTLIER_FAR,
    *,
    m_acc: int = 4096,
    n_trials: int = _N_TRIALS,
    seed: int = 20260913,
) -> float:
    """``k`` giving a per-channel FAR of ``far`` for the bandpass test.

    ``m_acc`` is the accumulation depth of the spectrum being tested —
    the full-cube ``S1`` that :mod:`dsart.rfi.combine` feeds the
    detector, i.e. ``max(m_values)``.
    """
    key = ("bp", int(n_chan), float(far), int(m_acc))
    if key not in _K_CACHE:
        _K_CACHE[key] = _solve_k(
            int(n_chan), float(far), int(m_acc),
            n_trials=n_trials, seed=seed)
        LOG.info(
            "bandpass FAR calibration: n_chan=%d far=%g M=%d -> k=%.3f",
            n_chan, far, m_acc, _K_CACHE[key])
    return _K_CACHE[key]


def group_threshold_k(
    n_ant: int,
    far: float = DEFAULT_OUTLIER_FAR,
    *,
    n_chan: int = 384,
    m_acc: int = 4096,
    n_trials: int = _N_TRIALS,
    seed: int = 20260913,
) -> float:
    """``k`` giving a per-antenna FAR of ``far`` for the group test.

    The group statistic averages over ``n_chan`` channels first, so the
    effective accumulation is ``m_acc * n_chan`` and the population is
    far closer to Gaussian than the bandpass case.
    """
    eff = max(1, int(m_acc) * int(n_chan))
    key = ("grp", int(n_ant), float(far), eff)
    if key not in _K_CACHE:
        _K_CACHE[key] = _solve_k(
            int(n_ant), float(far), eff, n_trials=n_trials, seed=seed)
        LOG.info(
            "group FAR calibration: n_ant=%d far=%g M_eff=%d -> k=%.3f",
            n_ant, far, eff, _K_CACHE[key])
    return _K_CACHE[key]


def mad_outlier_far(
    n: int, k: float, m_acc: int, *,
    n_trials: int = _N_TRIALS, seed: int = 20260913,
) -> float:
    """Realised per-cell FAR of threshold ``k`` under the thermal null.

    The inverse of :func:`_solve_k`; used by the tests and by the
    null-rate watchdog to state what a legacy hand-chosen ``k`` was
    actually buying.
    """
    rng = np.random.default_rng(seed)
    per_trial = max(1, int(_CHUNK_BYTES // (n * 8)))
    hits = 0
    total = 0
    done = 0
    while done < n_trials:
        take = min(per_trial, n_trials - done)
        x = rng.gamma(shape=float(m_acc), scale=1.0 / float(m_acc),
                      size=(take, n))
        z = _robust_z(x)
        hits += int((z > k).sum())
        total += z.size
        done += take
    return hits / float(total)

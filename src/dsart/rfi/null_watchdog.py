"""Null-rate watchdog for the RFI detectors (R9).

What it is for
==============

Each detector is configured with a false-alarm rate (SK via
``--sk-far``; bandpass and group via :mod:`dsart.rfi.far_calibration`).
That number is a *promise about a clean channel*. This module checks
the promise against what the fleet actually flagged.

A detector drifting off its null is the first symptom of a bandpass
change, a gain problem, or a threshold that has stopped meaning what
it says — all of which show up here long before they show up as a
missed burst.

What it reads
=============

Nothing new. :mod:`dsart.services.rfi_mon_shm` already publishes
per-detector flag fractions (``frac_sk``, ``frac_bp``, ``frac_grp``,
``frac_sumthr``, ``frac_fa``) and per-channel mask counts
(``mask_count_sk`` and friends). This module is a pure function over
those, so it runs in the monitoring path and never on the RT path —
corr_fast has ~1.8 ms of margin on the tightest node and must not pay
for diagnostics.

Why "on a clean channel range"
==============================

The fleet-wide flag fraction is dominated by real RFI, so comparing it
to the configured FAR is meaningless. The comparison only works on
channels that are actually clean. :func:`quietest_channels` picks them
empirically — the lowest-occupancy quantile of the band — rather than
hard-coding a range that a band or an interferer could invalidate.

Caveat worth stating plainly: the configured FAR is with respect to a
thermal null with a flat bandpass and identical antennas (see
:mod:`dsart.rfi.far_calibration`). The real null is not that, so a
realised rate ABOVE the configured FAR is expected. The watchdog looks
for order-of-magnitude departures, not agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TOL_FACTOR",
    "DetectorVerdict",
    "check_null_rates",
    "quietest_channels",
]

#: A detector is reported only when its realised clean-channel rate is
#: outside ``[far / tol, far * tol]``. An order of magnitude either way
#: is deliberately loose: the null model is idealised, so a factor of
#: a few is normal and only a gross departure is actionable.
DEFAULT_TOL_FACTOR: Final[float] = 10.0

#: Fraction of the band treated as "clean" for the comparison.
DEFAULT_QUIET_QUANTILE: Final[float] = 0.5


@dataclass(frozen=True)
class DetectorVerdict:
    """One detector's null-rate check."""

    detector: str
    configured_far: float
    realised: float
    ratio: float          # realised / configured
    ok: bool
    note: str = ""

    def __str__(self) -> str:
        state = "ok" if self.ok else ("HOT" if self.ratio > 1 else "COLD")
        return (
            f"{self.detector:9s} far={self.configured_far:.1e} "
            f"realised={self.realised:.2e} ({self.ratio:6.2f}x) {state}"
            + (f"  {self.note}" if self.note else "")
        )


def quietest_channels(
    mask_count_final: np.ndarray,
    *,
    quantile: float = DEFAULT_QUIET_QUANTILE,
) -> np.ndarray:
    """Bool mask over channels: the lowest-occupancy ``quantile``.

    Args:
        mask_count_final: per-channel flag counts, any shape whose LAST
            axis is channels (the monitor publishes ``(ant, chan, pol)``
            counts; they are summed over everything but channel).
        quantile: fraction of channels to treat as clean.

    Picked empirically rather than hard-coded: a fixed "known clean"
    range stops being clean the moment a new interferer lands in it,
    and then the watchdog silently measures the interferer instead of
    the null.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile={quantile}, expected in (0, 1]")
    counts = np.asarray(mask_count_final)
    if counts.ndim > 1:
        axes = tuple(range(counts.ndim - 1))
        counts = counts.sum(axis=axes)
    if counts.size == 0:
        raise ValueError("mask_count_final has no channels")
    cut = np.quantile(counts, quantile)
    return counts <= cut


def check_null_rates(
    fracs: Mapping[str, float],
    configured: Mapping[str, float],
    *,
    tol_factor: float = DEFAULT_TOL_FACTOR,
    n_cells: int | None = None,
) -> list[DetectorVerdict]:
    """Compare realised per-detector rates against their configured FARs.

    Args:
        fracs: ``detector -> realised flag fraction`` on the clean
            channel set (e.g. from ``frac_sk`` / ``frac_bp`` / ...
            restricted by :func:`quietest_channels`).
        configured: ``detector -> configured FAR``. Detectors absent
            here are skipped — ``flagants`` is a static overlay with no
            FAR, and ``array_burst`` is time-resolved and not part of
            the cube-cadence null.
        tol_factor: report outside ``[far/tol, far*tol]``.
        n_cells: cells behind each fraction. When given, a detector
            whose expected count is below ~5 is reported as
            inconclusive rather than COLD — with 1e-4 over a few
            thousand clean cells, zero flags is the *likely* outcome
            and says nothing.

    Returns:
        One :class:`DetectorVerdict` per detector in ``configured``,
        in that order.
    """
    if tol_factor <= 1.0:
        raise ValueError(f"tol_factor={tol_factor}, expected > 1")
    out: list[DetectorVerdict] = []
    for name, far in configured.items():
        if far <= 0.0:
            raise ValueError(f"{name}: configured far={far}, expected > 0")
        realised = float(fracs.get(name, float("nan")))
        if not np.isfinite(realised):
            out.append(DetectorVerdict(
                name, far, realised, float("nan"), True,
                "not reported by the monitor"))
            continue
        expected_n = None if n_cells is None else far * float(n_cells)
        if expected_n is not None and expected_n < 5.0:
            out.append(DetectorVerdict(
                name, far, realised, realised / far, True,
                f"inconclusive: only {expected_n:.1f} flags expected"))
            continue
        ratio = realised / far if far else float("inf")
        ok = (1.0 / tol_factor) <= ratio <= tol_factor
        out.append(DetectorVerdict(name, far, realised, ratio, ok))
    return out


def format_report(verdicts: Sequence[DetectorVerdict]) -> str:
    """Human-readable block, worst offender first."""
    ranked = sorted(
        verdicts,
        key=lambda v: (v.ok, -abs(np.log10(max(v.ratio, 1e-30)))),
    )
    lines = [str(v) for v in ranked]
    bad = [v for v in verdicts if not v.ok]
    lines.append(
        "all detectors within tolerance" if not bad else
        f"{len(bad)} detector(s) off null: "
        + ", ".join(v.detector for v in bad)
    )
    return "\n".join(lines)

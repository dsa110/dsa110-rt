"""Per-(chgroup, coarse-DM) stage-2 inter-chgroup time-alignment shifts.

Plan §3.6.2 describes a three-stage corr-side dedispersion pipeline:

  * stage-1 — per-channel integer-bin shifts inside each chgroup,
    aligning ch-edge to chgroup-bot frequency at each coarse-DM trial
    (already wired: ``coarse_dm.stage1``).
  * stage-2 — inter-chgroup time-alignment: each chgroup ``g`` advances
    its output time axis so all chgroups land on a common reference
    frequency ``ν_bot_proc`` (= chgroup-15 band-bottom). This is a
    PER-(``g``, coarse-DM ``c``) integer-sample shift in the corr's
    fast-vis cadence.
  * stage-3 — per-fine-DM differential corrections, applied on the
    search side (already wired: ``fine_dm.combiner.compute_time_shift_search``
    with ``include_coarse_offset=False``).

This module owns the *math* of stage-2: given a chgroup index ``g``
and a coarse-DM trial list, it returns the per-coarse-DM integer-sample
shift that must be applied at the TX boundary of the corr_fast service
on the corr node that owns chgroup ``g``.

The companion container :class:`dsart.coarse_dm.stage2_per_dm_fifo.PerCoarseDmStage2FIFO`
implements the *plumbing*: a bank of N_coarse FIFOs each at its own
target depth.

M7.4 production status
======================

The M7.4 250924mptq burst replay currently runs with stage-2 baked into
the SEARCH side ``time_shift_search`` table (``include_coarse_offset=True``
in :func:`dsart.fine_dm.combiner.compute_time_shift_search`). That works
end-to-end but inflates the search-side rolling ring buffer
(``ProductionRxRing._t_stream``) by ~50% because the shifts now have to
absorb the full per-chgroup inter-band delay rather than just the
fine-DM residual.

Moving stage-2 to the corr_fast TX side (the production-correct path,
"Option A") restores ``_t_stream`` to the minimal stage-3-only size,
which matters for the 16x4 production fleet's RAM budget. The math
helpers in this module are the first step; the TX-side wire-in lives
in a follow-up change to ``corr_fast_integration`` and the
``Stage2FIFO`` / ``async_tx`` plumbing.

References
==========

* Plan §3.6.2 lines ~726-770 — stage-2 inter-chgroup alignment.
* :func:`dsart.fine_dm.combiner.compute_time_shift_search` — the
  search-side counterpart; docstring describes the ``ν_bot_proc``
  reference frame and the ``include_coarse_offset`` escape hatch.
* :mod:`dsart.coarse_dm.stage2_fifo` — uniform-depth ring container
  (legacy: was used as a timing buffer before Option A).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from dsart.common.constants import N_CHGROUP, NU_CHGROUP_BOT_GHZ
from dsart.common.dispersion import delta_tau_us


__all__ = [
    "Stage2ShiftTable",
    "compute_stage2_shifts",
    "compute_stage2_shifts_all_chgroups",
]


# Default fast-vis cadence on the corr side (262.144 µs = 32 × 8.192 µs
# native), matching ``t_int_fast_native=32`` in the M7.4 launcher. The
# helpers accept ``t_int_corr_us`` so benches at other cadences can
# override.
T_INT_CORR_US_DEFAULT: float = 262.144


@dataclass(frozen=True, slots=True)
class Stage2ShiftTable:
    """Result of :func:`compute_stage2_shifts`.

    Attributes
    ----------
    chgroup : int
        The chgroup index this table was built for (``0..N_CHGROUP-1``).
    coarse_dm_pc_cm3 : np.ndarray
        ``(N_coarse,) float64`` coarse-DM trial values the shifts were
        computed against. Kept for provenance + downstream sanity checks.
    nu_chgroup_bot_GHz : float
        The chgroup's band-bottom frequency used in the math.
    nu_bot_proc_GHz : float
        The processed-band bottom frequency = chgroup-15's band-bottom.
    t_int_corr_us : float
        The corr-side sample period the shifts are expressed in.
    shifts_samples : np.ndarray
        ``(N_coarse,) int32`` integer-sample shifts. ``shifts_samples[c]``
        is the non-negative number of corr-side fast-vis samples that
        chgroup ``g`` must DELAY its coarse-DM-``c`` output by so the
        signal at ``ν_bot_g`` lines up with the signal at
        ``ν_bot_proc`` after the delay. Sign: chgroup-0 (band top)
        sees a high-DM burst BEFORE chgroup-15 (band bottom), so it
        must DELAY (positive shift) by ``round(Δτ(ν_bot_proc, ν_bot_g,
        coarse_dm[c]) / t_int_corr_us)``.

    Invariants
    ----------
    * ``len(shifts_samples) == len(coarse_dm_pc_cm3)``.
    * ``shifts_samples[c] >= 0`` for every ``c`` (chgroup ``g`` sees the
      burst first; the alignment direction is always a delay, never an
      advance, when shifting toward ``ν_bot_proc``).
    * For ``chgroup == N_CHGROUP - 1`` (band-bottom chgroup) every entry
      is identically zero (``ν_bot_g == ν_bot_proc`` by construction).
    """

    chgroup: int
    coarse_dm_pc_cm3: np.ndarray
    nu_chgroup_bot_GHz: float
    nu_bot_proc_GHz: float
    t_int_corr_us: float
    shifts_samples: np.ndarray

    def __post_init__(self) -> None:
        if self.chgroup < 0 or self.chgroup >= N_CHGROUP:
            raise ValueError(
                f"chgroup={self.chgroup} out of range [0, {N_CHGROUP})"
            )
        if self.shifts_samples.dtype != np.int32:
            raise TypeError(
                f"shifts_samples.dtype={self.shifts_samples.dtype}, "
                "expected int32 (production lock; matches "
                "TimeShiftSearchTable)"
            )
        if self.shifts_samples.shape != self.coarse_dm_pc_cm3.shape:
            raise ValueError(
                f"shifts_samples.shape={self.shifts_samples.shape} != "
                f"coarse_dm_pc_cm3.shape={self.coarse_dm_pc_cm3.shape}"
            )
        if np.any(self.shifts_samples < 0):
            raise ValueError(
                "stage-2 shifts must all be >= 0 (delays toward "
                f"ν_bot_proc); got min={int(self.shifts_samples.min())}"
            )
        if self.chgroup == N_CHGROUP - 1:
            if not np.all(self.shifts_samples == 0):
                bad = self.shifts_samples[self.shifts_samples != 0]
                raise ValueError(
                    f"chgroup={self.chgroup} is the band-bottom; all "
                    f"stage-2 shifts must be zero. Found {bad.size} "
                    f"non-zero entries (first: {int(bad[0])})."
                )

    @property
    def n_coarse(self) -> int:
        return int(self.shifts_samples.shape[0])

    @property
    def max_shift(self) -> int:
        """Largest delay across all coarse DMs (in corr-cadence samples)."""
        return int(self.shifts_samples.max())


def compute_stage2_shifts(
    *,
    chgroup: int,
    coarse_dm_pc_cm3: np.ndarray,
    t_int_corr_us: float = T_INT_CORR_US_DEFAULT,
    nu_chgroup_bot_GHz: Optional[Sequence[float]] = None,
    nu_bot_proc_GHz: Optional[float] = None,
) -> Stage2ShiftTable:
    """Build the per-coarse-DM stage-2 delay table for a single chgroup.

    Math (plan §3.6.2):

        Δt_samples_corr_stage2[g, c] = rint(
            Δτ_us(ν_bot_proc, ν_chgroup_bot[g], coarse_dm[c]) / t_int_corr_us
        )

    Rounding rule: ``numpy.rint`` (half-to-even) to match
    :func:`dsart.fine_dm.combiner.compute_time_shift_search`. This pins
    the search-side stage-3 differentials to the same rounding lattice
    as the corr-side stage-2 delays, so the cross-stage residual stays
    bounded by ±0.5 sample regardless of DM.

    Parameters
    ----------
    chgroup : int
        Local chgroup index for the corr node calling this. ``0..15``;
        chgroup-0 is the top of the band, chgroup-15 the bottom
        (= ``ν_bot_proc`` by construction).
    coarse_dm_pc_cm3 : np.ndarray
        ``(N_coarse,)`` coarse-DM trial values (pc / cm³).
    t_int_corr_us : float, optional
        Corr-side fast-vis sample period in µs. Default 262.144 µs
        (the M7.4 production cadence).
    nu_chgroup_bot_GHz : sequence of float, optional
        Per-chgroup band-bottom frequencies (16-tuple). Default uses
        :data:`dsart.common.constants.NU_CHGROUP_BOT_GHZ`.
    nu_bot_proc_GHz : float, optional
        Processed-band bottom frequency. Default is
        ``nu_chgroup_bot_GHz[N_CHGROUP - 1]`` (chgroup-15's band-bottom).

    Returns
    -------
    Stage2ShiftTable

    Raises
    ------
    ValueError
        If ``chgroup`` is out of range, ``coarse_dm_pc_cm3`` is not 1D,
        any coarse-DM is negative, or the rounding produces a negative
        shift (which can only happen if the caller passes wrong
        frequencies).
    """
    if chgroup < 0 or chgroup >= N_CHGROUP:
        raise ValueError(
            f"chgroup={chgroup} out of range [0, {N_CHGROUP})"
        )
    if coarse_dm_pc_cm3.ndim != 1:
        raise ValueError(
            f"coarse_dm_pc_cm3 must be 1D, got ndim={coarse_dm_pc_cm3.ndim}"
        )
    if np.any(coarse_dm_pc_cm3 < 0):
        raise ValueError(
            "coarse_dm_pc_cm3 contains negative entries; coarse DMs "
            "must be >= 0 (physical)"
        )
    if t_int_corr_us <= 0:
        raise ValueError(
            f"t_int_corr_us={t_int_corr_us} must be > 0"
        )

    chgroup_bot = np.asarray(
        nu_chgroup_bot_GHz
        if nu_chgroup_bot_GHz is not None
        else NU_CHGROUP_BOT_GHZ,
        dtype=np.float64,
    )
    if chgroup_bot.shape != (N_CHGROUP,):
        raise ValueError(
            f"nu_chgroup_bot_GHz must be a {N_CHGROUP}-tuple; got "
            f"shape {chgroup_bot.shape}"
        )

    nu_bot_proc = (
        float(nu_bot_proc_GHz)
        if nu_bot_proc_GHz is not None
        else float(chgroup_bot[N_CHGROUP - 1])
    )
    nu_g = float(chgroup_bot[chgroup])

    n_coarse = int(coarse_dm_pc_cm3.shape[0])
    shifts = np.zeros(n_coarse, dtype=np.int32)
    for c in range(n_coarse):
        # delta_tau_us(nu_low, nu_high, dm): positive when nu_high > nu_low.
        # We want: how much does ν_bot_g lead ν_bot_proc for a given DM?
        # Since nu_g >= nu_bot_proc (chgroup-0 ≥ chgroup-15), the higher-
        # frequency chgroup sees the signal FIRST and must DELAY by the
        # interval Δτ(ν_bot_proc, ν_g, DM) ≥ 0 to align with chgroup-15.
        d_us = delta_tau_us(nu_bot_proc, nu_g, float(coarse_dm_pc_cm3[c]))
        if d_us < 0:
            # Sanity: should not happen given nu_g >= nu_bot_proc and DM >= 0.
            raise ValueError(
                f"compute_stage2_shifts: negative Δτ={d_us:.6g} µs at "
                f"chgroup={chgroup} coarse_dm={float(coarse_dm_pc_cm3[c])} "
                f"nu_g={nu_g} nu_bot_proc={nu_bot_proc}. Check the "
                f"chgroup table ordering."
            )
        shifts[c] = int(np.rint(d_us / t_int_corr_us))

    # Force chgroup-15 row to zero (paranoia: floating-point noise in
    # delta_tau at the construction frequency could yield ±1-sample
    # noise away from zero on some platforms; matches the same
    # treatment in compute_time_shift_search).
    if chgroup == N_CHGROUP - 1:
        shifts[:] = 0

    return Stage2ShiftTable(
        chgroup=int(chgroup),
        coarse_dm_pc_cm3=np.ascontiguousarray(coarse_dm_pc_cm3, dtype=np.float64),
        nu_chgroup_bot_GHz=nu_g,
        nu_bot_proc_GHz=nu_bot_proc,
        t_int_corr_us=float(t_int_corr_us),
        shifts_samples=shifts,
    )


def compute_stage2_shifts_all_chgroups(
    *,
    coarse_dm_pc_cm3: np.ndarray,
    t_int_corr_us: float = T_INT_CORR_US_DEFAULT,
    nu_chgroup_bot_GHz: Optional[Sequence[float]] = None,
    nu_bot_proc_GHz: Optional[float] = None,
) -> Tuple[Stage2ShiftTable, ...]:
    """Convenience: compute :func:`compute_stage2_shifts` for all 16 chgroups.

    Returns a tuple indexed by chgroup. Useful for benchmarks and
    design-time analysis (e.g., sizing the per-(g, c) FIFO depths in
    the production transport-TX path).
    """
    return tuple(
        compute_stage2_shifts(
            chgroup=g,
            coarse_dm_pc_cm3=coarse_dm_pc_cm3,
            t_int_corr_us=t_int_corr_us,
            nu_chgroup_bot_GHz=nu_chgroup_bot_GHz,
            nu_bot_proc_GHz=nu_bot_proc_GHz,
        )
        for g in range(N_CHGROUP)
    )

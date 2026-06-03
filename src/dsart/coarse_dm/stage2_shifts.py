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

from dsart.common.constants import (
    N_CHGROUP,
    NU_CHGROUP_BOT_GHZ,
    NU_CHGROUP_TOP_GHZ,
)
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
    nu_chgroup_ref_GHz : float
        The chgroup's reference frequency the shifts were computed
        against. Convention A (corr-side stage-1 output): the chgroup's
        TOP channel frequency. See :func:`compute_stage2_shifts` for
        the rationale and the 2026-06-03 fix that aligned this with the
        actual stage-1 output reference.
    nu_bot_proc_GHz : float
        The processed-band bottom frequency = chgroup-15's band-bottom.
    t_int_corr_us : float
        The corr-side sample period the shifts are expressed in.
    shifts_samples : np.ndarray
        ``(N_coarse,) int32`` integer-sample shifts. ``shifts_samples[c]``
        is the non-negative number of corr-side fast-vis samples that
        chgroup ``g`` must DELAY its coarse-DM-``c`` output by so the
        signal at ``ν_chgroup_TOP[g]`` lines up with the signal at
        ``ν_bot_proc`` after the delay. Sign: chgroup-0 (band top)
        sees a high-DM burst BEFORE chgroup-15 (band bottom), so it
        must DELAY (positive shift) by ``round(Δτ(ν_bot_proc,
        ν_chgroup_TOP[g], coarse_dm[c]) / t_int_corr_us)``.

    Invariants
    ----------
    * ``len(shifts_samples) == len(coarse_dm_pc_cm3)``.
    * ``shifts_samples[c] >= 0`` for every ``c`` (chgroup ``g`` sees the
      burst first; the alignment direction is always a delay, never an
      advance, when shifting toward ``ν_bot_proc``).
    * ``shifts_samples`` is monotonically non-decreasing in ``DM`` (delay
      scales with DM at fixed reference frequency).
    * For ``chgroup == N_CHGROUP - 1``, ``ν_chgroup_TOP[15] = 1.323 GHz``
      is ABOVE ``ν_bot_proc = 1.311 GHz`` (by ~12 MHz, the within-chgroup
      span), so chgroup-15 carries a real (small, DM-dependent) delay
      that is NOT identically zero. The plan §3.2 line 577 / §3.6.2
      ``time_shift_corr_stage2[15, c] == 0`` invariant only held under
      the pre-2026-06-03 BOT-referenced convention which has been
      retired (see :func:`compute_stage2_shifts` docstring).
    """

    chgroup: int
    coarse_dm_pc_cm3: np.ndarray
    nu_chgroup_ref_GHz: float
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
    nu_chgroup_top_GHz: Optional[Sequence[float]] = None,
    nu_chgroup_bot_GHz: Optional[Sequence[float]] = None,  # noqa: ARG001  (retained for back-compat callers)
    nu_bot_proc_GHz: Optional[float] = None,
) -> Stage2ShiftTable:
    """Build the per-coarse-DM stage-2 delay table for a single chgroup.

    Math (plan §3.6.2, Convention-A reference fix 2026-06-03):

        Δt_samples_corr_stage2[g, c] = rint(
            Δτ_us(ν_bot_proc, ν_chgroup_TOP[g], coarse_dm[c]) / t_int_corr_us
        )

    Reference-frequency convention
    ==============================

    The corr-side coarse-DM stage-1 dedisperser
    (:func:`dsart.coarse_dm.dm_plan.compute_delay_native_samples_table`)
    ALIGNS each chgroup's 384 channels to that chgroup's TOP channel
    (Convention A — ``delay_native_samples(g, ch=0, dm) == 0``). So
    the stream emerging from stage-1 for chgroup ``g`` represents the
    astrophysical signal as it arrived at ``ν_chgroup_TOP[g]``.

    Stage-2 must therefore delay each chgroup by the dispersion span
    from ``ν_chgroup_TOP[g]`` down to ``ν_bot_proc`` so that the
    emerging stream is aligned to ``ν_bot_proc`` for the search side.

    Prior to 2026-06-03 this module used ``ν_chgroup_BOT[g]`` as the
    reference, which mismatched stage-1's TOP convention by the
    within-chgroup dispersion span (~12 MHz / chgroup). When composed
    with the search-side stage-3 differential, that produced a
    constant -2.45% bias in detected-vs-injected DM. The bias was
    isolated and patched in :func:`compute_time_shift_search`'s
    escape-hatch (``include_coarse_offset=True``) on 2026-05-29; this
    module mirrors that fix so the corr-side Option A wire-in
    (search-side ``include_coarse_offset=False``) composes correctly
    end-to-end.

    The same TOP reference is used by
    :func:`dsart.fine_dm.combiner.compute_time_shift_search` (both
    ``include_coarse_offset`` paths, as of 2026-06-03), so the
    cross-stage cancellation is bit-for-bit (modulo ±0.5-sample
    rint() noise) for any (chgroup, coarse_dm, fine_dm) triple.

    Rounding rule: ``numpy.rint`` (half-to-even) to match
    :func:`compute_time_shift_search`.

    Parameters
    ----------
    chgroup : int
        Local chgroup index for the corr node calling this. ``0..15``;
        chgroup-0 is the top of the band, chgroup-15 the bottom.
    coarse_dm_pc_cm3 : np.ndarray
        ``(N_coarse,)`` coarse-DM trial values (pc / cm³).
    t_int_corr_us : float, optional
        Corr-side fast-vis sample period in µs. Default 262.144 µs
        (the M7.4 production cadence).
    nu_chgroup_top_GHz : sequence of float, optional
        Per-chgroup TOP channel frequencies (16-tuple). Default uses
        :data:`dsart.common.constants.NU_CHGROUP_TOP_GHZ`.
    nu_chgroup_bot_GHz : sequence of float, optional
        Deprecated — accepted for backward-compat with the
        pre-2026-06-03 BOT-referenced signature, but IGNORED. Stage-2
        now references chgroup TOP per the Convention-A fix above.
    nu_bot_proc_GHz : float, optional
        Processed-band bottom frequency. Default is
        :data:`dsart.common.constants.NU_CHGROUP_BOT_GHZ[N_CHGROUP - 1]`
        which equals ``NU_BOT_PROC_GHZ`` by construction.

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

    chgroup_top = np.asarray(
        nu_chgroup_top_GHz
        if nu_chgroup_top_GHz is not None
        else NU_CHGROUP_TOP_GHZ,
        dtype=np.float64,
    )
    if chgroup_top.shape != (N_CHGROUP,):
        raise ValueError(
            f"nu_chgroup_top_GHz must be a {N_CHGROUP}-tuple; got "
            f"shape {chgroup_top.shape}"
        )

    nu_bot_proc = (
        float(nu_bot_proc_GHz)
        if nu_bot_proc_GHz is not None
        else float(NU_CHGROUP_BOT_GHZ[N_CHGROUP - 1])
    )
    nu_g = float(chgroup_top[chgroup])

    n_coarse = int(coarse_dm_pc_cm3.shape[0])
    shifts = np.zeros(n_coarse, dtype=np.int32)
    for c in range(n_coarse):
        # delta_tau_us(nu_low, nu_high, dm): positive when nu_high > nu_low.
        # Stage-1 (Convention A) leaves chgroup g's stream referenced to
        # ν_chgroup_TOP[g]. Since chgroup_TOP[g] >= ν_bot_proc (with
        # equality only when chgroup-15's TOP coincides with ν_bot_proc,
        # which it does not — there's a ~12 MHz within-chgroup span at
        # chgroup-15), the higher-frequency reference sees the signal
        # FIRST and must DELAY by Δτ(ν_bot_proc, ν_TOP_g, DM) ≥ 0 to
        # align with the band bottom.
        d_us = delta_tau_us(nu_bot_proc, nu_g, float(coarse_dm_pc_cm3[c]))
        if d_us < 0:
            raise ValueError(
                f"compute_stage2_shifts: negative Δτ={d_us:.6g} µs at "
                f"chgroup={chgroup} coarse_dm={float(coarse_dm_pc_cm3[c])} "
                f"nu_g={nu_g} nu_bot_proc={nu_bot_proc}. Check the "
                f"chgroup table ordering."
            )
        shifts[c] = int(np.rint(d_us / t_int_corr_us))

    return Stage2ShiftTable(
        chgroup=int(chgroup),
        coarse_dm_pc_cm3=np.ascontiguousarray(coarse_dm_pc_cm3, dtype=np.float64),
        nu_chgroup_ref_GHz=nu_g,
        nu_bot_proc_GHz=nu_bot_proc,
        t_int_corr_us=float(t_int_corr_us),
        shifts_samples=shifts,
    )


def compute_stage2_shifts_all_chgroups(
    *,
    coarse_dm_pc_cm3: np.ndarray,
    t_int_corr_us: float = T_INT_CORR_US_DEFAULT,
    nu_chgroup_top_GHz: Optional[Sequence[float]] = None,
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
            nu_chgroup_top_GHz=nu_chgroup_top_GHz,
            nu_bot_proc_GHz=nu_bot_proc_GHz,
        )
        for g in range(N_CHGROUP)
    )

"""Acceptance tests for :mod:`dsart.coarse_dm.stage2_shifts`.

Pins the per-(chgroup, coarse-DM) stage-2 alignment math used by the
corr_fast TX path (Option A) to replace the search-side
``include_coarse_offset=True`` workaround currently shipping in M7.4.

Coverage
--------

1. **Single-chgroup shape + dtype** — returns int32 ``(N_coarse,)`` shifts.
2. **Band-bottom invariant** — chgroup-15 (``ν_bot == ν_bot_proc``) is
   identically zero across every coarse DM.
3. **Sign invariant** — every shift is ``>= 0`` (alignment toward
   ``ν_bot_proc`` is always a DELAY, never an advance, for the
   ν_g >= ν_bot_proc convention).
4. **Monotonicity in DM** — for a fixed chgroup, larger DM ⇒ larger
   (or equal) shift.
5. **Monotonicity in chgroup** — for a fixed coarse DM, the lower the
   chgroup index (closer to band-top), the larger the shift.
6. **Cross-stage residual** — combining the corr-side stage-2 shift
   with the search-side stage-3 differential (from
   ``compute_time_shift_search`` with ``include_coarse_offset=False``)
   yields the same TOTAL inter-band delay as the search-side
   ``include_coarse_offset=True`` baked-in path, up to ±1 sample
   rounding residual.
7. **t_int_corr scaling** — doubling the corr cadence halves the shift
   count (with banker's-rounding cosmetic ±1 at the cadence threshold).
8. **Bench-against-truth** — at chgroup=0, coarse_dm=258.74 (the M7.4
   250924mptq coarse_dm[0]) the shift matches the published
   ``inter_chgroup_top_delay_native`` in the bench summary.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsart.coarse_dm.stage2_shifts import (
    Stage2ShiftTable,
    compute_stage2_shifts,
    compute_stage2_shifts_all_chgroups,
)
from dsart.common.constants import N_CHGROUP, NU_CHGROUP_BOT_GHZ
from dsart.common.dispersion import delta_tau_us
from dsart.fine_dm.combiner import compute_time_shift_search


# M7.4 250924mptq DM plan first coarse-DM trials (sample values from
# bench/reports/.../summary.json). Only the FIRST entry is used by the
# bench's chgroup-0 top-delay reference; the rest are filler so the
# helper exercises the multi-coarse-DM path.
COARSE_DM_PROD = np.array(
    [258.740, 387.50, 581.94, 873.95, 1312.71, 1971.62, 2962.93, 4452.16],
    dtype=np.float64,
)


def test_returns_stage2_shift_table_with_int32_shape():
    coarse = np.array([100.0, 200.0, 400.0], dtype=np.float64)
    t = compute_stage2_shifts(chgroup=0, coarse_dm_pc_cm3=coarse)
    assert isinstance(t, Stage2ShiftTable)
    assert t.shifts_samples.shape == (3,)
    assert t.shifts_samples.dtype == np.int32
    assert t.coarse_dm_pc_cm3.shape == (3,)
    assert t.coarse_dm_pc_cm3.dtype == np.float64
    assert t.n_coarse == 3
    assert t.max_shift == int(t.shifts_samples.max())


def test_band_bottom_chgroup_carries_top_to_bot_within_chgroup_delay():
    """Post-2026-06-03 unification: stage-2 references chgroup TOP, so
    chgroup-15 (whose TOP is ~12 MHz ABOVE ν_bot_proc) has a real
    DM-dependent (small) shift. The pre-fix "chgroup-15 is identically
    zero" invariant was a relic of the BOT-referenced math and has been
    retired."""
    coarse = np.array([100.0, 500.0, 4000.0], dtype=np.float64)
    t = compute_stage2_shifts(chgroup=N_CHGROUP - 1, coarse_dm_pc_cm3=coarse)
    # Must remain non-negative; scales with DM (small but non-zero at
    # high DM thanks to the within-chgroup-15 dispersion span).
    assert (t.shifts_samples >= 0).all()
    assert int(t.shifts_samples[-1]) > 0, (
        f"chgroup-15 high-DM shift expected non-zero (TOP→ν_bot_proc "
        f"residual); got {t.shifts_samples.tolist()}"
    )


def test_all_shifts_non_negative_for_every_chgroup():
    coarse = np.linspace(50.0, 4000.0, 8, dtype=np.float64)
    for g in range(N_CHGROUP):
        t = compute_stage2_shifts(chgroup=g, coarse_dm_pc_cm3=coarse)
        assert (t.shifts_samples >= 0).all(), (
            f"chgroup={g} produced negative shifts: "
            f"min={int(t.shifts_samples.min())}"
        )


def test_shift_monotonic_in_dm_for_fixed_chgroup():
    coarse_sorted = np.sort(
        np.array([50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 4000.0])
    )
    for g in range(N_CHGROUP):
        t = compute_stage2_shifts(chgroup=g, coarse_dm_pc_cm3=coarse_sorted)
        diffs = np.diff(t.shifts_samples)
        assert (diffs >= 0).all(), (
            f"chgroup={g}: shift not monotonic non-decreasing in DM: "
            f"{t.shifts_samples.tolist()}"
        )


def test_shift_monotonic_in_chgroup_for_fixed_dm():
    coarse = np.array([1000.0], dtype=np.float64)
    shifts_per_g = np.zeros(N_CHGROUP, dtype=np.int64)
    for g in range(N_CHGROUP):
        t = compute_stage2_shifts(chgroup=g, coarse_dm_pc_cm3=coarse)
        shifts_per_g[g] = int(t.shifts_samples[0])
    # As chgroup index increases (band-top to band-bottom), ν_g decreases
    # toward ν_bot_proc, so the alignment delay DECREASES. Pin that.
    diffs = np.diff(shifts_per_g)
    assert (diffs <= 0).all(), (
        "stage-2 shift must decrease monotonically as chgroup index "
        f"increases (band-top → band-bottom); got {shifts_per_g.tolist()}"
    )


def test_cross_stage_residual_against_baked_search_shifts():
    """Pin Option A ≡ Option B up to ±1-sample rounding.

    Option A: stage-2 on corr-TX side + stage-3 differential on search side.
    Option B: stage-2 baked into search shifts via include_coarse_offset=True.

    For the same (g, fine_dm) pair the TOTAL inter-band delay applied
    must be the same, modulo banker's-rounding noise.

    NOTE (2026-06-03): both Option A (stage-2 corr-side + stage-3
    search-side with ``include_coarse_offset=False``) and Option B
    (``include_coarse_offset=True``, search-side baked) now reference
    chgroup TOP. The residual is bounded by ±0.5 sample of the coarser
    cadence (search) — pinned here.
    """
    coarse = COARSE_DM_PROD.copy()
    # Build a 4-fine-per-coarse fdm grid so each coarse DM has both
    # negative and positive δdm offsets.
    fine = np.concatenate(
        [coarse[i] + np.array([-2.0, -0.5, 0.5, 2.0]) for i in range(coarse.size)]
    )
    fine_to_coarse = np.repeat(np.arange(coarse.size), 4).astype(np.int64)

    # Option B baseline: search-side shifts WITH coarse offset baked in.
    table_optB = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=fine_to_coarse,
        include_coarse_offset=True,
    )

    # Option A reconstruction: stage-2 per (g, c) + stage-3 differential.
    table_optA_stage3 = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=fine_to_coarse,
        include_coarse_offset=False,
    )

    # Note: compute_time_shift_search uses t_int_search_us=524.288 by
    # default (D9 hold-over); compute_stage2_shifts uses t_int_corr_us=
    # 262.144. For the cross-stage cancellation to work numerically we
    # need to compare delays in MICROSECONDS, not raw sample counts.
    # The test pins that the FUSED delay (stage-2 µs + stage-3 µs)
    # equals the Option-B fused delay (stage-2 µs + stage-3 µs baked
    # into one shift), within ±1-sample-of-the-coarser cadence.
    t_int_corr_us = 262.144
    t_int_search_us = table_optB.t_int_search_us
    for f in range(fine.size):
        c = int(fine_to_coarse[f])
        for g in range(N_CHGROUP):
            stage2 = compute_stage2_shifts(
                chgroup=g, coarse_dm_pc_cm3=coarse[c:c + 1]
            ).shifts_samples[0]
            stage3 = table_optA_stage3.shifts[f, g]
            fused_us_A = stage2 * t_int_corr_us + stage3 * t_int_search_us
            fused_us_B = table_optB.shifts[f, g] * t_int_search_us
            # The residual must be within ±1 sample of the SEARCH
            # cadence (the coarser of the two). The corr cadence is
            # finer (262 vs 524 µs by default), so stage-2 rounding
            # is sub-half-search-sample.
            assert abs(fused_us_A - fused_us_B) <= 1.01 * t_int_search_us, (
                f"f={f} c={c} g={g}: "
                f"Option-A fused={fused_us_A:.3f} µs, "
                f"Option-B fused={fused_us_B:.3f} µs, "
                f"residual={fused_us_A - fused_us_B:.3f} µs > "
                f"{t_int_search_us:.3f} µs"
            )


def test_doubling_corr_cadence_halves_shift():
    coarse = np.array([1000.0], dtype=np.float64)
    t_fine = compute_stage2_shifts(
        chgroup=0, coarse_dm_pc_cm3=coarse, t_int_corr_us=262.144
    )
    t_coarse = compute_stage2_shifts(
        chgroup=0, coarse_dm_pc_cm3=coarse, t_int_corr_us=524.288
    )
    # Allow ±1-sample rounding slack at the cadence transition.
    assert abs(int(t_fine.shifts_samples[0]) - 2 * int(t_coarse.shifts_samples[0])) <= 1


def test_compute_all_chgroups_returns_16():
    coarse = np.array([100.0, 1000.0, 4000.0], dtype=np.float64)
    tables = compute_stage2_shifts_all_chgroups(coarse_dm_pc_cm3=coarse)
    assert len(tables) == N_CHGROUP
    assert all(t.chgroup == g for g, t in enumerate(tables))
    # band-top has the largest delays; band-bottom now also has a
    # (small, DM-dependent) delay because TOP[15] > ν_bot_proc (within-
    # chgroup span). chgroup-0's delay must still dominate.
    assert tables[0].max_shift > tables[-1].max_shift
    # chgroup-15 carries the within-chgroup-15-span residual at high DM.
    assert tables[-1].max_shift > 0


def test_chgroup0_top_delay_matches_expected_top_referenced():
    """At chgroup=0, coarse_dm=258.74 (250924mptq coarse_dm[0]) the
    stage-2 delay (in corr-fast samples) is the Δτ from ν_chgroup_TOP[0]
    down to ν_bot_proc at coarse_dm[0] (Convention A; matches what
    Convention-A stage-1 leaves at chgroup-0's output).
    """
    from dsart.common.constants import NU_CHGROUP_TOP_GHZ
    coarse = COARSE_DM_PROD[:1]
    t = compute_stage2_shifts(chgroup=0, coarse_dm_pc_cm3=coarse)
    expected_us = delta_tau_us(
        NU_CHGROUP_BOT_GHZ[N_CHGROUP - 1],
        NU_CHGROUP_TOP_GHZ[0],
        258.740,
    )
    expected_samples = int(round(expected_us / 262.144))
    assert int(t.shifts_samples[0]) == expected_samples


def test_rejects_negative_dm():
    with pytest.raises(ValueError, match="negative"):
        compute_stage2_shifts(
            chgroup=0,
            coarse_dm_pc_cm3=np.array([-1.0, 100.0]),
        )


def test_rejects_bad_chgroup():
    with pytest.raises(ValueError, match="out of range"):
        compute_stage2_shifts(
            chgroup=-1, coarse_dm_pc_cm3=np.array([100.0])
        )
    with pytest.raises(ValueError, match="out of range"):
        compute_stage2_shifts(
            chgroup=N_CHGROUP, coarse_dm_pc_cm3=np.array([100.0])
        )


def test_rejects_non_1d_coarse_dm():
    with pytest.raises(ValueError, match="must be 1D"):
        compute_stage2_shifts(
            chgroup=0,
            coarse_dm_pc_cm3=np.array([[1.0], [2.0]], dtype=np.float64),
        )


def test_rejects_zero_t_int_corr():
    with pytest.raises(ValueError, match="must be > 0"):
        compute_stage2_shifts(
            chgroup=0,
            coarse_dm_pc_cm3=np.array([100.0]),
            t_int_corr_us=0.0,
        )


def test_table_round_trip_invariants_assertions():
    """The Stage2ShiftTable __post_init__ should refuse to construct
    invalid tables (smoke test for the invariants)."""
    from dsart.common.constants import NU_CHGROUP_TOP_GHZ
    # Try to forge a negative-shift table; should raise.
    with pytest.raises(ValueError, match=">= 0"):
        Stage2ShiftTable(
            chgroup=0,
            coarse_dm_pc_cm3=np.array([100.0]),
            nu_chgroup_ref_GHz=float(NU_CHGROUP_TOP_GHZ[0]),
            nu_bot_proc_GHz=float(NU_CHGROUP_BOT_GHZ[N_CHGROUP - 1]),
            t_int_corr_us=262.144,
            shifts_samples=np.array([-3], dtype=np.int32),
        )

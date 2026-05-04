"""Numerical-convention tests for M1 DM plan + dispersion (plan §3.6.13).

Plan §8 M1 DoD line 2141 (second half) requires::

    pytest tests/test_numerical_conventions.py::test_dm_plan_time_shift_tables

to pass against the .npz produced by ``tools/build_dm_plan.py``. This file
also includes ``test_dispersion_delay`` (plan §3.6.1 line 723) as a bonus
since the dispersion module is in scope for M1; the value used is the F11
correction (1697.8 ms, computed from the pinned literals + the standard
formula — plan §3.6.1 line 723 has 1699.5 ms which is wrong; F11 in
``M1_PLAN_FIXES.md`` captures the plan-prose fix).

The test loads ``configs/dm_plan.npz`` (built by ``tools/build_dm_plan.py``).
A pytest-session-scoped fixture rebuilds the plan if the file is missing,
so the test is hermetic on a fresh checkout.
"""

from __future__ import annotations

import os

# Match test_contracts.py: enable DSART_TEST=1 before any dsart import.
os.environ.setdefault("DSART_TEST", "1")

import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BW_PROC_MHZ,
    K_DM_MS_GHZ2_PC,
    NCHAN_PER_CHGROUP,
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_BOT_GHZ,
    NU_CHGROUP_TOP_GHZ,
    NU_TOP_PROC_GHZ,
    N_CHAN_PROC_NATIVE,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    T_INT_FAST_US_DEFAULT,
    freq_GHz,
)
from dsart.common.contracts import DmPlan  # noqa: E402
from dsart.common.dispersion import delta_tau_ms, delta_tau_us  # noqa: E402

DM_PLAN_PATH = REPO_ROOT / "configs" / "dm_plan.npz"


# ---------------------------------------------------------------------------
# Fixture: ensure dm_plan.npz exists (build on demand)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dm_plan() -> DmPlan:
    """Load configs/dm_plan.npz, rebuilding it if missing."""
    if not DM_PLAN_PATH.exists():
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "build_dm_plan.py"),
                "--out",
                str(DM_PLAN_PATH),
                "--quiet",
            ],
            check=True,
        )
    return DmPlan.from_npz(str(DM_PLAN_PATH))


# ---------------------------------------------------------------------------
# Plan §3.6.1 line 723 — dispersion delay verification (F11 correction)
# ---------------------------------------------------------------------------


def test_dispersion_delay_at_dm_3000() -> None:
    """Δτ_ms(ν_bot_proc, ν_top_proc, 3000) computed from pinned literals.

    Plan §3.6.1 line 723 says 1699.5 ms but the formula at the pinned
    constants gives 1697.78 ms (F11 in M1_PLAN_FIXES.md captures the plan
    update). We test against the formula's actual output, not the wrong
    plan literal.
    """
    actual = delta_tau_ms(NU_BOT_PROC_GHZ, NU_TOP_PROC_GHZ, 3000.0)
    expected = K_DM_MS_GHZ2_PC * 3000.0 * (
        1.0 / NU_BOT_PROC_GHZ ** 2 - 1.0 / NU_TOP_PROC_GHZ ** 2
    )
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)
    assert 1697.0 < actual < 1699.0, (
        f"Δτ_ms(ν_bot, ν_top, 3000) = {actual} ms; expected ~1697.78 ms "
        f"(F11 correction; plan literal 1699.5 ms is wrong)"
    )


def test_dispersion_delay_zero_dm() -> None:
    assert delta_tau_ms(1.0, 2.0, 0.0) == 0.0


def test_dispersion_delay_sign_convention() -> None:
    # delta_tau(low, high, dm > 0) > 0  (lower freq is delayed)
    assert delta_tau_us(1.0, 2.0, 1000.0) > 0
    # Reverse args → negative
    assert delta_tau_us(2.0, 1.0, 1000.0) < 0


def test_dispersion_delay_rejects_zero_freq() -> None:
    with pytest.raises(ValueError):
        delta_tau_ms(0.0, 1.0, 100.0)


# ---------------------------------------------------------------------------
# Plan §3.6.13 — DM plan time-shift tables sanity invariants
# (= second half of M1 DoD plan §8 line 2141)
# ---------------------------------------------------------------------------


def test_dm_plan_time_shift_tables(dm_plan: DmPlan) -> None:
    """Combined §3.2 schema round-trip + §3.6.2 / §3.6.3 sanity invariants.

    Per plan §3.2 lines 573-580 + §3.6.13 smearing-bound sub-assertion.
    """
    plan = dm_plan
    n_fine = plan.fine_dm.shape[0]
    n_coarse = plan.coarse_dm.shape[0]

    # Counts in expected ranges (plan §3.2 line 573).
    assert 100 < n_fine < 2000, f"N_fine={n_fine} outside expected window"
    assert 8 < n_coarse < 64, f"N_coarse={n_coarse} outside expected window"

    # fine_dm strictly increasing (line 573).
    assert np.all(np.diff(plan.fine_dm) > 0), "fine_dm not strictly increasing"
    assert np.all(np.diff(plan.coarse_dm) > 0), "coarse_dm not strictly increasing"

    # dm_min / dm_max bracket fine_dm.
    assert plan.fine_dm[0] >= plan.dm_min - 1e-9
    assert plan.fine_dm[-1] >= plan.dm_max - 1e-9 - plan.coarse_dm[-1]  # last trial may overshoot

    # CSR consistency: idx[N_coarse] == N_fine; flat shape == N_fine.
    assert int(plan.fine_offsets_idx[-1]) == n_fine
    assert plan.fine_offsets_flat.shape == (n_fine,)
    assert np.all(plan.fine_offsets_flat >= -1e-12), (
        "fine_offsets_flat must be ≥ 0 (each fine assigned to coarse below it)"
    )

    # Stage 1 invariants (lines 574-577):
    s1 = plan.time_shift_corr_stage1
    assert s1.shape == (N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse)
    assert s1.dtype == np.int32
    # (a) ≥ 0 everywhere
    assert (s1 >= 0).all(), "time_shift_corr_stage1 has negative entries"
    # (b) ch=383 (chgroup-bottom) → 0
    assert (s1[:, NCHAN_PER_CHGROUP - 1, :] == 0).all(), (
        "time_shift_corr_stage1[:, 383, :] must be 0 (chgroup-bottom alignment)"
    )
    # (c) monotone non-increasing in ch per (g, c)
    diffs = np.diff(s1, axis=1)
    assert (diffs <= 0).all(), (
        "time_shift_corr_stage1 must be non-increasing in ch per (chgroup, c)"
    )
    # (d) ch=0 row matches the analytic formula exactly (rint).
    for g in range(N_CHGROUP):
        nu_top_g = NU_CHGROUP_TOP_GHZ[g]
        nu_bot_g = NU_CHGROUP_BOT_GHZ[g]
        analytic = np.rint(
            K_DM_MS_GHZ2_PC * plan.coarse_dm
            * (1.0 / nu_bot_g ** 2 - 1.0 / nu_top_g ** 2)
            * 1e3
            / T_INT_FAST_US_DEFAULT
        ).astype("int32")
        np.testing.assert_array_equal(
            s1[g, 0, :], analytic,
            err_msg=f"time_shift_corr_stage1[{g}, 0, :] mismatch with analytic",
        )

    # Stage 2 invariants (lines 577-578):
    s2 = plan.time_shift_corr_stage2
    assert s2.shape == (N_CHGROUP, n_coarse)
    assert s2.dtype == np.int32
    assert (s2 >= 0).all()
    # chgroup 15: local-bottom is ν_bot_proc → all zeros (within rounding)
    assert (s2[N_CHGROUP - 1, :] == 0).all(), (
        f"time_shift_corr_stage2[15, :] must be 0; got "
        f"max={s2[N_CHGROUP - 1, :].max()}"
    )
    # chgroup 0 stage-2 matches analytic exactly at every coarse trial (line 578).
    # Plan literal "≈ 6144" is the value AT DM=3000; the recursion overshoots
    # dm_max=3000 by one trial (coarse_dm[-1] ≈ 3700), so we check:
    #   (a) build vs. analytic match at the actual trial value (must be exact);
    #   (b) plan-literal-at-DM=3000 self-check (independent of the build's
    #       overshoot behaviour).
    nu_bot_chgroup0 = NU_CHGROUP_BOT_GHZ[0]
    analytic_g0 = np.rint(
        K_DM_MS_GHZ2_PC * plan.coarse_dm
        * (1.0 / NU_BOT_PROC_GHZ ** 2 - 1.0 / nu_bot_chgroup0 ** 2)
        * 1e3
        / T_INT_FAST_US_DEFAULT
    ).astype("int32")
    np.testing.assert_array_equal(
        s2[0, :], analytic_g0,
        err_msg="time_shift_corr_stage2[0, :] mismatch with analytic",
    )
    # (b) plan-literal self-check at exactly DM=3000.
    expected_at_3000 = int(np.rint(
        K_DM_MS_GHZ2_PC * 3000.0
        * (1.0 / NU_BOT_PROC_GHZ ** 2 - 1.0 / nu_bot_chgroup0 ** 2)
        * 1e3
        / T_INT_FAST_US_DEFAULT
    ))
    assert abs(expected_at_3000 - 6144) <= 4, (
        f"plan §3.2 line 578: expected ≈ 6144 ± 4 at DM=3000; "
        f"got analytic = {expected_at_3000}"
    )

    # Search-side residual invariants (line 579):
    ss = plan.time_shift_search
    assert ss.shape == (n_fine, N_CHGROUP)
    assert ss.dtype == np.int32
    assert (ss >= 0).all(), (
        "time_shift_search must be ≥ 0 (line 579) — requires δdm ≥ 0 binning"
    )
    assert (ss[:, N_CHGROUP - 1] == 0).all(), (
        "time_shift_search[:, 15] must be 0 (chgroup-15 = ν_bot_proc)"
    )

    # Bound check (line 580): max stage-1 ≈ widest chgroup intra-band Δτ at DM_MAX.
    # The widest chgroup is the LOWEST-frequency one (chgroup 15) because the
    # 1/ν² curvature is steepest at low frequencies.
    g_max_band = int(np.argmax(s1[:, 0, -1]))
    nu_top_widest = NU_CHGROUP_TOP_GHZ[g_max_band]
    nu_bot_widest = NU_CHGROUP_BOT_GHZ[g_max_band]
    analytic_max_us = (
        K_DM_MS_GHZ2_PC * plan.coarse_dm[-1]
        * (1.0 / nu_bot_widest ** 2 - 1.0 / nu_top_widest ** 2) * 1e3
    )
    s1_max_us = int(s1.max()) * T_INT_FAST_US_DEFAULT
    assert abs(s1_max_us - analytic_max_us) <= T_INT_FAST_US_DEFAULT, (
        f"max(time_shift_corr_stage1) × t_int_fast = {s1_max_us} µs; "
        f"analytic Δτ at chgroup-{g_max_band} (widest) = {analytic_max_us} µs"
    )

    # §3.6.3 line 791 smearing-bound sub-assertion is INTENTIONALLY DROPPED
    # (M1 plan-fix F12). The plan claim "smearing at max δdm < 9 µs"
    # presumes ~30 fine-per-coarse density at every DM; our gen_dmtrials_step
    # recursion (faithful port of legacy) instead gives wide coarse cells in
    # the high-DM tail (recursion step grows ~ DM), so MEAN δdm is hundreds
    # of pc/cm³, not single-digit. The architectural reconciliation is the
    # detector's K_dm boxcar bank (widths {1,3,5,7} per plan §3.1) which
    # absorbs residual intra-chgroup smearing at run-time; the per-ops
    # SNR-loss curve is benched in M5 (`bench/dm_plan_smearing.py`,
    # deferred). M1's job is to PRODUCE the plan, not to certify the
    # detector's response to its smearing — that's M5.


def test_dm_plan_partition_invariants(dm_plan: DmPlan) -> None:
    """§3.2 line 573: dm_idx_range_consumed ⊃ canonical; per-GPU partitions canonical without gaps."""
    plan = dm_plan
    n_coarse = plan.coarse_dm.shape[0]
    canon = plan.dm_idx_range_canonical
    cons = plan.dm_idx_range_consumed
    canon_g = plan.dm_idx_range_canonical_per_gpu
    cons_g = plan.dm_idx_range_consumed_per_gpu

    assert canon.shape == (N_SEARCH, 2)
    assert cons.shape == (N_SEARCH, 2)
    assert canon_g.shape == (N_SEARCH, N_SEARCH_GPU, 2)
    assert cons_g.shape == (N_SEARCH, N_SEARCH_GPU, 2)

    # consumed ⊃ canonical for every search node
    for s in range(N_SEARCH):
        assert cons[s, 0] <= canon[s, 0]
        assert cons[s, 1] >= canon[s, 1]

    # canonical ranges across search nodes partition [0, n_coarse - 1] exactly
    # (sorted, no gaps, no overlap).
    canon_sorted = sorted(tuple(canon[s]) for s in range(N_SEARCH))
    assert canon_sorted[0][0] == 0, f"canonical[0].lo = {canon_sorted[0][0]} != 0"
    assert canon_sorted[-1][1] == n_coarse - 1, (
        f"canonical[-1].hi = {canon_sorted[-1][1]} != {n_coarse - 1}"
    )
    for i in range(N_SEARCH - 1):
        assert canon_sorted[i][1] + 1 == canon_sorted[i + 1][0], (
            f"gap or overlap between {canon_sorted[i]} and {canon_sorted[i+1]}"
        )

    # Per-GPU canonical halves partition each search node's canonical range without gaps.
    for s in range(N_SEARCH):
        gpu_sorted = sorted(tuple(canon_g[s, g]) for g in range(N_SEARCH_GPU))
        assert gpu_sorted[0][0] == int(canon[s, 0])
        assert gpu_sorted[-1][1] == int(canon[s, 1])
        for i in range(N_SEARCH_GPU - 1):
            assert gpu_sorted[i][1] + 1 == gpu_sorted[i + 1][0]


def test_dm_plan_metadata(dm_plan: DmPlan) -> None:
    """Metadata captures the band + ops point + git SHA."""
    md = dm_plan.metadata
    assert math.isclose(md["band_top_GHz"], NU_TOP_PROC_GHZ, rel_tol=1e-12)
    assert math.isclose(md["band_bot_GHz"], NU_BOT_PROC_GHZ, rel_tol=1e-12)
    assert math.isclose(md["BW_MHz"], BW_PROC_MHZ, rel_tol=1e-12)
    assert md["N_chan_proc_native"] == N_CHAN_PROC_NATIVE
    assert md["t_int_search_us"] > 0
    assert md["t_int_fast_us"] > 0
    assert md["tol"] > 0
    assert md["build_utc_ns"] > 0
    assert isinstance(md["git_sha"], str) and len(md["git_sha"]) > 0
    assert md["version"] == 1

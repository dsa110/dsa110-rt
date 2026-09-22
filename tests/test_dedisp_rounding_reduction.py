"""Two roundings instead of three, and sub-bin stage 1.

Both changes target the whole-sample quantisation in the coarse+fine
dedispersion chain, measured in ``_inspect/sensitivity/``:

  * ``compute_time_shift_search(merge_coarse_rounding=True)`` collapses
    the corr-side stage-2 rounding and the search-side fine rounding
    into one. Table-only, +2.7% recovered S/N at W <= 1 ms.
  * ``apply_stage1_shifts_subbin`` shifts at ``t_int_fast / n_sub``
    instead of whole fast-vis bins. x1.098 combined, but the caller has
    to produce n_sub x finer visibilities, which is the expensive part.

Both default OFF; these tests pin the new behaviour AND that the
defaults are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The repo is installed editable from the main checkout, so a worktree's
# tests would import the main checkout's dsart and miss the change under
# test. Same pattern as tests/test_dsart_pipeline_rt_yaml.py.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dsart.common.constants import (
    N_CHGROUP,
    NATIVE_SAMPLE_US,
    NBASE,
)
from dsart.fine_dm.combiner import compute_time_shift_search

T = 1048.576


def _grid():
    coarse = np.array([200.0, 400.0, 600.0])
    fine = np.concatenate([
        c + np.linspace(-40.0, 40.0, 9) for c in coarse
    ])
    f2c = np.repeat(np.arange(3), 9)
    return coarse, fine, f2c


# ---------------------------------------------------------------------------
# merge_coarse_rounding
# ---------------------------------------------------------------------------


def test_merge_is_off_by_default():
    """The shipped path must be untouched."""
    coarse, fine, f2c = _grid()
    a = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
    )
    b = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
        merge_coarse_rounding=False, t_int_corr_us=T,
    )
    assert np.array_equal(a.shifts, b.shifts)


def test_merged_total_is_a_single_rounding_at_the_fine_dm():
    """corr-applied integer + our shift == rint(full fine delay / T).

    This is the whole point: the TOTAL per-chgroup shift must be one
    rounding of the fine-DM delay, not the sum of two roundings.
    """
    from dsart.common.constants import NU_BOT_PROC_GHZ, NU_CHGROUP_TOP_GHZ
    from dsart.common.dispersion import delta_tau_us

    coarse, fine, f2c = _grid()
    merged = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
        merge_coarse_rounding=True, t_int_corr_us=T,
    )
    top = np.asarray(NU_CHGROUP_TOP_GHZ, dtype=np.float64)
    for f in range(len(fine)):
        c = int(f2c[f])
        for g in range(N_CHGROUP):
            # what the corr side applied (stage2_shifts.py formula)
            s2 = np.rint(
                delta_tau_us(float(NU_BOT_PROC_GHZ), float(top[g]),
                             float(coarse[c])) / T
            )
            want = np.rint(
                delta_tau_us(float(NU_BOT_PROC_GHZ), float(top[g]),
                             float(fine[f])) / T
            )
            assert s2 + merged.shifts[f, g] == want


def test_merged_shifts_move_by_at_most_one_sample():
    """|rint(a) - rint(b) - (a-b)| <= 1, so the ring history barely moves.

    ProductionRxRing sizes its stream from max|shifts|; a change larger
    than one sample would mean re-sizing the history window.
    """
    coarse, fine, f2c = _grid()
    old = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
    )
    new = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
        merge_coarse_rounding=True, t_int_corr_us=T,
    )
    d = np.abs(new.shifts.astype(np.int64) - old.shifts.astype(np.int64))
    assert d.max() <= 1


def test_merge_rejects_include_coarse_offset():
    """That path already has one rounding; merging is meaningless."""
    coarse, fine, f2c = _grid()
    with pytest.raises(ValueError, match="meaningless"):
        compute_time_shift_search(
            coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
            fine_to_coarse=f2c, t_int_search_us=T,
            include_coarse_offset=True, merge_coarse_rounding=True,
            t_int_corr_us=T,
        )


def test_merge_requires_the_corr_cadence():
    coarse, fine, f2c = _grid()
    with pytest.raises(ValueError, match="t_int_corr_us"):
        compute_time_shift_search(
            coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
            fine_to_coarse=f2c, t_int_search_us=T,
            merge_coarse_rounding=True,
        )


def test_merge_rejects_mismatched_cadence():
    """A corr sample that is not a search sample makes the subtraction
    meaningless, so it must fail loudly rather than mis-shift."""
    coarse, fine, f2c = _grid()
    with pytest.raises(ValueError, match="t_int_search_us"):
        compute_time_shift_search(
            coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
            fine_to_coarse=f2c, t_int_search_us=T,
            merge_coarse_rounding=True, t_int_corr_us=T / 4.0,
        )


def test_merged_residual_error_is_smaller_on_average():
    """One rounding beats two on the MEAN absolute arrival error.

    This is the mechanism behind the measured +2.7%. Note the worst
    case can be marginally worse -- two roundings sometimes cancel --
    so this asserts the mean, which is what the S/N gain follows.
    """
    from dsart.common.constants import NU_BOT_PROC_GHZ, NU_CHGROUP_TOP_GHZ
    from dsart.common.dispersion import delta_tau_us

    coarse, fine, f2c = _grid()
    top = np.asarray(NU_CHGROUP_TOP_GHZ, dtype=np.float64)
    old = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
    )
    new = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c, t_int_search_us=T,
        merge_coarse_rounding=True, t_int_corr_us=T,
    )
    e_old, e_new = [], []
    for f in range(len(fine)):
        c = int(f2c[f])
        for g in range(N_CHGROUP):
            exact = delta_tau_us(float(NU_BOT_PROC_GHZ), float(top[g]),
                                 float(fine[f])) / T
            s2 = np.rint(
                delta_tau_us(float(NU_BOT_PROC_GHZ), float(top[g]),
                             float(coarse[c])) / T
            )
            e_old.append(abs(s2 + old.shifts[f, g] - exact))
            e_new.append(abs(s2 + new.shifts[f, g] - exact))
    assert np.mean(e_new) < np.mean(e_old)
    # one rounding can never exceed half a sample
    assert max(e_new) <= 0.5 + 1e-9


# ---------------------------------------------------------------------------
# sub-bin stage 1
# ---------------------------------------------------------------------------

def _torch():
    """Import torch or skip -- only the sub-bin tests need it, and
    the h23 dsart env has no torch."""
    return pytest.importorskip("torch")


def _plan(n_coarse=3, t_int_fast_native=32, chan_sum_factor=8):
    """A real DMPlan with a NON-zero delay table.

    build_synthetic_summed_plan zeroes every shift, which cannot
    exercise sub-bin resolution. chan_sum_factor defaults to 8 to match
    production (48 summed channels per chgroup, per the deployed plan's
    fine_chan_sum_factor).
    """
    from dsart.coarse_dm.dm_plan import (
        DMPlan, compute_delay_native_samples_table,
    )
    from dsart.common.constants import NCHAN_PER_CHGROUP, NATIVE_SAMPLE_US

    nch = NCHAN_PER_CHGROUP // chan_sum_factor
    dm = np.linspace(200.0, 600.0, n_coarse).astype(np.float64)
    # summed-channel band CENTRES, top-down within each chgroup
    freqs = np.stack([
        1.53 - (1024 + 384 * g
                + (np.arange(nch) + 0.5) * chan_sum_factor)
        * 250.0 / 8192.0 / 1e3
        for g in range(N_CHGROUP)
    ])
    return DMPlan(
        dm_pc_cc=dm,
        n_fine_per_coarse=4,
        t_int_fast_us=float(t_int_fast_native) * NATIVE_SAMPLE_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(
            dm, freqs),
        chan_sum_factor=chan_sum_factor,
    )


def test_subbin_table_is_finer_than_whole_bin():
    _torch()   # dm_plan imports dedisp, which needs torch
    p = _plan()
    whole = p.delay_bins_per_chgroup(3)
    native = p.delay_subbins_per_chgroup(3, p.t_int_fast_native)
    # the native-resolution table is exactly the stored delay table
    assert np.array_equal(
        native, p.delay_native_samples_per_chgroup(3)
    )
    # and the whole-bin table is that, divided down and rounded again
    assert np.array_equal(
        whole, np.rint(native / p.t_int_fast_native).astype(np.int64)
    )


@pytest.mark.parametrize("n_sub", [0, 3, 64])
def test_subbin_table_rejects_bad_n_sub(n_sub):
    _torch()
    p = _plan()
    with pytest.raises(ValueError):
        p.delay_subbins_per_chgroup(0, n_sub)


def test_subbin_n_sub_1_matches_wholebin():
    """n_sub=1 must be bit-identical to the shipped primitive."""
    torch = _torch()
    from dsart.coarse_dm.stage1 import (
        apply_stage1_shifts, apply_stage1_shifts_subbin,
    )
    p = _plan()
    nch, n_fv = 6, 64
    rng = np.random.default_rng(0)
    vis = torch.as_tensor(
        rng.standard_normal((n_fv, NBASE, nch))
        + 1j * rng.standard_normal((n_fv, NBASE, nch)),
        dtype=torch.complex64,
    )
    t_dedisp = 8
    a = apply_stage1_shifts(vis, p, chgroup=0, dm_idx=0, t_dedisp=t_dedisp)
    b = apply_stage1_shifts_subbin(
        vis, p, chgroup=0, dm_idx=0, n_sub=1, t_dedisp=t_dedisp,
    )
    assert torch.equal(a, b)


def test_subbin_sums_n_sub_samples_per_output():
    """With a zero-DM trial the sub-bin path is a plain n_sub:1 time sum."""
    torch = _torch()
    from dsart.coarse_dm.stage1 import apply_stage1_shifts_subbin
    from dsart.coarse_dm.dm_plan import (
        DMPlan, compute_delay_native_samples_table,
    )
    from dsart.common.constants import NATIVE_SAMPLE_US

    nch_plan = 48
    dm = np.array([0.0], dtype=np.float64)
    freqs = np.stack([
        1.53 - (1024 + 384 * g + (np.arange(nch_plan) + 0.5) * 8)
        * 250.0 / 8192.0 / 1e3
        for g in range(N_CHGROUP)
    ])
    p = DMPlan(
        dm_pc_cc=dm, n_fine_per_coarse=4,
        t_int_fast_us=32.0 * NATIVE_SAMPLE_US,
        chgroup_freqs_GHz=freqs,
        _delay_native_samples_table=compute_delay_native_samples_table(
            dm, freqs),
        chan_sum_factor=8,
    )
    nch, n_sub, t_dedisp = 4, 4, 5
    vis = torch.ones((t_dedisp * n_sub, NBASE, nch), dtype=torch.complex64)
    out = apply_stage1_shifts_subbin(
        vis, p, chgroup=0, dm_idx=0, n_sub=n_sub, t_dedisp=t_dedisp,
    )
    assert out.shape == (t_dedisp, NBASE, nch)
    assert torch.allclose(out.real, torch.full_like(out.real, float(n_sub)))


def test_subbin_rejects_time_axis_not_multiple_of_n_sub():
    torch = _torch()
    from dsart.coarse_dm.stage1 import apply_stage1_shifts_subbin
    p = _plan()
    vis = torch.zeros((63, NBASE, 4), dtype=torch.complex64)
    with pytest.raises(ValueError, match="multiple of"):
        apply_stage1_shifts_subbin(
            vis, p, chgroup=0, dm_idx=0, n_sub=4,
        )


def test_subbin_peak_memory_is_one_output_not_n_sub():
    """The implementation must accumulate, not materialise n_sub gathers.

    A (t_dedisp*n_sub, NBASE, NCHAN) intermediate is ~900 MB * n_sub at
    the production op-point and OOMs an 11 GB card. Guard the shape
    contract that makes the accumulate form observable.
    """
    torch = _torch()
    from dsart.coarse_dm.stage1 import apply_stage1_shifts_subbin
    p = _plan()
    nch, n_sub, t_dedisp = 4, 8, 4
    vis = torch.zeros((t_dedisp * n_sub + 64, NBASE, nch),
                      dtype=torch.complex64)
    out = torch.empty((t_dedisp, NBASE, nch), dtype=torch.complex64)
    got = apply_stage1_shifts_subbin(
        vis, p, chgroup=0, dm_idx=0, n_sub=n_sub, t_dedisp=t_dedisp,
        out=out,
    )
    assert got.data_ptr() == out.data_ptr()

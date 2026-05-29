"""Tests for ``dsart.fine_dm.combiner`` (M5 chunk 6a).

Covers:
  * ``compute_time_shift_search`` against the §3.6.3 closed-form
    expression — chgroup-15 row identically zero, sign convention
    non-negative for δdm ≥ 0, monotone in δdm, exact for the
    coarse-DM-only case (δdm = 0 ⇒ all rows zero).
  * ``TimeShiftSearchTable`` validators reject malformed input.
  * ``sparse_to_dense_grid`` handles empty / duplicate / out-of-range
    indices and matches a numpy reference scatter.
  * ``combine_chgroups`` reproduces the closed-form sum across
    chgroups, applies the integer-sample shifts in the §3.6.3 sign
    direction, zero-fills out-of-range reads, and obeys the cube-time
    window bounds.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.constants import (  # noqa: E402
    K_DM_MS_GHZ2_PC,
    NCHAN_PER_CHGROUP,
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_BOT_GHZ,
    NU_TOP_PROC_GHZ,
    N_CHGROUP,
    T_INT_SEARCH_US_DEFAULT,
    freq_GHz,
)
from dsart.common.dispersion import delta_tau_us  # noqa: E402
from dsart.fine_dm.combiner import (  # noqa: E402
    TimeShiftSearchTable,
    combine_chgroups,
    compute_time_shift_search,
    sparse_to_dense_grid,
)


# ---------------------------------------------------------------------------
# compute_time_shift_search
# ---------------------------------------------------------------------------


def _coarse_fine_grid(n_coarse: int = 4, n_fine_per_coarse: int = 8):
    coarse = np.linspace(50.0, 200.0, n_coarse, dtype=np.float64)
    spacing = (coarse[1] - coarse[0]) / n_fine_per_coarse
    fine = np.concatenate(
        [coarse[c] + np.arange(n_fine_per_coarse) * spacing for c in range(n_coarse)]
    )
    fine_to_coarse = np.repeat(np.arange(n_coarse, dtype=np.int64), n_fine_per_coarse)
    return coarse, fine, fine_to_coarse


def test_compute_time_shift_search_chgroup15_row_is_zero() -> None:
    """Sign convention: chgroup-15 row is identically zero (it IS ν_bot_proc)."""
    coarse, fine, f2c = _coarse_fine_grid()
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
    )
    assert table.shifts.dtype == np.int32
    assert table.shifts.shape == (len(fine), N_CHGROUP)
    assert np.all(table.shifts[:, N_CHGROUP - 1] == 0)


def test_include_coarse_offset_uses_chgroup_top_reference_zero_dm_offset() -> None:
    """DM-offset fix (2026-05-29): the include_coarse_offset escape hatch
    must reference each chgroup's TOP channel (matching the Convention-A
    corr-side stage-1 output), NOT its bottom.

    Regression for the constant -2.45% detected-vs-injected DM bias: a
    delay-arithmetic forward model that dedisperses a signal at DM_inj
    using corr's actual per-chgroup TOP reference (chan-summed top-channel
    band-center) + the escape-hatch shift table must recover DM_inj to
    well within one fine-DM step. Also pins that chgroup-15 is no longer
    zero (its TOP freq sits above ν_bot_proc) and the table flags
    ``coarse_offset_baked``.
    """
    # corr Convention-A per-chgroup reference under chan_sum_factor=8:
    # band-center of the top 8 fine channels in each chgroup.
    fine_freqs = np.array(
        [[freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)] for g in range(N_CHGROUP)]
    )
    nu_sum_top = fine_freqs[:, 0:8].mean(axis=1)  # (16,)
    t_int_ms = T_INT_SEARCH_US_DEFAULT / 1000.0

    def arrival_samples(nu, dm):
        return K_DM_MS_GHZ2_PC * dm * (1.0 / nu**2 - 1.0 / NU_TOP_PROC_GHZ**2) / t_int_ms

    for dm_inj in (258.0, 576.0, 1300.0, 3000.0):
        grid = np.linspace(dm_inj * 0.9, dm_inj * 1.1, 4001)
        table = compute_time_shift_search(
            coarse_dm_pc_cm3=np.array([dm_inj]),
            fine_dm_pc_cm3=grid,
            fine_to_coarse=np.zeros(len(grid), dtype=np.int64),
            include_coarse_offset=True,
        )
        assert table.coarse_offset_baked is True
        # chgroup-15 carries a real (nonzero) alignment shift now.
        assert table.shifts[len(grid) // 2, N_CHGROUP - 1] != 0
        # Forward model: corr emits chgroup g peak at arrival_samples(top[g]);
        # escape-hatch shift advances it. Coherent sum peaks at the fine-DM
        # that minimises the cross-chgroup time variance.
        tpeak = arrival_samples(nu_sum_top, dm_inj)  # (16,)
        var = (tpeak[None, :] + table.shifts.astype(float)).var(axis=1)
        dm_det = float(grid[int(np.argmin(var))])
        # recovered within 0.4% (integer-sample rounding dominates at low
        # DM; vs the -2.45% bottom-reference bias this is a >6x improvement
        # and far below the 5% injection-match tolerance / fine-DM step).
        assert abs(dm_det - dm_inj) / dm_inj < 0.004, (
            f"DM={dm_inj}: detected {dm_det:.2f} "
            f"({100 * (dm_det - dm_inj) / dm_inj:+.3f}%)"
        )


def test_compute_time_shift_search_zero_at_dm_zero_offset() -> None:
    """δdm = 0 ⇒ every row identically zero."""
    coarse = np.array([100.0, 200.0], dtype=np.float64)
    fine = coarse.copy()
    f2c = np.array([0, 1], dtype=np.int64)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
    )
    assert np.all(table.shifts == 0)


def test_compute_time_shift_search_matches_closed_form() -> None:
    """Element-wise match against ``rint(Δτ_us / t_int)`` for a few cells."""
    coarse, fine, f2c = _coarse_fine_grid()
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
    )
    # Check a handful of cells explicitly. delta_tau_us takes
    # (nu_low, nu_high, dm); for g < 15 chgroup_bot[g] > nu_bot_proc,
    # so nu_low IS nu_bot_proc and shift = +rint(Δτ / t_int).
    for f_idx in (0, 5, 17, len(fine) - 1):
        c_idx = int(f2c[f_idx])
        ddm = float(fine[f_idx] - coarse[c_idx])
        for g in range(N_CHGROUP - 1):
            d_us = delta_tau_us(
                float(NU_BOT_PROC_GHZ),
                float(NU_CHGROUP_BOT_GHZ[g]),
                ddm,
            )
            expected = int(np.rint(d_us / T_INT_SEARCH_US_DEFAULT))
            assert table.shifts[f_idx, g] == expected, (
                f"f={f_idx} g={g}: got {table.shifts[f_idx, g]}, expected {expected}"
            )


def test_compute_time_shift_search_monotone_in_chgroup() -> None:
    """For δdm > 0 the per-chgroup shift increases with ν_bot_proc separation:
    chgroup 0 (top of band) requires the largest forward-shift; chgroup 15
    is zero by construction. Equivalently shifts decrease with chgroup index.
    """
    coarse = np.array([100.0], dtype=np.float64)
    fine = np.array([110.0], dtype=np.float64)
    f2c = np.array([0], dtype=np.int64)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
    )
    diffs = np.diff(table.shifts[0, :])
    assert np.all(diffs <= 0), f"shifts not monotone-decreasing: {table.shifts[0]}"


def test_compute_time_shift_search_rejects_bad_inputs() -> None:
    coarse = np.array([100.0, 200.0], dtype=np.float64)
    fine = np.array([110.0, 210.0], dtype=np.float64)
    f2c = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError):
        compute_time_shift_search(
            coarse_dm_pc_cm3=coarse,
            fine_dm_pc_cm3=fine.reshape(1, -1),
            fine_to_coarse=f2c,
        )
    with pytest.raises(ValueError):
        compute_time_shift_search(
            coarse_dm_pc_cm3=coarse,
            fine_dm_pc_cm3=fine,
            fine_to_coarse=np.array([0, 5], dtype=np.int64),
        )


def test_time_shift_search_table_accepts_signed_shifts() -> None:
    """v2 (2026-05-18): shifts are SIGNED. The TimeShiftSearchTable must
    accept both positive (read FUTURE) and negative (read PAST) shifts;
    only the chgroup-15 reference column is constrained to zero.

    Pre-v2 this test was ``rejects_negative``; under the v2 even-K
    partition K/2 fines per coarse sit BELOW the coarse (δdm < 0) and
    K/2 sit ABOVE (δdm > 0), so the table must round-trip both signs.
    """
    signed = np.zeros((4, N_CHGROUP), dtype=np.int32)
    signed[0, 3] = -7        # fine below coarse
    signed[1, 7] = +11       # fine above coarse
    signed[2, 14] = -1       # near chgroup-15 boundary
    # chgroup 15 column must stay 0 (still enforced under v2)
    table = TimeShiftSearchTable(
        shifts=signed,
        fine_to_coarse=np.zeros(4, dtype=np.int64),
        t_int_search_us=T_INT_SEARCH_US_DEFAULT,
    )
    assert int(table.shifts.min()) == -7
    assert int(table.shifts.max()) == +11


def test_time_shift_search_table_rejects_chgroup15_nonzero() -> None:
    bad = np.zeros((4, N_CHGROUP), dtype=np.int32)
    bad[2, N_CHGROUP - 1] = 1
    with pytest.raises(ValueError, match="chgroup 15"):
        TimeShiftSearchTable(
            shifts=bad,
            fine_to_coarse=np.zeros(4, dtype=np.int64),
            t_int_search_us=T_INT_SEARCH_US_DEFAULT,
        )


def test_time_shift_search_table_rejects_wrong_dtype() -> None:
    bad = np.zeros((4, N_CHGROUP), dtype=np.float32)
    with pytest.raises(TypeError, match="dtype"):
        TimeShiftSearchTable(
            shifts=bad,
            fine_to_coarse=np.zeros(4, dtype=np.int64),
            t_int_search_us=T_INT_SEARCH_US_DEFAULT,
        )


# ---------------------------------------------------------------------------
# sparse_to_dense_grid
# ---------------------------------------------------------------------------


def test_sparse_to_dense_grid_empty() -> None:
    grid = sparse_to_dense_grid(
        linear_indices=np.zeros(0, dtype=np.int64),
        values=np.zeros(0, dtype=np.complex64),
        n_grid=8,
    )
    assert grid.shape == (8, 8)
    assert grid.dtype == np.complex64
    assert np.allclose(grid, 0.0)


def test_sparse_to_dense_grid_simple_scatter() -> None:
    n_grid = 4
    indices = np.array([0, 5, 15], dtype=np.int64)
    values = np.array([1 + 1j, 2 - 1j, -3.0 + 0j], dtype=np.complex64)
    grid = sparse_to_dense_grid(
        linear_indices=indices,
        values=values,
        n_grid=n_grid,
    )
    flat = grid.reshape(-1)
    for ix, val in zip(indices, values):
        assert flat[ix] == val


def test_sparse_to_dense_grid_duplicates_sum() -> None:
    """``np.add.at`` semantics: duplicate indices accumulate."""
    n_grid = 2
    indices = np.array([0, 0, 3], dtype=np.int64)
    values = np.array([1, 2, 4], dtype=np.complex64)
    grid = sparse_to_dense_grid(
        linear_indices=indices,
        values=values,
        n_grid=n_grid,
    )
    assert grid.flat[0] == 3
    assert grid.flat[3] == 4


def test_sparse_to_dense_grid_rejects_oob() -> None:
    with pytest.raises(ValueError, match="n_grid"):
        sparse_to_dense_grid(
            linear_indices=np.array([16], dtype=np.int64),
            values=np.array([1], dtype=np.complex64),
            n_grid=4,
        )


# ---------------------------------------------------------------------------
# combine_chgroups
# ---------------------------------------------------------------------------


def _flat_streams(
    *,
    n_chgroup: int = N_CHGROUP,
    t_stream: int = 32,
    n_grid: int = 8,
    rng_seed: int = 0,
):
    rng = np.random.default_rng(rng_seed)
    streams = {}
    for g in range(n_chgroup):
        re = rng.standard_normal((t_stream, n_grid, n_grid)).astype(np.float32)
        im = rng.standard_normal((t_stream, n_grid, n_grid)).astype(np.float32)
        streams[g] = (re + 1j * im).astype(np.complex64)
    return streams


def test_combine_chgroups_zero_shift_is_pure_sum() -> None:
    """When all shifts are zero, the combiner is a per-cell sum across chgroups."""
    n_grid = 4
    t_stream = 16
    streams = _flat_streams(t_stream=t_stream, n_grid=n_grid)
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    out = combine_chgroups(
        per_chgroup_streams=streams,
        time_shift_per_chgroup=shifts,
        t_window=(0, t_stream),
        n_grid=n_grid,
    )
    expected = np.sum(np.stack([streams[g] for g in range(N_CHGROUP)], axis=0), axis=0)
    assert out.shape == (t_stream, n_grid, n_grid)
    assert out.dtype == np.complex64
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_combine_chgroups_applies_integer_shift_sign_convention() -> None:
    """Per §3.6.3: at cube-time t, chgroup g is read at stream[g][t - shift[g]];
    so a shift of +k delays chgroup-g's contribution by k samples in cube-time.
    """
    n_grid = 2
    t_stream = 16
    rng = np.random.default_rng(123)
    streams = {}
    for g in range(N_CHGROUP):
        streams[g] = np.zeros((t_stream, n_grid, n_grid), dtype=np.complex64)
    # Inject a single-sample δ at stream-time 4 in chgroup 0 only.
    streams[0][4, 0, 0] = 7.0 + 0.0j
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    shifts[0] = 3  # cube-time read at t=7 should pick up stream[0][4]
    out = combine_chgroups(
        per_chgroup_streams=streams,
        time_shift_per_chgroup=shifts,
        t_window=(0, t_stream),
        n_grid=n_grid,
    )
    # Cube-time index where the spike lands:
    expected_t = 4 + shifts[0]
    assert out[expected_t, 0, 0] == 7.0 + 0.0j
    # Verify other cube-times are zero at (0, 0) for chgroup-0's contribution
    # (other chgroups all-zero, so a strict equality test holds):
    mask = np.ones(t_stream, dtype=bool)
    mask[expected_t] = False
    assert np.all(out[mask, 0, 0] == 0.0)


def test_combine_chgroups_zero_fills_out_of_range() -> None:
    """When (t - shift[g]) is out of [0, T_stream), the contribution is zero."""
    n_grid = 2
    t_stream = 4
    streams = {0: np.full((t_stream, n_grid, n_grid), 1.0 + 0j, dtype=np.complex64)}
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    shifts[0] = 10  # t - 10 < 0 for all t in [0, 4)
    out = combine_chgroups(
        per_chgroup_streams=streams,
        time_shift_per_chgroup=shifts,
        t_window=(0, t_stream),
        n_grid=n_grid,
    )
    assert np.all(out == 0.0)


def test_combine_chgroups_t_window_bounds_partial() -> None:
    """If t_window is wider than what chgroup g supplies, only the valid
    sub-range is filled and the remainder stays at 0."""
    n_grid = 2
    t_stream = 8
    streams = {0: np.full((t_stream, n_grid, n_grid), 1.0 + 0j, dtype=np.complex64)}
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    # cube-time range [0, 12) but stream is only 8 long, no shift:
    out = combine_chgroups(
        per_chgroup_streams=streams,
        time_shift_per_chgroup=shifts,
        t_window=(0, 12),
        n_grid=n_grid,
    )
    assert out.shape == (12, n_grid, n_grid)
    assert np.all(out[:t_stream] == 1.0 + 0j)
    assert np.all(out[t_stream:] == 0.0 + 0j)


def test_combine_chgroups_missing_chgroup_zero_contribution() -> None:
    """A chgroup absent from the stream dict ⇒ 0 contribution at every cube-time."""
    n_grid = 2
    t_stream = 4
    streams = {7: np.full((t_stream, n_grid, n_grid), 2.0 + 0j, dtype=np.complex64)}
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    out = combine_chgroups(
        per_chgroup_streams=streams,
        time_shift_per_chgroup=shifts,
        t_window=(0, t_stream),
        n_grid=n_grid,
    )
    np.testing.assert_array_equal(out, streams[7])


def test_combine_chgroups_rejects_wrong_shift_shape() -> None:
    streams = {0: np.zeros((4, 2, 2), dtype=np.complex64)}
    bad = np.zeros(N_CHGROUP - 1, dtype=np.int32)
    with pytest.raises(ValueError, match="N_CHGROUP|shape"):
        combine_chgroups(
            per_chgroup_streams=streams,
            time_shift_per_chgroup=bad,
            t_window=(0, 4),
            n_grid=2,
        )


def test_combine_chgroups_rejects_non_integer_shifts() -> None:
    streams = {0: np.zeros((4, 2, 2), dtype=np.complex64)}
    bad = np.zeros(N_CHGROUP, dtype=np.float32)
    with pytest.raises(TypeError, match="dtype"):
        combine_chgroups(
            per_chgroup_streams=streams,
            time_shift_per_chgroup=bad,
            t_window=(0, 4),
            n_grid=2,
        )


def test_combine_chgroups_rejects_empty_window() -> None:
    streams = {0: np.zeros((4, 2, 2), dtype=np.complex64)}
    shifts = np.zeros(N_CHGROUP, dtype=np.int32)
    with pytest.raises(ValueError, match="non-positive"):
        combine_chgroups(
            per_chgroup_streams=streams,
            time_shift_per_chgroup=shifts,
            t_window=(5, 5),
            n_grid=2,
        )

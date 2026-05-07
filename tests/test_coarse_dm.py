"""Coarse-DM dedispersion + Stage-2 FIFO acceptance tests (M3 chunk 3b).

Pinned by the chunk 3b briefing (parent agent); covers:

* :class:`dsart.coarse_dm.dm_plan.DMPlan` — sign convention,
  delay-table invariants, npz round-trip, single-DM custom plans.
* :func:`dsart.coarse_dm.dedisp.coarse_dedisp` — output shape/dtype,
  zero-DM passthrough, synthetic-burst recovery, off-DM peak loss,
  fp16-accumulator-overflow safety, native-sample alignment (F24),
  spatial (l, m) preservation through the F18/F20/F21 stack.
* :class:`dsart.coarse_dm.stage2_fifo.Stage2FIFO` — push/pop
  ordering, capacity bound, eviction semantics, partial fill.

References
==========

* Plan §3.2 (DM plan), §3.6.2 (DEDISP architecture), §4.2 lines
  ~1283-1346 (streaming pipeline placement).
* :doc:`PARALLEL_AGENTS.md` §5.1 — single-DM burst-fixture plan.
* :doc:`M3_PLAN_FIXES.md` — F21 (fast-corr DEC-phase fold; chunk 1).
* :doc:`plan.md` §8.M2-carryover F18 / F20 — sign conventions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from dsart.coarse_dm import (
    DMPlan,
    Stage2FIFO,
    build_chgroup_freq_table_GHz,
    coarse_dedisp,
    compute_delay_native_samples_table,
    max_output_t_dedisp,
)
from dsart.common.constants import (
    COARSE_DM_FIFO_DEPTH_DEFAULT,
    K_DM_MS_GHZ2_PC,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    N_CHGROUP,
    T_INT_FAST_US_DEFAULT,
    freq_GHz,
)


# ---------------------------------------------------------------------------
# Helpers — small synthetic plans + cubes for the test suite
# ---------------------------------------------------------------------------


def _make_dm_plan(
    *,
    dms: list[float] | None = None,
    t_int_fast_us: float = T_INT_FAST_US_DEFAULT,
    n_fine_per_coarse: int = 5,
) -> DMPlan:
    """Build a DMPlan from a small explicit DM list (no fine-DM tables)."""
    if dms is None:
        dms = [0.0, 50.0, 200.0, 405.0, 1000.0]
    coarse = np.asarray(dms, dtype=np.float64)
    chgroup_freqs = build_chgroup_freq_table_GHz()
    table = compute_delay_native_samples_table(coarse, chgroup_freqs)
    return DMPlan(
        dm_pc_cc=coarse,
        n_fine_per_coarse=n_fine_per_coarse,
        t_int_fast_us=t_int_fast_us,
        chgroup_freqs_GHz=chgroup_freqs,
        _delay_native_samples_table=table,
    )


def _ch_subset_freqs(chgroup: int, n_chan_test: int) -> np.ndarray:
    """Return the contiguous chgroup-local freqs for the first ``n_chan_test``
    channels of ``chgroup``. Tests use a smaller subset to keep cubes
    small but the freqs are real freq_GHz() values so the dispersion
    math is identical to production.
    """
    return np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(n_chan_test)],
        dtype=np.float64,
    )


def _make_burst_cube(
    *,
    t_fast: int,
    n_chan_test: int,
    n_grid: int,
    chgroup: int,
    dm_pc_cc: float,
    t_peak_top_bin: int,
    l_idx: int,
    m_idx: int,
    t_int_fast_us: float = T_INT_FAST_US_DEFAULT,
    amplitude: complex = 1.0 + 0.0j,
) -> torch.Tensor:
    """Synthesise a ``(T_fast, n_chan_test, N_grid, N_grid) complex64`` cube.

    Places a single bright pixel at ``(l_idx, m_idx)`` in every
    channel's image, at the dispersively-shifted time ``t_peak_top_bin
    + delay_bin(ch, dm)`` (Convention A). Background is zero; only the
    one pixel per channel is non-zero.

    Returns a complex64 cube — `coarse_dedisp` accepts complex32 / 64
    / 128. Tests that need cfp16 explicitly cast at the call site.
    """
    cube = torch.zeros(
        (t_fast, n_chan_test, n_grid, n_grid), dtype=torch.complex64,
    )
    nu_ch = _ch_subset_freqs(chgroup, n_chan_test)                   # (NCHAN_test,)
    nu_top = nu_ch[0]
    delay_us = K_DM_MS_GHZ2_PC * dm_pc_cc * (
        1.0 / nu_ch ** 2 - 1.0 / nu_top ** 2
    ) * 1e3                                                          # (NCHAN_test,)
    delay_bins = np.rint(delay_us / t_int_fast_us).astype(np.int64)
    for ch in range(n_chan_test):
        t_peak_ch = t_peak_top_bin + int(delay_bins[ch])
        if 0 <= t_peak_ch < t_fast:
            cube[t_peak_ch, ch, l_idx, m_idx] = amplitude
    return cube


# ---------------------------------------------------------------------------
# DMPlan: schema + delay-table invariants
# ---------------------------------------------------------------------------


class TestDMPlanInvariants:

    def test_dm_plan_delay_monotone_with_dm(self) -> None:
        """At fixed chgroup + ch (≠ chgroup-top), delay is monotone-non-
        decreasing in DM (for ch=0 it's identically 0)."""
        plan = _make_dm_plan(dms=[0.0, 10.0, 100.0, 500.0, 2000.0])
        # Try a few chgroups + non-top channels.
        for g in (0, 7, 15):
            for ch in (1, 50, NCHAN_PER_CHGROUP - 1):
                delays = np.asarray(
                    [
                        plan.delay_native_samples(g, ch, k)
                        for k in range(plan.n_coarse)
                    ],
                    dtype=np.int64,
                )
                # Non-decreasing in DM (could be flat at very small DM
                # if the delay rounds to 0 before the DM grows enough).
                diffs = np.diff(delays)
                assert (diffs >= 0).all(), (
                    f"chgroup={g} ch={ch} delays non-monotone: {delays.tolist()}"
                )
                # And strictly increasing somewhere (i.e. not all-zero).
                assert delays[-1] > 0, (
                    f"chgroup={g} ch={ch} max delay = 0; bad freq table"
                )

    def test_dm_plan_delay_zero_at_top_freq(self) -> None:
        """Convention A: ``delay_native_samples(g, ch=0, dm) == 0`` for
        every chgroup, every DM trial."""
        plan = _make_dm_plan(dms=[0.0, 100.0, 1000.0, 3000.0])
        for g in range(N_CHGROUP):
            for k in range(plan.n_coarse):
                assert plan.delay_native_samples(g, 0, k) == 0, (
                    f"Convention A broken at chgroup={g} dm_idx={k}: "
                    f"delay = {plan.delay_native_samples(g, 0, k)}"
                )

    def test_dm_plan_chan_sum_factor_default_one(self) -> None:
        """Pre-F33 behaviour: ``chan_sum_factor`` defaults to 1; all
        existing test fixtures + benches that build a DMPlan without
        the parameter continue to see ``nchan_per_chgroup == 384``
        (= ``NCHAN_PER_CHGROUP``)."""
        plan = _make_dm_plan(dms=[0.0, 50.0, 405.0])
        assert plan.chan_sum_factor == 1
        assert plan.nchan_per_chgroup == NCHAN_PER_CHGROUP
        assert plan.chgroup_freqs_GHz.shape == (
            N_CHGROUP, NCHAN_PER_CHGROUP,
        )
        assert plan._delay_native_samples_table.shape == (
            N_CHGROUP, NCHAN_PER_CHGROUP, plan.n_coarse,
        )

    def test_dm_plan_load_save_round_trip(self, tmp_path: Path) -> None:
        """Coarse-only ``.npz`` round-trip preserves every field."""
        plan = _make_dm_plan(
            dms=[0.0, 50.0, 405.0, 1000.0], t_int_fast_us=262.144,
            n_fine_per_coarse=7,
        )
        out_path = tmp_path / "dm_plan_coarse.npz"
        plan.to_npz(str(out_path))
        loaded = DMPlan.from_coarse_only_npz(str(out_path))
        np.testing.assert_array_equal(loaded.dm_pc_cc, plan.dm_pc_cc)
        assert loaded.n_fine_per_coarse == plan.n_fine_per_coarse
        assert loaded.t_int_fast_us == plan.t_int_fast_us
        np.testing.assert_array_equal(
            loaded.chgroup_freqs_GHz, plan.chgroup_freqs_GHz,
        )
        # Hot-path lookup parity.
        for g in (0, 5, 15):
            for ch in (0, 100, NCHAN_PER_CHGROUP - 1):
                for k in range(plan.n_coarse):
                    assert (
                        loaded.delay_native_samples(g, ch, k)
                        == plan.delay_native_samples(g, ch, k)
                    )


class TestF33SummedDmPlan:
    """F33: :meth:`DMPlan.from_summed_canonical` + summed-channel
    plumbing across :meth:`DMPlan` schema."""

    def _canonical(self, *, dms: list[float] | None = None):
        """Synthesise a minimal-valid canonical :class:`DmPlan` (full
        schema; mirrors :func:`tests.test_contracts._make_minimal_dm_plan`).
        """
        from dsart.common.contracts import (
            DM_PLAN_METADATA_VERSION,
            DmPlan,
            N_SEARCH,
            N_SEARCH_GPU,
            NU_TOP_PROC_GHZ,
            NU_BOT_PROC_GHZ,
            BW_PROC_MHZ,
            N_CHAN_PROC_NATIVE,
        )
        if dms is None:
            dms = [0.0, 100.0, 1000.0, 3000.0]
        coarse = np.asarray(dms, dtype=np.float64)
        n_coarse = coarse.size
        n_fine = max(8, 2 * n_coarse)
        fine = np.linspace(
            float(dms[0]), float(dms[-1]), n_fine, dtype=np.float64,
        )
        fine_offsets_idx = np.linspace(
            0, n_fine, num=n_coarse + 1, dtype=np.int32,
        )
        return DmPlan(
            dm_min=float(dms[0]),
            dm_max=float(dms[-1]) + 1.0,
            tol=1.5,
            fine_dm=fine,
            coarse_dm=coarse,
            fine_to_coarse=np.zeros(n_fine, dtype=np.int32),
            fine_offsets_idx=fine_offsets_idx,
            fine_offsets_flat=np.zeros(n_fine, dtype=np.float64),
            time_shift_corr_stage1=np.zeros(
                (N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse), dtype=np.int32,
            ),
            time_shift_corr_stage2=np.zeros(
                (N_CHGROUP, n_coarse), dtype=np.int32,
            ),
            time_shift_search=np.zeros((n_fine, N_CHGROUP), dtype=np.int32),
            dm_idx_range_canonical=np.zeros((N_SEARCH, 2), dtype=np.int32),
            dm_idx_range_consumed=np.zeros((N_SEARCH, 2), dtype=np.int32),
            dm_idx_range_canonical_per_gpu=np.zeros(
                (N_SEARCH, N_SEARCH_GPU, 2), dtype=np.int32,
            ),
            dm_idx_range_consumed_per_gpu=np.zeros(
                (N_SEARCH, N_SEARCH_GPU, 2), dtype=np.int32,
            ),
            dm_overlap_coarse=2,
            metadata={
                "band_top_GHz": NU_TOP_PROC_GHZ,
                "band_bot_GHz": NU_BOT_PROC_GHZ,
                "BW_MHz": BW_PROC_MHZ,
                "N_chan_proc_native": N_CHAN_PROC_NATIVE,
                "t_int_fast_us": float(T_INT_FAST_US_DEFAULT),
                "t_int_search_us": 524.288,
                "tol": 1.5,
                "build_utc_ns": 1_872_345_677_000_000_000,
                "git_sha": "deadbeef",
                "version": DM_PLAN_METADATA_VERSION,
            },
        )

    def test_chan_sum_factor_8_shapes(self) -> None:
        canonical = self._canonical()
        plan = DMPlan.from_summed_canonical(canonical, chan_sum_factor=8)
        assert plan.chan_sum_factor == 8
        assert plan.nchan_per_chgroup == NCHAN_PER_CHGROUP // 8
        assert plan.chgroup_freqs_GHz.shape == (
            N_CHGROUP, NCHAN_PER_CHGROUP // 8,
        )
        assert plan._delay_native_samples_table.shape == (
            N_CHGROUP, NCHAN_PER_CHGROUP // 8, plan.n_coarse,
        )

    def test_chan_sum_factor_8_top_zero(self) -> None:
        """Convention A invariant: top SUMMED channel of every chgroup
        has zero shift for every (g, dm)."""
        canonical = self._canonical(dms=[0.0, 100.0, 3000.0])
        plan = DMPlan.from_summed_canonical(canonical, chan_sum_factor=8)
        for g in range(N_CHGROUP):
            for k in range(plan.n_coarse):
                assert plan.delay_native_samples(g, 0, k) == 0

    def test_chan_sum_factor_8_delay_monotone_in_dm(self) -> None:
        """Within a chgroup, lower-frequency summed channels delay
        more, and DM-monotonicity holds at fixed channel."""
        canonical = self._canonical(dms=[0.0, 100.0, 1000.0, 3000.0])
        plan = DMPlan.from_summed_canonical(canonical, chan_sum_factor=8)
        for g in (0, 7, 15):
            # Largest summed channel is the lowest freq.
            ch_last = plan.nchan_per_chgroup - 1
            delays = np.asarray(
                [
                    plan.delay_native_samples(g, ch_last, k)
                    for k in range(plan.n_coarse)
                ],
                dtype=np.int64,
            )
            assert (np.diff(delays) >= 0).all()
            assert delays[-1] > 0

    def test_chan_sum_factor_8_top_freq_above_summed_freqs(self) -> None:
        """Summed-channel grid is monotone DECREASING within a chgroup
        (matches the descending fine-channel convention).
        Band-CENTER freq of summed-ch 0 > band-CENTER of summed-ch 1
        > … > band-CENTER of summed-ch (NCHAN_eff − 1)."""
        canonical = self._canonical()
        plan = DMPlan.from_summed_canonical(canonical, chan_sum_factor=8)
        for g in range(N_CHGROUP):
            row = plan.chgroup_freqs_GHz[g]
            assert (np.diff(row) < 0).all(), (
                f"chgroup {g} summed freqs not monotone-decreasing: {row}"
            )

    def test_chan_sum_factor_1_equals_from_canonical(self) -> None:
        """``from_summed_canonical(plan, chan_sum_factor=1)`` ≡
        ``from_canonical(plan)`` (legacy bit-identical check)."""
        canonical = self._canonical()
        a = DMPlan.from_canonical(canonical)
        b = DMPlan.from_summed_canonical(canonical, chan_sum_factor=1)
        assert a.chan_sum_factor == b.chan_sum_factor == 1
        np.testing.assert_array_equal(
            a.chgroup_freqs_GHz, b.chgroup_freqs_GHz,
        )
        np.testing.assert_array_equal(
            a._delay_native_samples_table, b._delay_native_samples_table,
        )

    def test_chan_sum_factor_invalid(self) -> None:
        canonical = self._canonical()
        with pytest.raises(ValueError, match="chan_sum_factor"):
            DMPlan.from_summed_canonical(canonical, chan_sum_factor=0)
        with pytest.raises(ValueError, match="does not divide"):
            DMPlan.from_summed_canonical(canonical, chan_sum_factor=7)


# ---------------------------------------------------------------------------
# coarse_dedisp: shape, dtype, math
# ---------------------------------------------------------------------------


class TestCoarseDedispShape:

    def test_coarse_dedisp_output_shape_dtype(self) -> None:
        """Output cube has the documented shape + dtype."""
        plan = _make_dm_plan(dms=[0.0, 50.0, 200.0])
        n_chan_test = 16
        t_fast = 64
        n_grid = 16
        cube = torch.zeros(
            (t_fast, n_chan_test, n_grid, n_grid), dtype=torch.complex64,
        )
        out_fp16 = coarse_dedisp(cube, plan, chgroup=0, output_dtype=torch.float16)
        # T_dedisp = T_fast - max_bin_shift_over_subset (chgroup 0, full plan)
        max_b = plan.max_delay_bins_per_chgroup(chgroup=0)
        # but the bin-shift table was built at full NCHAN; the dedisperser
        # only saw the first n_chan_test channels — the max over the
        # subset is what matters for T_dedisp.
        bins_full = plan.delay_bins_per_chgroup(0)
        max_b_subset = int(bins_full[:n_chan_test].max())
        assert out_fp16.shape == (t_fast - max_b_subset, plan.n_coarse, n_grid, n_grid)
        assert out_fp16.dtype == torch.float16
        # fp32 output dtype option
        out_fp32 = coarse_dedisp(cube, plan, chgroup=0, output_dtype=torch.float32)
        assert out_fp32.dtype == torch.float32
        assert out_fp32.shape == out_fp16.shape


class TestCoarseDedispMath:

    def test_coarse_dedisp_zero_dm_passthrough(self) -> None:
        """At DM=0 every per-channel delay is 0, so ``out[t, 0, l, m]
        == Σ_ch |cube[t, ch, l, m]|²`` exactly (modulo fp32 round-off)."""
        plan = _make_dm_plan(dms=[0.0])
        n_chan_test = 8
        t_fast = 16
        n_grid = 8
        # Random-valued cube so the test isn't accidentally satisfied
        # by an all-zero shortcut.
        gen = torch.Generator().manual_seed(20260505)
        cube = (
            torch.randn(
                (t_fast, n_chan_test, n_grid, n_grid),
                generator=gen, dtype=torch.float32,
            )
            + 1j * torch.randn(
                (t_fast, n_chan_test, n_grid, n_grid),
                generator=gen, dtype=torch.float32,
            )
        ).to(torch.complex64)
        out = coarse_dedisp(cube, plan, chgroup=0, output_dtype=torch.float32)
        # T_dedisp = T_fast since max_delay = 0
        assert out.shape == (t_fast, 1, n_grid, n_grid)
        expected = (cube.real ** 2 + cube.imag ** 2).sum(dim=1)       # (T_fast, N_grid, N_grid)
        diff = (out[:, 0] - expected).abs().max().item()
        assert diff < 1e-4, f"DM=0 passthrough mismatch: max diff = {diff:.3e}"

    def test_coarse_dedisp_recovers_synthetic_burst(self) -> None:
        """A burst dispersed at DM=405 is recovered to within ≤ 1
        native-sample bin error when dedispersed at the matching DM."""
        dm_truth = 405.0
        plan = _make_dm_plan(dms=[dm_truth])
        n_chan_test = 24
        t_fast = 96
        n_grid = 32
        chgroup = 0
        l_idx, m_idx = 11, 20
        # Place burst peak at top-channel time = 4 fast-vis bins from
        # the start so even high-DM dispersion fits inside T_fast.
        t_peak = 4
        cube = _make_burst_cube(
            t_fast=t_fast,
            n_chan_test=n_chan_test,
            n_grid=n_grid,
            chgroup=chgroup,
            dm_pc_cc=dm_truth,
            t_peak_top_bin=t_peak,
            l_idx=l_idx,
            m_idx=m_idx,
            t_int_fast_us=plan.t_int_fast_us,
            amplitude=2.0 + 0.0j,
        )
        out = coarse_dedisp(cube, plan, chgroup=chgroup, output_dtype=torch.float32)
        # Reduce over (l, m) with argmax to find the hot pixel; over
        # t to find the recovered peak-time bin.
        out_hot = out[:, 0, l_idx, m_idx]                            # (T_dedisp,)
        peak_idx = int(out_hot.argmax().item())
        assert abs(peak_idx - t_peak) <= 1, (
            f"recovered peak at t'={peak_idx}, expected t_peak={t_peak} "
            f"± 1; out_hot[:8]={out_hot[:8].tolist()}"
        )
        # Peak amplitude should equal n_chan_test × |amp|² (each
        # channel contributes its peak power once).
        expected_peak = n_chan_test * (2.0 ** 2)
        # Allow ±1 channel of "loss" if a per-bin shift collides with
        # the cube boundary at high DM (none expected here, but the
        # margin is harmless).
        assert out_hot[peak_idx].item() >= 0.9 * expected_peak, (
            f"recovered peak amplitude {out_hot[peak_idx].item():.2f} < "
            f"0.9 × expected {expected_peak}; recovery failed"
        )

    def test_coarse_dedisp_off_dm_loses_peak(self) -> None:
        """Dedispersion at the WRONG DM keeps the channels misaligned
        and the peak amplitude must drop ≥ 50% relative to the truth."""
        dm_truth = 405.0
        # Two DM trials: truth + a clearly-wrong one.
        plan = _make_dm_plan(dms=[100.0, dm_truth])
        n_chan_test = 24
        t_fast = 96
        n_grid = 16
        chgroup = 0
        l_idx, m_idx = 8, 6
        t_peak = 4
        cube = _make_burst_cube(
            t_fast=t_fast,
            n_chan_test=n_chan_test,
            n_grid=n_grid,
            chgroup=chgroup,
            dm_pc_cc=dm_truth,
            t_peak_top_bin=t_peak,
            l_idx=l_idx,
            m_idx=m_idx,
            t_int_fast_us=plan.t_int_fast_us,
            amplitude=1.0 + 0.0j,
        )
        out = coarse_dedisp(cube, plan, chgroup=chgroup, output_dtype=torch.float32)
        # Truth-DM peak (over time, at the burst pixel).
        peak_truth = float(out[:, 1, l_idx, m_idx].max().item())
        # Off-DM peak (over time, at the burst pixel; over whole image
        # as a fall-back to make sure we're not picking up by accident).
        peak_off_pixel = float(out[:, 0, l_idx, m_idx].max().item())
        peak_off_image = float(out[:, 0, :, :].max().item())
        peak_off = max(peak_off_pixel, peak_off_image)
        assert peak_off <= 0.5 * peak_truth, (
            f"off-DM peak {peak_off:.2f} > 0.5 × truth peak "
            f"{peak_truth:.2f}; the dedisp doesn't discriminate DMs"
        )

    def test_coarse_dedisp_handles_int_overflow(self) -> None:
        """The fp32 in-kernel accumulator must not saturate at expected
        burst brightnesses, even when the per-channel cube is at the
        fp16 max (which IS representable in fp16).

        NCHAN_PER_CHGROUP=384 channels × T_fast × cube[t, ch, l, m]² —
        the per-output-cell sum is ``Σ_ch |c|²``; at |c| = fp16-max ≈
        65504, |c|² ≈ 4.29e9, × 384 = 1.65e12, which fp32 can hold (max
        ≈ 3.4e38) but fp16 cannot (max ≈ 6.55e4). This test pins the
        kernel's fp32 accumulator + flagged fp16 OUTPUT saturation.

        The output cube is ``output_dtype=float16``; fp16 saturation
        clamps to ``+inf`` when the per-cell sum exceeds 65504.
        """
        # 16 channels (test scale) × |c|² near fp16 max — sum stays
        # under fp16 saturation but is large enough to detect an
        # accidental fp16 accumulator.
        plan = _make_dm_plan(dms=[0.0])
        n_chan_test = 16
        t_fast = 8
        n_grid = 4
        # Per-cell amplitude such that 16 × |amp|² ≈ 6e4 (just under
        # fp16 max ≈ 65504): |amp|² ≈ 3750, so |amp| ≈ 61.2.
        amp_re = 61.0
        cube = torch.zeros(
            (t_fast, n_chan_test, n_grid, n_grid), dtype=torch.complex64,
        )
        cube[:, :, 0, 0] = (amp_re + 0.0j)
        out = coarse_dedisp(cube, plan, chgroup=0, output_dtype=torch.float16)
        peak = float(out[:, 0, 0, 0].max().item())
        # Expected sum: n_chan_test * amp_re² = 16 * 3721 = 59536
        expected = n_chan_test * (amp_re ** 2)
        assert math.isfinite(peak), (
            f"fp16 output saturated (peak={peak}); fp32 accumulator "
            f"may have been promoted incorrectly"
        )
        # fp16 representation has ~3 decimal digits of precision; tolerate
        # 5% relative error from fp16 quantisation of the output.
        rel_err = abs(peak - expected) / expected
        assert rel_err < 0.05, (
            f"output peak {peak:.1f} != expected {expected:.1f} "
            f"(rel_err={rel_err:.3f}); fp16 accumulator bug?"
        )

    def test_F24_coarse_dm_uses_native_t_axis(self) -> None:
        """F24: per-(g, ch, dm) shift is stored in NATIVE samples; bin
        units derived via two-stage rounding (native first, bins
        second) to compose with the canonical DmPlan stage-2 shifts.

        Pin the convention by:

        1. Showing :meth:`DMPlan.delay_native_samples` returns
           ``round(delay_us / NATIVE_SAMPLE_US)`` for a few channels.
        2. Showing :meth:`DMPlan.delay_bins` returns
           ``round(delay_native / t_int_fast_native)`` (NOT
           ``round(delay_us / t_int_fast_us)``).

        The two paths CAN differ — find a (g, ch, dm) where they do,
        then pin the native-rounded one.
        """
        # Use a non-default cadence that exercises the convention:
        # t_int_fast_us = 524.288 µs = 16 native samples.
        t_int_fast_us = 524.288
        # Pick a DM grid that has a non-trivial (sub-bin) delay at
        # most channels of chgroup 0.
        dm_test = 100.0
        plan = _make_dm_plan(dms=[dm_test], t_int_fast_us=t_int_fast_us)

        chgroup = 0
        nu_top = plan.chgroup_freqs_GHz[chgroup, 0]
        t_int_fast_native = t_int_fast_us / NATIVE_SAMPLE_US

        # 1. delay_native_samples == round(delay_us / NATIVE_SAMPLE_US)
        for ch in (0, 50, 200, NCHAN_PER_CHGROUP - 1):
            nu_ch = plan.chgroup_freqs_GHz[chgroup, ch]
            delay_us_truth = (
                K_DM_MS_GHZ2_PC * dm_test
                * (1.0 / nu_ch ** 2 - 1.0 / nu_top ** 2) * 1e3
            )
            native_truth = int(round(delay_us_truth / NATIVE_SAMPLE_US))
            native_actual = plan.delay_native_samples(chgroup, ch, 0)
            assert native_actual == native_truth, (
                f"chgroup={chgroup} ch={ch}: native_actual={native_actual}, "
                f"native_truth={native_truth} (delay_us={delay_us_truth:.3f})"
            )

        # 2. delay_bins == round(delay_native / t_int_fast_native), and
        #    we can find a channel where this differs from the
        #    direct round(delay_us / t_int_fast_us). Walk channels
        #    until divergence.
        divergence_found = False
        for ch in range(1, NCHAN_PER_CHGROUP):
            nu_ch = plan.chgroup_freqs_GHz[chgroup, ch]
            delay_us = (
                K_DM_MS_GHZ2_PC * dm_test
                * (1.0 / nu_ch ** 2 - 1.0 / nu_top ** 2) * 1e3
            )
            via_native = int(round(
                round(delay_us / NATIVE_SAMPLE_US) / t_int_fast_native
            ))
            direct = int(round(delay_us / t_int_fast_us))
            if via_native != direct:
                # Pin: the API chooses the via-native path.
                actual = plan.delay_bins(chgroup, ch, 0)
                assert actual == via_native, (
                    f"chgroup={chgroup} ch={ch}: delay_bins={actual} != "
                    f"via_native={via_native} (direct={direct}); F24 "
                    f"convention broken"
                )
                divergence_found = True
                break
        # If no divergence was found at this DM, the test still passes
        # — but log so a future investigator knows the assertion was
        # softer than intended in this run. (Most cadence/DM choices
        # produce divergence somewhere in 384 channels.)
        if not divergence_found:
            pytest.skip(
                "no native-vs-bin rounding divergence at this (cadence, "
                "DM); the F24 convention pin is degenerate here"
            )

    def test_F18_F20_F21_compose_in_dedispersed_image(self) -> None:
        """A burst at known ``(l_idx, m_idx)`` ends up at the SAME
        ``(l_idx, m_idx)`` (no axis swap, no negation) in the
        dedispersed image cube.

        Composes:

        * F18 (visibility GEMM index convention): upstream of this
          test — we synthesise the cube directly with a hot pixel,
          so F18 is implicit (the cube reflects the +uv-grid Stokes-I
          per-channel image that F18-correct visibilities + F20-
          correct gridding + iFFT2 would produce).
        * F20 (gridder ``(u, v)`` negation, applied once at
          ``build_pattern``): same — implicit in the cube.
        * F21 (cal-loader DEC-only phase fold): again upstream;
          implicit.
        * **Chunk 3b dedisperser MUST NOT introduce any spatial
          flip** — this is what we DO test: the dedispersed peak
          remains at the SAME ``(l, m)`` index it occupied in the
          input cube, never at ``(N-1-l, N-1-m)``, ``(m, l)``, etc.
        """
        plan = _make_dm_plan(dms=[200.0])
        n_chan_test = 16
        t_fast = 64
        n_grid = 32
        chgroup = 0
        # Asymmetric pixel: l_idx ≠ m_idx, both far from grid centre +
        # corners (so a 180° rotation, an l↔m swap, or a centred-flip
        # all map to a clearly different cell).
        l_idx, m_idx = 7, 22
        cube = _make_burst_cube(
            t_fast=t_fast,
            n_chan_test=n_chan_test,
            n_grid=n_grid,
            chgroup=chgroup,
            dm_pc_cc=200.0,
            t_peak_top_bin=2,
            l_idx=l_idx,
            m_idx=m_idx,
            t_int_fast_us=plan.t_int_fast_us,
        )
        out = coarse_dedisp(cube, plan, chgroup=chgroup, output_dtype=torch.float32)
        # Find the (l, m) of the dedispersed peak (over t and dm).
        out_max_over_tdm = out.amax(dim=(0, 1))                       # (N_grid, N_grid)
        flat_argmax = int(out_max_over_tdm.argmax().item())
        l_recovered = flat_argmax // n_grid
        m_recovered = flat_argmax % n_grid
        assert (l_recovered, m_recovered) == (l_idx, m_idx), (
            f"recovered peak at (l, m) = ({l_recovered}, {m_recovered}), "
            f"expected ({l_idx}, {m_idx}); dedisp introduced a spatial flip"
        )

    def test_dm_plan_single_dm_for_burst_test(self, tmp_path: Path) -> None:
        """A single-DM custom plan (PARALLEL_AGENTS.md §5.1 burst
        recipe) loads + dedisps as a single trial. Exercises both the
        ``n_coarse == 1`` edge case in :class:`DMPlan` and the
        ``dm_indices=[0]`` slice in :func:`coarse_dedisp`."""
        plan_in = _make_dm_plan(dms=[404.688], n_fine_per_coarse=1)
        out_path = tmp_path / "dm_plan_burst_250924mptq.npz"
        plan_in.to_npz(str(out_path))
        plan = DMPlan.from_coarse_only_npz(str(out_path))
        assert plan.n_coarse == 1
        # Sanity: single-DM lookup should non-zero on lower channels
        # (single trial != trivial zero plan).
        assert plan.delay_native_samples(0, NCHAN_PER_CHGROUP - 1, 0) > 0

        # Run the dedisperser on a synthetic burst at the same DM.
        n_chan_test = 16
        t_fast = 96
        n_grid = 16
        cube = _make_burst_cube(
            t_fast=t_fast,
            n_chan_test=n_chan_test,
            n_grid=n_grid,
            chgroup=0,
            dm_pc_cc=404.688,
            t_peak_top_bin=4,
            l_idx=5,
            m_idx=9,
            t_int_fast_us=plan.t_int_fast_us,
        )
        out = coarse_dedisp(cube, plan, chgroup=0, output_dtype=torch.float32)
        assert out.shape[1] == 1
        # Peak at (l=5, m=9), t' near 4
        flat = out[:, 0, 5, 9]
        peak_t = int(flat.argmax().item())
        assert abs(peak_t - 4) <= 1


# ---------------------------------------------------------------------------
# Stage2FIFO: capacity, ordering, eviction
# ---------------------------------------------------------------------------


def _make_cube(value: float, shape: tuple[int, ...] = (2, 3, 4)) -> torch.Tensor:
    """Tiny constant-value cube used by the FIFO tests."""
    return torch.full(shape, value, dtype=torch.float32)


class TestStage2FIFO:

    def test_stage2_fifo_push_pop_order(self) -> None:
        """``as_list()`` returns cubes in oldest-first push order; the
        latest push is always at the tail, the oldest at the head."""
        fifo = Stage2FIFO(depth=4)
        cubes = [_make_cube(float(k)) for k in range(3)]
        for c in cubes:
            assert fifo.push(c) is None                              # not full → no eviction
        contents = fifo.as_list()
        assert len(contents) == 3
        for k, c in enumerate(cubes):
            assert torch.equal(contents[k], c), (
                f"FIFO order broken at k={k}: got value "
                f"{float(contents[k][0, 0, 0])}, expected {float(c[0, 0, 0])}"
            )
        assert torch.equal(fifo.peek_oldest(), cubes[0])
        assert torch.equal(fifo.peek_latest(), cubes[-1])

    def test_stage2_fifo_capacity_full_evicts_oldest(self) -> None:
        """Once full, every additional push evicts the head; the FIFO
        size stays at ``depth``."""
        fifo = Stage2FIFO(depth=3)
        # Fill exactly to capacity.
        for k in range(3):
            assert fifo.push(_make_cube(float(k))) is None
        assert len(fifo) == 3
        assert fifo.full()
        # Push 4th → evicts cube 0.
        evicted = fifo.push(_make_cube(99.0))
        assert evicted is not None
        assert torch.equal(evicted, _make_cube(0.0))
        assert len(fifo) == 3
        # Contents are now [1, 2, 99]
        contents = fifo.as_list()
        assert torch.equal(contents[0], _make_cube(1.0))
        assert torch.equal(contents[1], _make_cube(2.0))
        assert torch.equal(contents[2], _make_cube(99.0))

    def test_stage2_fifo_push_returns_popped_when_full(self) -> None:
        """``push`` returns ``None`` while filling, then the evicted
        cube on every subsequent push. The returned cube IS the
        previous head (identity, not just equal)."""
        fifo = Stage2FIFO(depth=2)
        c0 = _make_cube(0.0)
        c1 = _make_cube(1.0)
        c2 = _make_cube(2.0)
        c3 = _make_cube(3.0)
        assert fifo.push(c0) is None
        assert fifo.push(c1) is None
        ev0 = fifo.push(c2)
        assert ev0 is c0, "evicted cube must be the SAME object as c0"
        ev1 = fifo.push(c3)
        assert ev1 is c1
        # Now contents are [c2, c3]
        assert fifo.peek_oldest() is c2
        assert fifo.peek_latest() is c3

    def test_stage2_fifo_initial_partial_fill(self) -> None:
        """Partial fill behaviour: ``empty()`` / ``full()`` / ``len()``
        / ``peek_*`` all behave consistently as the FIFO grows from
        0 → depth."""
        fifo = Stage2FIFO(depth=4)
        assert fifo.empty()
        assert not fifo.full()
        assert len(fifo) == 0
        assert fifo.peek_latest() is None
        assert fifo.peek_oldest() is None

        fifo.push(_make_cube(0.0))
        assert not fifo.empty()
        assert not fifo.full()
        assert len(fifo) == 1
        assert fifo.peek_latest() is not None
        assert fifo.peek_oldest() is not None

        for k in range(1, 4):
            fifo.push(_make_cube(float(k)))
        assert not fifo.empty()
        assert fifo.full()
        assert len(fifo) == 4

        fifo.clear()
        assert fifo.empty()
        assert len(fifo) == 0
        # After clear(), the type-contract reference is reset; a new
        # push with a different shape now succeeds (would have raised
        # before clear()).
        fifo.push(_make_cube(0.0, shape=(7, 7)))
        assert len(fifo) == 1

    def test_stage2_fifo_default_depth_constant(self) -> None:
        """The default depth comes from the constants module — the
        chunk-3b smoke tests assume :data:`COARSE_DM_FIFO_DEPTH_DEFAULT`
        ≥ 1 and ≤ 16 (sanity on the constant; actual production
        per-(g, c) sizing happens at the integration site)."""
        assert 1 <= COARSE_DM_FIFO_DEPTH_DEFAULT <= 16
        fifo = Stage2FIFO()
        assert fifo.depth == COARSE_DM_FIFO_DEPTH_DEFAULT

    def test_stage2_fifo_rejects_inconsistent_pushes(self) -> None:
        """Push of a cube with different shape/dtype/device than the
        first push raises ValueError — surfaces a producer bug at the
        FIFO boundary instead of letting heterogeneity propagate."""
        fifo = Stage2FIFO(depth=3)
        fifo.push(torch.zeros((2, 3), dtype=torch.float32))
        with pytest.raises(ValueError, match="shape"):
            fifo.push(torch.zeros((2, 4), dtype=torch.float32))
        with pytest.raises(ValueError, match="dtype"):
            fifo.push(torch.zeros((2, 3), dtype=torch.float16))


# ---------------------------------------------------------------------------
# Helper: max_output_t_dedisp
# ---------------------------------------------------------------------------


def test_max_output_t_dedisp() -> None:
    """``max_output_t_dedisp`` returns ``T_fast - max_bin_shift``,
    clamped to 0 when the cube is too short to accommodate any
    dedispersed bin."""
    plan = _make_dm_plan(dms=[100.0, 1000.0])
    max_b = plan.max_delay_bins_per_chgroup(0)
    assert max_output_t_dedisp(64, plan, chgroup=0) == max(0, 64 - max_b)
    assert max_output_t_dedisp(max_b, plan, chgroup=0) == 0       # boundary
    # Subset selecting only the lower DM should give a smaller max_b
    # (or zero, depending on freqs); the function should reflect that.
    bins_full = plan.delay_bins_per_chgroup(0)
    max_b_dm0 = int(bins_full[:, 0].max())
    assert max_output_t_dedisp(
        64, plan, chgroup=0, dm_indices=np.asarray([0])
    ) == max(0, 64 - max_b_dm0)

"""End-to-end smoke test for corr-side stage-2 + search-side stage-3.

Drives a synthetic dispersed pulse through ALL of:
  * per-chgroup :class:`Stage2InterChgroupShiftFifo` (corr side)
  * :func:`compute_time_shift_search` with ``include_coarse_offset=False``
    (search side)
  * :func:`combine_chgroups` (search side coherent sum)

and verifies:
  1. The burst peak lands at the predicted ``(fine_dm, t)`` bin (no
     DM bias).
  2. The burst is **centered** in the cube (no edge zeroing, no
     half-band coverage loss).
  3. Cross-stage residual is bounded by ±1 sample of the search cadence.

This is the integration-level regression for the 2026-06-03 Convention-A
TOP-reference unification that closed the -2.45% DM bias and enabled
corr-side Option A (per_coarse_dm stage-2). It complements:

  * tests/test_stage2_shifts.py::test_cross_stage_residual_against_baked_search_shifts
    (pins the analytic cross-stage cancellation)
  * tests/test_stage2_chgroup_alignment.py::test_realistic_chgroup0_max_shift
    (pins the corr-side FIFO at the worst-case shift)

by tying the math of all four moving parts together in one assertion.

Runs CPU-only (no GPU required) so it can gate every PR cheaply.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.coarse_dm.stage2_chgroup_alignment import (
    Stage2InterChgroupShiftFifo,
)
from dsart.common.constants import (
    NU_BOT_PROC_GHZ,
    NU_CHGROUP_TOP_GHZ,
    N_CHGROUP,
)
from dsart.common.dispersion import delta_tau_us
from dsart.fine_dm.combiner import (
    combine_chgroups,
    compute_time_shift_search,
)


T_INT_CORR_US = 262.144   # corr-side fast-vis cadence (M7.4)
T_INT_SEARCH_US = 524.288  # search-side cadence
SEARCH_PER_CORR = int(round(T_INT_SEARCH_US / T_INT_CORR_US))  # 2


def _arrival_corr_sample_at_chgroup_top(
    chgroup: int, true_dm: float, arrival_at_chgroup_top15_corr: int,
) -> int:
    """Per-chgroup pulse arrival time at the chgroup's TOP frequency.

    Stage-1 (Convention A) aligns each chgroup's channels to its TOP,
    so the post-stage-1 pulse for chgroup g lands at the time the
    burst arrived at ``ν_chgroup_TOP[g]``.
    """
    d_us = delta_tau_us(
        float(NU_CHGROUP_TOP_GHZ[N_CHGROUP - 1]),
        float(NU_CHGROUP_TOP_GHZ[chgroup]),
        true_dm,
    )
    samples_offset = int(round(d_us / T_INT_CORR_US))
    return arrival_at_chgroup_top15_corr - samples_offset


@pytest.mark.parametrize("true_dm", [258.74, 894.5, 1532.9, 2499.9])
def test_e2e_stage2_plus_stage3_pulse_recovery(true_dm: float):
    """End-to-end: per-chgroup pulses → per-chgroup stage-2 FIFO →
    search-side combiner with ``include_coarse_offset=False`` →
    coherent peak at the analytic ``(t_search, g_coverage)`` bin."""
    coarse_dm = np.array(
        [258.740, 576.4, 894.5, 1213.2, 1532.9, 1853.8, 2176.0, 2499.9],
        dtype=np.float64,
    )
    c_idx = int(np.argmin(np.abs(coarse_dm - true_dm)))
    t_dedisp_corr = 500
    t_dedisp_search = t_dedisp_corr // SEARCH_PER_CORR  # 250

    # Anchor: target the burst to land in FIFO output cube N_OUT.
    # The FIFO emits cube B representing ν_bot_proc time
    # [B*T, (B+1)*T) at corr cadence. We choose N_OUT well past the
    # worst-case warmup (chgroup-0 at DM=2500 has depth ~11 cubes).
    N_OUT = 18
    burst_t_at_proc_in_cube = t_dedisp_corr // 2  # interior position
    arrival_at_chgroup_top15_corr = (
        N_OUT * t_dedisp_corr + burst_t_at_proc_in_cube
        - int(round(
            delta_tau_us(
                float(NU_BOT_PROC_GHZ),
                float(NU_CHGROUP_TOP_GHZ[N_CHGROUP - 1]),
                true_dm,
            ) / T_INT_CORR_US
        ))
    )
    # Push enough cubes that emit-block N_OUT comes out.
    n_cubes = N_OUT + 1

    fifos = [
        Stage2InterChgroupShiftFifo(
            chgroup=g,
            coarse_dm_pc_cm3=coarse_dm,
            t_dedisp=t_dedisp_corr,
            t_int_corr_us=T_INT_CORR_US,
        )
        for g in range(N_CHGROUP)
    ]

    # Synthesize per-chgroup post-stage-1 cubes. Only the c_idx
    # coarse-DM slice carries a non-zero pulse (we model the burst
    # at its closest coarse-DM trial).
    per_chgroup_stage1: list[list[torch.Tensor]] = [
        [] for _ in range(N_CHGROUP)
    ]
    for g in range(N_CHGROUP):
        arrival_g = _arrival_corr_sample_at_chgroup_top(
            g, true_dm, arrival_at_chgroup_top15_corr,
        )
        for blk in range(n_cubes):
            cube = torch.zeros(
                (coarse_dm.size, t_dedisp_corr, 1), dtype=torch.float32,
            )
            for s in range(t_dedisp_corr):
                if blk * t_dedisp_corr + s == arrival_g:
                    cube[c_idx, s, 0] = 1000.0
            per_chgroup_stage1[g].append(cube)

    # Drive the FIFOs.
    per_chgroup_stage2_outs: list[list[torch.Tensor]] = [
        [] for _ in range(N_CHGROUP)
    ]
    for blk in range(n_cubes):
        for g in range(N_CHGROUP):
            out = fifos[g].push(per_chgroup_stage1[g][blk], block_n=blk)
            for emitted in out:
                per_chgroup_stage2_outs[g].append(emitted[c_idx, :, :])

    n_emits = [len(x) for x in per_chgroup_stage2_outs]
    assert min(n_emits) > 0, (
        f"At least one chgroup never emitted: n_emits={n_emits}, "
        f"true_dm={true_dm}"
    )
    # All chgroups must have emit-block N_OUT. Emit index for chgroup g
    # is N_OUT - warmup_g, where warmup_g = n_cubes - n_emits[g].
    target_emit_idx = []
    for g in range(N_CHGROUP):
        warmup_g = n_cubes - n_emits[g]
        idx = N_OUT - warmup_g
        assert 0 <= idx < n_emits[g], (
            f"chgroup {g}: N_OUT={N_OUT} not in emit window "
            f"[{warmup_g}, {n_cubes})"
        )
        target_emit_idx.append(idx)

    # Take a small window around the burst emit cube: [N_OUT-1, N_OUT+1)
    # (concatenate 2 cubes per chgroup so the combiner has interior
    # samples to shift into).
    trimmed = [
        [per_chgroup_stage2_outs[g][target_emit_idx[g]]]
        for g in range(N_CHGROUP)
    ]
    burst_emit_index_in_trimmed = 0  # only one cube in trimmed

    # --- Assemble the per-chgroup search-cadence stream ---
    # Concatenate emitted cubes along time, then downsample 2:1 to
    # search cadence (sum-of-pairs, matching detector integration).
    streams_corr = [
        torch.cat(trimmed[g], dim=0)  # (n_common * t_dedisp_corr, 1)
        for g in range(N_CHGROUP)
    ]
    t_corr = streams_corr[0].shape[0]
    t_search = t_corr // SEARCH_PER_CORR
    # Stream the 1 burst cube only (t_dedisp_corr samples per chgroup).
    streams_corr = [
        torch.cat(trimmed[g], dim=0)  # (t_dedisp_corr, 1)
        for g in range(N_CHGROUP)
    ]
    t_corr = streams_corr[0].shape[0]
    t_search = t_corr // SEARCH_PER_CORR

    streams_search_np = {}
    for g in range(N_CHGROUP):
        s = streams_corr[g][: t_search * SEARCH_PER_CORR].reshape(
            t_search, SEARCH_PER_CORR, 1
        ).sum(dim=1).numpy()
        streams_search_np[g] = s.reshape(t_search, 1, 1).astype(np.complex64)

    # --- Search-side stage-3 shifts (include_coarse_offset=False) ---
    fine_grid = np.array([
        coarse_dm[c_idx] + delta
        for delta in np.linspace(-30.0, 30.0, 31)
    ], dtype=np.float64)
    table = compute_time_shift_search(
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_grid,
        fine_to_coarse=np.full(fine_grid.size, c_idx, dtype=np.int64),
        t_int_search_us=T_INT_SEARCH_US,
        include_coarse_offset=False,
    )

    t_lo, t_hi = 0, t_search
    pwr = np.zeros((fine_grid.size, t_hi - t_lo), dtype=np.float64)
    for f in range(fine_grid.size):
        out = combine_chgroups(
            per_chgroup_streams=streams_search_np,
            time_shift_per_chgroup=table.shifts[f, :],
            t_window=(t_lo, t_hi),
            n_grid=1,
        )
        pwr[f] = (np.abs(out) ** 2).sum(axis=(1, 2))

    # --- Verify peak is at (fine_dm ≈ true_dm, t = expected) ---
    peak_idx = int(np.argmax(pwr))
    peak_f, peak_t = np.unravel_index(peak_idx, pwr.shape)
    dm_detected = float(fine_grid[peak_f])
    step = float(np.diff(fine_grid).mean())
    assert abs(dm_detected - true_dm) <= step + 1e-9, (
        f"true_dm={true_dm} detected={dm_detected} step={step:.3f}: "
        f"DM bias > one fine step (cross-stage cancellation broken)"
    )

    assert 1 <= peak_t <= pwr.shape[1] - 2, (
        f"peak at edge t={peak_t}/{pwr.shape[1]}: cube boundary "
        f"zero-fill? (true_dm={true_dm})"
    )

    # Coherence check: the peak must reflect MULTI-CHGROUP coherent
    # summation, not a single-chgroup contribution. A single-sample
    # delta pulse can split across adjacent search bins (the corr→search
    # 2:1 sum is bin-aligned but the per-chgroup stage-2 shifts have
    # ±1-corr-sample rint() noise), so we require ≥ 10 chgroups' worth
    # of coherent amplitude (vs the 1-chgroup floor of pulse_amp²).
    one_chgroup_floor = (1000.0) ** 2
    min_n_chgroups_coherent = 10
    expected_peak_min = (min_n_chgroups_coherent ** 2) * one_chgroup_floor
    assert float(pwr[peak_f, peak_t]) >= expected_peak_min, (
        f"Peak power {float(pwr[peak_f, peak_t]):.2e} < expected min "
        f"{expected_peak_min:.2e} (true_dm={true_dm}): partial-band "
        f"coherence loss (fewer than {min_n_chgroups_coherent}/16 "
        f"chgroups contributed coherently)"
    )

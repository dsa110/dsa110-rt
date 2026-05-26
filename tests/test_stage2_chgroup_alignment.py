"""Acceptance tests for
:class:`dsart.coarse_dm.stage2_chgroup_alignment.Stage2InterChgroupShiftFifo`.

Pins the corr-side stage-2 inter-chgroup time alignment FIFO. These
tests are the gating evidence for the Option A wire-in in
``corr_fast_integration``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.coarse_dm.stage2_chgroup_alignment import (
    Stage2InterChgroupShiftFifo,
)
from dsart.coarse_dm.stage2_shifts import compute_stage2_shifts
from dsart.common.constants import N_CHGROUP


COARSE_DM_PROD = np.array(
    [258.740, 387.50, 581.94, 873.95, 1312.71, 1971.62, 2962.93, 4452.16],
    dtype=np.float64,
)


def _identity_cube(
    n_dm: int, t_dedisp: int, n_filled: int = 4,
    block_offset: int = 0,
) -> torch.Tensor:
    """Build a cube where ``cube[c, s, f] = global_sample_index + c*1e6``.

    The +c*1e6 keeps per-coarse-DM channels distinguishable, and the
    sample-index payload lets us verify the roll math by inspecting
    output values without needing an oracle.
    """
    arr = np.empty((n_dm, t_dedisp, n_filled), dtype=np.float32)
    for c in range(n_dm):
        for s in range(t_dedisp):
            global_s = block_offset * t_dedisp + s
            arr[c, s, :] = float(global_s) + c * 1e6
    return torch.from_numpy(arr)


def test_chgroup15_is_identity():
    """The band-bottom chgroup has zero shift for every coarse-DM: every
    push emits the input cube unchanged (no warm-up)."""
    coarse = COARSE_DM_PROD.copy()
    f = Stage2InterChgroupShiftFifo(
        chgroup=N_CHGROUP - 1,
        coarse_dm_pc_cm3=coarse,
        t_dedisp=8,
    )
    for blk in range(4):
        cube = _identity_cube(coarse.size, 8, 4, block_offset=blk)
        out = f.push(cube, block_n=blk)
        assert len(out) == 1, f"chgroup-15 should emit on every push; got len={len(out)} at blk={blk}"
        assert torch.equal(out[0], cube)


def test_chgroup15_warmed_up_immediately():
    f = Stage2InterChgroupShiftFifo(
        chgroup=N_CHGROUP - 1,
        coarse_dm_pc_cm3=np.array([100.0]),
        t_dedisp=4,
    )
    cube = _identity_cube(1, 4, 2, block_offset=0)
    f.push(cube, block_n=0)
    assert f.warmed_up() is True


def test_synthetic_pure_block_shift():
    """Build a 1-coarse-DM, 1-chgroup FIFO with shift exactly equal to
    T_dedisp so the output for block_n=k should be the cube that was
    pushed at block_n=k-1. Tests the r==0 path.
    """
    t_dedisp = 16
    # delta_tau is what compute_stage2_shifts computes; we instead
    # construct a synthetic shift_table by hand so we can pin the math.
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    table = Stage2ShiftTable(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.5,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        shifts_samples=np.array([t_dedisp], dtype=np.int32),
    )
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0]),
        t_dedisp=t_dedisp,
        shift_table=table,
    )
    cubes = [
        _identity_cube(1, t_dedisp, 2, block_offset=b) for b in range(4)
    ]
    # block 0 push: ring=[cube0], not warm (need ring[1] for r=0, k=1).
    assert f.push(cubes[0], block_n=0) == []
    # block 1 push: ring=[cube1, cube0], warmed up.
    out = f.push(cubes[1], block_n=1)
    assert len(out) == 1
    # Output at block 1 should be cube0 (k=1, r=0 → output = ring[1]).
    assert torch.equal(out[0], cubes[0])
    out = f.push(cubes[2], block_n=2)
    assert torch.equal(out[0], cubes[1])
    out = f.push(cubes[3], block_n=3)
    assert torch.equal(out[0], cubes[2])


def test_synthetic_sub_block_shift_crosses_boundary():
    """Shift of T/4 means the output is mostly from the previous cube,
    with the LAST T/4 samples coming from the current cube. Tests the
    r>0 path that crosses cube boundaries.
    """
    t_dedisp = 16
    r = 4   # output[0:4] from ring[1][12:16]; output[4:16] from ring[0][0:12]
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    table = Stage2ShiftTable(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.5,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        shifts_samples=np.array([r], dtype=np.int32),
    )
    f = Stage2InterChgroupShiftFifo(
        chgroup=0, coarse_dm_pc_cm3=np.array([100.0]),
        t_dedisp=t_dedisp, shift_table=table,
    )
    cubes = [
        _identity_cube(1, t_dedisp, 1, block_offset=b) for b in range(3)
    ]
    # Push 0: ring=[c0], k=0,r=4 → need ring[1], not warm.
    assert f.push(cubes[0], block_n=0) == []
    # Push 1: ring=[c1, c0], warmed.
    out = f.push(cubes[1], block_n=1)
    assert len(out) == 1
    out_arr = out[0][0, :, 0].numpy()
    # Expected: out[0:4] = c0[12:16] = global samples [12,13,14,15];
    # out[4:16] = c1[0:12] = global samples [16, 17, ..., 27].
    expected = np.array(
        [12, 13, 14, 15] + list(range(16, 28)),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out_arr, expected)
    # Push 2: ring=[c2, c1, c0] but ring depth (k+1)+1=2 so c0 popped.
    out = f.push(cubes[2], block_n=2)
    out_arr = out[0][0, :, 0].numpy()
    # Expected: out[0:4] = c1[12:16] = [28, 29, 30, 31];
    # out[4:16] = c2[0:12] = [32..43].
    expected = np.array(
        [28, 29, 30, 31] + list(range(32, 44)),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out_arr, expected)


def test_per_coarse_dm_shifts_are_independent():
    """Two coarse-DMs in one cube; check each gets ITS shift, not the
    other's."""
    t_dedisp = 8
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    table = Stage2ShiftTable(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0, 200.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.5,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        shifts_samples=np.array([2, 5], dtype=np.int32),
    )
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0, 200.0]),
        t_dedisp=t_dedisp,
        shift_table=table,
    )
    cubes = [_identity_cube(2, t_dedisp, 1, block_offset=b) for b in range(3)]
    # Need 2 cubes of history (for c=0 shift=2: k=0, r=2, max_back=1;
    # for c=1 shift=5: k=0, r=5, max_back=1).
    # So after 1 push the ring depth is 1 → not warmed yet (we need ring[1]).
    f.push(cubes[0], block_n=0)
    out = f.push(cubes[1], block_n=1)
    assert len(out) == 1
    # c=0: shift=2 → out[0:2] = c0[6:8]=[6,7]; out[2:8] = c1[0:6]=[8..13]
    # c=1: shift=5 → out[0:5] = c0[3:8]=[3,4,5,6,7]; out[5:8] = c1[0:3]=[8,9,10]
    # Plus the per-coarse-DM +c*1e6 offset.
    out0 = out[0][0, :, 0].numpy()
    out1 = out[0][1, :, 0].numpy()
    expected_c0 = np.array([6, 7, 8, 9, 10, 11, 12, 13], dtype=np.float32)
    expected_c1 = (
        np.array([3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        + 1e6  # c=1 offset
    )
    np.testing.assert_array_equal(out0, expected_c0)
    np.testing.assert_array_equal(out1, expected_c1)


def test_zero_shift_coarse_dm_passthrough_after_warmup():
    """When a coarse-DM has shift==0, its slice is the just-pushed
    slice (k=0, r=0)."""
    t_dedisp = 4
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    table = Stage2ShiftTable(
        chgroup=N_CHGROUP - 1,   # band-bottom: shifts must be zero
        coarse_dm_pc_cm3=np.array([100.0, 200.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.4,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        shifts_samples=np.array([0, 0], dtype=np.int32),
    )
    f = Stage2InterChgroupShiftFifo(
        chgroup=N_CHGROUP - 1,
        coarse_dm_pc_cm3=np.array([100.0, 200.0]),
        t_dedisp=t_dedisp,
        shift_table=table,
    )
    # Zero shift → k=0, r=0 → required depth = max_back-1 = 0. Warmed
    # up after first push.
    cube0 = _identity_cube(2, t_dedisp, 1, block_offset=0)
    out = f.push(cube0, block_n=0)
    assert len(out) == 1
    assert torch.equal(out[0], cube0)


def test_realistic_chgroup0_max_shift():
    """At chgroup=0 with the M7.4 production DM plan the worst shift
    is ~9100 samples = 17-18 cubes of T_dedisp=512. Validate that
    warm-up completes by push #19 and the output for block_n=19 is
    a sample-shifted version of the cubes pushed during warm-up.
    """
    t_dedisp = 512
    # Use just one coarse DM at the largest plan value to keep the
    # test focused + memory-cheap (one slice per cube).
    coarse = COARSE_DM_PROD[-1:].copy()
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=coarse,
        t_dedisp=t_dedisp,
    )
    delta = int(f.shifts_samples[0])
    assert delta > 2 * t_dedisp, (
        f"chgroup=0 shift={delta} samples should span multiple cubes "
        f"at the M7.4 plan"
    )
    k = delta // t_dedisp
    r = delta - k * t_dedisp
    # Push enough cubes to fill the ring + warm up. n_filled small to
    # keep RAM tiny.
    cubes = [
        _identity_cube(1, t_dedisp, 1, block_offset=b)
        for b in range(k + 2)
    ]
    for blk in range(k + 1):
        out = f.push(cubes[blk], block_n=blk)
        if not f.warmed_up():
            assert out == [], (
                f"push {blk}: should not emit before warm-up; got {len(out)}"
            )
    # By push k+1 we should be warmed up if r>0 (need ring[k+1]); for
    # r==0 we'd be warmed at push k.
    out = f.push(cubes[k + 1], block_n=k + 1)
    assert f.warmed_up() is True
    assert len(out) == 1
    # The output sample at out[0, 0, 0] equals the global sample at
    # absolute position (k+1)*T - delta = (k+1)*T - (k*T + r) = T - r.
    expected_first = float(t_dedisp - r)
    actual_first = float(out[0][0, 0, 0])
    assert actual_first == expected_first, (
        f"first output sample = {actual_first}, expected {expected_first}"
    )
    # The output last sample is at absolute position (k+1)*T + T - 1 - delta
    # = (k+1)*T + T - 1 - k*T - r = 2*T - 1 - r.
    expected_last = float(2 * t_dedisp - 1 - r)
    actual_last = float(out[0][0, -1, 0])
    assert actual_last == expected_last, (
        f"last output sample = {actual_last}, expected {expected_last}"
    )


def test_rejects_wrong_n_dm():
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0, 200.0]),
        t_dedisp=8,
    )
    bad = _identity_cube(3, 8, 1)  # wrong n_dm
    with pytest.raises(ValueError, match="shape\\[0\\]=3"):
        f.push(bad, block_n=0)


def test_rejects_wrong_t_dedisp():
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0]),
        t_dedisp=8,
    )
    bad = _identity_cube(1, 4, 1)  # wrong t_dedisp
    with pytest.raises(ValueError, match="shape\\[1\\]=4"):
        f.push(bad, block_n=0)


def test_rejects_wrong_dtype_after_first_push():
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0]),
        t_dedisp=8,
    )
    f.push(_identity_cube(1, 8, 1).to(torch.float32), block_n=0)
    bad = _identity_cube(1, 8, 1).to(torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        f.push(bad, block_n=1)


def test_rejects_shift_table_mismatch():
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    bad_table = Stage2ShiftTable(
        chgroup=1,   # different from constructor arg
        coarse_dm_pc_cm3=np.array([100.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.5,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        shifts_samples=np.array([5], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="chgroup"):
        Stage2InterChgroupShiftFifo(
            chgroup=0,
            coarse_dm_pc_cm3=np.array([100.0]),
            t_dedisp=8,
            shift_table=bad_table,
        )


def test_partial_emit_when_some_coarse_dms_warmed():
    """Two coarse-DMs with very different shifts. The smaller shift is
    warm by push #2; the larger by push #5. Between pushes 2 and 4 the
    FIFO emits a cube where the smaller coarse-DM has the rolled
    slice and the larger has zeros.
    """
    t_dedisp = 8
    from dsart.coarse_dm.stage2_shifts import Stage2ShiftTable
    table = Stage2ShiftTable(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0, 200.0], dtype=np.float64),
        nu_chgroup_bot_GHz=1.5,
        nu_bot_proc_GHz=1.4,
        t_int_corr_us=262.144,
        # shift 2 (warm at depth 2) vs shift 30 (k=3 r=6 → depth 5).
        shifts_samples=np.array([2, 30], dtype=np.int32),
    )
    f = Stage2InterChgroupShiftFifo(
        chgroup=0,
        coarse_dm_pc_cm3=np.array([100.0, 200.0]),
        t_dedisp=t_dedisp,
        shift_table=table,
    )
    cubes = [_identity_cube(2, t_dedisp, 1, block_offset=b) for b in range(6)]
    # Push 0: nothing warm.
    assert f.push(cubes[0], block_n=0) == []
    assert f.any_warmed_up() is False
    # Push 1: coarse_dm[0] warm (depth=2 for shift=2 → req=2). coarse_dm[1]
    # still needs depth 5.
    out = f.push(cubes[1], block_n=1)
    assert len(out) == 1, "expected partial-cube emit at push 1"
    assert f.any_warmed_up() is True
    assert f.warmed_up() is False
    assert f.per_dm_warmed() == [True, False]
    # coarse_dm[0]: rolled (non-zero) values; coarse_dm[1]: identically zero.
    out0 = out[0][0, :, 0].numpy()
    out1 = out[0][1, :, 0].numpy()
    assert (out0 != 0).any()
    np.testing.assert_array_equal(out1, np.zeros(t_dedisp, dtype=np.float32))
    # Push through till the big-shift coarse-DM warms too.
    for b in range(2, 5):
        f.push(cubes[b], block_n=b)
    out = f.push(cubes[5], block_n=5)
    # After push 5 the slow ring has 6 entries ≥ required 5.
    assert f.per_dm_warmed() == [True, True]
    assert (out[0][1, :, 0] != 0).any()


def test_max_ring_depth_matches_max_shift():
    coarse = COARSE_DM_PROD.copy()
    for g in range(N_CHGROUP):
        f = Stage2InterChgroupShiftFifo(
            chgroup=g,
            coarse_dm_pc_cm3=coarse,
            t_dedisp=512,
        )
        max_shift = int(f.shifts_samples.max())
        expected = max_shift // 512 + 1
        assert f.max_ring_depth_in_cubes == expected, (
            f"chgroup={g}: max_ring_depth={f.max_ring_depth_in_cubes}, "
            f"expected {expected} (max_shift={max_shift})"
        )

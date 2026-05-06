"""M3 chunk 7: 16-chgroup alignment preview — acceptance tests.

Pin the per-block intra-cube alignment that the chunk-4 service
inherits from the synchronous fada → unpack → kernel pipeline:
across 16 chgroups, fast-vis tile ``k`` corresponds to the same
native-sample window. This is what makes the production stage-2
``time_shift_corr_stage2`` (chunk 9) need only to compensate for
band-dependent geometric / dispersion residuals — the bulk-block
alignment is already correct.

The heavy multi-chgroup tests are GPU-gated (skipped on CPU because
the chunk-4 fast-corr GEMM at the production block size = 2048
packets × 96 ants × 384 chans takes ~90 s / chgroup on CPU; on GPU
the same correlation completes in ~50 ms). On h01 the dsa110-rt
env always has CUDA; on h23 / CI without GPU the heavy tests skip
cleanly. The arithmetic / synth-byte / single-chgroup-tile-shape
tests are lightweight and run unconditionally.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from chgroup_alignment_preview import (                                  # noqa: E402
    _expected_fast_vis_tile,
    _run_one_chgroup,
    _synth_block_with_impulse,
)
from dsart.common.constants import (                                     # noqa: E402
    FADA_BYTES_PER_BLOCK,
    N_CHGROUP,
)
from dsart.services.slow_corr_kernel import (                            # noqa: E402
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


def _pick_test_device() -> torch.device:
    """GPU when available; else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="multi-chgroup chunk-4 pipeline GEMMs are too slow on CPU "
           "(~90 s / chgroup); skipping until run on h01 GPU",
)


# ---------------------------------------------------------------------------
# _synth_block_with_impulse
# ---------------------------------------------------------------------------


def test_synth_block_size_matches_fada_block() -> None:
    raw = _synth_block_with_impulse(impulse_packet=100)
    assert raw.shape == (FADA_BYTES_PER_BLOCK,)
    assert raw.dtype == np.uint8


def test_synth_block_impulse_packet_has_max_int4() -> None:
    """Bytes at the impulse packet offset should be 0x77 (= int4 +7 +7j)."""
    impulse_pkt = 1234
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)
    from dsart.common.constants import (
        NANTS, NCHAN_PER_CHGROUP, NPOL,
    )
    packet_stride = NANTS * NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * NPOL
    impulse_bytes = raw[
        impulse_pkt * packet_stride:(impulse_pkt + 1) * packet_stride
    ]
    assert np.all(impulse_bytes == 0x77)


def test_synth_block_rejects_out_of_range_impulse() -> None:
    with pytest.raises(ValueError, match="impulse_packet"):
        _synth_block_with_impulse(impulse_packet=NPACKETS_PER_BLOCK)
    with pytest.raises(ValueError, match="impulse_packet"):
        _synth_block_with_impulse(impulse_packet=-1)


def test_synth_block_deterministic_under_same_seed() -> None:
    a = _synth_block_with_impulse(impulse_packet=500, seed=99)
    b = _synth_block_with_impulse(impulse_packet=500, seed=99)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# _expected_fast_vis_tile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "impulse_packet, t_int_fast_native, expected_tile",
    [
        # t_int_fast_native = 8 native samples = 4 packets per tile
        (0, 8, 0),
        (3, 8, 0),
        (4, 8, 1),
        (1000, 8, 250),
        (NPACKETS_PER_BLOCK - 1, 8, (NPACKETS_PER_BLOCK - 1) // 4),
        # t_int_fast_native = 32 native samples = 16 packets per tile
        # (4× burst-test cadence)
        (15, 32, 0),
        (16, 32, 1),
        (1000, 32, 62),
        # full-block boundary: t_int_fast_native = 4096 → 1 tile
        (0, 4096, 0),
        (NPACKETS_PER_BLOCK - 1, 4096, 0),
    ],
)
def test_expected_fast_vis_tile_arithmetic(
    impulse_packet: int, t_int_fast_native: int, expected_tile: int,
) -> None:
    assert _expected_fast_vis_tile(
        impulse_packet, t_int_fast_native,
    ) == expected_tile


# ---------------------------------------------------------------------------
# _run_one_chgroup — single-chgroup smoke
# ---------------------------------------------------------------------------


def _free_gpu_between_chgroups(device: torch.device) -> None:
    """Drop GPU caches so per-chgroup pipeline contexts (kernel +
    gridder + cal tensors) don't pile up across chgroups. Production
    runs reuse one IntegrationContext per chgroup; the test loop
    builds N contexts back-to-back, which would otherwise OOM the GPU.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


# Tile width used by the GPU-gated tests: 128 native samples = 32
# fast-vis tiles per block. The fast-corr kernel batches the per-block
# GEMM as ``(n_tiles * NCHAN * NTIMES_PER_PACKET * NPOL,
# packets_per_tile, NANTS)``; smaller ``t_int_fast_native`` means more
# parallel batches (and more GPU memory). At t_int=128 the batch
# tensors stay under ~1 GiB; at t_int=8 (the production cadence) the
# kernel currently allocates ~13 GiB before tiling, which OOMs the
# 11 GiB h01 GPU. The alignment property pinned by these tests is
# independent of t_int — see chunk 9's stage-2 alignment for the
# production-cadence acceptance.
_TEST_T_INT_FAST_NATIVE = 128


@_NEEDS_GPU
def test_run_one_chgroup_returns_tile_power_with_correct_shape() -> None:
    impulse_pkt = 1000
    t_int = _TEST_T_INT_FAST_NATIVE
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)
    device = _pick_test_device()
    vis_power = _run_one_chgroup(
        chgroup=0, raw=raw,
        n_grid=32, t_int_fast_native=t_int, obs_dec_deg=53.85,
        device=device,
    )
    _free_gpu_between_chgroups(device)
    expected_n_fv = (NPACKETS_PER_BLOCK * NTIMES_PER_PACKET) // t_int
    assert vis_power.shape == (expected_n_fv,)
    assert vis_power.dtype == torch.float32
    assert torch.all(vis_power >= 0)


@_NEEDS_GPU
def test_run_one_chgroup_peak_at_expected_tile_for_chgroup_0() -> None:
    """Single-chgroup baseline: the impulse should peak at the tile
    containing the impulse packet for the simplest case (chgroup 0).
    """
    impulse_pkt = 500                                                     # mid-block-ish
    t_int = _TEST_T_INT_FAST_NATIVE
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)
    device = _pick_test_device()
    vis_power = _run_one_chgroup(
        chgroup=0, raw=raw,
        n_grid=32, t_int_fast_native=t_int, obs_dec_deg=53.85,
        device=device,
    )
    _free_gpu_between_chgroups(device)
    expected_tile = _expected_fast_vis_tile(impulse_pkt, t_int)
    peak_tile = int(torch.argmax(vis_power).item())
    assert abs(peak_tile - expected_tile) <= 1, (
        f"impulse should peak at tile {expected_tile} (±1); peaked at {peak_tile}. "
        f"vis_power neighborhood: "
        f"{vis_power[max(0, expected_tile - 2):expected_tile + 3].tolist()}"
    )


# ---------------------------------------------------------------------------
# 16-chgroup alignment — the headline pin
# ---------------------------------------------------------------------------


@_NEEDS_GPU
def test_16chgroups_all_peak_at_same_tile() -> None:
    """The headline chunk-7 pin: when we feed the SAME synthetic
    impulse-block to all 16 chgroups, the per-chgroup peak fast-vis
    tile must be identical (or differ by ≤ 1 tile) across all 16
    chgroups. This proves the per-block intra-cube alignment that
    chunk 9's stage-2 alignment can rely on.

    Uses a small grid (n_grid=16) for speed; the alignment property
    is independent of n_grid.
    """
    impulse_pkt = 800
    t_int = _TEST_T_INT_FAST_NATIVE
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)
    expected_tile = _expected_fast_vis_tile(impulse_pkt, t_int)

    device = _pick_test_device()
    peak_tiles: list[int] = []
    for chg in range(N_CHGROUP):
        vis_power = _run_one_chgroup(
            chgroup=chg, raw=raw,
            n_grid=16, t_int_fast_native=t_int, obs_dec_deg=53.85,
            device=device,
        )
        peak_tiles.append(int(torch.argmax(vis_power).item()))
        _free_gpu_between_chgroups(device)

    offsets = [pt - expected_tile for pt in peak_tiles]
    max_abs_offset = max(abs(o) for o in offsets)
    assert max_abs_offset <= 1, (
        f"chgroup-alignment FAIL: peak tiles per chgroup = {peak_tiles}; "
        f"expected = {expected_tile}; max abs offset = {max_abs_offset} "
        f"(criterion ≤ 1)"
    )
    # All 16 peaks should concentrate within a 3-tile window.
    assert max(peak_tiles) - min(peak_tiles) <= 2, (
        f"peak tiles span > 2 across 16 chgroups: {peak_tiles}"
    )


@_NEEDS_GPU
def test_16chgroups_peak_tile_invariant_under_t_int_change() -> None:
    """The alignment property holds at the burst-test 4×-coarser
    cadence (t_int_fast_native = 4 × _TEST_T_INT_FAST_NATIVE). The
    invariance shows that chunk-9's stage-2 alignment can swap
    cadences without disturbing intra-cube alignment.
    """
    impulse_pkt = 800
    t_int = 4 * _TEST_T_INT_FAST_NATIVE                                   # = 512 native samples
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)
    expected_tile = _expected_fast_vis_tile(impulse_pkt, t_int)

    device = _pick_test_device()
    # Sample every 4th chgroup for speed (still validates the property).
    peak_tiles: list[int] = []
    for chg in (0, 4, 8, 12, 15):
        vis_power = _run_one_chgroup(
            chgroup=chg, raw=raw,
            n_grid=16, t_int_fast_native=t_int, obs_dec_deg=53.85,
            device=device,
        )
        peak_tiles.append(int(torch.argmax(vis_power).item()))
        _free_gpu_between_chgroups(device)

    offsets = [pt - expected_tile for pt in peak_tiles]
    max_abs_offset = max(abs(o) for o in offsets)
    assert max_abs_offset <= 1, (
        f"peak tiles at t_int={t_int} = {peak_tiles}; "
        f"expected = {expected_tile}; max abs offset = {max_abs_offset}"
    )


@_NEEDS_GPU
def test_chgroup_0_and_15_peak_within_1_tile() -> None:
    """Targeted edge-case test: chgroup 0 (lowest band) and chgroup 15
    (highest band) — the largest possible band-dependent delay
    difference — should still align within 1 fast-vis tile.
    """
    impulse_pkt = 1500
    t_int = _TEST_T_INT_FAST_NATIVE
    raw = _synth_block_with_impulse(impulse_packet=impulse_pkt)

    device = _pick_test_device()
    p0 = _run_one_chgroup(
        chgroup=0, raw=raw,
        n_grid=16, t_int_fast_native=t_int, obs_dec_deg=53.85,
        device=device,
    )
    _free_gpu_between_chgroups(device)
    p15 = _run_one_chgroup(
        chgroup=15, raw=raw,
        n_grid=16, t_int_fast_native=t_int, obs_dec_deg=53.85,
        device=device,
    )
    _free_gpu_between_chgroups(device)

    pt0 = int(torch.argmax(p0).item())
    pt15 = int(torch.argmax(p15).item())
    assert abs(pt0 - pt15) <= 1, (
        f"chgroup 0 vs 15 peak tile mismatch: {pt0} vs {pt15} "
        f"(differ by {pt15 - pt0})"
    )

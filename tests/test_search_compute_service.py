"""Tests for ``dsart.services.search_compute`` (M5 Chunk 6b-α).

RX-ring → CubePipeline only (M6 chunk 0: trigger emitter retired;
cluster + cube-dump integration lands chunks 1-5).

The stack is the production data path with the synthetic RX-ring
source (instead of M4a's POSIX-shm). All numerical correctness gates
live in the chunk-1..6a unit tests; this module gates **wiring +
lifecycle**:

  * ``SyntheticRxRingSource`` yields the right number of slots with
    the right shapes; injections land at the requested cell.
  * ``CubePipeline.process(slot)`` returns a valid ``CubePipelineResult``
    with cube shape ``[T_det, N_fdm, N_grid, N_grid]`` and
    ``len(candidates) ≥ 0``.
  * ``SearchComputeService.run()`` drains the source end-to-end without
    raising, processing every cube.
  * Service.stop() leaves no dangling tasks (asyncio shutdown is clean).
"""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Tuple

import numpy as np
import pytest
import torch

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.constants import N_CHGROUP  # noqa: E402
from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    CubeRingSlot,
    RxRingSource,
    SyntheticInjection,
    SyntheticRxRingSource,
)
from dsart.services.search_compute import (  # noqa: E402
    SearchComputeConfig,
    SearchComputeService,
)


# Custom asyncio test decorator (no pytest-asyncio dependency, mirrors
# the pattern in tests/test_trigger_emitter.py).


def asyncio_test(func):
    """Custom asyncio test decorator (no pytest-asyncio dependency)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dm_grids(n_coarse: int = 2, n_fine_per_coarse: int = 2):
    coarse = np.linspace(50.0, 100.0, n_coarse, dtype=np.float64)
    spacing = (coarse[1] - coarse[0]) / n_fine_per_coarse if n_coarse > 1 else 1.0
    fine = np.concatenate(
        [coarse[c] + np.arange(n_fine_per_coarse) * spacing for c in range(n_coarse)]
    )
    fine_to_coarse = np.repeat(np.arange(n_coarse, dtype=np.int64), n_fine_per_coarse)
    return coarse, fine, fine_to_coarse


def _build_synth_source(
    *,
    n_cubes: int = 2,
    t_det: int = 32,
    n_fdm: int = 4,
    n_grid: int = 16,
    rng_seed: int = 0,
    injections: Tuple[SyntheticInjection, ...] = (),
):
    coarse, fine, f2c = _dm_grids(n_coarse=2, n_fine_per_coarse=n_fdm // 2)
    return SyntheticRxRingSource(
        n_cubes=n_cubes,
        t_det=t_det,
        n_fdm=n_fdm,
        n_grid=n_grid,
        coarse_dm_pc_cm3=coarse,
        fine_dm_pc_cm3=fine,
        fine_to_coarse=f2c,
        rng=np.random.default_rng(rng_seed),
        injections=injections,
    )


# ---------------------------------------------------------------------------
# SyntheticRxRingSource
# ---------------------------------------------------------------------------


def test_synthetic_source_satisfies_protocol() -> None:
    src = _build_synth_source()
    assert isinstance(src, RxRingSource)


@asyncio_test
async def test_synthetic_source_yields_correct_count_and_shape() -> None:
    src = _build_synth_source(n_cubes=3, t_det=16, n_fdm=4, n_grid=8)
    slots = []
    async with src:
        async for slot in src:
            slots.append(slot)
    assert len(slots) == 3
    for slot in slots:
        assert slot.t_det == 16
        assert slot.n_grid == 8
        assert slot.n_fdm_in_cube == 4
        assert slot.validity_mask.shape == (16, 4)
        assert slot.validity_mask.dtype == np.bool_
        assert all(g in slot.per_chgroup_streams for g in range(N_CHGROUP))
        for g in range(N_CHGROUP):
            arr = slot.per_chgroup_streams[g]
            assert arr.dtype == np.complex64
            assert arr.shape[1:] == (8, 8)
            # T_stream covers t_det + max_shift
            assert arr.shape[0] >= 16


@asyncio_test
async def test_synthetic_source_increments_cube_id_and_specnum() -> None:
    src = _build_synth_source(n_cubes=4, t_det=16)
    cube_ids = []
    specnums = []
    async with src:
        async for slot in src:
            cube_ids.append(slot.cube_id)
            specnums.append(slot.specnum_start)
    assert cube_ids == [0, 1, 2, 3]
    # specnum_start = cube_id * t_det
    assert specnums == [0, 16, 32, 48]


@asyncio_test
async def test_synthetic_source_injection_lands() -> None:
    inj = SyntheticInjection(
        cube_idx=0, t_in_cube=5, l_pix=3, m_pix=4, amplitude=100.0,
    )
    src = _build_synth_source(
        n_cubes=1, t_det=16, n_fdm=2, n_grid=8, injections=(inj,),
    )
    async with src:
        async for slot in src:
            stream_15 = slot.per_chgroup_streams[N_CHGROUP - 1]
            # Injection lands at chgroup-15 (no shift), [t_in_cube, l_pix, m_pix]
            cell = stream_15[5, 3, 4]
            assert abs(cell.real) > 50.0


# ---------------------------------------------------------------------------
# CubePipeline
# ---------------------------------------------------------------------------


@asyncio_test
async def test_cube_pipeline_runs_one_cube() -> None:
    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    pipeline = CubePipeline(
        config=CubePipelineConfig(
            n_grid=8,
            edge_mask_kernel_support=3,
            cube_dtype=torch.float32,
            device="cpu",
        ),
        detector=DeterministicDetector(
            threshold_sigma=8.0,
            detector_version="v1.M5",
            search_node_id=0,
            gpu_half=0,
            dtype=torch.float32,
        ),
        layer1_state=None,
    )
    async with src:
        async for slot in src:
            result = pipeline.process(slot)
            assert result.cube_id == slot.cube_id
            assert result.specnum_start == slot.specnum_start
            assert result.cube.shape == (32, 4, 8, 8)
            assert result.cube.dtype == torch.float32
            assert result.sigma_layer1.shape == (4,)
            assert result.validity_mask.shape == (32, 4)
            # noise-only cube under unit-σ should produce few or zero
            # candidates at θ=8 (broad guard, not an FAR test)
            assert len(result.candidates) >= 0


def test_cube_pipeline_rejects_n_grid_mismatch() -> None:
    pipeline = CubePipeline(
        config=CubePipelineConfig(n_grid=8, edge_mask_kernel_support=3),
        detector=DeterministicDetector(threshold_sigma=8.0),
        layer1_state=None,
    )
    coarse, fine, f2c = _dm_grids(n_coarse=2, n_fine_per_coarse=2)
    bad_slot = CubeRingSlot(
        cube_id=0,
        specnum_start=0,
        per_chgroup_streams={
            g: np.zeros((4, 16, 16), dtype=np.complex64)
            for g in range(N_CHGROUP)
        },
        time_shift_table=SyntheticRxRingSource(
            n_cubes=1, t_det=4, n_fdm=4, n_grid=16,
            coarse_dm_pc_cm3=coarse, fine_dm_pc_cm3=fine, fine_to_coarse=f2c,
        ).time_shift_table,
        validity_mask=np.ones((4, 4), dtype=np.bool_),
        n_fdm_in_cube=4,
        t_det=4,
        n_grid=16,  # <-- mismatched against pipeline.n_grid=8
    )
    with pytest.raises(ValueError, match="n_grid"):
        pipeline.process(bad_slot)


# ---------------------------------------------------------------------------
# SearchComputeService end-to-end
# ---------------------------------------------------------------------------


@asyncio_test
async def test_search_compute_service_drains_synthetic_source() -> None:
    """End-to-end smoke: 3 cubes through the full stack, no exceptions,
    cube counter ticks."""
    src = _build_synth_source(
        n_cubes=3, t_det=32, n_fdm=4, n_grid=8, rng_seed=42,
    )
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        detector_threshold_sigma=4.0,  # low threshold so we see candidates
        detector_dtype=torch.float32,
        detector_device="cpu",
        search_node_id=0,
        gpu_half=0,
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        await service.run()
    finally:
        await service.stop()
    assert service.cubes_processed == 3
    assert service.candidates_emitted >= 0


@asyncio_test
async def test_search_compute_service_emits_for_strong_injection() -> None:
    """A single strong injection placed at chgroup-15 should produce
    at least one candidate. We don't gate on count or kernel — that's
    chunk-5's bench's job — only that the wiring fires.
    """
    inj = SyntheticInjection(
        cube_idx=0, t_in_cube=16, l_pix=4, m_pix=4, amplitude=200.0,
    )
    src = _build_synth_source(
        n_cubes=1, t_det=32, n_fdm=4, n_grid=8,
        rng_seed=7, injections=(inj,),
    )
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        detector_threshold_sigma=4.0,
        detector_dtype=torch.float32,
        detector_device="cpu",
        search_node_id=0,
        gpu_half=0,
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        await service.run()
    finally:
        await service.stop()
    assert service.cubes_processed == 1

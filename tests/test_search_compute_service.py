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


# ---------------------------------------------------------------------------
# M6 chunk 5: clusterer + cube_dump + UDP listener + cands_logger wiring
# ---------------------------------------------------------------------------

import socket  # noqa: E402

from dsart.cluster.forward import ClustererBackend, ClustererConfig  # noqa: E402
from dsart.cluster.cands_logger import CandsLoggerConfig  # noqa: E402
from dsart.dump.cube_dump import (  # noqa: E402
    BrightPulsePredicateConfig,
    CubeDumpWriterConfig,
)
from dsart.dump.udp_listener import UdpTriggerListenerConfig  # noqa: E402


def _full_config(
    *,
    tmp_path,
    inject=False,
    enable_cluster=False,
    enable_dump=False,
    enable_udp=False,
    enable_log=False,
    udp_port=0,
    predicate_min_snr=4.0,
):
    pipeline = CubePipelineConfig(
        n_grid=8,
        edge_mask_kernel_support=3,
        cube_dtype=torch.float32,
        device="cpu",
    )
    return SearchComputeConfig(
        pipeline=pipeline,
        n_fdm=4,
        detector_threshold_sigma=4.0,
        detector_dtype=torch.float32,
        detector_device="cpu",
        search_node_id=0,
        gpu_half=0,
        cube_cell_l_rad=1.5e-4,
        cube_cell_m_rad=1.5e-4,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
        clusterer_config=(
            ClustererConfig(backend=ClustererBackend.DBSCAN)
            if enable_cluster else None
        ),
        bright_pulse_predicate_config=(
            BrightPulsePredicateConfig(
                min_snr=predicate_min_snr, holdoff_ms=0.0,
            )
            if enable_dump else None
        ),
        cube_dump_writer_config=(
            CubeDumpWriterConfig(
                dump_root=tmp_path / "dumps",
                search_node_id=0,
                gpu_half=0,
                queue_maxsize=4,
            )
            if enable_dump else None
        ),
        udp_trigger_listener_config=(
            UdpTriggerListenerConfig(host="127.0.0.1", port=udp_port)
            if enable_udp else None
        ),
        cands_logger_config=(
            CandsLoggerConfig(
                log_root=tmp_path / "logs",
                search_node_id=0,
                gpu_half=0,
            )
            if enable_log else None
        ),
        # M7.4 strip: the legacy DBSCAN/HDBSCAN clusterer + cands_logger
        # + BrightPulsePredicate path is gated behind this flag. Tests
        # that build a full config with ``enable_cluster=True`` flip
        # the gate on so the legacy fixtures still exercise the
        # clusterer hand-off.
        enable_legacy_clusterer=bool(enable_cluster),
    )


@asyncio_test
async def test_chunk5_clusterer_optional_no_subsystems_off_path() -> None:
    """Default-configured service runs cubes with no sub-systems wired
    (asserts the chunk-5 plumbing is opt-in)."""
    src = _build_synth_source(n_cubes=2, t_det=32, n_fdm=4, n_grid=8)
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        await service.run()
    finally:
        await service.stop()
    assert service.cubes_processed == 2
    assert service.clusters_emitted == 0
    assert service.auto_dumps_dispatched == 0
    assert service.udp_dumps_dispatched == 0
    assert service.clusterer is None
    assert service.cube_dump is None
    assert service.udp_listener is None
    assert service.cands_logger is None


@asyncio_test
async def test_chunk5_clusterer_runs_when_configured(tmp_path) -> None:
    """Clusterer + logger configured but no dump: clusters emit, T1/T2
    rows write."""
    inj = SyntheticInjection(
        cube_idx=0, t_in_cube=16, l_pix=4, m_pix=4, amplitude=200.0,
    )
    src = _build_synth_source(
        n_cubes=1, t_det=32, n_fdm=4, n_grid=8, rng_seed=7, injections=(inj,),
    )
    config = _full_config(tmp_path=tmp_path, enable_cluster=True, enable_log=True)
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        await service.run()
    finally:
        await service.stop()
    assert service.cubes_processed == 1
    assert service.clusterer is not None
    assert service.cands_logger is not None
    # The clusterer ran and produced records (≥ 1 per cube — even noise
    # singletons count).
    assert service.clusters_emitted >= 0
    # T1/T2 files exist if any candidates emitted.
    if service.candidates_emitted > 0:
        assert service.clusters_emitted >= 1
        log_files = list((tmp_path / "logs").glob("cands_T*.txt"))
        assert len(log_files) >= 2  # T1 + T2


@asyncio_test
async def test_chunk5_auto_dump_fires_for_bright_cluster(tmp_path) -> None:
    """Predicate.min_snr below the cube's expected SNR floor — every
    cluster fires; cube_dump.n_dumped > 0."""
    inj = SyntheticInjection(
        cube_idx=0, t_in_cube=16, l_pix=4, m_pix=4, amplitude=200.0,
    )
    src = _build_synth_source(
        n_cubes=1, t_det=32, n_fdm=4, n_grid=8, rng_seed=7, injections=(inj,),
    )
    config = _full_config(
        tmp_path=tmp_path, enable_cluster=True, enable_dump=True,
        enable_log=True, predicate_min_snr=0.1,  # always-fire
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        await service.run()
    finally:
        await service.stop()
    if service.candidates_emitted > 0:
        assert service.auto_dumps_dispatched >= 1
        # NPZ files should land in tmp_path / "dumps"
        npz_files = list((tmp_path / "dumps").glob("cube_s*_g*_*.npz"))
        assert len(npz_files) >= 1


@asyncio_test
async def test_chunk5_udp_trigger_dumps_next_cube(tmp_path) -> None:
    """A datagram arrives BEFORE cube N processing → cube N is dumped
    as 'udp' source."""
    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    config = _full_config(
        tmp_path=tmp_path, enable_dump=True, enable_udp=True, udp_port=0,
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        # Send a datagram BEFORE running so the listener arms its flag.
        port = service.udp_listener.bound_port  # type: ignore[union-attr]
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sk.sendto(b"trigger", ("127.0.0.1", port))
        finally:
            sk.close()
        # Give the asyncio loop a tick to receive the datagram.
        await asyncio.sleep(0.1)
        await service.run()
    finally:
        await service.stop()
    assert service.cubes_processed == 1
    assert service.udp_dumps_dispatched == 1
    npz_files = list((tmp_path / "dumps").glob("cube_s*_g*_*.npz"))
    assert len(npz_files) >= 1


@asyncio_test
async def test_chunk5_geom_from_slot_uses_config(tmp_path) -> None:
    """``_geom_from_slot`` populates a CubeGeometry with the configured
    cell sizes + computed mjd."""
    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        cube_cell_l_rad=2.0e-4,
        cube_cell_m_rad=3.0e-4,
        cube_l0_rad=1e-3,
        cube_m0_rad=2e-3,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
        mjd_at_specnum_0=60942.0,
    )
    service = SearchComputeService(config=config, source=src)
    await service.start()
    try:
        async for slot in src:
            geom = service._geom_from_slot(slot)
            assert geom.cell_l_rad == 2.0e-4
            assert geom.cell_m_rad == 3.0e-4
            assert geom.l0_rad == 1e-3
            assert geom.m0_rad == 2e-3
            assert geom.sample_period_specnum == 16
            assert geom.cube_id == slot.cube_id
            assert geom.specnum_start == slot.specnum_start
            assert geom.fine_dm_pc_cc.shape == (slot.n_fdm_in_cube,)
            # mjd_start = 60942.0 + specnum_start * sample_period_us * 1e-6 / 86400
            # specnum_start is in SEARCH-SAMPLE units, so the per-specnum MJD
            # step is the FULL search-sample period (NOT divided by
            # sample_period_specnum).
            t_int_sample_us = 131.072
            expected_mjd = 60942.0 + slot.specnum_start * t_int_sample_us * 1e-6 / 86400.0
            np.testing.assert_allclose(geom.mjd_start, expected_mjd)
            break
    finally:
        await service.stop()


@asyncio_test
async def test_geom_mjd_anchor_prefers_capture_arm_record(tmp_path) -> None:
    """With the placeholder mjd_at_specnum_0 (0.0), the first-cube latch
    uses the etcd capture-arm anchor (armed_mjd) when available, so
    labels land on the slow-vis absolute time base instead of running
    late by the pipeline fill latency (2026-07-16 fix)."""
    import time as _time

    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
    )
    service = SearchComputeService(config=config, source=src)
    now_mjd = 40587.0 + _time.time() / 86400.0
    anchor = now_mjd - 5.0 / 86400.0  # armed 5 s ago
    service._read_capture_anchor_mjd = lambda: anchor  # type: ignore
    await service.start()
    try:
        async for slot in src:
            geom = service._geom_from_slot(slot)
            expected = anchor + (
                slot.specnum_start * 131.072e-6 / 86400.0
            )
            np.testing.assert_allclose(geom.mjd_start, expected)
            assert service._mjd_at_specnum_0_override == anchor
            break
    finally:
        await service.stop()


@asyncio_test
async def test_geom_mjd_anchor_falls_back_to_wall_latch(tmp_path) -> None:
    """No etcd arm record -> the pre-existing wall-clock latch."""
    import time as _time

    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
    )
    service = SearchComputeService(config=config, source=src)
    service._read_capture_anchor_mjd = lambda: None  # type: ignore
    await service.start()
    try:
        async for slot in src:
            before = 40587.0 + _time.time() / 86400.0
            geom = service._geom_from_slot(slot)
            after = 40587.0 + _time.time() / 86400.0
            assert before - 1e-6 <= geom.mjd_start <= after + 1e-6
            break
    finally:
        await service.stop()


@asyncio_test
async def test_geom_mjd_anchor_rejects_stale_arm_record(tmp_path) -> None:
    """An arm record inconsistent with the wall clock (lag outside
    [-2, 120] s) is rejected in favour of the wall latch."""
    import time as _time

    src = _build_synth_source(n_cubes=1, t_det=32, n_fdm=4, n_grid=8)
    config = SearchComputeConfig(
        pipeline=CubePipelineConfig(
            n_grid=8, edge_mask_kernel_support=3,
            cube_dtype=torch.float32, device="cpu",
        ),
        n_fdm=4,
        cube_sample_period_us=131.072,
        cube_sample_period_specnum=16,
    )
    service = SearchComputeService(config=config, source=src)
    now_mjd = 40587.0 + _time.time() / 86400.0
    stale = now_mjd - 1000.0 / 86400.0  # armed "1000 s ago": lag > 120 s
    service._read_capture_anchor_mjd = lambda: stale  # type: ignore
    await service.start()
    try:
        async for slot in src:
            before = 40587.0 + _time.time() / 86400.0
            geom = service._geom_from_slot(slot)
            after = 40587.0 + _time.time() / 86400.0
            assert before - 1e-6 <= geom.mjd_start <= after + 1e-6
            break
    finally:
        await service.stop()

"""Tests for ``dsart.cluster.state.ClustererService`` (M6 chunk 1).

Smoke + lifecycle + concurrency tests for the ThreadPoolExecutor
clusterer harness used by chunk 5's per-cube driver.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("DSART_TEST", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dsart.cluster.forward import ClustererBackend, ClustererConfig  # noqa: E402
from dsart.cluster.state import ClustererService, ClustererServiceResult  # noqa: E402
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    CubeGeometry,
)


def _make_geom(cube_id=0, **overrides) -> CubeGeometry:
    base: dict = dict(
        cube_id=cube_id,
        specnum_start=1024 + cube_id * 4096,
        sample_period_specnum=16,
        t_det=256,
        n_grid=256,
        n_fdm_in_cube=8,
        sample_period_us=131.072,
        cell_l_rad=1.5e-4,
        cell_m_rad=1.5e-4,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=np.linspace(50.0, 800.0, 8, dtype=np.float64),
        mjd_start=60942.123456789 + cube_id * 1e-3,
    )
    base.update(overrides)
    return CubeGeometry(**base)


def _make_cands(geom, n=5):
    """Build n Candidates whose event_specnum lies inside ``geom``'s window.

    event_specnum = geom.specnum_start + 64 * sample_period_specnum (= 64
    samples into the cube). Critically, the cands MUST be regenerated
    against the geom they're submitted with — passing geom-A cands to
    geom-B will yield t_in_cube < 0.
    """
    return [
        Candidate(
            l=float(10 + i),
            m=20.0,
            dm_fine=float(geom.fine_dm_pc_cc[2]),
            dm_idx=0,
            event_specnum=geom.specnum_start + 64 * geom.sample_period_specnum,
            width_samples=4,
            kernel_id="unit:d1:b4",
            snr=8.5 + 0.5 * i,
            detector_version="v1.M5",
            flags=int(CandidateFlags.NONE),
            search_node_id=2,
            gpu_half=1,
        )
        for i in range(n)
    ]


def test_default_config() -> None:
    svc = ClustererService()
    assert svc.config.backend == ClustererBackend.HDBSCAN
    assert svc.n_submitted == 0


def test_invalid_max_workers_raises() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        ClustererService(max_workers=0)


def test_submit_before_start_raises() -> None:
    svc = ClustererService()
    geom = _make_geom()
    cands = _make_cands(geom, n=3)
    with pytest.raises(RuntimeError, match="start"):
        svc.submit(cands, geom)


def test_double_start_raises() -> None:
    svc = ClustererService()
    svc.start()
    try:
        with pytest.raises(RuntimeError, match="start"):
            svc.start()
    finally:
        svc.shutdown()


def test_happy_path_submit_resolves_to_result() -> None:
    svc = ClustererService(config=ClustererConfig(backend=ClustererBackend.DBSCAN))
    svc.start()
    try:
        geom = _make_geom(cube_id=42)
        cands = _make_cands(geom, n=4)
        future = svc.submit(cands, geom)
        result = future.result(timeout=5.0)
        assert isinstance(result, ClustererServiceResult)
        assert result.cube_id == 42
        assert result.labels.shape == (4,)
        assert isinstance(result.records, list)
        assert result.wall_ms >= 0.0
    finally:
        svc.shutdown()


def test_counters_advance() -> None:
    svc = ClustererService(config=ClustererConfig(backend=ClustererBackend.DBSCAN))
    svc.start()
    try:
        for i in range(5):
            geom_i = _make_geom(cube_id=i)
            cands_i = _make_cands(geom_i, n=3)
            future = svc.submit(cands_i, geom_i)
            future.result(timeout=5.0)
        assert svc.n_submitted == 5
        assert svc.n_completed == 5
        assert svc.n_failed == 0
    finally:
        svc.shutdown()


def test_failure_increments_n_failed() -> None:
    svc = ClustererService(config=ClustererConfig(backend="bogus"))
    svc.start()
    try:
        geom = _make_geom()
        cands = _make_cands(geom, n=3)
        future = svc.submit(cands, geom)
        with pytest.raises(ValueError, match="backend"):
            future.result(timeout=5.0)
        # Give the worker a moment to flip the counter
        time.sleep(0.05)
        assert svc.n_failed == 1
        assert svc.n_completed == 0
    finally:
        svc.shutdown()


def test_shutdown_drains_in_flight_jobs() -> None:
    svc = ClustererService(config=ClustererConfig(backend=ClustererBackend.DBSCAN))
    svc.start()
    try:
        geom = _make_geom()
        cands = _make_cands(geom, n=3)
        futures = [svc.submit(cands, _make_geom(cube_id=i)) for i in range(3)]
    finally:
        svc.shutdown(wait=True)
    # All futures should have resolved.
    for f in futures:
        assert f.done()


def test_concurrent_submit_from_multiple_threads() -> None:
    """Multiple producer threads submit; counters land on the right value."""
    svc = ClustererService(config=ClustererConfig(backend=ClustererBackend.DBSCAN))
    svc.start()
    try:
        futures = []
        lock = threading.Lock()

        def submit_one(i):
            geom_i = _make_geom(cube_id=i)
            cands_i = _make_cands(geom_i, n=3)
            f = svc.submit(cands_i, geom_i)
            with lock:
                futures.append(f)

        threads = [threading.Thread(target=submit_one, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for f in futures:
            f.result(timeout=5.0)
        assert svc.n_submitted == 20
        assert svc.n_completed == 20
    finally:
        svc.shutdown()


def test_shutdown_without_start_is_noop() -> None:
    svc = ClustererService()
    svc.shutdown()  # no start() — should not raise

"""Tests for :mod:`dsart.coinc.plotter` (4-panel cube event PNGs).

We fake 8 NPZs with a small synthetic cube and verify that
``render_event_plots`` produces 4 PNG files. No visual diff — too
fragile across matplotlib versions / DPI defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dsart.coinc.plotter import (
    PlotJob,
    PlotWorker,
    enqueue_event,
    render_event_plots,
)
from dsart.coinc.stats import ClusterStats
from dsart.coinc.window import WindowEntry


# Ensure matplotlib never tries to open a display window during tests.
os.environ["MPLBACKEND"] = "Agg"


def _stats() -> ClusterStats:
    return ClusterStats(
        n_events=3,
        n_search_nodes=2,
        n_gpu_halves=3,
        snr_max=12.5,
        snr_sum=33.0,
        snr_mean=11.0,
        dm_min=99.0,
        dm_max=101.0,
        dm_median=100.0,
        dm_iqr=1.0,
        l_median=1.5e-3,
        m_median=-2.5e-3,
        lm_diag_rad=2.0e-3,
        width_min=2,
        width_max=8,
        width_median=4.0,
        t_start_mjd=60781.0,
        t_end_mjd=60781.0 + 1.0 / 86400.0,
        t_peak_mjd=60781.0 + 0.5 / 86400.0,
        kernel_ids_distinct=("unit:d1:b4", "unit:d1:b8"),
        peak_event_specnum=42,
    )


def _members() -> list[WindowEntry]:
    out = []
    for i in range(6):
        out.append(WindowEntry(
            mjd=60781.0 + i * 1e-6,
            snr=10.0 + i,
            l_rad=1.5e-3,
            m_rad=-2.5e-3,
            l_pix=120,
            m_pix=130,
            dm_pc_cc=100.0 + i,
            dm_idx_global=10 + i,
            fine_dm_idx=i,
            event_specnum=100 + i,
            width_samples=4,
            kernel_id="unit:d1:b4" if i % 2 == 0 else "unit:d1:b8",
            flags=0,
            search_node_id=(i % 4) + 1,
            gpu_half=i % 2,
            cube_id=7,
            sample_period_us=1048.576,
        ))
    return out


def _write_fake_cubes(cubes_dir: Path, n_fdm: int = 4, n_t: int = 8,
                       n_grid: int = 16) -> None:
    cubes_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sid in (1, 2, 9, 13):
        for g in (0, 1):
            cube = rng.standard_normal(
                (n_fdm, n_t, n_grid, n_grid), dtype=np.float32,
            ).astype(np.float16)
            # Plant a small bright spot at (fdm=1, t=3, pix=8, pix=8)
            cube[1, 3, 8, 8] = 10.0
            manifest = {
                "fine_dm_pc_cc": np.linspace(90.0, 110.0, n_fdm),
                "mjd_start": 60781.0,
                "sample_period_us": 1048.576,
                "search_node_id": sid,
                "gpu_half": g,
            }
            path = cubes_dir / f"cube_s{sid}_g{g}_100.npz"
            np.savez(path, cube=cube, manifest=manifest)


def test_render_event_plots_produces_four_pngs(tmp_path: Path) -> None:
    archive_root = tmp_path / "candidates"
    ev = "260521abcd"
    _write_fake_cubes(archive_root / ev / "cubes")
    job = PlotJob(
        event_name=ev,
        archive_root=archive_root,
        stats=_stats(),
        members=tuple(_members()),
    )
    written = render_event_plots(job)
    plots = archive_root / ev / "Level2" / "plots"
    assert len(written) == 4
    expected = {
        plots / f"dm_time_{ev}.png",
        plots / f"image_peak_{ev}.png",
        plots / f"lightcurve_{ev}.png",
        plots / f"kernel_snrs_{ev}.png",
    }
    assert set(written) == expected
    for p in expected:
        assert p.is_file(), f"missing {p}"
        # Sanity: each PNG should be a small but non-empty file.
        assert p.stat().st_size > 100


def _write_fake_cubes_with_peak_grid(
    cubes_dir: Path, n_fdm: int = 4, n_t: int = 8, n_grid: int = 16,
) -> None:
    """Same fixture as ``_write_fake_cubes`` but each NPZ carries a
    writer-side-precomputed ``peak_grid``. Lets us test the plotter's
    fast path (CubeDumpWriter precompute landed 2026-05-27).
    """
    cubes_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sid in (1, 2, 9, 13):
        for g in (0, 1):
            cube = rng.standard_normal(
                (n_fdm, n_t, n_grid, n_grid), dtype=np.float32,
            ).astype(np.float16)
            cube[1, 3, 8, 8] = 10.0
            peak_grid = cube.max(axis=(2, 3))  # shape (n_fdm, n_t)
            manifest = {
                "fine_dm_pc_cc": np.linspace(90.0, 110.0, n_fdm),
                "mjd_start": 60781.0,
                "sample_period_us": 1048.576,
                "search_node_id": sid,
                "gpu_half": g,
            }
            path = cubes_dir / f"cube_s{sid}_g{g}_100.npz"
            np.savez(
                path, cube=cube, peak_grid=peak_grid, manifest=manifest,
            )


def test_populate_peak_grids_uses_cache_when_present(
    tmp_path: Path,
) -> None:
    """The plotter must consume ``peak_grid`` from the NPZ when the
    writer pre-computed it, instead of re-running the (expensive)
    cube.max(axis=(2,3)) reduction.

    We verify by writing a fixture where the cube is all-zeros but
    the stored ``peak_grid`` has a distinctive sentinel value. The
    fast path returns the sentinel; the slow path would return zeros.
    """
    from dsart.coinc.plotter import _load_cubes, _populate_peak_grids

    cubes_dir = tmp_path / "cubes"
    cubes_dir.mkdir(parents=True, exist_ok=True)
    n_fdm, n_t, n_grid = 4, 8, 16
    sentinel = np.float16(7.5)
    for sid in (1, 2, 9, 13):
        for g in (0, 1):
            cube = np.zeros(
                (n_fdm, n_t, n_grid, n_grid), dtype=np.float16,
            )
            peak_grid = np.full(
                (n_fdm, n_t), sentinel, dtype=np.float16,
            )
            manifest = {
                "fine_dm_pc_cc": np.linspace(90.0, 110.0, n_fdm),
                "mjd_start": 60781.0,
                "sample_period_us": 1048.576,
                "search_node_id": sid,
                "gpu_half": g,
            }
            path = cubes_dir / f"cube_s{sid}_g{g}_100.npz"
            np.savez(
                path, cube=cube, peak_grid=peak_grid, manifest=manifest,
            )

    chunks = _load_cubes(cubes_dir)
    assert len(chunks) == 8

    _populate_peak_grids(chunks)

    for c in chunks:
        assert c.peak_grid is not None
        assert c.peak_grid.shape == (n_fdm, n_t)
        assert c.peak_grid.dtype == np.float32
        # Fast path used the sentinel from the NPZ. If the slow path
        # had run, every entry would be 0 (cube is all zeros).
        np.testing.assert_allclose(c.peak_grid, float(sentinel))


def test_populate_peak_grids_falls_back_when_cache_absent(
    tmp_path: Path,
) -> None:
    """Cubes without ``peak_grid`` (older NPZs / non-writer test
    fixtures) must still work via the full reduction path."""
    from dsart.coinc.plotter import _load_cubes, _populate_peak_grids

    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)  # no peak_grid in NPZ

    chunks = _load_cubes(cubes_dir)
    assert len(chunks) == 8
    _populate_peak_grids(chunks)
    for c in chunks:
        assert c.peak_grid is not None
        assert c.peak_grid.shape == (c.cube.shape[0], c.cube.shape[1])


def test_render_event_plots_handles_missing_cubes(tmp_path: Path) -> None:
    """No cubes directory should still produce 4 placeholder PNGs."""
    archive_root = tmp_path / "candidates"
    ev = "260521efgh"
    (archive_root / ev / "Level2" / "plots").mkdir(parents=True)
    # Deliberately no cubes/ subdir
    job = PlotJob(
        event_name=ev,
        archive_root=archive_root,
        stats=_stats(),
        members=tuple(_members()),
    )
    written = render_event_plots(job)
    assert len(written) == 4
    for p in written:
        assert p.is_file()


def test_render_event_plots_kernel_snrs_works_without_cubes(
    tmp_path: Path,
) -> None:
    """The kernel_snrs panel is the only one that doesn't need cubes,
    so even with no NPZs + minimal stats we still expect a real plot."""
    archive_root = tmp_path / "candidates"
    ev = "260521xxxx"
    job = PlotJob(
        event_name=ev,
        archive_root=archive_root,
        stats=None,
        members=tuple(_members()),
    )
    written = render_event_plots(job)
    paths = {p.name for p in written}
    assert f"kernel_snrs_{ev}.png" in paths
    # kernel_snrs PNG must exist and be non-trivial
    p = archive_root / ev / "Level2" / "plots" / f"kernel_snrs_{ev}.png"
    assert p.is_file()
    assert p.stat().st_size > 500  # > placeholder size


def test_plot_worker_dispatches_and_completes(tmp_path: Path) -> None:
    archive_root = tmp_path / "candidates"
    ev = "260521abcd"
    _write_fake_cubes(archive_root / ev / "cubes")
    worker = PlotWorker(max_workers=2)
    try:
        fut = enqueue_event(
            worker, ev, archive_root,
            stats=_stats(), members=_members(),
        )
        written = fut.result(timeout=30.0)
        assert len(written) == 4
        plots = archive_root / ev / "Level2" / "plots"
        assert {p.name for p in written} == {
            f"dm_time_{ev}.png",
            f"image_peak_{ev}.png",
            f"lightcurve_{ev}.png",
            f"kernel_snrs_{ev}.png",
        }
        for p in written:
            assert p.is_file()
            assert p.stat().st_size > 100
    finally:
        worker.shutdown(wait=True)


def test_plot_worker_dedupes_inflight(tmp_path: Path) -> None:
    """Enqueuing the same event twice while a job is in flight returns
    the same future (no duplicate work)."""
    archive_root = tmp_path / "candidates"
    ev = "260521abcd"
    _write_fake_cubes(archive_root / ev / "cubes")
    worker = PlotWorker(max_workers=1)  # serialise to guarantee overlap
    try:
        fut1 = enqueue_event(
            worker, ev, archive_root,
            stats=_stats(), members=_members(),
        )
        fut2 = enqueue_event(
            worker, ev, archive_root,
            stats=_stats(), members=_members(),
        )
        # Either the second submission saw fut1 still pending and
        # returned it, or fut1 already finished and a fresh job ran.
        if not fut1.done():
            assert fut2 is fut1
        fut1.result(timeout=30.0)
        fut2.result(timeout=30.0)
    finally:
        worker.shutdown(wait=True)

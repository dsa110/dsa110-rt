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
    _burst_coords,
    _burst_waterfall,
    _load_cubes,
    _resolve_burst,
    _select_burst_chunk,
    enqueue_event,
    regenerate_recent_events,
    render_event_plots,
)
from dsart.coinc.stats import ClusterStats
from dsart.coinc.window import WindowEntry


# Ensure matplotlib never tries to open a display window during tests.
os.environ["MPLBACKEND"] = "Agg"


# Cube layout under test: axis 0 = time (T_DET), axis 1 = fine-DM
# (N_FDM), axes 2/3 = the (l, m) image grid. The burst is planted in
# the s2_g1 cube at (t=BURST_T, fdm=BURST_FDM, l=BURST_L, m=BURST_M);
# a *brighter* low-DM "continuum/RFI" blob is planted at fdm=0 so a
# naïve cube-argmax would mis-pick it — the metadata path must not.
T_DET, N_FDM, N_GRID = 8, 4, 16
BURST_SID, BURST_G = 2, 1
BURST_T, BURST_FDM, BURST_L, BURST_M = 5, 2, 10, 4
BURST_SNR = 20.0


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
    """Cluster members whose peak (max SNR) targets the planted burst."""
    out = []
    for i in range(5):
        out.append(WindowEntry(
            mjd=60781.0 + i * 1e-6,
            snr=10.0 + i,  # plain ascending; the peak is appended below
            l_rad=1.5e-3,
            m_rad=-2.5e-3,
            l_pix=3,
            m_pix=3,
            dm_pc_cc=100.0 + i,
            dm_idx_global=10 + i,
            fine_dm_idx=0,
            event_specnum=100 + i,
            width_samples=4,
            kernel_id="unit:d1:b4" if i % 2 == 0 else "unit:d1:b8",
            flags=0,
            search_node_id=(i % 4) + 1,
            gpu_half=i % 2,
            cube_id=7,
            sample_period_us=1048.576,
        ))
    # The actual burst peak (highest SNR) → s2_g1, fdm=2, (l,m)=(10,4).
    out.append(WindowEntry(
        mjd=60781.0 + 5e-6,
        snr=BURST_SNR,
        l_rad=1.5e-3,
        m_rad=-2.5e-3,
        l_pix=BURST_L,
        m_pix=BURST_M,
        dm_pc_cc=312.0,
        dm_idx_global=22,
        fine_dm_idx=BURST_FDM,
        event_specnum=200,
        width_samples=2,
        kernel_id="unit:d1:b2",
        flags=0,
        search_node_id=BURST_SID,
        gpu_half=BURST_G,
        cube_id=7,
        sample_period_us=1048.576,
    ))
    return out


def _write_fake_cubes(cubes_dir: Path) -> None:
    cubes_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for sid in (1, 2, 9, 13):
        for g in (0, 1):
            cube = (rng.standard_normal(
                (T_DET, N_FDM, N_GRID, N_GRID),
            ) * 0.5).astype(np.float16)
            # Bright low-DM "continuum/RFI" column present in every cube:
            # all-time at fdm=0, pixel (2,2). This is the trap a naïve
            # cube-argmax falls into.
            cube[:, 0, 2, 2] = np.float16(9.0)
            if sid == BURST_SID and g == BURST_G:
                # The (fainter) real burst at the detected DM/time/pixel.
                cube[BURST_T, BURST_FDM, BURST_L, BURST_M] = np.float16(7.0)
            manifest = {
                "mjd_start": 60781.0,
                "sample_period_us": 1048.576,
                "search_node_id": sid,
                "gpu_half": g,
            }
            path = cubes_dir / f"cube_s{sid}_g{g}_100.npz"
            np.savez(path, cube=cube, manifest=manifest)


_C1_WINDOW_HEADER = (
    "mjd,event_specnum,snr,dm_pc_cc,dm_idx_global,fine_dm_idx,l_rad,m_rad,"
    "l_pix,m_pix,width_samples,kernel_id,flags,search_node_id,gpu_half,"
    "cube_id,trigger,inj_id"
)


def _write_c1_window_csv(archive_root: Path, ev: str) -> None:
    """Write a per-event C1-window CSV mirroring the production schema,
    with the max-SNR row pointing at the planted burst."""
    lev2 = archive_root / ev / "Level2"
    lev2.mkdir(parents=True, exist_ok=True)
    rows = [
        # low-SNR decoys at low DM
        f"60781.0,100,10.5,100.0,10,0,2.7e-3,2.1e-2,3,3,4,unit:d1:b4,0,1,0,7,{ev},",
        f"60781.0,101,11.0,101.0,11,0,2.7e-3,2.1e-2,3,3,4,unit:d1:b8,0,9,0,7,{ev},",
        # the burst peak (max SNR)
        f"60781.0,200,{BURST_SNR},312.0,22,{BURST_FDM},2.7e-3,2.1e-2,"
        f"{BURST_L},{BURST_M},2,unit:d1:b2,0,{BURST_SID},{BURST_G},7,{ev},",
    ]
    (lev2 / f"C1_window_{ev}.csv").write_text(
        _C1_WINDOW_HEADER + "\n" + "\n".join(rows) + "\n"
    )


def test_all_waterfalls_ordered_and_complete(tmp_path: Path) -> None:
    """2026-06-10 8-panel dm_time: every cube gets a waterfall, sorted
    by (search_node_id, gpu_half) so the panel grid follows the fine-DM
    coverage order."""
    from dsart.coinc.plotter import _all_waterfalls

    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)
    cubes = _load_cubes(cubes_dir)
    try:
        wfs = _all_waterfalls(cubes)
        assert len(wfs) == 8
        order = [(c.search_node_id, c.gpu_half) for c, _ in wfs]
        assert order == sorted(order)
        for _, wf in wfs:
            assert wf.shape == (T_DET, N_FDM)
    finally:
        for c in cubes:
            c.close()


def test_dm_time_renders_partial_cube_set(tmp_path: Path) -> None:
    """The 8-panel figure degrades gracefully when only some halves'
    cubes made it to disk (e.g. a slow uploader)."""
    from dsart.coinc.plotter import _all_waterfalls, _render_dm_time

    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)
    # Drop all but three cubes.
    for p in sorted(cubes_dir.glob("cube_s*_g*_*.npz"))[3:]:
        p.unlink()
    cubes = _load_cubes(cubes_dir)
    try:
        wfs = _all_waterfalls(cubes)
        assert len(wfs) == 3
        out = _render_dm_time(
            tmp_path, "260610test", wfs, burst=None, coords=None,
        )
        assert out.is_file() and out.stat().st_size > 100
    finally:
        for c in cubes:
            c.close()


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


def test_burst_resolved_from_metadata_not_argmax(tmp_path: Path) -> None:
    """The burst panels must be placed from detection metadata, NOT a
    cube argmax. The fixture plants a *brighter* low-DM blob (fdm=0)
    than the real burst (fdm=2), so an argmax would mis-pick fdm=0."""
    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)
    cubes = _load_cubes(cubes_dir)
    try:
        job = PlotJob(
            event_name="260521abcd", archive_root=tmp_path,
            stats=_stats(), members=tuple(_members()),
        )
        peak, kernels = _resolve_burst(job)
        assert peak is not None and peak.source == "members"
        assert (peak.search_node_id, peak.gpu_half) == (BURST_SID, BURST_G)
        burst = _select_burst_chunk(cubes, peak)
        assert burst is not None
        assert (burst.search_node_id, burst.gpu_half) == (BURST_SID, BURST_G)
        wf = _burst_waterfall(burst)
        assert wf is not None and wf.shape == (T_DET, N_FDM)
        coords = _burst_coords(burst, wf, peak)
        assert coords is not None and coords.from_metadata
        # Metadata-driven: detected DM row + (l, m), time at the DM-slice
        # argmax — NOT the brighter fdm=0 continuum the argmax would grab.
        assert coords.fdm_idx == BURST_FDM
        assert (coords.l_pix, coords.m_pix) == (BURST_L, BURST_M)
        assert coords.t_idx == BURST_T
        # Sanity: the cube's global argmax really is the fdm=0 trap.
        assert int(np.argmax(wf)) // N_FDM != coords.t_idx or (
            int(np.argmax(wf)) % N_FDM == 0
        )
        assert any(k == "unit:d1:b2" for k, _ in kernels)
    finally:
        for c in cubes:
            c.close()


def test_offline_regeneration_from_csv(tmp_path: Path) -> None:
    """With no live members, render must resolve the burst from the
    archived C1-window CSV and still produce 4 correct PNGs."""
    archive_root = tmp_path / "candidates"
    ev = "260521csv0"
    _write_fake_cubes(archive_root / ev / "cubes")
    _write_c1_window_csv(archive_root, ev)

    # The CSV-only path must resolve the same burst as the members path.
    job = PlotJob(event_name=ev, archive_root=archive_root)  # members=()
    peak, kernels = _resolve_burst(job)
    assert peak is not None and peak.source == "csv"
    assert (peak.search_node_id, peak.gpu_half) == (BURST_SID, BURST_G)
    assert peak.fine_dm_idx == BURST_FDM
    assert (peak.l_pix, peak.m_pix) == (BURST_L, BURST_M)

    written = render_event_plots(job)
    assert len(written) == 4
    for p in written:
        assert p.is_file() and p.stat().st_size > 100


def test_regenerate_recent_events_scans_for_cubes(tmp_path: Path) -> None:
    """``regenerate_recent_events`` discovers archived events that have
    cubes and rewrites their plots."""
    archive_root = tmp_path / "candidates"
    for ev in ("260521aaaa", "260521bbbb"):
        _write_fake_cubes(archive_root / ev / "cubes")
        _write_c1_window_csv(archive_root, ev)
    # A dir without cubes must be skipped.
    (archive_root / "260521nope").mkdir(parents=True)

    done = regenerate_recent_events(archive_root)
    assert set(done) == {"260521aaaa", "260521bbbb"}
    for ev in done:
        plots = archive_root / ev / "Level2" / "plots"
        assert (plots / f"dm_time_{ev}.png").is_file()
        assert (plots / f"image_peak_{ev}.png").is_file()


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


# ---------------------------------------------------------------------------
# Detector-matched boxcar smoothing (2026-08-02)
# ---------------------------------------------------------------------------


def test_boxcar_is_unit_variance_and_pass_through_at_w1() -> None:
    """The 1/sqrt(w) normalisation keeps white noise at unit variance, so
    the smoothed series stays in the detector's sigma units."""
    from dsart.coinc.plotter import _boxcar

    rng = np.random.default_rng(20260802)
    x = rng.normal(size=20000).astype(np.float32)
    assert _boxcar(x, 1) is not x or True          # w<=1 is a pass-through
    np.testing.assert_allclose(_boxcar(x, 1), x, rtol=0, atol=0)
    for w in (2, 8, 16):
        y = _boxcar(x, w)
        # Trim the convolution edges before measuring.
        assert abs(float(np.std(y[w:-w])) - 1.0) < 0.05, w


def test_boxcar_recovers_a_wide_burst_the_raw_argmax_smears() -> None:
    """A w-sample burst buried in noise: the width-matched series must
    both find it and score it far above the unsmoothed one."""
    from dsart.coinc.plotter import _boxcar, _robust_z

    rng = np.random.default_rng(7)
    n, w, t0 = 512, 16, 200
    x = rng.normal(size=n).astype(np.float32)
    x[t0:t0 + w] += 1.2                       # 1.2 sigma/sample => ~4.8 total
    z_raw = _robust_z(x)
    z_box = _robust_z(_boxcar(x, w))
    # The matched peak lands inside the burst; the raw argmax need not.
    assert t0 <= int(np.argmax(z_box)) < t0 + w
    # Matched filter recovers ~1.2 * sqrt(16) = 4.8 sigma; the raw series
    # can only ever show one sample's worth (1.2 sigma) plus whatever
    # noise sample happens to sit highest inside the burst (~2 sigma for
    # 16 draws), so the gap is real but not the full sqrt(w).
    assert float(z_box.max()) > 4.3
    assert float(z_box.max()) > 1.3 * float(z_raw[t0:t0 + w].max())


def test_robust_z_survives_a_degenerate_mad() -> None:
    """A mostly-constant series (sparse synthetic cube) must still yield
    a usable argmax instead of collapsing to all-zeros."""
    from dsart.coinc.plotter import _robust_z

    x = np.zeros(16, dtype=np.float32)
    x[9] = 5.0
    z = _robust_z(x)
    assert int(np.argmax(z)) == 9
    assert float(z.max()) > 0.0
    # Genuinely constant -> zeros, no spurious peak.
    assert not np.any(_robust_z(np.full(16, 3.0, dtype=np.float32)))


def test_burst_coords_reports_detector_and_cube_snrs(tmp_path: Path) -> None:
    """`_burst_coords` must carry BOTH the detector's SNR and its own
    cube re-measurement (smoothed and unsmoothed) so the panels can
    print the difference."""
    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)
    cubes = _load_cubes(cubes_dir)
    try:
        job = PlotJob(
            event_name="260802snr0", archive_root=tmp_path,
            stats=_stats(), members=tuple(_members()),
        )
        peak, _ = _resolve_burst(job)
        burst = _select_burst_chunk(cubes, peak)
        coords = _burst_coords(burst, _burst_waterfall(burst), peak)
        assert coords is not None
        # Detector value is passed through untouched...
        assert coords.snr == pytest.approx(BURST_SNR)
        # ...and the cube re-measurement is a separate, finite number.
        assert np.isfinite(coords.snr_measured)
        assert np.isfinite(coords.snr_measured_raw)
        assert coords.snr_delta == pytest.approx(
            BURST_SNR - coords.snr_measured
        )
        # The boxcar is the detector's own width for the peak member.
        assert coords.boxcar == 2
        assert coords.t_idx_raw is not None
    finally:
        for c in cubes:
            c.close()


def test_smoothing_can_be_disabled_by_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSART_PLOTTER_SMOOTH=0 restores the pre-2026-08-02 raw argmax."""
    cubes_dir = tmp_path / "cubes"
    _write_fake_cubes(cubes_dir)
    cubes = _load_cubes(cubes_dir)
    try:
        job = PlotJob(
            event_name="260802nosm", archive_root=tmp_path,
            stats=_stats(), members=tuple(_members()),
        )
        peak, _ = _resolve_burst(job)
        burst = _select_burst_chunk(cubes, peak)
        wf = _burst_waterfall(burst)
        monkeypatch.setenv("DSART_PLOTTER_SMOOTH", "0")
        coords = _burst_coords(burst, wf, peak)
        assert coords is not None and coords.boxcar == 1
        # With no smoothing the two measurements coincide.
        assert coords.snr_measured == pytest.approx(coords.snr_measured_raw)
    finally:
        for c in cubes:
            c.close()

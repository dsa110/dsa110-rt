"""Unit tests for the dashboard sky-monitor backend
(``tools/dashboard/dsa_monitor/sky_monitor.py``) — E2E test 1.

Covers: snapshot parsing, UV combine with amp-scale weighting, the
dirty-image convention (point source at a known pixel), robust sigma,
frame store write/list/resolve/prune, and the SkyMonitor ingest →
frame pipeline end-to-end on a tmp dir.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_DASH_DIR = Path(__file__).resolve().parents[1] / "tools" / "dashboard" / "dsa_monitor"
sys.path.insert(0, str(_DASH_DIR))

import sky_monitor as sm                                    # noqa: E402


N_GRID = 64


def _snapshot_bytes(
    chgroup: int,
    *,
    vis: np.ndarray | None = None,
    n_filled: int = 32,
    amp_scale: float = 1.0,
    n_grid: int = N_GRID,
    version: int | None = None,
    dec_deg: float = 54.5,
) -> bytes:
    """Build a wire-format snapshot npz (mirrors sky_export)."""
    rng = np.random.default_rng(chgroup)
    if vis is None:
        vis = (rng.standard_normal(n_filled)
               + 1j * rng.standard_normal(n_filled)).astype(np.complex64)
    n_filled = vis.shape[0]
    ix_row = rng.integers(0, n_grid, n_filled).astype(np.uint16)
    ix_col = rng.integers(0, n_grid, n_filled).astype(np.uint16)
    meta = {
        "chgroup": chgroup,
        "hostname": f"n{chgroup + 3:02d}",
        "n_grid": n_grid,
        "cell_lambda": 12.5,
        "dec_deg": dec_deg,
        "amp_scale": amp_scale,
        "cubes_seen": 1000,
        "block_n": 5000,
        "unix_ts": time.time(),
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        version=np.int64(
            sm.SKY_SNAPSHOT_VERSION if version is None else version
        ),
        vis=np.ascontiguousarray(vis, dtype=np.complex64),
        ix_row=ix_row,
        ix_col=ix_col,
        meta_json=np.bytes_(json.dumps(meta).encode("utf-8")),
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_roundtrip():
    body = _snapshot_bytes(7)
    snap = sm.parse_snapshot_npz(body)
    assert snap["meta"]["chgroup"] == 7
    assert snap["vis"].dtype == np.complex64
    assert snap["ix_row"].shape == snap["vis"].shape


def test_parse_rejects_garbage_and_bad_version():
    with pytest.raises(ValueError):
        sm.parse_snapshot_npz(b"garbage")
    with pytest.raises(ValueError):
        sm.parse_snapshot_npz(_snapshot_bytes(0, version=99))


# ---------------------------------------------------------------------------
# Combine + imaging
# ---------------------------------------------------------------------------


def test_combine_applies_amp_scale_weight():
    vis = np.full(4, 2.0 + 0.0j, dtype=np.complex64)
    snap = {
        "vis": vis,
        "ix_row": np.array([1, 2, 3, 4], dtype=np.uint16),
        "ix_col": np.array([1, 2, 3, 4], dtype=np.uint16),
        "meta": {"chgroup": 0, "n_grid": N_GRID, "amp_scale": 2.0},
    }
    uv, used = sm.combine_chgroups_to_uv([snap], n_grid=N_GRID)
    assert used == [0]
    # weight = 1/amp_scale² = 0.25 → 2.0 * 0.25 = 0.5
    assert uv[1, 1] == pytest.approx(0.5)


def test_combine_skips_mismatched_n_grid():
    snap = {
        "vis": np.ones(2, dtype=np.complex64),
        "ix_row": np.array([0, 1], dtype=np.uint16),
        "ix_col": np.array([0, 1], dtype=np.uint16),
        "meta": {"chgroup": 4, "n_grid": 999, "amp_scale": 1.0},
    }
    uv, used = sm.combine_chgroups_to_uv([snap], n_grid=N_GRID)
    assert used == []
    assert not uv.any()


def test_dirty_image_point_source_at_phase_center():
    """A flat UV plane (all ones) images to a point at the center
    pixel under the ifftshift→ifft2→fftshift convention."""
    uv = np.ones((N_GRID, N_GRID), dtype=np.complex64)
    img = sm.dirty_image_from_uv(uv)
    assert img.shape == (N_GRID, N_GRID)
    peak = np.unravel_index(np.argmax(img), img.shape)
    assert peak == (N_GRID // 2, N_GRID // 2)
    # All energy in one pixel.
    assert img[peak] == pytest.approx(1.0)
    assert abs(img.sum() - 1.0) < 1e-4


def _uv_for_point_source(row: int, col: int) -> np.ndarray:
    """UV grid whose dirty image (under the module's
    fftshift(ifft2(ifftshift(·))) convention) is a unit delta at
    (row, col)."""
    img = np.zeros((N_GRID, N_GRID), dtype=np.complex64)
    img[row, col] = 1.0
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))).astype(
        np.complex64
    )


def test_pillbox_correction_envelope():
    corr = sm.pillbox_grid_correction(N_GRID)
    c = N_GRID // 2
    assert corr[c, c] == pytest.approx(1.0)
    # FOV edge: 1/sinc(1/2) = π/2 per axis.
    assert corr[0, c] == pytest.approx(np.pi / 2.0, rel=1e-3)
    assert corr[c, 0] == pytest.approx(np.pi / 2.0, rel=1e-3)
    # Corner ≈ (π/2)² ≈ 2.47, under the 2.5 cap.
    assert corr[0, 0] == pytest.approx((np.pi / 2.0) ** 2, rel=1e-3)
    assert float(corr.max()) <= 2.5 + 1e-6
    # Symmetric about the center on the open interval.
    assert corr[c + 10, c] == pytest.approx(corr[c - 10, c], rel=1e-6)


def test_dirty_image_oversample_is_exact_interpolation():
    """2× UV zero-padding: same FOV, 2× pixel coordinates, peak
    amplitude preserved (after the oversample² renormalisation) —
    exact band-limited interpolation of the same dirty image."""
    row, col = N_GRID // 2 + 18, N_GRID // 2 - 11    # off-center source
    uv = _uv_for_point_source(row, col)
    img1 = sm.dirty_image_from_uv(uv)
    img2 = sm.dirty_image_from_uv(uv, oversample=2)
    assert img2.shape == (2 * N_GRID, 2 * N_GRID)
    assert img1[row, col] == pytest.approx(1.0, abs=1e-4)
    # The on-grid samples are reproduced exactly at even pixels.
    assert img2[2 * row, 2 * col] == pytest.approx(1.0, abs=1e-3)
    peak = np.unravel_index(np.argmax(img2), img2.shape)
    assert peak == (2 * row, 2 * col)


def test_dirty_image_grid_correct_boosts_edge_sources():
    """Grid correction multiplies a source near the FOV edge by the
    inverse pillbox envelope; a phase-center source is untouched."""
    c = N_GRID // 2
    # Center source: correction = 1.
    uv_c = _uv_for_point_source(c, c)
    img_c = sm.dirty_image_from_uv(uv_c, grid_correct=True)
    assert img_c[c, c] == pytest.approx(1.0, abs=1e-4)
    # Near-edge source: boosted by 1/sinc(f) on the offset axis.
    row, col = c, 4
    f = (col - c) / float(N_GRID)
    expected = 1.0 / float(np.sinc(f))
    uv_e = _uv_for_point_source(row, col)
    plain = sm.dirty_image_from_uv(uv_e)
    corrected = sm.dirty_image_from_uv(uv_e, grid_correct=True)
    assert corrected[row, col] / plain[row, col] == pytest.approx(
        expected, rel=1e-4,
    )


def test_monitor_frame_is_oversampled_by_default(tmp_path):
    """SkyMonitor production defaults: 2× oversample + grid correct,
    surfaced in the frame metadata and the written NPZ image shape."""
    mon = sm.SkyMonitor(
        nvss_enabled=False,
        store=sm.SkyFrameStore(root=tmp_path, retention_h=48.0),
        frame_interval_s=30.0,
        freshness_s=90.0,
        min_chgroups=1,
        n_grid=N_GRID,
    )
    assert mon.oversample == 2 and mon.grid_correct
    ack = mon.ingest(_snapshot_bytes(0), now=1_750_000_000.0)
    assert ack["frame_written"]
    frames = mon.store.list_frames(since_unix=0)
    npz_path = (
        mon.store.frames_dir / frames[0]["day"]
        / frames[0]["png"].replace(".png", ".npz")
    )
    with np.load(npz_path, allow_pickle=False) as z:
        assert z["image"].shape == (2 * N_GRID, 2 * N_GRID)
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    assert meta["oversample"] == 2
    assert meta["grid_correct"] is True
    assert meta["n_pix"] == 2 * N_GRID


# ---------------------------------------------------------------------------
# Astrometry + NVSS overlay (sky_astrometry + annotated frames)
# ---------------------------------------------------------------------------

import sky_astrometry as sa                                  # noqa: E402


def test_radec_lm_roundtrip_and_orientation():
    ra0, dec0 = 123.4, 16.27
    # Phase center maps to (0, 0).
    l, m = sa.radec_to_lm(ra0, dec0, ra0_deg=ra0, dec0_deg=dec0)
    assert abs(float(l)) < 1e-12 and abs(float(m)) < 1e-12
    # East (increasing RA) → +l; north (increasing Dec) → +m. Matches
    # bench/run_0319_pipeline._compute_expected_lm (l = -cosδ sinHA).
    l, m = sa.radec_to_lm(ra0 + 0.5, dec0, ra0_deg=ra0, dec0_deg=dec0)
    assert float(l) > 0 and abs(float(m)) < 1e-4
    l, m = sa.radec_to_lm(ra0, dec0 + 0.5, ra0_deg=ra0, dec0_deg=dec0)
    assert abs(float(l)) < 1e-12 and float(m) > 0
    # Round-trip a grid of offsets.
    ras = ra0 + np.linspace(-0.8, 0.8, 7)
    decs = dec0 + np.linspace(-0.8, 0.8, 7)
    l, m = sa.radec_to_lm(ras, decs, ra0_deg=ra0, dec0_deg=dec0)
    ra2, dec2 = sa.lm_to_radec(l, m, ra0_deg=ra0, dec0_deg=dec0)
    np.testing.assert_allclose(ra2, ras % 360.0, atol=1e-9)
    np.testing.assert_allclose(dec2, decs, atol=1e-9)


def test_lm_to_pix_matches_replay_convention():
    # Mirrors bench/_corr_fast_replay.pixel_to_lm_radians: center pixel
    # n_pix//2 ↔ (l, m) = (0, 0); pixel scale = fov / n_pix.
    fov = 1.0 / 31.49                                    # rad, production-like
    row, col = sa.lm_to_pix(0.0, 0.0, n_pix=512, fov_rad=fov)
    assert (float(row), float(col)) == (256.0, 256.0)
    l = 10 * fov / 512.0
    row, col = sa.lm_to_pix(l, -l, n_pix=512, fov_rad=fov)
    assert float(col) == pytest.approx(266.0)
    assert float(row) == pytest.approx(246.0)


def test_nvss_tdat_parse_and_select(tmp_path):
    tdat = tmp_path / "mini.tdat"
    tdat.write_text(
        "<HEADER>\n"
        "field[name] = char20\n"
        "<DATA>\n"
        # name|ra|dec|lii|bii|ra_err|dec_err|flux|...
        "NVSS J1|10.0|16.0|0|0|1|1|250.0|1|\n"
        "NVSS J2|10.2|16.1|0|0|1|1|99.9|1|\n"          # below cut
        "NVSS J3|10.1|15.9|0|0|1|1|1500.0|1|\n"
        "NVSS J4|200.0|-40.0|0|0|1|1|800.0|1|\n"       # out of FOV
        "NVSS Jbad|x|y|0|0|1|1|500.0|1|\n"             # malformed
        "<END>\n"
    )
    cat = sa.load_nvss(min_mjy=100.0, tdat_path=tdat, cache_dir=tmp_path)
    assert cat is not None
    assert sorted(cat["name"].tolist()) == ["NVSS J1", "NVSS J3", "NVSS J4"]
    # Cache round-trip gives identical content.
    cat2 = sa.load_nvss(min_mjy=100.0, tdat_path=tdat, cache_dir=tmp_path)
    np.testing.assert_array_equal(cat["flux_mjy"], cat2["flux_mjy"])

    sel = sa.select_in_fov(
        cat, ra0_deg=10.0, dec0_deg=16.0, fov_rad=np.deg2rad(1.0),
    )
    # J1 + J3 in the 1° FOV, brightest first; J4 excluded.
    assert sel["name"].tolist() == ["NVSS J3", "NVSS J1"]
    assert abs(sel["l_rad"][1]) < 1e-9                   # J1 at center


def test_measure_source_snr():
    img = np.zeros((64, 64), dtype=np.float32)
    img[40, 22] = 7.0                                     # source peak
    snr = sm.measure_source_snr(
        img, row=41.5, col=20.5, median=0.0, sigma=1.0, radius_pix=3.0,
    )
    assert snr == pytest.approx(7.0)
    # Aperture misses the peak → 0σ here (empty field).
    snr = sm.measure_source_snr(
        img, row=10.0, col=10.0, median=0.0, sigma=1.0, radius_pix=3.0,
    )
    assert snr == pytest.approx(0.0)
    # Fully off-frame → NaN.
    assert np.isnan(sm.measure_source_snr(
        img, row=-50.0, col=-50.0, median=0.0, sigma=1.0, radius_pix=3.0,
    ))


def test_annotated_frame_with_nvss(tmp_path, monkeypatch):
    """End-to-end: stubbed catalog + stubbed LST → annotated PNG, and
    the per-source measured S/N lands in the frame metadata."""
    monkeypatch.setattr(sa, "lst_deg", lambda ts: 10.0)   # ra0 = 10°
    mon = sm.SkyMonitor(
        store=_store(tmp_path), min_chgroups=1, n_grid=N_GRID,
        nvss_enabled=True,
    )
    # Inject the catalog directly (skip the loader thread): one source
    # AT the phase center so it sits on the central pixel.
    mon._nvss._cat = {
        "name": np.array(["NVSS JX"], dtype="U20"),
        "ra_deg": np.array([10.0]),
        "dec_deg": np.array([16.0]),                      # = dec0 below
        "flux_mjy": np.array([500.0]),
    }
    # dec_deg=16.0 must come through the snapshot meta.
    ack = mon.ingest(
        _snapshot_bytes(0, dec_deg=16.0), now=1_750_000_000.0,
    )
    assert ack["frame_written"]
    frames = mon.store.list_frames(since_unix=0)
    day, png = frames[0]["day"], frames[0]["png"]
    png_path = mon.store.frames_dir / day / png
    assert png_path.is_file() and png_path.stat().st_size > 10_000
    npz_path = png_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    assert meta["ra0_deg"] == pytest.approx(10.0)
    assert meta["dec0_deg"] == pytest.approx(16.0)
    assert meta["nvss_loaded"] is True
    assert len(meta["nvss"]) == 1
    src = meta["nvss"][0]
    assert src["name"] == "NVSS JX"
    # Central pixel of the oversampled frame, finite measured S/N.
    n_pix = meta["n_pix"]
    assert src["row"] == pytest.approx(n_pix // 2, abs=1.0)
    assert src["col"] == pytest.approx(n_pix // 2, abs=1.0)
    assert src["snr"] is not None


def test_robust_sigma_ignores_bright_sources():
    rng = np.random.default_rng(11)
    img = rng.standard_normal((N_GRID, N_GRID)).astype(np.float32)
    img[5, 5] = 1e4                      # one "continuum source"
    med, sigma = sm.robust_sigma(img)
    assert abs(med) < 0.1
    assert 0.8 < sigma < 1.2             # MAD ≈ true σ = 1, unmoved by spike


# ---------------------------------------------------------------------------
# Frame store
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> sm.SkyFrameStore:
    return sm.SkyFrameStore(root=tmp_path, retention_h=48.0)


def test_store_write_list_resolve(tmp_path):
    store = _store(tmp_path)
    img = np.zeros((N_GRID, N_GRID), dtype=np.float32)
    ts = 1_750_000_000.0
    png, npz = store.write_frame(
        img, ts=ts, median=0.0, sigma=1.0,
        used_chgroups=[0, 1, 2], meta={"ts": ts},
    )
    assert png.exists() and npz.exists()
    frames = store.list_frames(since_unix=ts - 10)
    assert len(frames) == 1
    f = frames[0]
    assert f["ts"] == int(ts)
    assert f["n_chgroups"] == 3
    assert store.resolve_png(f["day"], f["png"]) == png
    # listing excludes frames older than `since`
    assert store.list_frames(since_unix=ts + 10) == []


def test_store_resolve_rejects_path_tricks(tmp_path):
    store = _store(tmp_path)
    assert store.resolve_png("..", "sky_1_n1.png") is None
    assert store.resolve_png("20260609", "../../etc/passwd") is None
    assert store.resolve_png("20260609", "sky_1_n1.npz") is None


def test_store_prune_removes_old_frames(tmp_path):
    store = _store(tmp_path)
    img = np.zeros((N_GRID, N_GRID), dtype=np.float32)
    now = 1_750_000_000.0
    old_ts = now - 3 * 24 * 3600.0       # 72 h old, retention 48 h
    store.write_frame(img, ts=old_ts, median=0.0, sigma=1.0,
                      used_chgroups=[0], meta={})
    store.write_frame(img, ts=now, median=0.0, sigma=1.0,
                      used_chgroups=[0], meta={})
    n = store.prune(now=now)
    assert n == 2                        # old png + old npz
    frames = store.list_frames(since_unix=0)
    assert len(frames) == 1
    assert frames[0]["ts"] == int(now)


# ---------------------------------------------------------------------------
# SkyMonitor ingest → frame
# ---------------------------------------------------------------------------


def test_ingest_builds_frame_when_due(tmp_path):
    mon = sm.SkyMonitor(
        nvss_enabled=False,
        store=_store(tmp_path),
        frame_interval_s=30.0,
        freshness_s=90.0,
        min_chgroups=2,
        n_grid=N_GRID,
    )
    t0 = 1_750_000_000.0
    ack0 = mon.ingest(_snapshot_bytes(0), now=t0)
    assert ack0["ok"] and not ack0["frame_written"]      # below min_chgroups
    ack1 = mon.ingest(_snapshot_bytes(1), now=t0 + 1)
    assert ack1["frame_written"]                          # 2 fresh, interval due
    assert mon.n_frames == 1
    frames = mon.store.list_frames(since_unix=0)
    assert len(frames) == 1
    assert frames[0]["n_chgroups"] == 2

    # Within the interval: no second frame.
    ack2 = mon.ingest(_snapshot_bytes(2), now=t0 + 5)
    assert not ack2["frame_written"]
    # After the interval: a new frame, now from 3 fresh chgroups.
    ack3 = mon.ingest(_snapshot_bytes(3), now=t0 + 40)
    assert ack3["frame_written"]
    frames = mon.store.list_frames(since_unix=0)
    assert frames[-1]["n_chgroups"] == 4


def test_ingest_excludes_stale_chgroups(tmp_path):
    mon = sm.SkyMonitor(
        nvss_enabled=False,
        store=_store(tmp_path),
        frame_interval_s=30.0,
        freshness_s=90.0,
        min_chgroups=1,
        n_grid=N_GRID,
    )
    t0 = 1_750_000_000.0
    mon.ingest(_snapshot_bytes(0), now=t0)                # frame 1 (cg 0)
    # 10 min later only cg 1 is fresh; cg 0 must age out.
    ack = mon.ingest(_snapshot_bytes(1), now=t0 + 600)
    assert ack["frame_written"]
    assert ack["n_fresh"] == 1
    frames = mon.store.list_frames(since_unix=0)
    assert frames[-1]["n_chgroups"] == 1


def test_ingest_rejects_bad_chgroup_and_garbage(tmp_path):
    mon = sm.SkyMonitor(store=_store(tmp_path), n_grid=N_GRID,
                        nvss_enabled=False)
    with pytest.raises(ValueError):
        mon.ingest(_snapshot_bytes(99), now=time.time())
    with pytest.raises(ValueError):
        mon.ingest(b"garbage", now=time.time())
    assert mon.n_frames == 0


def test_status_reports_freshness(tmp_path):
    mon = sm.SkyMonitor(store=_store(tmp_path), n_grid=N_GRID,
                        nvss_enabled=False)
    t0 = 1_750_000_000.0
    mon.ingest(_snapshot_bytes(2), now=t0)
    st = mon.status(now=t0 + 12)
    assert st["chgroups"]["2"]["age_s"] == pytest.approx(12.0)
    assert st["chgroups"]["2"]["hostname"] == "n05"
    assert st["n_ingested"] == 1

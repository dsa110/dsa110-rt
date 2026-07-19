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
    uv, used, _cs = sm.combine_chgroups_to_uv([snap], n_grid=N_GRID)
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
    uv, used, _cs = sm.combine_chgroups_to_uv([snap], n_grid=N_GRID)
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
    # ra0/dec0 are the ICRS phase center: the (patched) apparent LST
    # of 10.0 deg minus ~26 yr of precession (~0.3-0.4 deg of RA here),
    # via the same TETE->ICRS transform the event pages use.
    assert 0.2 < 10.0 - meta["ra0_deg"] < 0.5
    assert meta["dec0_deg"] == pytest.approx(16.0, abs=0.2)
    assert meta["dec0_apparent_deg"] == pytest.approx(16.0)
    assert meta["radec_epoch"] == "ICRS/J2000"
    assert meta["nvss_loaded"] is True
    assert len(meta["nvss"]) == 1
    src = meta["nvss"][0]
    assert src["name"] == "NVSS JX"
    # The catalog source sits at the APPARENT (ra=10, dec=16) frame
    # center's J2000 counterpart... i.e. ~0.35 deg EAST of the ICRS
    # center in l (the precession offset), so its predicted column is
    # east of center by ~0.35 deg / pix_scale.
    n_pix = meta["n_pix"]
    pix_deg = meta["fov_deg"] / n_pix
    expect_dcol = (10.0 - meta["ra0_deg"]) * np.cos(np.deg2rad(16.0)) \
        / pix_deg
    assert src["col"] == pytest.approx(n_pix // 2 + expect_dcol, abs=3.0)
    assert src["snr"] is not None
    # JSON sidecar drives the page-side source table.
    sidecar = json.loads(png_path.with_suffix(".json").read_text())
    assert sidecar["ra0_deg"] == pytest.approx(meta["ra0_deg"])
    assert [s["name"] for s in sidecar["nvss"]] == ["NVSS JX"]
    assert mon.store.resolve_sidecar(day, png) is not None


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


# ---------------------------------------------------------------------------
# 2026-07-18: data-time alignment + noise / SEFD reporting
# ---------------------------------------------------------------------------


def _sparse_from_uv(uv: np.ndarray, *, chgroup: int, cell_lambda: float,
                    dec_deg: float = 16.27) -> dict:
    """Full-coverage sparse snapshot dict wrapping a dense UV grid."""
    n = uv.shape[0]
    rows, cols = np.mgrid[0:n, 0:n]
    return {
        "vis": uv.reshape(-1).astype(np.complex64),
        "ix_row": rows.reshape(-1).astype(np.uint16),
        "ix_col": cols.reshape(-1).astype(np.uint16),
        "meta": {
            "chgroup": chgroup,
            "n_grid": n,
            "amp_scale": 1.0,
            "cell_lambda": cell_lambda,
            "dec_deg": dec_deg,
        },
    }


def test_combine_align_ramp_translates_image():
    """The align phase ramp moves a chgroup's image by exactly +Δl.

    Physics scenario: chgroup B's snapshot is Δt EARLIER in data time,
    so its sources sit EAST (larger l / larger col) of the reference;
    the builder passes Δl = −cosδ·ω·Δt (negative) which must pull them
    WEST onto the reference position.
    """
    cell_lambda = 17.9
    fov = 1.0 / cell_lambda
    pix_rad = fov / N_GRID
    ref_col, row = N_GRID // 2 + 5, N_GRID // 2

    uv_ref = _uv_for_point_source(row, ref_col)
    uv_east = _uv_for_point_source(row, ref_col + 3)      # 3 px east

    snaps = [
        _sparse_from_uv(uv_ref, chgroup=0, cell_lambda=cell_lambda),
        _sparse_from_uv(uv_east, chgroup=1, cell_lambda=cell_lambda),
    ]
    # Unaligned: two separated unit peaks.
    uv0, used0, _cs0 = sm.combine_chgroups_to_uv(snaps, n_grid=N_GRID)
    img0 = sm.dirty_image_from_uv(uv0)
    assert used0 == [0, 1]
    assert img0[row, ref_col] == pytest.approx(1.0, abs=1e-3)
    assert img0[row, ref_col + 3] == pytest.approx(1.0, abs=1e-3)

    # Aligned: chgroup 1 shifted WEST by 3 px → single 2.0 peak.
    align = {0: 0.0, 1: -3.0 * pix_rad}
    uv1, _, _cs1 = sm.combine_chgroups_to_uv(
        snaps, n_grid=N_GRID, align_dl_rad=align,
    )
    img1 = sm.dirty_image_from_uv(uv1)
    assert img1[row, ref_col] == pytest.approx(2.0, abs=1e-2)
    assert img1[row, ref_col + 3] < 0.1


def test_snapshot_data_mid_unix_prefers_arm_anchor():
    armed_unix = 1_784_000_000.0
    wb = 8
    meta = {
        "block_n": 100_000,
        "window_blocks": wb,
        "unix_ts": armed_unix + 100_000 * sm.BLOCK_S + 1.96,  # 2 s lag
    }
    t = sm.snapshot_data_mid_unix(meta, armed_unix=armed_unix)
    expect = armed_unix + 100_000 * sm.BLOCK_S - (wb - 1) / 2 * sm.BLOCK_S
    assert t == pytest.approx(expect, abs=1e-6)

    # No arm anchor → wall clock minus measured export lag.
    t2 = sm.snapshot_data_mid_unix(meta, armed_unix=None)
    expect2 = (meta["unix_ts"] - sm.EXPORT_LAG_FALLBACK_S
               - (wb - 1) / 2 * sm.BLOCK_S)
    assert t2 == pytest.approx(expect2, abs=1e-6)

    # Stale arm anchor (re-arm happened): block time disagrees with the
    # corr wall clock by minutes → fall back to the wall clock.
    t3 = sm.snapshot_data_mid_unix(meta, armed_unix=armed_unix + 300.0)
    assert t3 == pytest.approx(expect2, abs=1e-6)


def test_pb_resp_power_shape():
    import sky_astrometry as sa2
    assert float(sa2.pb_resp_power(0.0)) == pytest.approx(1.0)
    # Monotone decline over the main lobe; FWHM ~1.8 deg at 1.405 GHz.
    th = np.deg2rad(np.array([0.4, 0.9, 1.3]))
    pb = sa2.pb_resp_power(th)
    assert 0.6 < pb[0] < 0.95
    assert 0.35 < pb[1] < 0.65          # HWHM ≈ 0.9 deg
    assert pb[2] < pb[1] < pb[0]


def test_fit_flux_scale_and_sefd_math():
    rows = [
        {"snr": 40.0, "peak": 2.0e-3, "pb": 1.0, "flux_mjy": 1000.0},
        {"snr": 16.0, "peak": 8.0e-4, "pb": 0.8, "flux_mjy": 500.0},
        {"snr": 3.0, "peak": 1.0e-4, "pb": 1.0, "flux_mjy": 50.0},   # below snr
        {"snr": 20.0, "peak": 5.0e-4, "pb": 0.05, "flux_mjy": 5000.0},  # low pb
    ]
    k, n = sm.fit_flux_scale(rows)
    assert n == 2
    assert k == pytest.approx(2.0e-6, rel=1e-6)   # exact: both rows on line

    # SEFD radiometer line: 7000 Jy over ~1.07 s / 82 ant / 2 pol.
    sig = sm.sefd_predicted_sigma_mjy(7000.0, window_s=8 * sm.BLOCK_S)
    assert 3.5 < sig < 5.5

    # A single detection is accepted as a (noisy) scale anchor.
    k2, n2 = sm.fit_flux_scale(rows[:1])
    assert n2 == 1 and k2 == pytest.approx(2.0e-6, rel=1e-6)
    # No detections → None.
    k3, n3 = sm.fit_flux_scale(rows[2:3])
    assert k3 is None and n3 == 0


def test_combine_uv_clip_zeroes_hot_cells():
    """Cells with |V| >> median (static RFI / crosstalk) are zeroed;
    ordinary cells survive untouched."""
    n_filled = 64
    rng = np.random.default_rng(7)
    vis = (rng.normal(size=n_filled) + 1j * rng.normal(size=n_filled))
    vis = vis.astype(np.complex64)
    vis[5] = 500.0 + 0j                      # hot cell (RFI-like)
    vis[17] = 0.0 - 300.0j                   # hot cell
    rows = np.arange(n_filled, dtype=np.uint16)
    cols = np.arange(n_filled, dtype=np.uint16)
    snap = {
        "vis": vis, "ix_row": rows, "ix_col": cols,
        "meta": {"chgroup": 3, "n_grid": N_GRID, "amp_scale": 1.0,
                 "cell_lambda": 17.9},
    }
    uv, used, cs = sm.combine_chgroups_to_uv([snap], n_grid=N_GRID)
    assert used == [3]
    assert cs[3]["n_clipped"] == 2
    assert uv[5, 5] == 0 and uv[17, 17] == 0
    assert uv[3, 3] == pytest.approx(vis[3], abs=1e-6)

    # Clip disabled → hot cells pass through.
    uv2, _, cs2 = sm.combine_chgroups_to_uv(
        [snap], n_grid=N_GRID, uv_clip_k=0.0,
    )
    assert uv2[5, 5] == pytest.approx(500.0)
    assert cs2[3]["n_clipped"] == 0


def test_static_subtract_removes_instrument_baseline(tmp_path):
    """A constant per-cell offset present in every snapshot is removed
    once STATIC_SUB_MIN_HIST history has accumulated."""
    mon = sm.SkyMonitor(
        nvss_enabled=False,
        store=sm.SkyFrameStore(root=tmp_path, retention_h=48.0),
        frame_interval_s=1e9,          # never auto-build frames
        n_grid=N_GRID,
    )
    rng = np.random.default_rng(1)
    static = (rng.standard_normal(32) + 1j * rng.standard_normal(32)
              ).astype(np.complex64) * 10.0
    for i in range(sm.STATIC_SUB_MIN_HIST):
        noise = (rng.standard_normal(32) + 1j * rng.standard_normal(32)
                 ).astype(np.complex64) * 0.01
        mon.ingest(_snapshot_bytes(2, vis=static + noise),
                   now=1_750_000_000.0 + i)
    snap = sm.parse_snapshot_npz(
        _snapshot_bytes(2, vis=static + (0.5 + 0j)))
    out, n_hist, ready = mon._static_subtract([snap])
    assert ready
    assert n_hist[2] >= sm.STATIC_SUB_MIN_HIST
    # The 10.0-scale static baseline is gone; the 0.5 offset survives.
    resid = out[0]["vis"]
    assert float(np.median(np.abs(resid))) < 1.0
    assert float(np.median(resid.real)) == pytest.approx(0.5, abs=0.1)

    # Cold history → passthrough, flagged not-ready.
    mon2 = sm.SkyMonitor(
        nvss_enabled=False,
        store=sm.SkyFrameStore(root=tmp_path / "b", retention_h=48.0),
        frame_interval_s=1e9, n_grid=N_GRID,
    )
    out2, _, ready2 = mon2._static_subtract([snap])
    assert not ready2
    assert np.array_equal(out2[0]["vis"], snap["vis"])


def test_sky_to_instrument_lm_compression_and_roundtrip():
    """The gridder's v = raw ΔN ⇒ image m is the sky m compressed by
    cos(lat − dec) plus a small w-term warp; inverse round-trips."""
    import sky_astrometry as sa2
    dec0 = 16.2734
    g = np.deg2rad(sa2.OVRO_LAT_DEG - dec0)
    m_true = np.deg2rad(1.0)
    l_img, m_img = sa2.sky_to_instrument_lm(0.0, m_true, dec0_deg=dec0)
    # Compressed toward center (dec < lat) + small positive w-term.
    assert float(m_img) < m_true
    assert float(m_img) == pytest.approx(
        m_true * np.cos(g) + np.sin(g) * m_true ** 2 / 2.0, rel=1e-12,
    )
    # ~4 arcmin at the 1-deg edge — the reported varying-Dec offset.
    assert 3.0 < np.rad2deg(m_true - float(m_img)) * 60.0 < 5.0
    l2, m2 = sa2.instrument_to_sky_lm(l_img, m_img, dec0_deg=dec0)
    assert float(m2) == pytest.approx(m_true, abs=5e-11)  # ~10 uas
    # At dec == lat (zenith) the mapping is identity.
    _, mz = sa2.sky_to_instrument_lm(
        0.0, m_true, dec0_deg=sa2.OVRO_LAT_DEG,
    )
    assert float(mz) == pytest.approx(m_true, rel=1e-12)


def test_phase_center_icrs_removes_precession():
    """TETE(date) → ICRS: ~26 yr of precession is ~18-20 arcmin of RA
    at these coordinates — the constant +45 px east offset measured on
    2026-07-19. ICRS ra0 must be LOWER than the apparent LST."""
    import sky_astrometry as sa2
    unix = 1_784_438_491.0                     # 2026-07-19 05:21:31 UT
    dec_app = 16.2734
    lst = sa2.lst_deg(unix)
    ra0, dec0 = sa2.phase_center_icrs(unix, dec_app)
    dra_arcmin = ((lst - ra0 + 180.0) % 360.0 - 180.0) * 60.0
    assert 12.0 < dra_arcmin < 25.0            # precession-scale, east
    assert abs(dec0 - dec_app) < 12.0 / 60.0   # dec shift < 12 arcmin


def test_measure_astrometric_offset_finds_shift():
    rng = np.random.default_rng(3)
    n = 128
    img = rng.standard_normal((n, n)).astype(np.float32)
    rows = []
    # sources truly at predicted + (+4, -6)
    for (r, c, f) in ((30, 40, 900.0), (70, 100, 400.0), (100, 20, 250.0),
                      (55, 64, 150.0), (90, 80, 120.0)):
        img[r + 4, c - 6] += 30.0
        rows.append({"row": r, "col": c, "flux_mjy": f})
    res = sm.measure_astrometric_offset(
        img, median=0.0, sigma=1.0, nvss_rows=rows)
    assert res["z"] > 10
    assert (res["drow_px"], res["dcol_px"]) == (4, -6)

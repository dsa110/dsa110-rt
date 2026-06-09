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
        "dec_deg": 54.5,
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
    mon = sm.SkyMonitor(store=_store(tmp_path), n_grid=N_GRID)
    with pytest.raises(ValueError):
        mon.ingest(_snapshot_bytes(99), now=time.time())
    with pytest.raises(ValueError):
        mon.ingest(b"garbage", now=time.time())
    assert mon.n_frames == 0


def test_status_reports_freshness(tmp_path):
    mon = sm.SkyMonitor(store=_store(tmp_path), n_grid=N_GRID)
    t0 = 1_750_000_000.0
    mon.ingest(_snapshot_bytes(2), now=t0)
    st = mon.status(now=t0 + 12)
    assert st["chgroups"]["2"]["age_s"] == pytest.approx(12.0)
    assert st["chgroups"]["2"]["hostname"] == "n05"
    assert st["n_ingested"] == 1

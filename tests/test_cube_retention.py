"""Cube-staging retention sweeper (2026-07-19 disk-full incident).

Covers the three sweep tiers of ``dsart.dump.cube_retention`` and the
uploaded-first preference under the size cap.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dsart.dump.cube_retention import (
    CubeRetentionConfig,
    CubeRetentionSweeper,
    sweep_once,
)

GB = 1e9
H = 3600.0


def _event(root: Path, name: str, *, age_h: float, n_bytes: int = 1000,
           uploaded: bool = False) -> Path:
    d = root / name
    d.mkdir()
    npz = d / f"cube_s1_g0_{name}.npz"
    npz.write_bytes(b"\0" * n_bytes)
    if uploaded:
        (d / "upload.log").write_text("# ...\n# ...Z UPLOAD_OK rc=0\n")
    ts = time.time() - age_h * H
    for f in list(d.iterdir()) + [d]:
        os.utime(f, (ts, ts))
    return d


def _cfg(root: Path, **kw) -> CubeRetentionConfig:
    defaults = dict(max_age_h=96.0, max_total_gb=1e6, low_water_gb=1e6,
                    tmp_age_h=2.0, sweep_interval_s=600.0)
    defaults.update(kw)
    return CubeRetentionConfig(dump_root=root, **defaults)


def test_tier0_hygiene_deletes_stale_tmp_and_zero_npz(tmp_path) -> None:
    old = time.time() - 3 * H
    tmp = tmp_path / "cube_s1_g0_9.npz.tmp"
    tmp.write_bytes(b"x")
    zero = tmp_path / "cube_s1_g0_8.npz"
    zero.write_bytes(b"")
    fresh_tmp = tmp_path / "cube_s1_g0_7.npz.tmp"
    fresh_tmp.write_bytes(b"x")
    for f in (tmp, zero):
        os.utime(f, (old, old))
    stats = sweep_once(_cfg(tmp_path))
    assert stats["n_tmp_deleted"] == 2
    assert not tmp.exists() and not zero.exists()
    assert fresh_tmp.exists()  # under tmp_age_h: kept


def test_tier1_age_cap_deletes_old_event_dirs(tmp_path) -> None:
    old = _event(tmp_path, "oldev", age_h=100)
    young = _event(tmp_path, "youngev", age_h=1)
    stats = sweep_once(_cfg(tmp_path, max_age_h=96.0))
    assert stats["n_age_deleted"] == 1
    assert not old.exists() and young.exists()


def test_tier2_size_cap_prefers_uploaded_oldest_first(tmp_path) -> None:
    # four 1-byte-scaled dirs; cap forces deletion of two. The two
    # UPLOADED dirs must go first (oldest first) even though an
    # un-uploaded dir is older than one of them.
    up_old = _event(tmp_path, "up_old", age_h=50, n_bytes=1000,
                    uploaded=True)
    up_new = _event(tmp_path, "up_new", age_h=10, n_bytes=1000,
                    uploaded=True)
    raw_mid = _event(tmp_path, "raw_mid", age_h=30, n_bytes=1000)
    raw_new = _event(tmp_path, "raw_new", age_h=1, n_bytes=1000)
    cfg = _cfg(tmp_path, max_total_gb=3.5e3 / GB, low_water_gb=2e3 / GB)
    stats = sweep_once(cfg)
    assert stats["n_size_deleted"] == 2
    assert stats["n_unuploaded_sacrificed"] == 0
    assert not up_old.exists() and not up_new.exists()
    assert raw_mid.exists() and raw_new.exists()


def test_tier2_sacrifices_unuploaded_only_when_needed(tmp_path) -> None:
    up = _event(tmp_path, "up", age_h=50, n_bytes=1000, uploaded=True)
    raw = _event(tmp_path, "raw", age_h=30, n_bytes=1000)
    keep = _event(tmp_path, "keep", age_h=1, n_bytes=1000)
    cfg = _cfg(tmp_path, max_total_gb=2.5e3 / GB, low_water_gb=1.5e3 / GB)
    stats = sweep_once(cfg)
    assert not up.exists()
    assert not raw.exists()
    assert keep.exists()
    assert stats["n_unuploaded_sacrificed"] == 1


def test_sweeper_sweep_now_locks_and_reports(tmp_path) -> None:
    _event(tmp_path, "oldev", age_h=100)
    sw = CubeRetentionSweeper(_cfg(tmp_path))
    stats = sw.sweep_now()
    assert stats["n_age_deleted"] == 1
    assert sw.n_sweeps == 1
    # lock file must not be treated as sweepable content
    assert (tmp_path / ".retention.lock").exists()
    stats2 = sw.sweep_now()
    assert stats2["n_age_deleted"] == 0

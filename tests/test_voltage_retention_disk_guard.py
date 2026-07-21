"""Disk-headroom + storm-containment guard (2026-07-21 ENOSPC fleet stall).

A C2 voltage-dump broadcast storm filled the root filesystem on 9/16
corr nodes because ``voltage_retention.py`` had NO free-space check
anywhere before writing a ~6.47 GiB fragment. These tests cover the
guard added at :func:`dsart.services.voltage_retention.check_disk_headroom`
plus the worker-loop wiring: refusal below headroom, refusal above the
cumulative staging cap, mid-write ENOSPC cleanup + continue, and that
the normal (unguarded) path is untouched.

Convention follows ``tests/test_voltage_retention_manifest.py``: a tiny
``VoltageRing`` (bytes_per_block=8) stands in for the real ~288 MiB fada
block so writes are cheap; the guard math itself uses the real
``FADA_BYTES_PER_BLOCK`` constant via ``RetentionConfig.
expected_fragment_bytes``, independent of the ring's test block size.
"""

from __future__ import annotations

import errno
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dsart.dump.voltage_ring import VoltageRing
from dsart.services import voltage_retention as vr


def _block(nbytes: int, fill: int) -> np.ndarray:
    return np.full(nbytes, fill % 256, dtype=np.uint8)


def _make_service(tmp_path, **cfg_overrides):
    cfg_kwargs = dict(
        fada_key=0xFADA,
        cn_id=3,
        chgroup=0,
        bind_host="127.0.0.1",
        bind_port=0,
        staging_dir=tmp_path,
        retention_blocks=8,
        n_pre=1,
        n_post=1,
        dump_wait_s=0.2,
        queue_max=8,
    )
    cfg_kwargs.update(cfg_overrides)
    cfg = vr.RetentionConfig(**cfg_kwargs)
    ring = VoltageRing(n_blocks=8, bytes_per_block=8)
    for b in range(4):
        ring.store(b, _block(8, b + 1))
    svc = vr.VoltageRetentionService(cfg, mon_store={}, ring=ring)
    return svc


def _drain_worker(svc, timeout: float = 2.0) -> None:
    """Run ``_worker_loop`` in a thread until the queue empties, then stop."""
    t = threading.Thread(target=svc._worker_loop, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while not svc._q.empty() and time.time() < deadline:
        time.sleep(0.01)
    # let the in-flight item finish processing
    time.sleep(0.1)
    svc._stop.set()
    t.join(timeout=timeout)
    assert not t.is_alive(), "worker thread did not exit (stuck / crashed?)"


# ---------------------------------------------------------------------------
# check_disk_headroom (pure function)
# ---------------------------------------------------------------------------


def test_check_disk_headroom_ok(tmp_path, monkeypatch):
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 100 * (1 << 30)
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)
    ok, free = vr.check_disk_headroom(tmp_path, required_bytes=10 * (1 << 30))
    assert ok is True
    assert free == 100 * (1 << 30)


def test_check_disk_headroom_refuses_below_required(tmp_path, monkeypatch):
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 1 * (1 << 30)
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)
    ok, free = vr.check_disk_headroom(tmp_path, required_bytes=10 * (1 << 30))
    assert ok is False
    assert free == 1 * (1 << 30)


# ---------------------------------------------------------------------------
# Worker loop: refusal below headroom
# ---------------------------------------------------------------------------


def test_worker_skips_dump_below_headroom(tmp_path, monkeypatch):
    svc = _make_service(tmp_path)
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 1  # far below any floor
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)

    svc._q.put_nowait(("dump", "stormev", 1 * 2048, 0.0))
    _drain_worker(svc)

    assert svc._counters["dumps_skipped_disk_full"] == 1
    assert svc._counters["dumps_done"] == 0
    out_path, json_path = vr.staged_paths(tmp_path, "stormev", 0)
    assert not out_path.exists()
    assert not json_path.exists()


# ---------------------------------------------------------------------------
# Worker loop: cumulative staging cap refusal
# ---------------------------------------------------------------------------


def test_worker_skips_dump_over_staging_cap(tmp_path, monkeypatch):
    # Plenty of free disk, but the cumulative cap is already exceeded.
    svc = _make_service(tmp_path, staging_max_total_bytes=1)
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 500 * (1 << 30)
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)

    svc._q.put_nowait(("dump", "capev", 1 * 2048, 0.0))
    _drain_worker(svc)

    assert svc._counters["dumps_skipped_staging_cap"] == 1
    assert svc._counters["dumps_done"] == 0
    out_path, _ = vr.staged_paths(tmp_path, "capev", 0)
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Worker loop: mid-write ENOSPC cleanup + continue
# ---------------------------------------------------------------------------


def test_worker_enospc_cleanup_and_continues(tmp_path, monkeypatch):
    svc = _make_service(tmp_path)
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 500 * (1 << 30)  # headroom guard passes
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)

    calls = {"n": 0}
    real_write = vr.write_window_to_staging

    def flaky_write(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise vr.DiskFullError("synthetic ENOSPC")
        return real_write(**kwargs)

    monkeypatch.setattr(vr, "write_window_to_staging", flaky_write)

    svc._q.put_nowait(("dump", "failev", 1 * 2048, 0.0))
    svc._q.put_nowait(("dump", "okev", 1 * 2048, 0.0))
    _drain_worker(svc)

    assert svc._counters["dumps_failed_enospc"] == 1
    # worker survived the ENOSPC and processed the next queued dump
    assert svc._counters["dumps_done"] == 1
    out_path, _ = vr.staged_paths(tmp_path, "okev", 0)
    assert out_path.exists()


def test_write_window_to_staging_enospc_cleans_up_partial(tmp_path, monkeypatch):
    """Unit-level: a real ENOSPC OSError during fh.write is turned into
    DiskFullError and the partial .tmp fragment is removed."""
    ring = VoltageRing(n_blocks=4, bytes_per_block=8)
    for b in range(3):
        ring.store(b, _block(8, b + 1))

    real_open = open

    def flaky_open(path, mode="r", *a, **kw):
        fh = real_open(path, mode, *a, **kw)
        if "tmp" in str(path) and "w" in mode:
            real_write_bytes = fh.write

            def bad_write(data):
                real_write_bytes(data)
                raise OSError(errno.ENOSPC, "No space left on device")

            fh.write = bad_write
        return fh

    monkeypatch.setattr(vr, "open", flaky_open, raising=False)

    with pytest.raises(vr.DiskFullError):
        vr.write_window_to_staging(
            ring=ring,
            event_name="enospcev",
            event_specnum=1 * 2048,
            cn_id=3,
            chgroup=0,
            staging_dir=tmp_path,
            n_pre=1,
            n_post=1,
        )

    out_path, json_path = vr.staged_paths(tmp_path, "enospcev", 0)
    tmp_path_out = out_path.with_suffix(out_path.suffix + ".tmp")
    assert not tmp_path_out.exists()
    assert not out_path.exists()
    assert not json_path.exists()


# ---------------------------------------------------------------------------
# Normal path untouched
# ---------------------------------------------------------------------------


def test_worker_normal_dump_still_succeeds(tmp_path, monkeypatch):
    svc = _make_service(tmp_path)
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 500 * (1 << 30)
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)

    svc._q.put_nowait(("dump", "normalev", 1 * 2048, 0.0))
    _drain_worker(svc)

    assert svc._counters["dumps_done"] == 1
    assert svc._counters["dumps_skipped_disk_full"] == 0
    assert svc._counters["dumps_skipped_staging_cap"] == 0
    assert svc._counters["dumps_failed_enospc"] == 0
    out_path, json_path = vr.staged_paths(tmp_path, "normalev", 0)
    assert out_path.exists()
    assert json_path.exists()
    # cumulative staged-bytes tracker picked up the write
    assert svc._staged_bytes == out_path.stat().st_size


def test_staged_bytes_seeded_from_existing_files(tmp_path):
    (tmp_path / "preexisting_sb00_data.out").write_bytes(b"x" * 4096)
    svc = _make_service(tmp_path)
    assert svc._staged_bytes == 4096


def test_delete_decrements_staged_bytes(tmp_path, monkeypatch):
    svc = _make_service(tmp_path)
    Usage = type("Usage", (), {})
    usage = Usage()
    usage.free = 500 * (1 << 30)
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)

    svc._q.put_nowait(("dump", "delev", 1 * 2048, 0.0))
    _drain_worker(svc)
    assert svc._staged_bytes > 0

    svc._stop.clear()
    svc._q.put_nowait(("delete", "delev", 0, 0.0))
    _drain_worker(svc)
    assert svc._staged_bytes == 0
    assert svc._counters["deletes_done"] == 1

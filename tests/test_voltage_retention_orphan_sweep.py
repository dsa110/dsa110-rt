"""Orphan-eviction guard (2026-07-24 stranded-staging fleet stall).

The 2026-07-21 cumulative-staging cap has no eviction, and C3 only ever
deletes events it adjudicates. A C2 dump-trigger whose event C2 then
DISCARDS (cube-incomplete under an RFI storm) strands its fragment in
staging forever; on 2026-07-23 31 such orphans (~1.81 TiB fleet-wide)
pinned the cap and voltage_retention skipped ~half of every dump for
~18 h. These tests cover the periodic sweep added at
:meth:`dsart.services.voltage_retention.VoltageRetentionService.
_sweep_orphans` + the ``evict`` worker branch: only over-TTL fragments
are evicted, eviction removes files + decrements ``_staged_bytes`` +
counts apart from C3 deletes, and the sweep is a no-op when disabled or
when everything is fresh.

Harness mirrors ``tests/test_voltage_retention_disk_guard.py``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

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
    return vr.VoltageRetentionService(cfg, mon_store={}, ring=ring)


def _drain_worker(svc, timeout: float = 2.0) -> None:
    t = threading.Thread(target=svc._worker_loop, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while not svc._q.empty() and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    svc._stop.set()
    t.join(timeout=timeout)
    assert not t.is_alive(), "worker thread did not exit"


def _stage_fragment(tmp_path, event, chgroup=0, nbytes=4096, age_s=0.0):
    """Write an ``{event}_sbNN`` .out/.json pair, optionally back-dated."""
    out_path, json_path = vr.staged_paths(tmp_path, event, chgroup)
    out_path.write_bytes(b"x" * nbytes)
    json_path.write_text("{}")
    if age_s:
        old = time.time() - age_s
        for p in (out_path, json_path):
            os.utime(p, (old, old))
    return out_path, json_path


# ---------------------------------------------------------------------------
# scan_staged_events (pure helper)
# ---------------------------------------------------------------------------


def test_scan_staged_events_lists_this_chgroup_only(tmp_path):
    _stage_fragment(tmp_path, "evA", chgroup=0)
    _stage_fragment(tmp_path, "evB", chgroup=0)
    # a different chgroup's fragment and stray files must be ignored
    _stage_fragment(tmp_path, "evC", chgroup=5)
    (tmp_path / "note.txt").write_text("ignore me")

    found = dict(vr.scan_staged_events(tmp_path, chgroup=0))
    assert set(found) == {"evA", "evB"}
    assert all(isinstance(m, float) for m in found.values())


def test_scan_staged_events_missing_dir(tmp_path):
    assert vr.scan_staged_events(tmp_path / "nope", chgroup=0) == []


# ---------------------------------------------------------------------------
# _sweep_orphans: age selection
# ---------------------------------------------------------------------------


def test_sweep_enqueues_only_over_ttl(tmp_path):
    svc = _make_service(tmp_path, staging_orphan_ttl_s=3600.0)
    _stage_fragment(tmp_path, "oldorphan", age_s=7200.0)   # 2 h > TTL
    _stage_fragment(tmp_path, "freshkeep", age_s=60.0)     # 1 min < TTL

    n = svc._sweep_orphans()
    assert n == 1
    queued = [svc._q.get_nowait() for _ in range(svc._q.qsize())]
    assert queued == [("evict", "oldorphan", 0, 0.0)]


def test_sweep_disabled_when_ttl_zero(tmp_path):
    svc = _make_service(tmp_path, staging_orphan_ttl_s=0.0)
    _stage_fragment(tmp_path, "ancient", age_s=1e6)
    assert svc._sweep_orphans() == 0
    assert svc._q.empty()


def test_sweep_noop_when_all_fresh(tmp_path):
    svc = _make_service(tmp_path, staging_orphan_ttl_s=3600.0)
    _stage_fragment(tmp_path, "fresh1", age_s=10.0)
    _stage_fragment(tmp_path, "fresh2", age_s=10.0)
    assert svc._sweep_orphans() == 0
    assert svc._q.empty()


# ---------------------------------------------------------------------------
# evict worker branch: removes + decrements + counts separately
# ---------------------------------------------------------------------------


def test_evict_removes_decrements_and_counts_apart_from_delete(tmp_path):
    _stage_fragment(tmp_path, "orphan", nbytes=4096, age_s=99999.0)
    svc = _make_service(tmp_path, staging_orphan_ttl_s=3600.0)
    # Seed counts the whole dir (.out + the 2-byte .json); the
    # delete/evict path decrements only the .out size (via
    # _staged_fragment_bytes) — a pre-existing byte-scale accounting
    # asymmetry shared with C3 deletes, harmless against a 120 GiB cap.
    seed = svc._staged_bytes
    assert seed == 4096 + len(b"{}")

    assert svc._sweep_orphans() == 1
    _drain_worker(svc)

    out_path, json_path = vr.staged_paths(tmp_path, "orphan", 0)
    assert not out_path.exists()
    assert not json_path.exists()
    assert svc._staged_bytes == seed - 4096       # only the .out is subtracted
    assert svc._counters["orphans_evicted"] == 1
    assert svc._counters["deletes_done"] == 0     # eviction != C3 delete


def test_sweep_respects_full_queue(tmp_path):
    svc = _make_service(tmp_path, staging_orphan_ttl_s=3600.0, queue_max=2)
    for i in range(5):
        _stage_fragment(tmp_path, f"orph{i}", age_s=99999.0)
    n = svc._sweep_orphans()
    # queue holds at most queue_max; sweep stops cleanly, no exception
    assert n <= 2
    assert svc._q.qsize() == n

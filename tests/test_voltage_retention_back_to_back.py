"""Back-to-back voltage dumps (2026-11-20).

A fragment takes ~18-25 s to write while the RAM ring holds ~25 s, so a
second trigger for the same burst used to be copied out of the ring only after
the first write finished, by when its window had lapped (261120czvd, 45 ms
after 261120lnkq: 0 blocks on n07/n08/n10). The worker now copies queued
windows out between the blocks of a write, sharing blocks with the window
before, and hard-links an identical window.

Here the ring "laps" (old blocks overwritten) right after the first block of
the first write, which is exactly when the old code would lose the second dump.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from dsart.dump.voltage_ring import VoltageRing
from dsart.services import voltage_retention as vr


def _block(fill: int) -> np.ndarray:
    return np.full(8, fill % 256, dtype=np.uint8)


def _service(tmp_path: Path, monkeypatch, **over) -> vr.VoltageRetentionService:
    kw = dict(fada_key=0xFADA, cn_id=3, chgroup=0, bind_host="127.0.0.1", bind_port=0,
              staging_dir=tmp_path, retention_blocks=8, n_pre=1, n_post=1,
              dump_wait_s=0.2, queue_max=8)
    kw.update(over)
    ring = VoltageRing(n_blocks=8, bytes_per_block=8)
    for b in range(4):
        ring.store(b, _block(b + 1))
    svc = vr.VoltageRetentionService(vr.RetentionConfig(**kw), mon_store={}, ring=ring)
    usage = type("U", (), {"free": 500 * (1 << 30)})()
    monkeypatch.setattr(vr.shutil, "disk_usage", lambda p: usage)
    return svc


def _lap_after_first_block(svc: vr.VoltageRetentionService) -> None:
    """Overwrite the whole ring with newer blocks after the first block of
    the first write (after the worker's chance to copy queued windows)."""
    orig = svc._absorb
    state = {"done": False}

    def wrapped() -> None:
        orig()
        if not state["done"]:
            state["done"] = True
            for b in range(4, 12):
                svc._ring.store(b, _block(b + 1))
    svc._absorb = wrapped


def _run(svc: vr.VoltageRetentionService, timeout: float = 5.0) -> None:
    t = threading.Thread(target=svc._worker_loop, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while (not svc._q.empty() or svc._pending or svc._on_deck) and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.15)
    svc._stop.set()
    t.join(timeout=timeout)
    assert not t.is_alive()


def _frag(tmp_path: Path, name: str) -> Path:
    return vr.staged_paths(tmp_path, name, 0)[0]


def test_old_behaviour_loses_the_second_dump(tmp_path, monkeypatch):
    # Control: with no window copied ahead, the second dump of the pair is
    # extracted after the ring lapped and writes nothing.
    svc = _service(tmp_path, monkeypatch, max_prepared_windows=0)
    _lap_after_first_block(svc)
    svc._q.put_nowait(("dump", "lnkq", 1 * 2048, 0.0))
    svc._q.put_nowait(("dump", "czvd", 1 * 2048 + 352, 0.0))
    _run(svc)
    assert _frag(tmp_path, "lnkq").stat().st_size == 24
    assert _frag(tmp_path, "czvd").stat().st_size == 0
    assert svc._counters["dumps_failed"] == 1


def test_same_burst_pair_is_copied_ahead_and_hard_linked(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    _lap_after_first_block(svc)
    svc._q.put_nowait(("dump", "lnkq", 1 * 2048, 0.0))
    svc._q.put_nowait(("dump", "czvd", 1 * 2048 + 352, 0.0))   # same block, 12 ms later
    _run(svc)
    a, b = _frag(tmp_path, "lnkq"), _frag(tmp_path, "czvd")
    assert a.read_bytes() == b.read_bytes() and a.stat().st_size == 24
    assert a.stat().st_ino == b.stat().st_ino                    # hard link, no second write
    c = svc._counters
    assert c["dumps_done"] == 2 and c["dumps_failed"] == 0
    assert c["dumps_prepared_early"] == 1 and c["blocks_shared"] == 3 and c["dumps_linked"] == 1
    assert svc._recent_blocks == {}                              # window RAM released


def test_overlapping_shifted_window_shares_blocks(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    _lap_after_first_block(svc)
    svc._q.put_nowait(("dump", "aaaa", 1 * 2048, 0.0))           # blocks 0,1,2
    svc._q.put_nowait(("dump", "bbbb", 2 * 2048, 0.0))           # blocks 1,2,3
    _run(svc)
    want = b"".join(_block(n + 1).tobytes() for n in (1, 2, 3))
    assert _frag(tmp_path, "bbbb").read_bytes() == want
    assert svc._counters["blocks_shared"] == 2 and svc._counters["dumps_linked"] == 0
    assert svc._counters["dumps_failed"] == 0


def test_delete_keeps_its_order_behind_the_dump(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    svc._q.put_nowait(("dump", "aaaa", 1 * 2048, 0.0))
    svc._q.put_nowait(("dump", "bbbb", 1 * 2048, 0.0))
    svc._q.put_nowait(("delete", "bbbb", 0, 0.0))
    _run(svc)
    assert _frag(tmp_path, "aaaa").exists()
    assert not _frag(tmp_path, "bbbb").exists()                  # written, then deleted
    assert svc._counters["deletes_done"] == 1


def test_prepared_windows_are_capped(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch, max_prepared_windows=1)
    for i, name in enumerate(("aaaa", "bbbb", "cccc")):
        svc._q.put_nowait(("dump", name, 1 * 2048 + i, 0.0))
    seen = []
    orig = svc._absorb

    def spy() -> None:
        orig()
        seen.append(svc._n_prepared())
    svc._absorb = spy
    _run(svc)
    assert max(seen) <= 1 and svc._counters["dumps_done"] == 3


def test_dump_not_ready_waits_on_deck(tmp_path, monkeypatch):
    # The second dump's post-window is not in the ring during the first write:
    # it waits on deck (not dropped) and is written once its blocks arrive.
    svc = _service(tmp_path, monkeypatch, dump_wait_s=2.0)
    orig = svc._absorb
    state = {"n": 0}

    def wrapped() -> None:
        orig()
        state["n"] += 1
        if state["n"] == 3:                                     # end of the first write
            for b in range(4, 7):
                svc._ring.store(b, _block(b + 1))
    svc._absorb = wrapped
    svc._q.put_nowait(("dump", "aaaa", 1 * 2048, 0.0))
    svc._q.put_nowait(("dump", "late", 5 * 2048, 0.0))           # needs block 6
    _run(svc)
    assert _frag(tmp_path, "late").stat().st_size == 24
    assert svc._counters["dumps_failed"] == 0

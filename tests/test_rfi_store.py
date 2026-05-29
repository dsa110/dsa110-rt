"""Unit tests for the dsa_monitor RFI window store (M7.6).

Focus: the producer-restart path. The corr-node ``rfi_monitor_export``
sidecar's ``seq`` counter is monotonic only within one process
lifetime; every fleet relaunch resets it to ~0. The long-lived h23
dashboard must drop its stale per-cn high-water mark on restart,
otherwise the ``r.seq <= last_seq`` dedup rejects every post-restart
record and the Antennas/RFI tab shows ``windows=0`` until the counter
climbs back past the pre-restart value (~hours).

The module under test lives in
``tools/dashboard/dsa_monitor/rfi_store.py`` (one directory off the
canonical test PYTHONPATH); we make it importable by inserting the
directory at the top of sys.path before the import.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    HERE, "..", "tools", "dashboard", "dsa_monitor",
))
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import rfi_store  # noqa: E402
from rfi_client import DecodedRFIMonRecord  # noqa: E402
from corr_topology import CORR_NODES  # noqa: E402


_CN = CORR_NODES[0].cn_id


def _rec(seq: int, *, publish_unix: float) -> DecodedRFIMonRecord:
    z = np.zeros((1, 1, 1), dtype=np.float32)
    u = np.zeros((1, 1, 1), dtype=np.uint8)
    return DecodedRFIMonRecord(
        cn_id=_CN,
        seq=seq,
        publish_unix=publish_unix,
        block_n_start=seq * 16,
        block_n_end=seq * 16 + 16,
        n_cubes=16,
        n_cubes_warmup=0,
        scalars={},
        s1_full_mean=z,
        mask_count_final=u,
        mask_count_sk=u,
        mask_count_bp=u,
        mask_count_grp=u,
        mask_count_sumthr=u,
        mask_count_fa=u,
    )


def _n_windows(store: "rfi_store.RFIWindowStore") -> int:
    snap = store.snapshot()
    for ring in snap.per_chgroup:
        if ring.cn.cn_id == _CN:
            return len(ring.records)
    raise AssertionError("cn not found in snapshot")


def test_normal_append_dedupes_monotonic_seq() -> None:
    store = rfi_store.RFIWindowStore()
    now = time.time()
    assert store.append([_rec(1, publish_unix=now)], cn_id=_CN) == 1
    # Re-delivering the same seq is a no-op.
    assert store.append([_rec(1, publish_unix=now)], cn_id=_CN) == 0
    assert store.append([_rec(2, publish_unix=now + 2)], cn_id=_CN) == 1
    assert _n_windows(store) == 2
    assert store.last_seq_for(_CN) == 2


def test_producer_restart_resets_high_water_and_ring() -> None:
    store = rfi_store.RFIWindowStore()
    now = time.time()
    # Pre-restart: counter climbed high.
    for s in range(1, 51):
        store.append([_rec(s, publish_unix=now + s)], cn_id=_CN)
    assert store.last_seq_for(_CN) == 50
    assert _n_windows(store) == 50

    # Producer restarts: seq resets to 1, fresh wall-clock. Without the
    # restart guard this record is <= 50 and would be dropped forever.
    store.append([_rec(1, publish_unix=now + 100)], cn_id=_CN)
    assert store.last_seq_for(_CN) == 1
    # Stale pre-restart ring was cleared; only the new record remains.
    assert _n_windows(store) == 1

    # And the post-restart stream now accumulates normally.
    store.append([_rec(2, publish_unix=now + 102)], cn_id=_CN)
    assert _n_windows(store) == 2
    assert store.last_seq_for(_CN) == 2


def test_restart_backfill_batch_replaces_stale_ring() -> None:
    store = rfi_store.RFIWindowStore()
    now = time.time()
    for s in range(1, 51):
        store.append([_rec(s, publish_unix=now + s)], cn_id=_CN)
    # Simulate a /api/recent re-backfill after restart: a batch whose
    # max seq is still below the stale high-water mark.
    backfill = [_rec(s, publish_unix=now + 200 + s) for s in range(1, 11)]
    store.append(backfill, cn_id=_CN)
    assert _n_windows(store) == 10
    assert store.last_seq_for(_CN) == 10

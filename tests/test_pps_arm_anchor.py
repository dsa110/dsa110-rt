"""Tests for the SNAP GPS-PPS capture-arm time anchoring (2026-07-20).

Covers:
  * ``compute_pps_armed_mjd`` — exact PPS arithmetic incl. the 2^35
    wrap count, validated against the real 2026-07 incident numbers.
  * ``RtOrchestrator._read_snap_pps_epoch`` — per-SNAP consensus,
    exclusion of the dsart-owned ``/mon/snap/1`` key, disagreement and
    absence fallbacks.
  * ``_verb_utc_start`` — writes a PPS-anchored ``armed_mjd`` with
    provenance, and falls back to the wall latch (with provenance)
    when no epoch consensus exists.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from dsart.services import dsart_rt
from dsart.services.dsart_rt import (
    RtOrchestrator,
    SNAP_SEQ_TICK_US,
    SNAP_SEQ_WRAP,
    compute_pps_armed_mjd,
)

# Real numbers from the 2026-07-20 wrap incident.
PPS_EPOCH = 61228.589212962965          # 2026-07-07 14:08:28.000 UT
ARM_SEQ = 27_332_965_236                # utc_start_rt of the 07-17 arm
WALL_LATCH = 61238.95547279             # what Time.now() stamped then
TICK_DAYS = SNAP_SEQ_TICK_US * 1e-6 / 86400.0


def test_compute_pps_armed_mjd_matches_incident_numbers():
    armed, k = compute_pps_armed_mjd(
        PPS_EPOCH, ARM_SEQ, now_mjd=WALL_LATCH,
    )
    assert k == 0
    # PPS-derived arm is ~1.76 s LATER than the wall latch (the arm
    # margin) — the systematic the upgrade removes.
    assert (armed - WALL_LATCH) * 86400.0 == pytest.approx(1.76, abs=0.3)
    assert armed == pytest.approx(
        PPS_EPOCH + ARM_SEQ * TICK_DAYS, abs=1e-12,
    )


def test_compute_pps_armed_mjd_wrap_aware():
    wrap_days = SNAP_SEQ_WRAP * TICK_DAYS          # 13.0312 d
    # An arm 13.5 days after the SNAP sync uses a small post-wrap seq;
    # the wrap count must land the result at the wall clock, not
    # 13 days in the past.
    seq = int(0.5 * 86400 / (SNAP_SEQ_TICK_US * 1e-6))   # 0.5 d of ticks
    now = PPS_EPOCH + wrap_days + 0.5
    armed, k = compute_pps_armed_mjd(PPS_EPOCH, seq, now_mjd=now)
    assert k == 1
    assert abs(armed - now) * 86400.0 < 1.0
    # Two wraps out.
    now2 = PPS_EPOCH + 2 * wrap_days + 0.5
    armed2, k2 = compute_pps_armed_mjd(PPS_EPOCH, seq, now_mjd=now2)
    assert k2 == 2
    assert abs(armed2 - now2) * 86400.0 < 1.0
    # A seq value >= 2^35 (already-unwrapped arm from the wrap-aware
    # capture's mon feed) is reduced modulo the wrap and re-lifted.
    armed3, k3 = compute_pps_armed_mjd(
        PPS_EPOCH, seq + SNAP_SEQ_WRAP, now_mjd=now,
    )
    assert (armed3, k3) == (armed, k)
    # k never goes negative (wall clock earlier than epoch+seq).
    armed4, k4 = compute_pps_armed_mjd(PPS_EPOCH, seq, now_mjd=PPS_EPOCH)
    assert k4 == 0


class _FakeStore:
    def __init__(self, docs: dict[str, Any]):
        self.docs = dict(docs)
        self.puts: list[tuple[str, dict]] = []

    def get_dict(self, key):
        if key not in self.docs:
            raise KeyError(key)
        return self.docs[key]

    def put_dict(self, key, value):
        self.puts.append((key, dict(value)))

    def last_put(self, key):
        for k, v in reversed(self.puts):
            if k == key:
                return v
        return None


def _orch(store) -> RtOrchestrator:
    o = object.__new__(RtOrchestrator)
    o._store = store
    o._armed_at = None
    o._send_utc_udp = lambda payload: None
    return o


def _epoch_docs(n_ids, mjd=PPS_EPOCH):
    return {
        f"/mon/snap/{n}/armed_mjd": {"armed_mjd": mjd} for n in n_ids
    }


def test_read_snap_pps_epoch_consensus_and_exclusions():
    # Consensus over >=3 snaps; the dsart-owned n=1 key must never be
    # consulted (poison it to prove it).
    docs = _epoch_docs([2, 3, 10, 11])
    docs["/mon/snap/1/armed_mjd"] = {"armed_mjd": 12345.0}
    o = _orch(_FakeStore(docs))
    assert o._read_snap_pps_epoch() == pytest.approx(PPS_EPOCH)

    # Too few epochs -> None.
    o2 = _orch(_FakeStore(_epoch_docs([2, 3])))
    assert o2._read_snap_pps_epoch() is None

    # Disagreement beyond tolerance -> None.
    docs3 = _epoch_docs([2, 3, 4])
    docs3["/mon/snap/4/armed_mjd"] = {"armed_mjd": PPS_EPOCH + 0.5}
    docs3["/mon/snap/5/armed_mjd"] = {"armed_mjd": PPS_EPOCH - 0.5}
    o3 = _orch(_FakeStore(docs3))
    assert o3._read_snap_pps_epoch() is None or (
        # 2 of 3 within tolerance is below SNAP_PPS_MIN_AGREE=3
        False
    )


def test_verb_utc_start_writes_pps_anchor_with_provenance(monkeypatch):
    now = time.time()
    now_mjd = now / 86400.0 + 40587.0
    # Choose an arm seq so epoch + seq*tick lands ~2 s after "now".
    seq = int(round(((now_mjd - PPS_EPOCH) * 86400.0 + 2.0)
                    / (SNAP_SEQ_TICK_US * 1e-6))) % SNAP_SEQ_WRAP
    store = _FakeStore(_epoch_docs([2, 3, 10]))
    o = _orch(store)
    o._verb_utc_start(seq)

    rec = store.last_put("/mon/snap/1/armed_mjd")
    assert rec is not None
    assert rec["source"] == "pps_epoch"
    assert rec["arm_seq"] == seq
    assert rec["pps_epoch_mjd"] == pytest.approx(PPS_EPOCH)
    assert rec["seq_tick_us"] == SNAP_SEQ_TICK_US
    # PPS anchor ~2 s after the verb wall clock, never equal to it.
    d_s = (rec["armed_mjd"] - rec["verb_wall_mjd"]) * 86400.0
    assert 0.5 < d_s < 5.0
    # Legacy dsamfs key refreshed alongside.
    assert store.last_put("/mon/snap/1/utc_start") == {"utc_start": 0}
    # utc_start_rt mirror keeps the raw seq.
    assert store.last_put("/mon/snap/1/utc_start_rt") == {"val": seq}


def test_verb_utc_start_falls_back_to_wall_latch(monkeypatch):
    store = _FakeStore({})                     # no SNAP epochs at all
    o = _orch(store)
    o._verb_utc_start(123456)
    rec = store.last_put("/mon/snap/1/armed_mjd")
    assert rec["source"] == "wall_latch"
    assert rec["pps_epoch_mjd"] is None
    now_mjd = time.time() / 86400.0 + 40587.0
    assert abs(rec["armed_mjd"] - now_mjd) * 86400.0 < 5.0


def test_verb_utc_start_rejects_stale_epoch(monkeypatch):
    # Epoch consensus exists but implies an arm time far from the wall
    # clock (e.g. SNAPs re-armed without refreshing etcd) -> fallback.
    store = _FakeStore(_epoch_docs([2, 3, 10], mjd=PPS_EPOCH - 100.0))
    o = _orch(store)
    o._verb_utc_start(1000)
    rec = store.last_put("/mon/snap/1/armed_mjd")
    assert rec["source"] == "wall_latch"

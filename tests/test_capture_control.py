"""Unit tests for `dsart.services.capture_control`.

Behavioural surface tested:

1. `_snap_to_mon_dict` produces the documented etcd payload schema.
2. `_mon_dict_unavailable` produces the documented placeholder when
   the shm is missing.
3. `CaptureControlService._tick`:
   a. attaches to the shm when present; publishes the snapshot.
   b. publishes the unavailable placeholder when the shm is missing.
   c. flags `degraded=true` when the snapshot is stale.
4. CLI argparse: ports parser accepts comma-separated lists.

These tests do not exercise the real /dev/shm path; we pass an
in-memory `MockStore` and patch the shm directory with a tmpdir,
exactly matching the pattern in test_capture_mon_shm.py.
"""

from __future__ import annotations

import os
import struct
import time
from typing import Any
from unittest import mock

import pytest

from dsart.capture import mon_shm
from dsart.services import capture_control


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _FakeStore:
    """Mock for the etcd put_dict surface."""

    def __init__(self) -> None:
        self.puts: list[tuple[str, dict[str, Any]]] = []

    def put_dict(self, key: str, value: dict[str, Any]) -> None:
        self.puts.append((key, dict(value)))

    def last_for(self, key: str) -> dict[str, Any] | None:
        for k, v in reversed(self.puts):
            if k == key:
                return v
        return None


def _write_fake_shm(
    tmp_path,
    udp_port: int,
    *,
    last_update_utc_ns: int | None = None,
    arm_state: int = int(mon_shm.ArmState.WRITING),
    **kw: Any,
) -> None:
    """Drop a binary-shaped fake shm file into ``tmp_path``."""
    if last_update_utc_ns is None:
        last_update_utc_ns = time.time_ns()
    payload = struct.pack(
        mon_shm._FMT,
        mon_shm._MON_MAGIC,
        mon_shm._MON_VERSION,
        udp_port,
        kw.get("control_port", 11223),
        kw.get("socket_rcvbuf_bytes", 268435456),
        arm_state,
        kw.get("pid", 99),
        kw.get("startup_utc_ns", 0),
        last_update_utc_ns,
        kw.get("utc_start_specnum", 100000),
        kw.get("utc_stop_specnum", 0),
        kw.get("last_seq_no", 1234),
        kw.get("n_recv_packets", 50),
        kw.get("n_recv_bytes", 50 * 4608),
        kw.get("n_dropped_payload", 0),
        kw.get("n_dropped_kernel", 0),
        kw.get("n_seq_skipped", 0),
        kw.get("n_too_late", 0),
        kw.get("n_wrong_size", 0),
        kw.get("n_recv_errors", 0),
        kw.get("n_block_writes", 1),
        kw.get("rate_gbps_milli", 1100),
        kw.get("rate_drop_milli", 0),
        kw.get("rate_kernel_drop_pps", 0),
        0, 0, 0, 0, 0,
    )
    path = os.path.join(str(tmp_path), f"dsart-capture-{udp_port}")
    with open(path, "wb") as f:
        f.write(payload)


# ---------------------------------------------------------------------------
# _snap_to_mon_dict / _mon_dict_unavailable
# ---------------------------------------------------------------------------


def test_snap_to_mon_dict_has_required_schema_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    _write_fake_shm(tmp_path, 4011)
    with mon_shm.MonShm.open(4011) as m:
        snap = m.snapshot()
    payload = capture_control._snap_to_mon_dict(
        snap, staleness_threshold_ms=1000.0
    )
    expected_keys = {
        "schema_version", "udp_port", "control_port", "pid",
        "arm_state", "arm_state_int",
        "utc_start_specnum", "utc_stop_specnum", "last_seq_no",
        "socket_rcvbuf_bytes",
        "rate_gbps", "rate_drop_mb_s", "rate_kernel_drop_pps",
        "n_recv_packets", "n_recv_bytes", "n_dropped_payload",
        "n_dropped_kernel", "n_seq_skipped", "n_too_late",
        "n_wrong_size", "n_recv_errors", "n_block_writes",
        "startup_utc_ns", "last_update_utc_ns", "age_ms", "degraded",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["schema_version"] == 1
    assert payload["arm_state"] == "WRITING"
    assert payload["arm_state_int"] == int(mon_shm.ArmState.WRITING)
    assert payload["degraded"] is False


def test_snap_to_mon_dict_flags_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    # 2 seconds ago -- well past the 1 s threshold
    old_ts = time.time_ns() - int(2.0 * 1e9)
    _write_fake_shm(tmp_path, 4011, last_update_utc_ns=old_ts)
    with mon_shm.MonShm.open(4011) as m:
        snap = m.snapshot()
    payload = capture_control._snap_to_mon_dict(
        snap, staleness_threshold_ms=1000.0
    )
    assert payload["degraded"] is True
    assert payload["age_ms"] > 1000.0


def test_mon_dict_unavailable_shape():
    payload = capture_control._mon_dict_unavailable(4011, "shm not present")
    assert payload["schema_version"] == 1
    assert payload["udp_port"] == 4011
    assert payload["arm_state"] == "UNAVAILABLE"
    assert payload["arm_state_int"] == -1
    assert payload["degraded"] is True
    assert payload["shm_status"] == "missing"
    assert payload["reason"] == "shm not present"


# ---------------------------------------------------------------------------
# CaptureControlService._tick
# ---------------------------------------------------------------------------


def test_tick_publishes_unavailable_when_shm_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    store = _FakeStore()
    svc = capture_control.CaptureControlService(
        udp_ports=(4011, 4012),
        cn_id=6,
        store=store,
    )
    svc._tick()
    # Both ports should produce an unavailable placeholder
    assert len(store.puts) == 2
    for key, payload in store.puts:
        assert payload["shm_status"] == "missing"
        assert payload["arm_state"] == "UNAVAILABLE"
        assert payload["degraded"] is True
    keys = sorted(k for k, _ in store.puts)
    assert keys == [
        "/mon/corr_rt/6/capture/4011",
        "/mon/corr_rt/6/capture/4012",
    ]


def test_tick_publishes_snapshot_when_shm_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    _write_fake_shm(tmp_path, 4011, last_seq_no=98765)
    # Leave port 4012 missing.
    store = _FakeStore()
    svc = capture_control.CaptureControlService(
        udp_ports=(4011, 4012),
        cn_id=6,
        store=store,
    )
    svc._tick()
    p4011 = store.last_for("/mon/corr_rt/6/capture/4011")
    p4012 = store.last_for("/mon/corr_rt/6/capture/4012")
    assert p4011 is not None
    assert p4012 is not None
    # 4011 should be the live snapshot
    assert p4011["arm_state"] == "WRITING"
    assert p4011["last_seq_no"] == 98765
    assert p4011["degraded"] is False
    # 4012 should be the placeholder
    assert p4012["shm_status"] == "missing"


def test_tick_marks_stale_snapshot_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    old_ts = time.time_ns() - int(2.0 * 1e9)
    _write_fake_shm(tmp_path, 4011, last_update_utc_ns=old_ts)
    store = _FakeStore()
    svc = capture_control.CaptureControlService(
        udp_ports=(4011,),
        cn_id=6,
        store=store,
    )
    svc._tick()
    payload = store.last_for("/mon/corr_rt/6/capture/4011")
    assert payload is not None
    assert payload["degraded"] is True


def test_tick_reattaches_after_detach(tmp_path, monkeypatch):
    """If the shm disappears mid-run, the next tick should re-attempt."""
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    _write_fake_shm(tmp_path, 4011)
    store = _FakeStore()
    svc = capture_control.CaptureControlService(
        udp_ports=(4011,),
        cn_id=6,
        store=store,
    )
    svc._tick()
    assert store.last_for("/mon/corr_rt/6/capture/4011")["arm_state"] == "WRITING"
    # Remove the shm + force a detach
    os.remove(os.path.join(str(tmp_path), "dsart-capture-4011"))
    svc._detach(4011)
    svc._tick()
    # Should publish a missing placeholder
    last = store.last_for("/mon/corr_rt/6/capture/4011")
    assert last["shm_status"] == "missing"
    # Bring it back -- next tick should reattach
    _write_fake_shm(tmp_path, 4011, arm_state=int(mon_shm.ArmState.ARMED))
    svc._tick()
    last = store.last_for("/mon/corr_rt/6/capture/4011")
    assert last["arm_state"] == "ARMED"


# ---------------------------------------------------------------------------
# CLI argparse
# ---------------------------------------------------------------------------


def test_parse_ports_handles_csv():
    assert capture_control._parse_ports("4011,4012") == (4011, 4012)
    assert capture_control._parse_ports("4011") == (4011,)
    assert capture_control._parse_ports(" 4011 , 4012 ") == (4011, 4012)


def test_parse_ports_rejects_invalid():
    with pytest.raises(Exception):  # argparse.ArgumentTypeError subclass
        capture_control._parse_ports("not-a-port")
    with pytest.raises(Exception):
        capture_control._parse_ports("70000")  # > 65535
    with pytest.raises(Exception):
        capture_control._parse_ports("")  # empty


def test_main_dispatches_to_service(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run(self: capture_control.CaptureControlService, **kw):
        captured["udp_ports"] = self.udp_ports
        captured["cn_id"] = self.cn_id
        captured["mon_cadence_s"] = self.mon_cadence_s
        return 0

    monkeypatch.setattr(capture_control, "_StoreWrapper", lambda: _FakeStore())
    monkeypatch.setattr(
        capture_control.CaptureControlService, "run", fake_run
    )
    rc = capture_control.main([
        "--udp-ports", "4011,4012",
        "--cn-id", "6",
        "--mon-cadence-s", "1.5",
        "--log-level", "WARNING",
    ])
    assert rc == 0
    assert captured == {
        "udp_ports": (4011, 4012),
        "cn_id": 6,
        "mon_cadence_s": 1.5,
    }


# ---------------------------------------------------------------------------
# Mon-key path convention
# ---------------------------------------------------------------------------


def test_mon_key_for_path():
    assert capture_control._mon_key_for(6, 4011) == "/mon/corr_rt/6/capture/4011"
    assert capture_control._mon_key_for(22, 4012) == "/mon/corr_rt/22/capture/4012"

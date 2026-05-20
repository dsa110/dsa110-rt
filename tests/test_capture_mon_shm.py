"""Unit tests for `dsart.capture.mon_shm`.

The C binary `dsart_capture_manythread` publishes a fixed-shape
POSIX-shm segment that the Python sidecar reads. These tests fence
the ABI (struct layout + magic + version) and the read path of the
Python wrapper, *without* actually running the C binary -- we fake
the shm by writing the binary struct ourselves and verifying the
``MonShm`` reader sees what we wrote.

The C binary is exercised end-to-end on the n06 hardware smoke; this
file is the cheap CI gate that catches binary-incompatible struct
edits before deploy.
"""

from __future__ import annotations

import ctypes
import os
import struct
import tempfile
import time

import pytest

from dsart.capture import mon_shm


# ---------------------------------------------------------------------------
# ABI invariants
# ---------------------------------------------------------------------------


def test_struct_size_pinned():
    """Module-level assertion: struct size must match C-side header.

    208 B = 6*uint32 (24) + 18*uint64 (144) + 5*uint64 reserved (40).
    No natural-alignment padding because the uint32 block is itself
    8-byte aligned (24 B = 3*8). Any change to the struct that bumps
    this requires bumping DSART_CAPTURE_MON_VERSION on both sides.
    """
    assert struct.calcsize(mon_shm._FMT) == mon_shm._MON_STRUCT_BYTES == 208


def test_arm_state_enum_values():
    """ArmState integer values must match dsart_arm_state_t in the header."""
    assert int(mon_shm.ArmState.WAITING_FOR_ARM) == 0
    assert int(mon_shm.ArmState.ARMED) == 1
    assert int(mon_shm.ArmState.WRITING) == 2
    assert int(mon_shm.ArmState.STOPPED) == 3


def test_shm_path_for_port_format():
    """Path follows the C-side DSART_CAPTURE_MON_SHM_FMT pattern."""
    assert mon_shm.shm_path_for_port(4011) == "/dev/shm/dsart-capture-4011"
    assert mon_shm.shm_path_for_port(4012) == "/dev/shm/dsart-capture-4012"


# ---------------------------------------------------------------------------
# Helpers to fake a populated shm segment
# ---------------------------------------------------------------------------


def _make_fake_shm(
    udp_port: int,
    *,
    magic: int = mon_shm._MON_MAGIC,
    version: int = mon_shm._MON_VERSION,
    control_port: int = 11223,
    socket_rcvbuf_bytes: int = 268435456,
    arm_state: int = int(mon_shm.ArmState.WRITING),
    pid: int = 12345,
    startup_utc_ns: int = 0,
    last_update_utc_ns: int | None = None,
    utc_start_specnum: int = 100000,
    utc_stop_specnum: int = 0,
    last_seq_no: int = 99,
    n_recv_packets: int = 1000,
    n_recv_bytes: int = 1000 * 4608,
    n_dropped_payload: int = 0,
    n_dropped_kernel: int = 0,
    n_seq_skipped: int = 0,
    n_too_late: int = 0,
    n_wrong_size: int = 0,
    n_recv_errors: int = 0,
    n_block_writes: int = 2,
    rate_gbps_milli: int = 1100,  # 1.1 Gb/s
    rate_drop_milli: int = 0,
    rate_kernel_drop_pps: int = 0,
) -> bytes:
    """Pack a `dsart_capture_mon_t` value as raw bytes."""
    if last_update_utc_ns is None:
        last_update_utc_ns = time.time_ns()
    return struct.pack(
        mon_shm._FMT,
        magic, version, udp_port, control_port, socket_rcvbuf_bytes, arm_state,
        pid, startup_utc_ns, last_update_utc_ns,
        utc_start_specnum, utc_stop_specnum, last_seq_no,
        n_recv_packets, n_recv_bytes, n_dropped_payload, n_dropped_kernel,
        n_seq_skipped, n_too_late, n_wrong_size, n_recv_errors,
        n_block_writes, rate_gbps_milli, rate_drop_milli, rate_kernel_drop_pps,
        0, 0, 0, 0, 0,  # _reserved[5]
    )


def _write_fake_shm_file(udp_port: int, payload: bytes, tmpdir: str) -> str:
    """Write `payload` to a /dev/shm-style file inside tmpdir.

    We use a tmpdir override (not /dev/shm) so tests don't pollute
    the real shm namespace. The caller MUST patch
    `mon_shm._SHM_DIR` to point at tmpdir before invoking MonShm.open.
    """
    path = os.path.join(tmpdir, f"dsart-capture-{udp_port}")
    with open(path, "wb") as f:
        f.write(payload)
    return path


# ---------------------------------------------------------------------------
# MonShm read path
# ---------------------------------------------------------------------------


def test_open_missing_shm_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    with pytest.raises(mon_shm.MonShmNotPresent):
        mon_shm.MonShm.open(4011)


def test_open_bad_magic_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    payload = _make_fake_shm(4011, magic=0xDEADBEEF)
    _write_fake_shm_file(4011, payload, str(tmp_path))
    with pytest.raises(mon_shm.MonShmAbiMismatch, match="bad magic"):
        mon_shm.MonShm.open(4011)


def test_open_version_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    payload = _make_fake_shm(4011, version=99)
    _write_fake_shm_file(4011, payload, str(tmp_path))
    with pytest.raises(mon_shm.MonShmAbiMismatch, match="version"):
        mon_shm.MonShm.open(4011)


def test_open_undersized_shm_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    path = os.path.join(str(tmp_path), "dsart-capture-4011")
    with open(path, "wb") as f:
        f.write(b"\x00" * 16)  # way under 256 bytes
    with pytest.raises(mon_shm.MonShmAbiMismatch, match="too small"):
        mon_shm.MonShm.open(4011)


def test_snapshot_round_trips_all_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    expected = dict(
        udp_port=4011,
        control_port=11223,
        socket_rcvbuf_bytes=268435456,
        arm_state=int(mon_shm.ArmState.WRITING),
        pid=42,
        startup_utc_ns=1_700_000_000_000_000_000,
        last_update_utc_ns=time.time_ns(),
        utc_start_specnum=1234567,
        utc_stop_specnum=9999999,
        last_seq_no=12345,
        n_recv_packets=2_000_000,
        n_recv_bytes=2_000_000 * 4608,
        n_dropped_payload=37,
        n_dropped_kernel=4,
        n_seq_skipped=2,
        n_too_late=8,
        n_wrong_size=1,
        n_recv_errors=0,
        n_block_writes=120,
        rate_gbps_milli=1100,
        rate_drop_milli=0,
        rate_kernel_drop_pps=2,
    )
    payload = _make_fake_shm(**expected)
    _write_fake_shm_file(4011, payload, str(tmp_path))

    with mon_shm.MonShm.open(4011) as m:
        snap = m.snapshot()

    assert snap.udp_port == expected["udp_port"]
    assert snap.control_port == expected["control_port"]
    assert snap.socket_rcvbuf_bytes == expected["socket_rcvbuf_bytes"]
    assert snap.arm_state == mon_shm.ArmState.WRITING
    assert snap.pid == expected["pid"]
    assert snap.utc_start_specnum == expected["utc_start_specnum"]
    assert snap.utc_stop_specnum == expected["utc_stop_specnum"]
    assert snap.last_seq_no == expected["last_seq_no"]
    assert snap.n_recv_packets == expected["n_recv_packets"]
    assert snap.n_recv_bytes == expected["n_recv_bytes"]
    assert snap.n_dropped_payload == expected["n_dropped_payload"]
    assert snap.n_dropped_kernel == expected["n_dropped_kernel"]
    assert snap.n_seq_skipped == expected["n_seq_skipped"]
    assert snap.n_too_late == expected["n_too_late"]
    assert snap.n_wrong_size == expected["n_wrong_size"]
    assert snap.n_recv_errors == expected["n_recv_errors"]
    assert snap.n_block_writes == expected["n_block_writes"]
    assert snap.rate_gbps_milli == expected["rate_gbps_milli"]
    assert snap.rate_drop_milli == expected["rate_drop_milli"]
    assert snap.rate_kernel_drop_pps == expected["rate_kernel_drop_pps"]


def test_snapshot_derived_properties(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    now = time.time_ns()
    payload = _make_fake_shm(
        4011,
        last_update_utc_ns=now,
        rate_gbps_milli=1500,  # 1.5 Gb/s
        rate_drop_milli=250,   # 0.25 MB/s
    )
    _write_fake_shm_file(4011, payload, str(tmp_path))
    with mon_shm.MonShm.open(4011) as m:
        snap = m.snapshot()
    assert snap.rate_gbps == pytest.approx(1.5)
    assert snap.rate_drop_mb_s == pytest.approx(0.25)
    assert snap.age_ms < 100.0  # just stamped
    assert not snap.is_stale


def test_snapshot_stale_when_old(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    old_ts = time.time_ns() - int(2.0 * 1e9)  # 2 seconds ago
    payload = _make_fake_shm(4011, last_update_utc_ns=old_ts)
    _write_fake_shm_file(4011, payload, str(tmp_path))
    with mon_shm.MonShm.open(4011) as m:
        snap = m.snapshot()
    assert snap.is_stale
    assert snap.age_ms > 1000.0


def test_snapshot_to_dict_includes_derived(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    payload = _make_fake_shm(4011)
    _write_fake_shm_file(4011, payload, str(tmp_path))
    with mon_shm.MonShm.open(4011) as m:
        d = m.snapshot().to_dict()
    assert d["arm_state_name"] == "WRITING"
    assert "rate_gbps" in d
    assert "age_ms" in d
    assert "is_stale" in d


def test_context_manager_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(mon_shm, "_SHM_DIR", str(tmp_path))
    payload = _make_fake_shm(4011)
    _write_fake_shm_file(4011, payload, str(tmp_path))
    m = mon_shm.MonShm.open(4011)
    assert m.udp_port == 4011
    m.close()
    # Idempotent close
    m.close()

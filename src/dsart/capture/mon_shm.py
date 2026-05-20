"""Python reader for the dsart_capture_mon POSIX-shm segment.

The C binary `dsart_capture_manythread` publishes a fixed-layout
shm segment at ``/dev/shm/dsart-capture-<udp_port>`` (POSIX ``shm_open``
name ``/dsart-capture-<udp_port>``) with atomic counters describing
the capture-process's live state. This module mmaps that segment
read-only and exposes the counters as a dataclass + ``MonShm`` reader.

ABI is pinned by ``dsart_capture_mon.h::dsart_capture_mon_t``. Any
change to that struct MUST bump ``DSART_CAPTURE_MON_VERSION`` (in
the header) AND ``_MON_VERSION`` (in this module).
"""

from __future__ import annotations

import dataclasses
import enum
import mmap
import os
import struct
import time
from typing import Optional

# Must match dsart_capture_mon.h
_MON_MAGIC = 0xCA77A1E1
_MON_VERSION = 1
# 6 * uint32 (24 B) + 18 * uint64 (144 B) + 5 * uint64 reserved (40 B) = 208 B.
# Layout has no natural-alignment padding because the uint64 fields
# start at offset 24 which is already 8-byte aligned.
_MON_STRUCT_BYTES = 208

_SHM_DIR = "/dev/shm"


class ArmState(enum.IntEnum):
    """Mirror of dsart_arm_state_t in dsart_capture_mon.h."""

    WAITING_FOR_ARM = 0
    ARMED = 1
    WRITING = 2
    STOPPED = 3


# Layout (all little-endian; matches the C atomic types on x86-64):
#   uint32 magic
#   uint32 version
#   uint32 udp_port
#   uint32 control_port
#   uint32 socket_rcvbuf_bytes
#   uint32 arm_state
#   uint64 pid
#   uint64 startup_utc_ns
#   uint64 last_update_utc_ns
#   uint64 utc_start_specnum
#   uint64 utc_stop_specnum
#   uint64 last_seq_no
#   uint64 n_recv_packets
#   uint64 n_recv_bytes
#   uint64 n_dropped_payload
#   uint64 n_dropped_kernel
#   uint64 n_seq_skipped
#   uint64 n_too_late
#   uint64 n_wrong_size
#   uint64 n_recv_errors
#   uint64 n_block_writes
#   uint64 rate_gbps_milli
#   uint64 rate_drop_milli
#   uint64 rate_kernel_drop_pps
#   uint64 _reserved[5]
_FMT = "<IIIIII" + "Q" * 18 + "Q" * 5
_EXPECTED_SIZE = struct.calcsize(_FMT)
# The header reserves the final 256-byte page; assert this matches the C side.
assert _EXPECTED_SIZE == _MON_STRUCT_BYTES, (
    f"struct size mismatch: python={_EXPECTED_SIZE} expected={_MON_STRUCT_BYTES}"
)


@dataclasses.dataclass(frozen=True)
class CaptureMonSnapshot:
    """One atomic snapshot of the capture process's shm counters."""

    udp_port: int
    control_port: int
    socket_rcvbuf_bytes: int
    arm_state: ArmState
    pid: int
    startup_utc_ns: int
    last_update_utc_ns: int
    utc_start_specnum: int
    utc_stop_specnum: int
    last_seq_no: int
    n_recv_packets: int
    n_recv_bytes: int
    n_dropped_payload: int
    n_dropped_kernel: int
    n_seq_skipped: int
    n_too_late: int
    n_wrong_size: int
    n_recv_errors: int
    n_block_writes: int
    rate_gbps_milli: int
    rate_drop_milli: int
    rate_kernel_drop_pps: int

    @property
    def rate_gbps(self) -> float:
        return self.rate_gbps_milli / 1000.0

    @property
    def rate_drop_mb_s(self) -> float:
        return self.rate_drop_milli / 1000.0

    @property
    def age_ms(self) -> float:
        """Wall-clock ms since the binary last stamped ``last_update_utc_ns``.

        The C stats_thread ticks the timestamp every ~100 ms. The
        sidecar's watchdog flags > 1000 ms as ``degraded``.
        """
        return (time.time_ns() - self.last_update_utc_ns) / 1e6

    @property
    def is_stale(self) -> bool:
        """True if the binary hasn't stamped a tick in over 1 s."""
        return self.age_ms > 1000.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["arm_state_name"] = self.arm_state.name
        d["rate_gbps"] = self.rate_gbps
        d["rate_drop_mb_s"] = self.rate_drop_mb_s
        d["age_ms"] = round(self.age_ms, 2)
        d["is_stale"] = self.is_stale
        return d


class MonShmAbiMismatch(RuntimeError):
    """Magic or version field of the shm segment did not match the python ABI."""


class MonShmNotPresent(FileNotFoundError):
    """Capture binary has not opened its shm segment yet (or has exited)."""


def shm_path_for_port(udp_port: int) -> str:
    """POSIX-shm file path for ``udp_port`` (e.g. ``/dev/shm/dsart-capture-4011``)."""

    return os.path.join(_SHM_DIR, f"dsart-capture-{udp_port}")


class MonShm:
    """Read-only mmap of a single capture-binary's shm segment.

    Usage:
        >>> mon = MonShm.open(4011)
        >>> snap = mon.snapshot()
        >>> print(snap.arm_state, snap.n_recv_packets, snap.is_stale)
        >>> mon.close()

    Or as a context manager:

        >>> with MonShm.open(4011) as mon:
        ...     while True:
        ...         snap = mon.snapshot()
        ...         ...
    """

    def __init__(self, fd: int, mm: mmap.mmap, udp_port: int):
        self._fd = fd
        self._mm = mm
        self._udp_port = udp_port

    @classmethod
    def open(cls, udp_port: int) -> "MonShm":
        """Attach to the capture binary's shm. Raises if not present."""
        path = shm_path_for_port(udp_port)
        if not os.path.exists(path):
            raise MonShmNotPresent(
                f"capture mon shm {path} does not exist -- is "
                f"dsart_capture_manythread running on UDP port {udp_port}?"
            )
        fd = os.open(path, os.O_RDONLY)
        try:
            sz = os.fstat(fd).st_size
            if sz < _EXPECTED_SIZE:
                raise MonShmAbiMismatch(
                    f"mon shm too small (got {sz} B, want {_EXPECTED_SIZE} B)"
                )
            mm = mmap.mmap(fd, _EXPECTED_SIZE, prot=mmap.PROT_READ)
        except Exception:
            os.close(fd)
            raise
        inst = cls(fd, mm, udp_port)
        try:
            inst._check_abi()
        except Exception:
            inst.close()
            raise
        return inst

    def _check_abi(self) -> None:
        # Read magic + version (first 8 bytes).
        raw = bytes(self._mm[:8])
        magic, version = struct.unpack("<II", raw)
        if magic != _MON_MAGIC:
            raise MonShmAbiMismatch(
                f"bad magic 0x{magic:08x} (want 0x{_MON_MAGIC:08x}) "
                f"-- shm segment is not a dsart_capture_mon_t"
            )
        if version != _MON_VERSION:
            raise MonShmAbiMismatch(
                f"version mismatch: shm={version} python={_MON_VERSION} "
                f"-- rebuild capture binary or update mon_shm.py"
            )

    def snapshot(self) -> CaptureMonSnapshot:
        """Atomically (x86-64-style) read all fields into a snapshot.

        Note: we do a single struct.unpack on the mmap buffer; on
        x86-64 every aligned 8-byte load is atomic, and the C binary
        only stores via `atomic_store_explicit(..., memory_order_release)`.
        Reads here are *not* synchronised across fields -- a snapshot
        may interleave a stats_thread update across its 100 ms
        cadence. That's fine for the 2 s mon-publisher consumer.
        """
        raw = bytes(self._mm[:_EXPECTED_SIZE])
        fields = struct.unpack(_FMT, raw)
        return CaptureMonSnapshot(
            udp_port=fields[2],
            control_port=fields[3],
            socket_rcvbuf_bytes=fields[4],
            arm_state=ArmState(fields[5]),
            pid=fields[6],
            startup_utc_ns=fields[7],
            last_update_utc_ns=fields[8],
            utc_start_specnum=fields[9],
            utc_stop_specnum=fields[10],
            last_seq_no=fields[11],
            n_recv_packets=fields[12],
            n_recv_bytes=fields[13],
            n_dropped_payload=fields[14],
            n_dropped_kernel=fields[15],
            n_seq_skipped=fields[16],
            n_too_late=fields[17],
            n_wrong_size=fields[18],
            n_recv_errors=fields[19],
            n_block_writes=fields[20],
            rate_gbps_milli=fields[21],
            rate_drop_milli=fields[22],
            rate_kernel_drop_pps=fields[23],
        )

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass

    @property
    def udp_port(self) -> int:
        return self._udp_port

    def __enter__(self) -> "MonShm":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

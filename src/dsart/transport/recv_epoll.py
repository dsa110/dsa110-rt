"""ctypes wrapper for the M4a chunk-6 C epoll receive loop.

Provides :class:`RxEpoll`, a drop-in replacement for the Python
``_RxLoop`` + ``TransportRxProd.ingest_datagram`` hot path. The C loop
runs in its own pthread and exposes atomic counters readable from
Python without locks.

Usage:

    from dsart.transport.recv_epoll import RxEpoll

    rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                      so_rcvbuf_bytes=256 * 1024 * 1024)
    rx.set_expected_pattern_id(chgroup=0, pattern_id=0xCAFE)
    rx.start()
    # ... wait for traffic ...
    print(rx.counters())
    rx.stop()
    rx.close()

The MVP does NOT write committed slots to a shm ring (that's Phase B);
it only counts. The intent is to validate that the C recv path can
absorb 7+ Gb/s of fragmented prod-frame UDP traffic before adding the
shm-ring write-through, which would replicate the chunk-3
``ring_write_cb`` semantics.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LOG = logging.getLogger("dsart.transport.recv_epoll")


# ---------------------------------------------------------------------------
# Library loader
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _find_lib() -> ctypes.CDLL:
    """Locate and load the ``_recv_epoll`` extension.

    setuptools builds it as a Python extension module
    (``_recv_epoll.cpython-*.so``) next to the package. ctypes can
    ``CDLL`` the resulting .so directly even though it doesn't expose
    a Python init function — we only use it for the plain-C symbols.
    """
    global _lib
    if _lib is not None:
        return _lib

    pkg_dir = Path(__file__).resolve().parent
    candidates = list(pkg_dir.glob("_recv_epoll*.so"))
    if not candidates:
        raise RuntimeError(
            f"recv_epoll: C extension _recv_epoll*.so not found in {pkg_dir}. "
            "Did you `pip install -e .` after pulling chunk 6? See setup.py."
        )
    _lib = ctypes.CDLL(str(candidates[0]))
    return _lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Bind argtypes / restype for every exported symbol."""
    lib.recv_epoll_open.argtypes = [
        ctypes.c_char_p,         # bind_host
        ctypes.c_uint16,         # bind_port
        ctypes.c_int,            # so_rcvbuf_bytes
        ctypes.POINTER(ctypes.c_uint16),  # out_actual_port
    ]
    lib.recv_epoll_open.restype = ctypes.c_int

    lib.recv_epoll_set_expected_pid.argtypes = [
        ctypes.c_uint32, ctypes.c_uint64,
    ]
    lib.recv_epoll_set_expected_pid.restype = ctypes.c_int

    lib.recv_epoll_clear_expected_pid.argtypes = [ctypes.c_uint32]
    lib.recv_epoll_clear_expected_pid.restype = ctypes.c_int

    lib.recv_epoll_start.argtypes = []
    lib.recv_epoll_start.restype = ctypes.c_int

    lib.recv_epoll_stop.argtypes = []
    lib.recv_epoll_stop.restype = ctypes.c_int

    lib.recv_epoll_close.argtypes = []
    lib.recv_epoll_close.restype = ctypes.c_int

    for name in (
        "n_received",
        "n_committed",
        "bad_magic_count",
        "bad_version_count",
        "bad_length_count",
        "bad_field_range_count",
        "reserved_bit_count",
        "pattern_mismatch_count",
        "window_slide_zerofill_count",
        "out_of_order_drop_count",
        "bytes_received_total",
    ):
        fn = getattr(lib, f"recv_epoll_get_{name}")
        fn.argtypes = []
        fn.restype = ctypes.c_uint64


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RxEpollCounters:
    """Snapshot of all C-side counters at a moment in time."""

    n_received: int
    n_committed: int
    bad_magic_count: int
    bad_version_count: int
    bad_length_count: int
    bad_field_range_count: int
    reserved_bit_count: int
    pattern_mismatch_count: int
    window_slide_zerofill_count: int
    out_of_order_drop_count: int
    bytes_received_total: int

    def to_dict(self) -> dict:
        return {
            "n_received": self.n_received,
            "n_committed": self.n_committed,
            "bad_magic_count": self.bad_magic_count,
            "bad_version_count": self.bad_version_count,
            "bad_length_count": self.bad_length_count,
            "bad_field_range_count": self.bad_field_range_count,
            "reserved_bit_count": self.reserved_bit_count,
            "pattern_mismatch_count": self.pattern_mismatch_count,
            "window_slide_zerofill_count": self.window_slide_zerofill_count,
            "out_of_order_drop_count": self.out_of_order_drop_count,
            "bytes_received_total": self.bytes_received_total,
        }


# ---------------------------------------------------------------------------
# RxEpoll class
# ---------------------------------------------------------------------------


class RxEpoll:
    """High-level wrapper around the C epoll receive loop.

    Singleton per process — the C side only supports one listening
    socket. Construct via :meth:`open`, run via :meth:`start` /
    :meth:`stop`, query via :meth:`counters` / :attr:`port`, tear
    down via :meth:`close`.
    """

    _instance: Optional["RxEpoll"] = None

    def __init__(self, lib: ctypes.CDLL, port: int) -> None:
        self._lib = lib
        self._port = port
        self._started = False
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        so_rcvbuf_bytes: int = 256 * 1024 * 1024,
    ) -> "RxEpoll":
        """Bind a UDP socket and prepare the epoll loop (not yet running).

        Args:
            bind_host: bind address. Empty string ⇒ ``INADDR_ANY``.
            bind_port: bind port. ``0`` ⇒ kernel-picked ephemeral.
                Read the actual port back from :attr:`port`.
            so_rcvbuf_bytes: requested ``SO_RCVBUF`` size. The kernel
                caps at ``/proc/sys/net/core/rmem_max``; failures here
                are silent (matches Python ``socket.setsockopt`` warn).

        Returns:
            A new :class:`RxEpoll` ready to :meth:`start`.

        Raises:
            RuntimeError: the C lib could not bind the socket or
                another instance is already open in this process.
        """
        if cls._instance is not None:
            raise RuntimeError(
                "RxEpoll is singleton; close the existing instance first"
            )
        lib = _find_lib()
        _bind_signatures(lib)
        out_port = ctypes.c_uint16(0)
        host_b = bind_host.encode() if bind_host else b""
        rc = lib.recv_epoll_open(host_b, bind_port, so_rcvbuf_bytes,
                                 ctypes.byref(out_port))
        if rc != 0:
            raise RuntimeError(
                f"recv_epoll_open failed: rc={rc} "
                f"(host={bind_host!r} port={bind_port} rcvbuf={so_rcvbuf_bytes})"
            )
        inst = cls(lib, int(out_port.value))
        cls._instance = inst
        return inst

    @property
    def port(self) -> int:
        """Actual bound port (relevant when caller passed ``bind_port=0``)."""
        return self._port

    def set_expected_pattern_id(self, chgroup: int, pattern_id: int) -> None:
        """Register the expected ``pattern_id`` for a chgroup.

        Mirrors :meth:`TransportRxProd.update_expected_pattern_id`.
        Datagrams from this chgroup whose ``pattern_id`` differs from
        the registered value are dropped and counted in
        ``pattern_mismatch_count``.
        """
        rc = self._lib.recv_epoll_set_expected_pid(
            ctypes.c_uint32(chgroup), ctypes.c_uint64(pattern_id),
        )
        if rc != 0:
            raise ValueError(
                f"recv_epoll_set_expected_pid: rc={rc} (chgroup={chgroup})"
            )

    def clear_expected_pattern_id(self, chgroup: int) -> None:
        """Unset the expected ``pattern_id`` for a chgroup."""
        rc = self._lib.recv_epoll_clear_expected_pid(
            ctypes.c_uint32(chgroup),
        )
        if rc != 0:
            raise ValueError(
                f"recv_epoll_clear_expected_pid: rc={rc} (chgroup={chgroup})"
            )

    def start(self) -> None:
        """Spin up the epoll pthread. Idempotent."""
        if self._started:
            return
        rc = self._lib.recv_epoll_start()
        if rc != 0:
            raise RuntimeError(f"recv_epoll_start: rc={rc}")
        self._started = True

    def stop(self) -> None:
        """Signal the pthread to exit and join it. Idempotent."""
        if not self._started:
            return
        rc = self._lib.recv_epoll_stop()
        if rc != 0:
            raise RuntimeError(f"recv_epoll_stop: rc={rc}")
        self._started = False

    def close(self) -> None:
        """Tear down the socket + epoll fd. After close the instance
        is unusable; create a new one via :meth:`open`."""
        if self._closed:
            return
        if self._started:
            self.stop()
        rc = self._lib.recv_epoll_close()
        if rc != 0:
            LOG.warning("recv_epoll_close: rc=%d (ignoring)", rc)
        self._closed = True
        type(self)._instance = None

    def __enter__(self) -> "RxEpoll":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def counters(self) -> RxEpollCounters:
        """Snapshot all atomic counters in one go."""
        return RxEpollCounters(
            n_received=int(self._lib.recv_epoll_get_n_received()),
            n_committed=int(self._lib.recv_epoll_get_n_committed()),
            bad_magic_count=int(self._lib.recv_epoll_get_bad_magic_count()),
            bad_version_count=int(self._lib.recv_epoll_get_bad_version_count()),
            bad_length_count=int(self._lib.recv_epoll_get_bad_length_count()),
            bad_field_range_count=int(
                self._lib.recv_epoll_get_bad_field_range_count()
            ),
            reserved_bit_count=int(self._lib.recv_epoll_get_reserved_bit_count()),
            pattern_mismatch_count=int(
                self._lib.recv_epoll_get_pattern_mismatch_count()
            ),
            window_slide_zerofill_count=int(
                self._lib.recv_epoll_get_window_slide_zerofill_count()
            ),
            out_of_order_drop_count=int(
                self._lib.recv_epoll_get_out_of_order_drop_count()
            ),
            bytes_received_total=int(
                self._lib.recv_epoll_get_bytes_received_total()
            ),
        )

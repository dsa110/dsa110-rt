"""ctypes wrapper for the M4a chunk-6 C epoll receive loop (Phase A: counters)
+ the M7.2 Phase B shm-ring write-through.

Provides :class:`RxEpoll`, a drop-in replacement for the Python
``_RxLoop`` + ``TransportRxProd.ingest_datagram`` hot path. The C loop
runs in its own pthread and exposes atomic counters readable from
Python without locks.

Phase A usage (counters-only; no ring publish):

    from dsart.transport.recv_epoll import RxEpoll

    rx = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                      so_rcvbuf_bytes=256 * 1024 * 1024)
    rx.set_expected_pattern_id(chgroup=0, pattern_id=0xCAFE)
    rx.start()
    # ... wait for traffic ...
    print(rx.counters())
    rx.stop()
    rx.close()

Phase B usage (multi-port bind + ring publish; M7.2 search-node fan-in):

    rx = RxEpoll.open(bind_host="10.41.0.205", bind_port=6625,
                      so_rcvbuf_bytes=256 * 1024 * 1024)
    # Add 15 more sockets to receive from all 16 chgroups (one per port).
    for chg in range(1, 16):
        rx.add_port(bind_host="10.41.0.205", bind_port=6625 + chg)
    # Open the producer side of the shm ring (owner=1 creates).
    rx.attach_ring(
        shm_name="/dsart-rx-n01",
        owner=True,
        n_corr=16, n_coarse_dm=5,
        t_buf_samples=4096,    # CONC-1 ring depth in search-cadence samples
        n_filled=5000,
        bytes_per_cell=2,      # cint8 complex
    )
    # Register the 16 expected pattern_ids per chgroup (one per corr node).
    for chg, pid in pids_by_chgroup.items():
        rx.set_expected_pattern_id(chgroup=chg, pattern_id=pid)
    rx.start()
    # ... soak ...
    print(rx.counters())  # both Phase A and Phase B counters

Lifecycle ordering: ``open`` -> ``add_port``* -> ``attach_ring`` ->
``set_expected_pattern_id``* -> ``start`` -> ... -> ``stop`` -> ``close``.
``attach_ring`` and ``add_port`` reject calls after ``start`` (the
epoll loop is single-drainer and we don't synchronise mid-loop
``epoll_ctl`` / ring swap-in); stop first, modify, restart.
"""

from __future__ import annotations

import ctypes
import logging
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

    # M7.2: multi-port bind.
    lib.recv_epoll_add_port.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint16,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.recv_epoll_add_port.restype = ctypes.c_int

    lib.recv_epoll_set_expected_pid.argtypes = [
        ctypes.c_uint32, ctypes.c_uint64,
    ]
    lib.recv_epoll_set_expected_pid.restype = ctypes.c_int

    lib.recv_epoll_clear_expected_pid.argtypes = [ctypes.c_uint32]
    lib.recv_epoll_clear_expected_pid.restype = ctypes.c_int

    # M7.2 Phase B: ring attach / detach.
    lib.recv_epoll_attach_ring.argtypes = [
        ctypes.c_char_p,   # shm_name
        ctypes.c_int,      # owner (0/1)
        ctypes.c_uint32,   # n_corr
        ctypes.c_uint32,   # n_coarse_dm
        ctypes.c_uint32,   # t_buf_samples
        ctypes.c_uint32,   # n_filled
        ctypes.c_uint32,   # bytes_per_cell
        ctypes.c_char_p,   # errbuf
        ctypes.c_size_t,   # errbuf_len
    ]
    lib.recv_epoll_attach_ring.restype = ctypes.c_int

    lib.recv_epoll_detach_ring.argtypes = []
    lib.recv_epoll_detach_ring.restype = ctypes.c_int

    lib.recv_epoll_start.argtypes = []
    lib.recv_epoll_start.restype = ctypes.c_int

    lib.recv_epoll_stop.argtypes = []
    lib.recv_epoll_stop.restype = ctypes.c_int

    lib.recv_epoll_close.argtypes = []
    lib.recv_epoll_close.restype = ctypes.c_int

    lib.recv_epoll_get_n_sockets.argtypes = []
    lib.recv_epoll_get_n_sockets.restype = ctypes.c_int

    lib.recv_epoll_get_ring_attached.argtypes = []
    lib.recv_epoll_get_ring_attached.restype = ctypes.c_int

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
        # M7.2 Phase B counters
        "ring_slots_written",
        "ring_data_present_count",
        "ring_pattern_mismatch_count",
        "ring_zerofill_slot_count",
        "ring_write_error_count",
    ):
        fn = getattr(lib, f"recv_epoll_get_{name}")
        fn.argtypes = []
        fn.restype = ctypes.c_uint64


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RxEpollCounters:
    """Snapshot of all C-side counters at a moment in time.

    Phase A counters (M4a chunk 6): n_received .. bytes_received_total.
    Phase B counters (M7.2): ring_* — only non-zero once
    :meth:`RxEpoll.attach_ring` has been called.
    """

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
    # Phase B (ring)
    ring_slots_written: int = 0
    ring_data_present_count: int = 0
    ring_pattern_mismatch_count: int = 0
    ring_zerofill_slot_count: int = 0
    ring_write_error_count: int = 0

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
            "ring_slots_written": self.ring_slots_written,
            "ring_data_present_count": self.ring_data_present_count,
            "ring_pattern_mismatch_count": self.ring_pattern_mismatch_count,
            "ring_zerofill_slot_count": self.ring_zerofill_slot_count,
            "ring_write_error_count": self.ring_write_error_count,
        }


# ---------------------------------------------------------------------------
# RxEpoll class
# ---------------------------------------------------------------------------


class RxEpoll:
    """High-level wrapper around the C epoll receive loop.

    Singleton per process — the C side holds one ``g_state`` struct.
    Construct via :meth:`open`, optionally bind extra ports via
    :meth:`add_port`, optionally arm shm-ring publish via
    :meth:`attach_ring`, run via :meth:`start` / :meth:`stop`,
    query via :meth:`counters` / :attr:`port` / :attr:`n_sockets`,
    tear down via :meth:`close`.
    """

    _instance: Optional["RxEpoll"] = None

    def __init__(self, lib: ctypes.CDLL, port: int) -> None:
        self._lib = lib
        self._port = port             # primary (first-bound) port
        self._extra_ports: list[int] = []
        self._ring_attached = False
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
        """Bind the first UDP socket and prepare the epoll loop (not yet running).

        Args:
            bind_host: bind address. Empty string ⇒ ``INADDR_ANY``.
            bind_port: bind port. ``0`` ⇒ kernel-picked ephemeral.
                Read the actual port back from :attr:`port`.
            so_rcvbuf_bytes: requested ``SO_RCVBUF`` size. The kernel
                caps at ``/proc/sys/net/core/rmem_max``; failures here
                are silent (matches Python ``socket.setsockopt`` warn).

        Returns:
            A new :class:`RxEpoll` ready to :meth:`add_port` /
            :meth:`attach_ring` / :meth:`start`.

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
        """Actual bound port of the primary socket (relevant when
        caller passed ``bind_port=0`` to :meth:`open`)."""
        return self._port

    @property
    def n_sockets(self) -> int:
        """How many UDP sockets are currently bound (primary +
        :meth:`add_port` additions). Reads through to the C side so
        it stays in sync with the source of truth."""
        return int(self._lib.recv_epoll_get_n_sockets())

    @property
    def ports(self) -> list[int]:
        """All currently bound ports in bind-order (primary first).
        Lives in Python — the C side does not expose a port list
        getter."""
        return [self._port, *self._extra_ports]

    @property
    def ring_attached(self) -> bool:
        """Whether a shm ring is currently armed for Phase B publish."""
        return bool(self._lib.recv_epoll_get_ring_attached())

    def add_port(
        self,
        *,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        so_rcvbuf_bytes: int = 256 * 1024 * 1024,
    ) -> int:
        """Bind another UDP socket on the same epoll fd.

        Use this to register the 16 production listen ports
        (``6625 + chgroup``) under one process so a single drainer
        thread handles the whole search-node fan-in.

        Returns the actual bound port (relevant when caller passed
        ``bind_port=0``).

        Raises:
            RuntimeError: the C lib could not bind, the maximum port
                count has been reached, or :meth:`start` has already
                been called (mid-loop ``epoll_ctl`` is racy with our
                single drainer; stop first to add ports).
        """
        if self._started:
            raise RuntimeError(
                "RxEpoll.add_port called after start; stop first"
            )
        out_port = ctypes.c_uint16(0)
        host_b = bind_host.encode() if bind_host else b""
        rc = self._lib.recv_epoll_add_port(host_b, bind_port,
                                           so_rcvbuf_bytes,
                                           ctypes.byref(out_port))
        if rc != 0:
            raise RuntimeError(
                f"recv_epoll_add_port failed: rc={rc} "
                f"(host={bind_host!r} port={bind_port})"
            )
        p = int(out_port.value)
        self._extra_ports.append(p)
        return p

    def attach_ring(
        self,
        *,
        shm_name: str,
        owner: bool,
        n_corr: int,
        n_coarse_dm: int,
        t_buf_samples: int,
        n_filled: int,
        bytes_per_cell: int,
    ) -> None:
        """Arm the M7.2 Phase B shm-ring publish path.

        Args:
            shm_name: POSIX shm name (e.g. ``/dsart-rx-n01``). The
                same name must be opened (with ``owner=False``) by the
                consumer side (``ProductionRxRingSource`` /
                :mod:`dsart.transport.recv_ring.RxRing`).
            owner: ``True`` creates the shm segment (``ftruncate``,
                zero-init header + data); ``False`` attaches read-only
                to an existing segment. M7.2 search nodes own the ring
                (one writer per process); compute halves attach.
            n_corr, n_coarse_dm, t_buf_samples, n_filled, bytes_per_cell:
                CONC-1 ring dimensions — must match what the consumer
                expects. ``t_buf_samples`` should be ≥ a few cube
                cadences (~2-4×) so a momentary RX stall doesn't
                trigger writer overrun.

        Raises:
            RuntimeError: the ring could not be opened/created, a ring
                is already attached, or :meth:`start` has been called.
        """
        if self._started:
            raise RuntimeError(
                "RxEpoll.attach_ring called after start; stop first"
            )
        if self._ring_attached:
            raise RuntimeError(
                f"RxEpoll: ring already attached (shm_name={shm_name!r}). "
                "Call detach_ring() before attaching a different ring."
            )
        errbuf = ctypes.create_string_buffer(512)
        rc = self._lib.recv_epoll_attach_ring(
            shm_name.encode(),
            1 if owner else 0,
            int(n_corr), int(n_coarse_dm), int(t_buf_samples),
            int(n_filled), int(bytes_per_cell),
            errbuf, len(errbuf),
        )
        if rc != 0:
            msg = errbuf.value.decode(errors="replace")
            raise RuntimeError(
                f"recv_epoll_attach_ring failed: rc={rc} shm_name={shm_name!r} "
                f"(n_corr={n_corr} n_coarse_dm={n_coarse_dm} "
                f"t_buf_samples={t_buf_samples} n_filled={n_filled} "
                f"bytes_per_cell={bytes_per_cell}): {msg or '(no error message)'}"
            )
        self._ring_attached = True

    def detach_ring(self) -> None:
        """Tear down the ring-publish path. No-op if not attached."""
        if self._started:
            raise RuntimeError(
                "RxEpoll.detach_ring called while running; stop first"
            )
        if not self._ring_attached:
            return
        rc = self._lib.recv_epoll_detach_ring()
        if rc != 0:
            LOG.warning("recv_epoll_detach_ring: rc=%d (ignoring)", rc)
        self._ring_attached = False

    def set_expected_pattern_id(self, chgroup: int, pattern_id: int) -> None:
        """Register the expected ``pattern_id`` for a chgroup.

        Mirrors :meth:`TransportRxProd.update_expected_pattern_id`.
        Datagrams from this chgroup whose ``pattern_id`` differs from
        the registered value are counted in ``pattern_mismatch_count``
        and either dropped (Phase A, no ring attached) or published as
        a zero-payload slot with ``VF_PATTERN_MISMATCH`` set (Phase B,
        ring attached).
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
        """Tear down all sockets + epoll fd + ring. After close the
        instance is unusable; create a new one via :meth:`open`."""
        if self._closed:
            return
        if self._started:
            self.stop()
        rc = self._lib.recv_epoll_close()
        if rc != 0:
            LOG.warning("recv_epoll_close: rc=%d (ignoring)", rc)
        # Python-side bookkeeping reset.
        self._ring_attached = False
        self._extra_ports.clear()
        self._closed = True
        type(self)._instance = None

    def __enter__(self) -> "RxEpoll":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def counters(self) -> RxEpollCounters:
        """Snapshot all atomic counters in one go (Phase A + Phase B)."""
        g = self._lib
        return RxEpollCounters(
            n_received=int(g.recv_epoll_get_n_received()),
            n_committed=int(g.recv_epoll_get_n_committed()),
            bad_magic_count=int(g.recv_epoll_get_bad_magic_count()),
            bad_version_count=int(g.recv_epoll_get_bad_version_count()),
            bad_length_count=int(g.recv_epoll_get_bad_length_count()),
            bad_field_range_count=int(g.recv_epoll_get_bad_field_range_count()),
            reserved_bit_count=int(g.recv_epoll_get_reserved_bit_count()),
            pattern_mismatch_count=int(g.recv_epoll_get_pattern_mismatch_count()),
            window_slide_zerofill_count=int(
                g.recv_epoll_get_window_slide_zerofill_count()
            ),
            out_of_order_drop_count=int(
                g.recv_epoll_get_out_of_order_drop_count()
            ),
            bytes_received_total=int(g.recv_epoll_get_bytes_received_total()),
            ring_slots_written=int(g.recv_epoll_get_ring_slots_written()),
            ring_data_present_count=int(
                g.recv_epoll_get_ring_data_present_count()
            ),
            ring_pattern_mismatch_count=int(
                g.recv_epoll_get_ring_pattern_mismatch_count()
            ),
            ring_zerofill_slot_count=int(
                g.recv_epoll_get_ring_zerofill_slot_count()
            ),
            ring_write_error_count=int(
                g.recv_epoll_get_ring_write_error_count()
            ),
        )

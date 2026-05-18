"""Single-producer / single-consumer cube ring backed by POSIX shm.

Used by the M7.2 **production async TX path** (see
:class:`dsart.transport.async_tx.AsyncTransportTx`): the corr-side GPU
pipeline writes a host-resident complex64 cube into a bounded ring of
shm slots and signals a worker subprocess; the worker reads the cube,
encodes + frames + ``sendto``s, then releases the slot.

The corr-side block budget is ~134 ms cadence at 32× integration. A
single Python TX thread spends ~74 ms per cube on encode + ``sendto``
at the N=8 op-point (M4a chunk-2 finding), which would blow the RT
budget if performed inline on the GPU-pipeline thread. Off-loading
that work to N worker subprocesses (one per DM-split) keeps the
GPU-pipeline thread on its ~110 ms ceiling, with TX latency overlapping
into the *next* block's GPU compute. The shm ring is the hand-off
between the two — the worker reads bytes that the GPU thread just
wrote, with no Python copy on the worker side.

Design pin (M7.2 corner-turn):
    The user's directive "all settings for all code will be production
    settings" rules out any inline-only fallback. The single source of
    truth for the cube → wire path on a corr node is the
    AsyncTransportTx + worker subprocess pair built on top of this
    ring. The ring carries complex64 (no quantisation on the producer);
    the worker subprocess owns the cint8 quantisation step so it can
    overlap with the *next* block's GPU compute.

Synchronisation:
    The shm segment carries the cube payload; a pair of
    :class:`multiprocessing.Queue` instances carry slot ownership::

        ready_q : producer ─→ worker  (slot has a fresh cube)
        done_q  : worker   ─→ producer (slot may be re-used)

    No locks or atomics on the shm itself. The producer recycles slots
    eagerly by draining ``done_q`` non-blockingly at the top of
    :meth:`CubeShmRing.reserve_slot`. The consumer marks slots free by
    posting on ``done_q`` AFTER the cube has been fully consumed (i.e.
    after the worker's TX send loop has returned).

    A slot is in exactly one of three states at any moment:

        FREE      — producer may overwrite
        PUBLISHED — written by producer, waiting in ``ready_q``
        IN_USE    — worker is reading; not yet posted on ``done_q``

    With ``n_slots`` slots and a single producer / single consumer,
    backpressure is provided by ``ready_q``'s ``put_nowait`` semantics:
    if the worker can't keep up, the producer's :meth:`reserve_slot`
    will block on the ``done_q.get`` recycle path (no FREE slots).

Failure modes:
    On producer crash, the worker's ``ready_q.get(timeout=...)`` will
    eventually time out; the worker then exits cleanly. On worker
    crash, the producer's ``reserve_slot`` will block on ``done_q``
    once all slots are exhausted; callers must surface this via a
    timeout (the AsyncTransportTx wrapper passes ``reserve_timeout_s``
    and raises :class:`TxRingBackpressureError` on time-out).

Wire formats:
    The ring is dtype-agnostic at the shm layer; the producer specifies
    ``dtype`` + ``shape`` at construction time, and both sides agree
    on these via the worker's startup config. The default for M7.2 is
    complex64 (8 B/cell); the worker quantises to cint8 (2 B/cell) on
    the consumer side as part of the prod-frame encode step.

Lifecycle:
    - Producer constructs ``CubeShmRing(name, dims, owner=True)`` which
      allocates the shm and registers it for unlink-on-close.
    - Worker constructs ``CubeShmRing(name, dims, owner=False)`` which
      attaches read-write to the producer's shm.
    - Either side may call :meth:`close` to drop their handle; the
      owner's :meth:`close` is what unlinks the shm.

C-extension parity:
    This module deliberately avoids the
    :mod:`dsart.transport.recv_ring` C extension because (a) we only
    need SPSC (not SPMC), (b) huge-page mapping is not required at the
    corr-side cube rate (one ~32 MiB slot every ~134 ms), and (c) the
    handoff is producer→worker on the SAME host (no NIC). If the
    M7.3 16x4 deployment needs cudaHostRegister + huge pages on the
    TX-side ring too, a C-extension drop-in implementing the same
    Python API is the right follow-up (M7.3 F-item).
"""

from __future__ import annotations

import logging
import queue as _stdlib_queue
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.queues import Queue as MPQueue
from typing import Any

import numpy as np

LOG = logging.getLogger("dsart.transport.tx_ring")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TxRingBackpressureError(RuntimeError):
    """Raised when :meth:`CubeShmRing.reserve_slot` cannot find a free
    slot within ``reserve_timeout_s``.

    Means the worker is not draining fast enough — TX-side encode is
    slower than the cube arrival rate. The AsyncTransportTx wrapper
    surfaces this as a ``tx_backpressure`` mon-key counter and a
    structured ERROR log so an operator can spot the wall-rate ceiling
    in production.
    """


class TxRingClosedError(RuntimeError):
    """Raised when a producer/consumer method is called on a closed ring."""


# ---------------------------------------------------------------------------
# Dims + slot metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeShmRingDims:
    """Geometry of a single :class:`CubeShmRing`.

    Args:
        n_slots: number of cube slots in the ring (≥ 2). With 4 slots
            at 134 ms cadence the ring buffers ~536 ms of cubes; the
            worker only needs >1 to overlap encode of cube N with the
            producer writing cube N+1.
        shape: per-slot cube shape, e.g. ``(n_dm_per_worker, n_fast_vis,
            n_filled)`` for the prod-frame path. Must be fixed at
            construction time; cubes that exceed it raise ValueError.
        dtype: numpy dtype of the per-slot cube. Default ``complex64``;
            the worker performs cint8 quantisation on its side so the
            producer does NOT pre-quantise.
    """

    n_slots: int
    shape: tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self) -> None:
        if self.n_slots < 2:
            raise ValueError(
                f"CubeShmRingDims.n_slots={self.n_slots} must be >= 2"
            )
        if any(s <= 0 for s in self.shape):
            raise ValueError(
                f"CubeShmRingDims.shape={self.shape} must be all positive"
            )

    @property
    def cube_nbytes(self) -> int:
        """Bytes per slot (cube payload only; no header)."""
        return int(np.prod(self.shape)) * int(np.dtype(self.dtype).itemsize)

    @property
    def slot_nbytes(self) -> int:
        """Bytes per slot including the 64 B header."""
        return self.cube_nbytes + _SLOT_HEADER_BYTES


# ---------------------------------------------------------------------------
# Slot header (64 B)
# ---------------------------------------------------------------------------
#
# Each slot in the shm carries a small header in front of the cube
# payload so the worker can read block_n + specnum + flags without a
# separate queue payload (the queue carries only the slot_idx).
#
# Layout (little-endian, 64 B total)::
#
#   off  size  field
#    0    8    block_n   uint64  (corr-side block counter)
#    8    8    specnum   uint64  (SNAP F-engine packet count at block start)
#   16    4    n_dm      uint32  (this slot's DM-axis length)
#   20    4    n_fv      uint32  (this slot's n_fast_vis axis)
#   24    4    n_filled  uint32  (this slot's N_filled axis)
#   28    1    flags     uint8   (bit 0 = rfi_warming_up)
#   29    3    pad
#   32   32    pad
#
# The header is bytes 0..64 of the slot; the cube payload starts at
# byte 64. The producer writes the header LAST so the worker never
# sees a stale block_n; in practice the worker only reads the slot
# after popping its index from ``ready_q``, which serialises with the
# producer's ``ready_q.put`` (which happens AFTER the header write).

_SLOT_HEADER_BYTES: int = 64
_HEADER_DTYPE = np.dtype([
    ("block_n", "<u8"),
    ("specnum", "<u8"),
    ("n_dm", "<u4"),
    ("n_fv", "<u4"),
    ("n_filled", "<u4"),
    ("flags", "<u1"),
    ("_pad_29", "<u1", 3),
    ("_pad_32", "<u1", 32),
])
assert _HEADER_DTYPE.itemsize == _SLOT_HEADER_BYTES, (
    f"header dtype = {_HEADER_DTYPE.itemsize} B, expected "
    f"{_SLOT_HEADER_BYTES} B; layout drift"
)


FLAG_RFI_WARMING_UP: int = 0x01


@dataclass(frozen=True)
class SlotMeta:
    """Per-slot metadata read by the worker after dequeue.

    Mirrors the fields the slot header carries, plus the slot index so
    the worker can post to ``done_q`` once it's done with the cube.
    """

    slot_idx: int
    block_n: int
    specnum: int
    n_dm: int
    n_fv: int
    n_filled: int
    flags: int

    @property
    def rfi_warming_up(self) -> bool:
        return bool(self.flags & FLAG_RFI_WARMING_UP)


# ---------------------------------------------------------------------------
# CubeShmRing
# ---------------------------------------------------------------------------


class CubeShmRing:
    """Single-producer / single-consumer cube ring backed by POSIX shm.

    Use one ring per worker subprocess. The producer (corr_fast) calls
    :meth:`reserve_slot` → :meth:`copy_to_slot` → :meth:`publish_slot`
    per cube; the worker calls :meth:`wait_slot` → reads the cube
    via :meth:`view_slot` → :meth:`release_slot` per cube.

    The producer and worker each construct their own
    :class:`CubeShmRing` handle. The first one to construct (with
    ``owner=True``) allocates the shm and the two Queues; the worker
    side attaches with ``owner=False`` using the same ``name``,
    ``dims``, ``ready_q``, and ``done_q`` (the producer passes the
    Queue handles to the worker via the mp.Process args).

    Args:
        name: POSIX-shm segment name. Convention:
            ``dsart-corr-tx-<corr_idx>-w<worker_idx>``. Must be < 32
            chars (POSIX shm name limit varies; 32 is safe).
        dims: :class:`CubeShmRingDims`.
        ready_q: :class:`multiprocessing.Queue` for producer → worker
            slot_idx hand-off. Created by the producer; passed to the
            worker via its process args.
        done_q: :class:`multiprocessing.Queue` for worker → producer
            slot_idx release. Same lifecycle as ``ready_q``.
        owner: when ``True``, allocate the shm; when ``False``, attach
            to an existing shm with the same name.

    Attributes:
        n_publish: number of publish calls (producer-side).
        n_consume: number of release calls (consumer-side).
        n_backpressure: number of times :meth:`reserve_slot` blocked
            waiting for a free slot (producer-side; recycling stalled
            because the worker is behind).
    """

    __slots__ = (
        "_name", "_dims", "_owner",
        "_shm", "_buf",
        "_header_views", "_cube_views",
        "_ready_q", "_done_q",
        "_free_slots",
        "_published_slots",
        "_closed",
        "_poisoned",
        "n_publish", "n_consume", "n_backpressure",
    )

    def __init__(
        self,
        name: str,
        dims: CubeShmRingDims,
        *,
        ready_q: MPQueue,
        done_q: MPQueue,
        owner: bool,
    ) -> None:
        if not name or len(name) > 250:
            raise ValueError(f"shm name length {len(name)} out of [1, 250]")
        self._name = str(name)
        self._dims = dims
        self._owner = bool(owner)

        total_bytes = dims.slot_nbytes * dims.n_slots
        if owner:
            try:
                self._shm = shared_memory.SharedMemory(
                    name=name, create=True, size=total_bytes,
                )
            except FileExistsError:
                LOG.warning(
                    "tx_ring shm %s already exists; unlinking + retrying",
                    name,
                )
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                self._shm = shared_memory.SharedMemory(
                    name=name, create=True, size=total_bytes,
                )
        else:
            self._shm = shared_memory.SharedMemory(
                name=name, create=False,
            )
            if self._shm.size < total_bytes:
                self._shm.close()
                raise ValueError(
                    f"tx_ring shm {name} size {self._shm.size} < required "
                    f"{total_bytes} (n_slots={dims.n_slots} × "
                    f"slot_nbytes={dims.slot_nbytes})"
                )

        self._buf: memoryview = self._shm.buf
        self._header_views: list[np.ndarray] = []
        self._cube_views: list[np.ndarray] = []
        for s in range(dims.n_slots):
            base = s * dims.slot_nbytes
            hdr_mv = self._buf[base : base + _SLOT_HEADER_BYTES]
            cube_mv = self._buf[
                base + _SLOT_HEADER_BYTES : base + dims.slot_nbytes
            ]
            hdr = np.ndarray(
                (1,), dtype=_HEADER_DTYPE, buffer=hdr_mv,
            )
            cube = np.ndarray(
                dims.shape, dtype=dims.dtype, buffer=cube_mv,
            )
            self._header_views.append(hdr)
            self._cube_views.append(cube)

        self._ready_q = ready_q
        self._done_q = done_q

        # Owner-side bookkeeping: which slots the producer believes are
        # currently FREE (recycled but not yet reused). At construction
        # all slots start FREE.
        self._free_slots: list[int] = list(range(dims.n_slots)) if owner else []
        self._published_slots: set[int] = set() if owner else set()

        self._closed: bool = False
        self._poisoned: bool = False
        self.n_publish: int = 0
        self.n_consume: int = 0
        self.n_backpressure: int = 0

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def reserve_slot(
        self,
        *,
        timeout_s: float = 1.0,
    ) -> int:
        """Reserve a free slot for writing. Producer-side.

        Drains ``done_q`` non-blockingly to recycle freed slots, then
        pops the next free slot. If none are free, blocks up to
        ``timeout_s`` waiting for the worker to post one on ``done_q``.

        Args:
            timeout_s: max time to wait for a free slot. ``0.0`` means
                fail immediately. The default 1.0 s is ~7 cube periods
                at 134 ms cadence; if backpressure persists that long
                the worker is genuinely stuck and the caller (Async
                TX) should surface the error.

        Returns:
            Slot index (0..n_slots-1) the producer now owns. Must be
            followed by exactly one :meth:`copy_to_slot` and one
            :meth:`publish_slot`, in that order.

        Raises:
            TxRingBackpressureError: no free slot within ``timeout_s``.
            TxRingClosedError: ring was closed.
        """
        if self._closed:
            raise TxRingClosedError(f"reserve_slot on closed ring {self._name}")
        if not self._owner:
            raise RuntimeError(
                "reserve_slot is producer-side only (this handle is owner=False)"
            )

        # Drain done_q non-blockingly to recycle returned slots.
        while True:
            try:
                slot_idx = self._done_q.get_nowait()
            except _stdlib_queue.Empty:
                break
            self._published_slots.discard(int(slot_idx))
            self._free_slots.append(int(slot_idx))

        if self._free_slots:
            return self._free_slots.pop(0)

        # No FREE slot; wait for the worker to post on done_q.
        self.n_backpressure += 1
        t_deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            remaining = t_deadline - time.monotonic()
            if remaining <= 0:
                raise TxRingBackpressureError(
                    f"tx_ring {self._name}: no free slot within "
                    f"{timeout_s:.3f}s (n_slots={self._dims.n_slots}, "
                    f"published={len(self._published_slots)}). The TX "
                    f"worker is behind the cube arrival rate; production "
                    f"target is encode_ms_per_cube < block_period_ms / "
                    f"n_workers."
                )
            try:
                slot_idx = self._done_q.get(timeout=min(0.05, remaining))
            except _stdlib_queue.Empty:
                continue
            self._published_slots.discard(int(slot_idx))
            return int(slot_idx)

    def copy_to_slot(
        self,
        slot_idx: int,
        cube: np.ndarray,
    ) -> None:
        """Copy ``cube`` into ``slot_idx``'s payload area. Producer-side.

        ``cube`` must match :attr:`dims.shape` and :attr:`dims.dtype`
        exactly. The copy is a single :func:`numpy.copyto` (≈ DRAM
        bandwidth, ~30 GB/s); at 32 MiB / cube that is ~1 ms.

        Args:
            slot_idx: returned by :meth:`reserve_slot`.
            cube: numpy array; must already be on the host (not a torch
                CUDA tensor). The producer is responsible for the D2H
                step before calling this method.

        Raises:
            ValueError: shape/dtype mismatch.
        """
        if self._closed:
            raise TxRingClosedError(f"copy_to_slot on closed ring {self._name}")
        if cube.shape != self._dims.shape:
            raise ValueError(
                f"copy_to_slot: cube.shape={cube.shape} != "
                f"dims.shape={self._dims.shape}"
            )
        if cube.dtype != self._dims.dtype:
            raise ValueError(
                f"copy_to_slot: cube.dtype={cube.dtype} != "
                f"dims.dtype={self._dims.dtype}"
            )
        np.copyto(self._cube_views[slot_idx], cube, casting="no")

    def publish_slot(
        self,
        slot_idx: int,
        *,
        block_n: int,
        specnum: int,
        n_dm: int,
        n_fv: int,
        n_filled: int,
        rfi_warming_up: bool = False,
        publish_timeout_s: float = 1.0,
    ) -> None:
        """Write the slot header, then hand the slot to the worker via
        ``ready_q``. Producer-side.

        Header write happens BEFORE the ``ready_q.put`` so the worker
        sees a consistent header on dequeue.

        Args:
            slot_idx: returned by :meth:`reserve_slot`.
            block_n: corr-side block counter (logged in mon keys).
            specnum: SNAP F-engine packet count at block start; carried
                into ProdFrameHeader.specnum on the wire.
            n_dm, n_fv, n_filled: this cube's logical shape (which may
                be smaller than :attr:`dims.shape` if the producer is
                emitting a short cube — e.g. during RFI warm-up).
            rfi_warming_up: sets the corresponding flag in the slot
                header; the worker reads it back into the prod-frame
                flags bitfield.
            publish_timeout_s: max time to wait on a ``ready_q.put``
                (which only blocks on Queue's internal buffer, not on
                consumer behaviour). The default 1 s is generous.
        """
        if self._closed:
            raise TxRingClosedError(f"publish_slot on closed ring {self._name}")
        hdr = self._header_views[slot_idx]
        hdr[0]["block_n"] = int(block_n) & 0xFFFF_FFFF_FFFF_FFFF
        hdr[0]["specnum"] = int(specnum) & 0xFFFF_FFFF_FFFF_FFFF
        hdr[0]["n_dm"] = int(n_dm) & 0xFFFF_FFFF
        hdr[0]["n_fv"] = int(n_fv) & 0xFFFF_FFFF
        hdr[0]["n_filled"] = int(n_filled) & 0xFFFF_FFFF
        flags = FLAG_RFI_WARMING_UP if rfi_warming_up else 0
        hdr[0]["flags"] = flags & 0xFF

        self._published_slots.add(slot_idx)
        self._ready_q.put(int(slot_idx), timeout=publish_timeout_s)
        self.n_publish += 1

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def wait_slot(
        self,
        *,
        timeout_s: float = 0.5,
    ) -> SlotMeta | None:
        """Block until the producer publishes a slot, then return its
        metadata. Consumer-side.

        Args:
            timeout_s: max time to wait. ``None`` is not allowed (the
                worker loop must wake periodically to check for poison-
                pill messages).

        Returns:
            :class:`SlotMeta` with the slot index + header fields, or
            ``None`` if no slot was published within ``timeout_s``.
        """
        if self._closed:
            raise TxRingClosedError(f"wait_slot on closed ring {self._name}")
        if self._poisoned:
            return None
        try:
            slot_idx = self._ready_q.get(timeout=timeout_s)
        except _stdlib_queue.Empty:
            return None
        # Poison pill: int sentinel < 0. Latch self._poisoned so the
        # caller's next ``wait_slot`` returns ``None`` immediately and
        # the worker can use ``ring.poisoned`` to distinguish "graceful
        # shutdown requested" from "no cubes yet".
        if int(slot_idx) < 0:
            self._poisoned = True
            return None
        hdr = self._header_views[int(slot_idx)][0]
        return SlotMeta(
            slot_idx=int(slot_idx),
            block_n=int(hdr["block_n"]),
            specnum=int(hdr["specnum"]),
            n_dm=int(hdr["n_dm"]),
            n_fv=int(hdr["n_fv"]),
            n_filled=int(hdr["n_filled"]),
            flags=int(hdr["flags"]),
        )

    def view_slot(self, slot_idx: int) -> np.ndarray:
        """Return a zero-copy view of the cube payload at ``slot_idx``.

        Consumer-side. The view's lifetime is bounded by the slot
        ownership window — i.e. between :meth:`wait_slot` returning the
        slot and :meth:`release_slot` releasing it.
        """
        return self._cube_views[slot_idx]

    def release_slot(self, slot_idx: int) -> None:
        """Mark ``slot_idx`` as consumed by posting to ``done_q``.
        Consumer-side. After this returns the producer is free to
        recycle the slot.
        """
        if self._closed:
            raise TxRingClosedError(f"release_slot on closed ring {self._name}")
        self._done_q.put(int(slot_idx))
        self.n_consume += 1

    # ------------------------------------------------------------------
    # Producer poison-pill
    # ------------------------------------------------------------------

    def signal_worker_exit(self) -> None:
        """Post a negative sentinel on ``ready_q`` so the worker's
        :meth:`wait_slot` returns ``None`` and the worker exits
        cleanly. Producer-side. Safe to call multiple times.
        """
        if self._closed:
            return
        try:
            self._ready_q.put(-1, timeout=0.5)
        except Exception:  # pragma: no cover — defensive on shutdown
            LOG.warning(
                "tx_ring %s: failed to post poison pill on ready_q",
                self._name,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop this handle on the shm. Owner's :meth:`close` also
        unlinks the shm. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Release numpy views BEFORE closing the shm so the underlying
        # buffer isn't pinned (avoids 'cannot close exported pointers
        # exist' warnings in Python ≥ 3.12).
        self._header_views.clear()
        self._cube_views.clear()
        try:
            # SharedMemory holds the memoryview alive via .buf; drop
            # our handle on it before calling .close().
            self._buf = None  # type: ignore[assignment]
            self._shm.close()
        except Exception:  # pragma: no cover
            LOG.exception("tx_ring %s: shm.close() raised", self._name)
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # pragma: no cover
                LOG.exception("tx_ring %s: shm.unlink() raised", self._name)

    def __enter__(self) -> "CubeShmRing":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Introspection (mon-key emitter consumers)
    # ------------------------------------------------------------------

    @property
    def dims(self) -> CubeShmRingDims:
        return self._dims

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_slots(self) -> int:
        return self._dims.n_slots

    @property
    def poisoned(self) -> bool:
        """Consumer-side: True once a poison sentinel has been observed
        on ``ready_q``. The worker checks this after :meth:`wait_slot`
        returns ``None`` to distinguish graceful shutdown from idle."""
        return self._poisoned

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of in-process counters for mon-key emit."""
        return {
            "name": self._name,
            "n_slots": self._dims.n_slots,
            "n_publish": int(self.n_publish),
            "n_consume": int(self.n_consume),
            "n_backpressure": int(self.n_backpressure),
            "published_inflight": (
                len(self._published_slots) if self._owner else -1
            ),
            "free_slots": (
                len(self._free_slots) if self._owner else -1
            ),
        }

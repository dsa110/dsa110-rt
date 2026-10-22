"""Ring-v4 consumer contract (SPEC §6) and an in-memory implementation.

The owner engine consumes ONLY:

* ``dms()`` -- owned global coarse DM indices;
* ``row_format(d)`` -- :class:`~dsart.x2.search.codec.RowFormat` of DM ``d``;
* ``wait_window(d, t_first, n_rows, timeout_s)`` -> ``(raw, valid)`` with
  ``raw uint8 [S, n_rows, row_bytes]`` and ``valid bool [S, n_rows]``
  (missing streams zero-filled and flagged; rows past retention invalid);
* ``counters()``;
* optionally ``write_pos(d)`` -- exclusive absolute row up to which the
  per-DM fan-in is complete (used by the scheduler; ``wait_window``
  blocking is the fallback).

Readers never write into the ring (the ``read_slot`` overrun-counter trap).
:class:`MemoryRing` is the CPU/test implementation (also the replay
target); WS-T's ``recv_ring`` v4 binding is a drop-in once it lands.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .codec import RowFormat

__all__ = ["RingV4Consumer", "MemoryRing", "RingTimeout"]


class RingTimeout(TimeoutError):
    pass


class RingV4Consumer(Protocol):
    def dms(self) -> list: ...

    def row_format(self, d: int) -> RowFormat: ...

    def wait_window(self, d: int, t_first: int, n_rows: int,
                    timeout_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]: ...

    def counters(self) -> dict: ...


class MemoryRing:
    """Per-DM circular buffers ``[S, t_buf, row_bytes]`` indexed by absolute
    stream row ``r % t_buf``; a writer API for tests / replay."""

    def __init__(self, formats: Dict[int, RowFormat], n_stream: int, t_buf: int) -> None:
        self._fmt = dict(formats)
        self._S = int(n_stream)
        self._T = int(t_buf)
        self._buf = {d: np.zeros((self._S, self._T, f.row_bytes), dtype=np.uint8)
                     for d, f in self._fmt.items()}
        self._valid = {d: np.zeros((self._S, self._T), dtype=bool) for d in self._fmt}
        self._rowid = {d: np.full(self._T, -1, dtype=np.int64) for d in self._fmt}
        self._hi = {d: 0 for d in self._fmt}
        self._n_wait = 0
        self._n_stale = 0

    # -- writer side ---------------------------------------------------------
    def push(self, d: int, row0: int, raw: np.ndarray, valid: Optional[np.ndarray] = None) -> None:
        S, n, nb = raw.shape
        if S != self._S or nb != self._fmt[d].row_bytes:
            raise ValueError(f"DM {d}: rows {raw.shape} vs S={self._S} "
                             f"row_bytes={self._fmt[d].row_bytes}")
        slots = np.mod(np.arange(row0, row0 + n), self._T)
        self._buf[d][:, slots] = raw
        self._valid[d][:, slots] = True if valid is None else np.asarray(valid, bool)
        self._rowid[d][slots] = np.arange(row0, row0 + n)
        self._hi[d] = max(self._hi[d], row0 + n)

    # -- consumer side -------------------------------------------------------
    def dms(self) -> list:
        return sorted(self._fmt)

    def row_format(self, d: int) -> RowFormat:
        return self._fmt[d]

    def write_pos(self, d: int) -> int:
        return int(self._hi[d])

    def wait_window(self, d: int, t_first: int, n_rows: int,
                    timeout_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while self._hi[d] < t_first + n_rows:
            self._n_wait += 1
            if deadline is None or time.monotonic() >= deadline:
                raise RingTimeout(f"DM {d}: rows [{t_first}, {t_first + n_rows}) not "
                                  f"available (write_pos {self._hi[d]})")
            time.sleep(0.001)
        rows = np.arange(t_first, t_first + n_rows)
        slots = np.mod(rows, self._T)
        current = self._rowid[d][slots] == rows
        self._n_stale += int((~current).sum())
        raw = self._buf[d][:, slots].copy()
        raw[:, ~current] = 0
        valid = self._valid[d][:, slots] & current[None, :]
        return raw, valid

    def counters(self) -> dict:
        return dict(n_wait=self._n_wait, n_stale_rows=self._n_stale,
                    write_pos={d: int(h) for d, h in self._hi.items()})

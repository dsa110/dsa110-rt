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
target); :class:`RingV4Source` binds WS-T's ``RxRingV4Reader`` (SPEC §6).

Fan-in in DATA time (:class:`RingV4Source`). The reader's own
``wait_window`` measures ``grace_s`` / ``deadline_s`` in wall time from the
CALL, so a consumer that is behind real time (catching up after a hiccup,
or a cutout reading the past) pays the full grace on every window while a
stream is missing -- with one corr node out that is 0.25 s per DM window,
~1.5 s per 201 ms cycle for 6 DMs: the engine could never catch up. Here a
window ``[t, t + n)`` is released, from ``heads(dm)["complete_through"]``
(``ct``, per stream) alone, when

* every stream is final through it (``complete``); or
* >= ``min_streams`` are (default ceil(15/16 S), today's 60/64) AND the
  leading stream is ``grace_s`` of DATA past its end (``fan_in``); or
* the leading stream is ``deadline_s`` of data past its end, whatever the
  count (``deadline``: no freeze when several corr nodes are out).

i.e. ``release_edge = max(min ct, min(ct_need, max ct - grace_rows),
max ct - deadline_rows)`` -- which is also :meth:`RingV4Source.write_pos`,
so the scheduler and the window agree. Every term is DATA time: a stream
that stopped (a corr node out) costs a live consumer a constant ``grace_s``
of latency for the whole outage -- never a wait per window -- and a lagging
consumer nothing. ``grace_s`` defaults to 0.1 s: the writer declares a
lost datagram's rows HOLE (final, zero-filled) once they are
``reorder_rows`` (64 rows: 34 / 67 ms tier A / B) behind the stream's lead
or after ``hole_timeout_s`` (50 ms), so a stream that is merely in flight is
waited for, not zero-filled at the window edge. Rows are then taken with ``copy_window`` (seqlock copy, every
non-PRESENT row zero-filled; ``access="mmap"`` C helper or ``"pread"``),
straight into the GPU backend's pinned staging when it asks
(:meth:`RingV4Source.wait_window_into`). A row is used iff ``RV_PRESENT``.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from .codec import RowFormat, StreamFormats, as_stream_formats

__all__ = ["RingV4Consumer", "MemoryRing", "RingTimeout", "RingV4Source", "RV_PRESENT",
           "RV_LAPPED"]

# SPEC §6 validity bits (frozen; WS-T's recv_ring.RV_*)
RV_PRESENT, RV_HOLE, RV_PATTERN_MISMATCH, RV_MISSING, RV_LAPPED = 1, 2, 4, 8, 16
NATIVE_SAMPLE_S = 32.768e-6


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


class RingV4Source:
    """WS-T's ``RxRingV4Reader`` (SPEC §6) as the engine's consumer, with the
    data-time fan-in rule of the module docstring. Read-only: only
    ``dms``/``fmt``/``streams``/``heads``/``copy_window``/``counters`` of the
    reader are called."""

    def __init__(self, reader, *, owned=None, min_streams: Optional[int] = None,
                 grace_s: float = 0.1, deadline_s: float = 1.0, poll_s: float = 0.001) -> None:
        self.r = reader
        infos = {int(i.dm): i for i in reader.dms()}
        own = sorted(infos) if owned is None else sorted(int(d) for d in owned)
        missing = [d for d in own if d not in infos]
        if missing:
            raise ValueError(f"ring {getattr(reader, 'shm_name', '?')} does not hold owned DMs "
                             f"{missing} (ring: {sorted(infos)})")
        self._own = own
        self._info = {d: infos[d] for d in own}
        n = int(reader.fmt.n_stream)
        self.n_stream = n
        self.need = (max(1, -(-15 * n // 16)) if min_streams is None
                     else max(1, min(int(min_streams), n)))
        self.grace_s, self.deadline_s, self.poll_s = float(grace_s), float(deadline_s), float(poll_s)
        self._grace: Dict[int, int] = {}
        self._dead: Dict[int, int] = {}
        for d, i in self._info.items():
            per = float(i.t_int_factor) * NATIVE_SAMPLE_S
            self._grace[d] = int(np.ceil(self.grace_s / per))
            self._dead[d] = max(self._grace[d], int(np.ceil(self.deadline_s / per)))
        self.fan = {d: dict(windows={}, last_status="", last_n_final=0, last_waited_ms=0.0,
                            rows_present=0, rows_missing=0, rows_lapped=0, timeouts=0)
                    for d in own}

    @classmethod
    def open(cls, shm_name: str, *, owned=None, access: str = "pread",
             attach_timeout_s: float = 0.0, **kw) -> "RingV4Source":
        from ...transport.recv_ring import RxRingV4Reader          # WS-T, SPEC §6
        return cls(RxRingV4Reader.open(shm_name, access=access,
                                       attach_timeout_s=attach_timeout_s), owned=owned, **kw)

    def close(self) -> None:
        self.r.close()

    # -- description ------------------------------------------------------
    def dms(self) -> list:
        return list(self._own)

    @property
    def row_bytes(self) -> int:
        return int(self.r.fmt.row_bytes)

    @property
    def access(self) -> str:
        return str(getattr(self.r, "access", "?"))

    def t_int_factor(self, d: int) -> int:
        return int(self._info[d].t_int_factor)

    def row_format(self, d: int) -> StreamFormats:
        f = self.r.fmt
        return StreamFormats(tuple(
            RowFormat("frame2", n_values=int(si.n_values), n_heavy=int(si.n_heavy),
                      bits=int(f.bits_main) // 2) for si in self.r.streams()))

    def validate(self, plan, inputs: Dict[int, object], vmap=None) -> List[str]:
        """Ring vs the owner plan + decode inputs (SPEC §7 "fail loudly"):
        tier clocks, stream count, per-stream value / heavy counts, bits,
        row size, decode steps and per-stream pattern ids. Raises ValueError
        listing every mismatch; returns [] when consistent."""
        f = self.r.fmt
        sinfo = self.r.streams()
        bad: List[str] = []
        for spec in plan.dms:
            d = spec.d
            if d not in self._info:
                bad.append(f"dm {d}: not in ring")
                continue
            if self.t_int_factor(d) != int(spec.t_native):
                bad.append(f"dm {d}: ring t_int_factor {self.t_int_factor(d)} != plan "
                           f"{spec.t_native}")
            if int(f.n_stream) != int(spec.n_stream):
                bad.append(f"dm {d}: ring n_stream {f.n_stream} != plan {spec.n_stream}")
                continue
            inp = inputs[d]
            fm = as_stream_formats(inp.fmt, spec.n_stream)
            if fm.kind != "frame2":
                bad.append(f"dm {d}: engine format {fm.kind} on a v4 ring")
                continue
            if int(f.row_bytes) < fm.row_bytes:
                bad.append(f"dm {d}: ring row_bytes {f.row_bytes} < engine {fm.row_bytes}")
            for s_, (si, rf) in enumerate(zip(sinfo, fm.per_stream)):
                if int(f.bits_main) != 2 * int(rf.bits):
                    bad.append(f"stream {s_}: ring bits_main {f.bits_main} != 2 x {rf.bits}")
                    break
                if int(si.n_values) and (int(si.n_values) != rf.n_values
                                         or int(si.n_heavy) != rf.n_heavy):
                    bad.append(f"stream {s_}: ring n_values/n_heavy {si.n_values}/{si.n_heavy} "
                               f"!= engine {rf.n_values}/{rf.n_heavy}")
            cv = np.asarray(inp.c_v, dtype=np.float32)
            for s_, rf in enumerate(fm.per_stream):
                if float(f.s_main) and rf.n_main and cv[s_, rf.n_heavy] != np.float32(f.s_main):
                    bad.append(f"dm {d}: decode s_main {cv[s_, rf.n_heavy]} != ring {f.s_main}")
                    break
                if float(f.s_heavy) and rf.n_heavy and cv[s_, 0] != np.float32(f.s_heavy):
                    bad.append(f"dm {d}: decode s_heavy {cv[s_, 0]} != ring {f.s_heavy}")
                    break
        pids = getattr(vmap, "stream_pattern_ids", None) if vmap is not None else None
        if pids is not None:
            for s_, si in enumerate(sinfo):
                if si.pattern_id is not None and int(si.pattern_id) != int(pids[s_]):
                    bad.append(f"stream {s_}: ring pattern_id {int(si.pattern_id):016x} != value "
                               f"map {int(pids[s_]):016x} (PATTERN_MISMATCH)")
        if bad:
            raise ValueError("ring v4 / engine mismatch: " + "; ".join(bad[:20]))
        return bad

    # -- fan-in -----------------------------------------------------------
    def _ct(self, d: int) -> np.ndarray:
        return np.asarray(self.r.heads(d)["complete_through"], dtype=np.int64)

    def _edge(self, ct: np.ndarray, d: int) -> int:
        s = np.sort(ct)
        lead = int(s[-1])
        return int(max(int(s[0]), min(int(s[s.size - self.need]), lead - self._grace[d]),
                       lead - self._dead[d]))

    def write_pos(self, d: int) -> int:
        """Exclusive release edge (rows below it may be taken now)."""
        return self._edge(self._ct(d), d)

    def quorum_pos(self, d: int) -> int:
        """Rows final on at least ``min_streams`` streams (the scheduler's
        start: never a cycle only the deadline rule releases)."""
        s = np.sort(self._ct(d))
        return int(s[s.size - self.need])

    def frontier(self, d: int) -> dict:
        ct = self._ct(d)
        return dict(complete_min=int(ct.min()), lead=int(ct.max()),
                    release_edge=self._edge(ct, d), n_stream=int(ct.size), need=self.need)

    def wait_window(self, d: int, t_first: int, n_rows: int,
                    timeout_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``(raw uint8 [S, n_rows, row_bytes], valid bool [S, n_rows])`` in a
        FRESH array (callers may keep it, e.g. a cutout)."""
        return self.wait_window_into(d, t_first, n_rows, out=None, timeout_s=timeout_s)

    def wait_window_into(self, d: int, t_first: int, n_rows: int, *,
                         out: Optional[np.ndarray] = None,
                         timeout_s: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """As :meth:`wait_window`, copied into ``out`` (e.g. pinned staging)."""
        end = int(t_first) + int(n_rows)
        t0 = time.monotonic()
        fan = self.fan[d]
        while True:
            ct = self._ct(d)
            if self._edge(ct, d) >= end:
                break
            if timeout_s is not None and time.monotonic() - t0 >= float(timeout_s):
                fan["timeouts"] += 1
                raise RingTimeout(f"dm {d}: rows [{t_first}, {end}) not released "
                                  f"(complete {int(ct.min())}, lead {int(ct.max())})")
            time.sleep(self.poll_s)
        n_final = int((ct >= end).sum())
        status = ("complete" if n_final == ct.size else
                  "fan_in" if n_final >= self.need else "deadline")
        if out is None:
            out = np.empty((self.n_stream, int(n_rows), self.row_bytes), dtype=np.uint8)
        w = self.r.copy_window(d, int(t_first), int(n_rows), out=out)
        v = np.asarray(w.validity)
        valid = (v & RV_PRESENT) != 0
        fan["windows"][status] = fan["windows"].get(status, 0) + 1
        fan["last_status"], fan["last_n_final"] = status, n_final
        fan["last_waited_ms"] = 1e3 * (time.monotonic() - t0)
        npres = int(valid.sum())
        fan["rows_present"] += npres
        fan["rows_missing"] += int(valid.size - npres)
        fan["rows_lapped"] += int(np.count_nonzero(v & RV_LAPPED))
        return w.view, valid

    def fan_in(self, d: int) -> dict:
        f = self.fan[d]
        return dict(f, windows=dict(f["windows"]))

    def counters(self) -> dict:
        try:
            rc = self.r.counters()
        except Exception as e:                                   # noqa: BLE001
            rc = dict(error=repr(e))
        return dict(reader=rc, access=self.access, need=self.need,
                    fan_in={d: self.fan_in(d) for d in self._own})

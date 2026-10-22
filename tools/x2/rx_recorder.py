#!/usr/bin/env python
"""Passive recorder of TODAY's (v3) search-node RX ring, for 1x-compat replay.

STRICTLY PASSIVE. The ring is opened ``O_RDONLY`` and mapped ``PROT_READ``:
the recorder cannot write a byte of the segment (no reader registration, no
read_seq / overrun counters, no control words — unlike ``RxRing.read_slot``,
which bumps overrun counters in the shared header). The writer may lap the
recorder; every copied row is re-validated against ``write_seq`` after the
copy and marked TORN when the writer could have overwritten it.

What is recorded (per recorded coarse DM ``d``; the node's active DMs are
detected from the slots' validity flags unless ``--dms`` is given):

* ``dm<d>.slots``  uint8 ``[T, n_corr, slot_stride]`` — the ring slots
  verbatim (payload ``n_filled * bpc`` B | scale f32 | offset f32 | vf u16 |
  pad), time-major, for consecutive ring sample indices ``t0 .. t0+T-1``.
* ``dm<d>.rows``   structured ``[T]`` (``ROW_DTYPE``): ``t`` (ring sample
  index = the wire seq the consumer treats as specnum), ``dm``, ``wall_ns``
  (UTC ns at copy), ``n_written`` / ``n_present`` / ``n_torn`` (fan-in
  counts) and ``corr_flags[n_corr]`` (``F_WRITTEN``: the writer had reached
  ``t`` for that corr when copied; ``F_TORN``: possibly overwritten during
  the copy; ``F_VF_PRESENT`` / ``F_VF_MISMATCH`` / ``F_VF_OVERRUN`` mirror the
  slot's vf bits for valid copies).
* ``header_start.bin`` / ``header_end.bin`` — the 4096-byte ring header.
* ``manifest.json`` — node, git sha, ring layout constants, args, ranges,
  counts, CPU time; written at start and rewritten at the end.

The capture front follows the production consumer
(``ProductionRxRingSource``): per corr ``written = write_seq // n_active``;
rows are copied up to the ``fan_in_min``-th fastest corr minus
``--lag-rows`` (so stragglers have landed), starting at the live edge.

Safety: run under ``nice -n 19 ionice -c3 taskset -c <core>`` (the script
also lowers its own priority and pins itself); memory is bounded by
``--chunk-rows``; it stops at ``--duration-s``, ``--max-gib`` written, the
``--max-disk-frac`` filesystem use, or ``--stall-s`` without writer progress.
``--dry-run`` attaches, samples for ``--dry-run-s`` (10 s) and prints rates
without writing anything.

Reading back: :class:`Recording` (``slots``, ``iter_cubes`` — the per-cube
compact payload/scale/validity that ``rx_ring_assemble_compact_block`` would
have produced — and ``replay_into_ring``).
"""
from __future__ import annotations

import os

# Single-threaded numerics BEFORE numpy is imported (TX-worker MKL lesson).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Iterator, Optional  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# v3 ring layout (src/dsart/transport/recv_ring.c, RX_RING_VERSION 3)
# ---------------------------------------------------------------------------

RING_MAGIC = 0xD5A1107E
RING_VERSION = 3
HEADER_SIZE = 4096
MAX_CORR = 64
OFF_WRITE_SEQ = 32
OFF_SLOT_STRIDE = 1088
OFF_DATA_OFFSET = 1096
SCALE_OFFSET_BYTES = 8

VF_DATA_PRESENT = 1 << 0
VF_PATTERN_MISMATCH = 1 << 1
VF_RX_OVERRUN = 1 << 4

F_WRITTEN = 1 << 0
F_TORN = 1 << 1
F_VF_PRESENT = 1 << 2
F_VF_MISMATCH = 1 << 3
F_VF_OVERRUN = 1 << 4

RECORDING_FORMAT = "dsart-x2-rxrec-v1"


def row_dtype(n_corr: int) -> np.dtype:
    return np.dtype([
        ("t", "<u8"), ("dm", "<u2"), ("n_written", "<u2"),
        ("n_present", "<u2"), ("n_torn", "<u2"), ("wall_ns", "<u8"),
        ("corr_flags", "u1", (n_corr,)),
    ])


@dataclass(frozen=True)
class RingLayout:
    n_corr: int
    n_coarse_dm: int
    t_buf: int
    n_filled: int
    bytes_per_cell: int
    slot_stride: int
    data_offset: int
    shm_size: int

    @property
    def payload_bytes(self) -> int:
        return self.n_filled * self.bytes_per_cell

    @property
    def vf_offset(self) -> int:
        return self.payload_bytes + SCALE_OFFSET_BYTES

    def to_dict(self) -> dict:
        return dict(self.__dict__, payload_bytes=self.payload_bytes,
                    vf_offset=self.vf_offset, ring_version=RING_VERSION,
                    header_size=HEADER_SIZE)


def _shm_path(name: str) -> str:
    return "/dev/shm/" + name.lstrip("/")


class PassiveRing:
    """Read-only (O_RDONLY + PROT_READ) view of a v3 ring segment."""

    def __init__(self, shm_name: str) -> None:
        self.shm_name = shm_name
        fd = os.open(_shm_path(shm_name), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            self._mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                                 prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        buf = np.frombuffer(self._mm, dtype=np.uint8)
        hdr32 = np.frombuffer(self._mm, dtype="<u4", count=8, offset=0)
        if int(hdr32[0]) != RING_MAGIC:
            raise RuntimeError(f"{shm_name}: bad magic 0x{int(hdr32[0]):08x}")
        if int(hdr32[1]) != RING_VERSION:
            raise RuntimeError(f"{shm_name}: ring version {int(hdr32[1])} != {RING_VERSION}")
        stride = int(np.frombuffer(self._mm, "<u8", 1, OFF_SLOT_STRIDE)[0])
        doff = int(np.frombuffer(self._mm, "<u8", 1, OFF_DATA_OFFSET)[0])
        self.layout = RingLayout(
            n_corr=int(hdr32[2]), n_coarse_dm=int(hdr32[3]), t_buf=int(hdr32[4]),
            n_filled=int(hdr32[5]), bytes_per_cell=int(hdr32[6]),
            slot_stride=stride, data_offset=doff, shm_size=size)
        L = self.layout
        need = doff + L.n_corr * L.n_coarse_dm * L.t_buf * stride
        if not (1 <= L.n_corr <= MAX_CORR) or need > size:
            raise RuntimeError(f"{shm_name}: inconsistent header {L}")
        self.data = buf[doff:need].reshape(L.n_corr, L.n_coarse_dm, L.t_buf, stride)
        self._wseq = np.frombuffer(self._mm, "<u8", L.n_corr, OFF_WRITE_SEQ)

    def write_seq(self) -> np.ndarray:
        """Snapshot of write_seq_per_corr (aligned 8-byte loads)."""
        return self._wseq.copy()

    def header_bytes(self) -> bytes:
        return bytes(self._mm[:HEADER_SIZE])

    def vf(self, corr: int, dm: int) -> np.ndarray:
        """vf u16 of every slot of (corr, dm) (copy)."""
        off = self.layout.vf_offset
        col = self.data[corr, dm, :, off:off + 2]
        return np.ascontiguousarray(col).view("<u2").reshape(-1)

    def close(self) -> None:
        self.data = None
        self._wseq = None
        try:
            self._mm.close()
        except BufferError:
            pass


def detect_active_dms(ring: PassiveRing, n_probe_corr: int = 8) -> list[int]:
    """Coarse DMs whose slots ever carried data (vf != 0) on probed corrs."""
    L = ring.layout
    active = []
    for d in range(L.n_coarse_dm):
        for c in range(min(L.n_corr, n_probe_corr)):
            if ring.vf(c, d).any():
                active.append(d)
                break
    return active


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


@dataclass
class RecorderConfig:
    shm_name: str
    out_dir: Optional[Path]
    duration_s: float = 30.0
    dms: Optional[list[int]] = None
    n_active: Optional[int] = None
    fan_in_min: int = 60
    lag_rows: int = 64
    chunk_rows: int = 64
    poll_s: float = 0.05
    max_gib: float = 40.0
    max_disk_frac: float = 0.80
    stall_s: float = 10.0
    torn_margin_rows: int = 64
    sync_every_mb: float = 256.0
    label: str = ""
    git_sha: str = ""


@dataclass
class _DmSink:
    dm: int
    slots_f: object = None
    rows_f: object = None
    t0: Optional[int] = None
    t_next: Optional[int] = None
    n_rows: int = 0
    n_torn_rows: int = 0
    n_unwritten_slots: int = 0


class Recorder:
    """Step-wise passive recorder (``poll_once``) plus a ``run`` loop."""

    def __init__(self, cfg: RecorderConfig, ring: Optional[PassiveRing] = None) -> None:
        self.cfg = cfg
        self.ring = ring if ring is not None else PassiveRing(cfg.shm_name)
        L = self.ring.layout
        self.dms = list(cfg.dms) if cfg.dms is not None else detect_active_dms(self.ring)
        if not self.dms:
            raise RuntimeError("no active coarse DM found in the ring (pass --dms)")
        bad = [d for d in self.dms if not 0 <= d < L.n_coarse_dm]
        if bad:
            raise ValueError(f"dms {bad} outside [0, {L.n_coarse_dm})")
        self.n_active = int(cfg.n_active) if cfg.n_active else len(self.dms)
        self.fan_in_min = max(1, min(int(cfg.fan_in_min), L.n_corr))
        self.rdt = row_dtype(L.n_corr)
        self.sinks = {d: _DmSink(d) for d in self.dms}
        self.bytes_written = 0
        self._synced_bytes = 0
        self.stop_reason: Optional[str] = None
        self.t_start_unix = time.time()
        self._last_progress = time.monotonic()
        self._last_wseq = self.ring.write_seq()
        self._after_copy_hook: Optional[Callable[[], None]] = None   # tests
        if cfg.out_dir is not None:
            cfg.out_dir.mkdir(parents=True, exist_ok=False)
            (cfg.out_dir / "header_start.bin").write_bytes(self.ring.header_bytes())
            for d, s in self.sinks.items():
                s.slots_f = open(cfg.out_dir / f"dm{d}.slots", "wb")
                s.rows_f = open(cfg.out_dir / f"dm{d}.rows", "wb")
            self.write_manifest(final=False)

    # -- capture -------------------------------------------------------------

    def positions(self, wseq: np.ndarray) -> np.ndarray:
        """Per-corr sample position (rows the writer has reached)."""
        return wseq.astype(np.int64) // self.n_active

    def front(self, wseq: np.ndarray) -> int:
        pos = np.sort(self.positions(wseq))
        n = pos.size
        ready = pos[n - self.fan_in_min]
        return int(ready) - int(self.cfg.lag_rows)

    def _disk_ok(self, nbytes: int) -> bool:
        cfg = self.cfg
        if cfg.out_dir is None:
            return True
        if self.bytes_written + nbytes > cfg.max_gib * (1 << 30):
            self.stop_reason = "max_gib"
            return False
        st = os.statvfs(cfg.out_dir)
        used = 1.0 - (st.f_bavail * st.f_frsize) / max(1, st.f_blocks * st.f_frsize)
        if used >= cfg.max_disk_frac:
            self.stop_reason = f"disk_use {used:.3f} >= {cfg.max_disk_frac}"
            return False
        return True

    def _copy_chunk(self, d: int, t0: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Copy rows [t0, t0+n) of dm d for every corr; return (slots, rows)."""
        ring, L = self.ring, self.ring.layout
        wseq_before = ring.write_seq()
        idx = (np.arange(t0, t0 + n) % L.t_buf)
        slots = np.ascontiguousarray(ring.data[:, d, idx, :].transpose(1, 0, 2))
        wall = time.time_ns()
        if self._after_copy_hook is not None:
            self._after_copy_hook()
        wseq_after = ring.write_seq()
        pos_b = self.positions(wseq_before)
        pos_a = self.positions(wseq_after)
        t = np.arange(t0, t0 + n, dtype=np.int64)
        written = t[:, None] < pos_b[None, :]                       # [n, n_corr]
        # The writer overwrites row t when it writes t + t_buf; allow a margin
        # for per-DM interleave and the integer position estimate.
        torn = (pos_a[None, :] + self.cfg.torn_margin_rows) >= (t[:, None] + L.t_buf)
        vf = slots[:, :, L.vf_offset:L.vf_offset + 2].copy().view("<u2")[..., 0]
        flags = np.zeros((n, L.n_corr), dtype=np.uint8)
        flags |= np.where(written, F_WRITTEN, 0).astype(np.uint8)
        flags |= np.where(torn, F_TORN, 0).astype(np.uint8)
        good = written & ~torn
        flags |= np.where(good & ((vf & VF_DATA_PRESENT) != 0), F_VF_PRESENT, 0).astype(np.uint8)
        flags |= np.where(good & ((vf & VF_PATTERN_MISMATCH) != 0), F_VF_MISMATCH, 0).astype(np.uint8)
        flags |= np.where(good & ((vf & VF_RX_OVERRUN) != 0), F_VF_OVERRUN, 0).astype(np.uint8)
        rows = np.zeros(n, dtype=self.rdt)
        rows["t"] = t
        rows["dm"] = d
        rows["wall_ns"] = wall
        rows["n_written"] = written.sum(axis=1)
        rows["n_torn"] = torn.sum(axis=1)
        rows["n_present"] = ((flags & F_VF_PRESENT) != 0).sum(axis=1)
        rows["corr_flags"] = flags
        return slots, rows

    def _maybe_flush(self, force: bool = False) -> None:
        """Bound dirty / cached pages: every ``sync_every_mb`` written,
        fdatasync and drop the written pages from the page cache, so a long
        recording never bloats node memory (the node keeps ~40 GiB of shm)."""
        mb = self.cfg.sync_every_mb
        if mb <= 0 or self.cfg.out_dir is None:
            return
        if not force and self.bytes_written - self._synced_bytes < mb * (1 << 20):
            return
        for s in self.sinks.values():
            for f in (s.slots_f, s.rows_f):
                if f is None or f.closed:
                    continue
                f.flush()
                os.fdatasync(f.fileno())
                try:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
        self._synced_bytes = self.bytes_written

    def poll_once(self) -> int:
        """Copy every complete row up to the front; return rows recorded."""
        wseq = self.ring.write_seq()
        if (wseq != self._last_wseq).any():
            self._last_progress = time.monotonic()
            self._last_wseq = wseq
        front = self.front(wseq)
        total = 0
        for d, s in self.sinks.items():
            if s.t_next is None:
                if front <= 0:
                    continue
                s.t_next = front                      # start at the live edge
                s.t0 = front
            while s.t_next < front and self.stop_reason is None:
                n = min(self.cfg.chunk_rows, front - s.t_next)
                slots, rows = self._copy_chunk(d, s.t_next, n)
                if self.cfg.out_dir is not None:
                    if not self._disk_ok(slots.nbytes + rows.nbytes):
                        break
                    s.slots_f.write(memoryview(slots).cast("B"))
                    s.rows_f.write(rows.tobytes())
                    self.bytes_written += slots.nbytes + rows.nbytes
                    self._maybe_flush()
                s.n_rows += n
                s.n_torn_rows += int((rows["n_torn"] > 0).sum())
                s.n_unwritten_slots += int((rows["corr_flags"] & F_WRITTEN == 0).sum())
                s.t_next += n
                total += n
        return total

    def run(self) -> str:
        cfg = self.cfg
        t_end = time.monotonic() + cfg.duration_s
        while self.stop_reason is None:
            now = time.monotonic()
            if now >= t_end:
                self.stop_reason = "duration"
                break
            if now - self._last_progress > cfg.stall_s:
                self.stop_reason = f"writer stalled > {cfg.stall_s} s"
                break
            self.poll_once()
            time.sleep(cfg.poll_s)
        self.close()
        return self.stop_reason

    # -- bookkeeping -----------------------------------------------------------

    def manifest(self, final: bool) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ru.ru_utime + ru.ru_stime
        wall = max(1e-9, time.time() - self.t_start_unix)
        return {
            "format": RECORDING_FORMAT,
            "final": final,
            "node": socket.gethostname(),
            "shm_name": self.cfg.shm_name,
            "label": self.cfg.label,
            "git_sha": self.cfg.git_sha or _git_sha(),
            "recorder_sha256": _file_sha256(Path(__file__)),
            "start_utc": _dt.datetime.fromtimestamp(self.t_start_unix, _dt.timezone.utc).isoformat(),
            "end_utc": _dt.datetime.now(_dt.timezone.utc).isoformat() if final else None,
            "layout": self.ring.layout.to_dict(),
            "dms": self.dms,
            "n_active_dms_per_corr": self.n_active,
            "fan_in_min": self.fan_in_min,
            "lag_rows": self.cfg.lag_rows,
            "torn_margin_rows": self.cfg.torn_margin_rows,
            "row_dtype": self.rdt.descr,
            "per_dm": {str(d): {"t0": s.t0, "n_rows": s.n_rows,
                                "n_torn_rows": s.n_torn_rows,
                                "n_unwritten_slots": s.n_unwritten_slots}
                       for d, s in self.sinks.items()},
            "bytes_written": self.bytes_written,
            "stop_reason": self.stop_reason,
            "cpu_s": cpu,
            "cpu_frac": cpu / wall,
            "maxrss_kib": ru.ru_maxrss,
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in self.cfg.__dict__.items()},
        }

    def write_manifest(self, final: bool) -> None:
        if self.cfg.out_dir is None:
            return
        p = self.cfg.out_dir / "manifest.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest(final), indent=1))
        tmp.replace(p)

    def close(self) -> None:
        if self.cfg.out_dir is not None:
            self._maybe_flush(force=True)
            for s in self.sinks.values():
                for f in (s.slots_f, s.rows_f):
                    if f is not None and not f.closed:
                        f.close()
            (self.cfg.out_dir / "header_end.bin").write_bytes(self.ring.header_bytes())
            self.write_manifest(final=True)


def dry_run(shm_name: str, seconds: float = 10.0, n_active: Optional[int] = None,
            dms: Optional[list[int]] = None) -> dict:
    """Attach, sample write_seq for ``seconds``, report rates. Writes nothing."""
    ring = PassiveRing(shm_name)
    try:
        L = ring.layout
        act = list(dms) if dms is not None else detect_active_dms(ring)
        na = int(n_active) if n_active else max(1, len(act))
        w0, t0 = ring.write_seq(), time.monotonic()
        time.sleep(seconds)
        w1, t1 = ring.write_seq(), time.monotonic()
        dt = max(1e-9, t1 - t0)
        d_slots = (w1 - w0).astype(np.int64)
        rows_s = d_slots / na / dt
        rec_bps = float(np.median(rows_s)) * L.n_corr * L.slot_stride * len(act)
        return {
            "shm_name": shm_name, "layout": L.to_dict(), "active_dms": act,
            "n_active_dms_per_corr": na, "seconds": dt,
            "slots_per_s_per_corr": (d_slots / dt).round(1).tolist(),
            "rows_per_s_median": float(np.median(rows_s)),
            "corrs_stalled": [int(i) for i in np.nonzero(d_slots == 0)[0]],
            "recording_MB_per_s": rec_bps / 1e6,
            "recording_GB_per_min": rec_bps * 60 / 1e9,
            "retention_s": L.t_buf / max(1e-9, float(np.median(rows_s))),
        }
    finally:
        ring.close()


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


class Recording:
    """Read a recording directory (memory-mapped)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != RECORDING_FORMAT:
            raise ValueError(f"{path}: not a {RECORDING_FORMAT} recording")
        lay = self.manifest["layout"]
        self.layout = RingLayout(**{k: lay[k] for k in (
            "n_corr", "n_coarse_dm", "t_buf", "n_filled", "bytes_per_cell",
            "slot_stride", "data_offset", "shm_size")})
        self.dms = [int(d) for d in self.manifest["dms"]]
        self.rdt = row_dtype(self.layout.n_corr)
        self._slots: dict[int, np.ndarray] = {}
        self._rows: dict[int, np.ndarray] = {}
        L = self.layout
        for d in self.dms:
            rows = np.fromfile(self.path / f"dm{d}.rows", dtype=self.rdt)
            n = rows.size
            sp = self.path / f"dm{d}.slots"
            if n:
                sl = np.memmap(sp, dtype=np.uint8, mode="r",
                               shape=(n, L.n_corr, L.slot_stride))
            else:
                sl = np.zeros((0, L.n_corr, L.slot_stride), np.uint8)
            self._rows[d] = rows
            self._slots[d] = sl

    def t_range(self, dm: int) -> tuple[int, int]:
        r = self._rows[dm]
        if not r.size:
            return (0, 0)
        return int(r["t"][0]), int(r["t"][-1]) + 1

    def rows(self, dm: int) -> np.ndarray:
        return self._rows[dm]

    def raw_slots(self, dm: int) -> np.ndarray:
        """uint8 ``[T, n_corr, slot_stride]`` (memory-mapped, read-only)."""
        return self._slots[dm]

    def slots(self, dm: int, t0: int, t1: int) -> dict:
        """Decoded fields of rows [t0, t1): payload int8, scale, offset, vf,
        corr_flags (rows outside the recording raise)."""
        a, b = self.t_range(dm)
        if not (a <= t0 <= t1 <= b):
            raise IndexError(f"[{t0}, {t1}) outside recorded [{a}, {b}) for dm {dm}")
        L = self.layout
        raw = np.asarray(self._slots[dm][t0 - a:t1 - a])
        pb = L.payload_bytes
        return {
            "payload": raw[:, :, :pb].view(np.int8),
            "scale": raw[:, :, pb:pb + 4].copy().view("<f4")[..., 0],
            "offset": raw[:, :, pb + 4:pb + 8].copy().view("<f4")[..., 0],
            "vf": raw[:, :, pb + 8:pb + 10].copy().view("<u2")[..., 0],
            "corr_flags": self._rows[dm]["corr_flags"][t0 - a:t1 - a],
        }

    def iter_cubes(self, dm: int, *, t_det: int = 256, cube_cadence: int = 192,
                   start: Optional[int] = None,
                   n_filled_per_corr: Optional[np.ndarray] = None) -> Iterator[dict]:
        """Yield what ``RxRing.assemble_compact_block`` (search_compute's
        compact path) would return for each cube window [s, s + t_det),
        s = start, start + cadence, ... within the recording.

        Slots the recorder flagged unwritten or torn count as overruns
        (the consumer would have read a stale or overwritten slot)."""
        L = self.layout
        if L.bytes_per_cell != 2:
            raise NotImplementedError("compact path is cint8 only")
        nfp = (np.full(L.n_corr, L.n_filled, np.int32) if n_filled_per_corr is None
               else np.asarray(n_filled_per_corr, np.int32))
        a, b = self.t_range(dm)
        s = a if start is None else int(start)
        while s + t_det <= b:
            f = self.slots(dm, s, s + t_det)
            cf = f["corr_flags"]                                  # [t, corr]
            vf = f["vf"]
            live = nfp >= 0
            rec_bad = ((cf & F_WRITTEN) == 0) | ((cf & F_TORN) != 0)
            ovr = rec_bad | ((vf & VF_RX_OVERRUN) != 0)
            pat = ~ovr & ((vf & VF_PATTERN_MISMATCH) != 0)
            nod = ~ovr & ~pat & ((vf & VF_DATA_PRESENT) == 0)
            bad = (ovr | pat | nod) & live[None, :]
            good = ~bad & live[None, :]
            cells = np.where(good.T[:, :, None], f["payload"].transpose(1, 0, 2), 0).astype(np.int8)
            scale = np.where(good, f["scale"], 0).T.astype(np.float32)
            off = np.where(good, f["offset"], 0).T.astype(np.float32)
            yield {
                "specnum_start": s,
                "cells_packed": np.ascontiguousarray(cells),
                "scale": np.ascontiguousarray(scale),
                "offset_re": np.ascontiguousarray(off),
                "offset_im": np.ascontiguousarray(off.copy()),
                "validity": ~bad.any(axis=1),
                "n_overrun": int((ovr & live[None, :]).sum()),
                "n_pattern_mismatch": int((pat & live[None, :]).sum()),
                "n_no_data_present": int((nod & live[None, :]).sum()),
            }
            s += cube_cadence

    def replay_into_ring(self, ring, *, rebase: bool = True,
                         dm_map: Optional[dict[int, int]] = None) -> int:
        """Write the recording into a v3 ``RxRing`` (owner handle) in time
        order, all recorded DMs interleaved per sample as the RX does. Rows
        flagged unwritten/torn are written as holes (vf 0, zero payload).
        ``rebase`` makes the first recorded sample t = 0 (the consumer
        derives sample positions from write_seq). Returns slots written."""
        L = self.layout
        pb = L.payload_bytes
        ranges = {d: self.t_range(d) for d in self.dms}
        t_lo = max(r[0] for r in ranges.values())
        t_hi = min(r[1] for r in ranges.values())
        n = 0
        for t in range(t_lo, t_hi):
            for d in self.dms:
                a = ranges[d][0]
                raw = np.asarray(self._slots[d][t - a])
                cf = self._rows[d]["corr_flags"][t - a]
                for c in range(L.n_corr):
                    tt = t - t_lo if rebase else t
                    dd = dm_map.get(d, d) if dm_map else d
                    if (cf[c] & F_WRITTEN) == 0 or (cf[c] & F_TORN):
                        ring.write_slot(c, dd, tt, None, 0, scale=0.0, offset=0.0)
                    else:
                        sl = raw[c]
                        ring.write_slot(
                            c, dd, tt, np.ascontiguousarray(sl[:pb]),
                            int(sl[pb + 8:pb + 10].view("<u2")[0]),
                            scale=float(sl[pb:pb + 4].view("<f4")[0]),
                            offset=float(sl[pb + 4:pb + 8].view("<f4")[0]))
                    n += 1
        return n


# ---------------------------------------------------------------------------
# Helpers + CLI
# ---------------------------------------------------------------------------


def _file_sha256(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _lower_priority(cpu: Optional[int]) -> None:
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    if shutil.which("ionice"):
        subprocess.run(["ionice", "-c3", "-p", str(os.getpid())],
                       capture_output=True, check=False)
    try:
        ncpu = os.cpu_count() or 1
        # default: the second-to-last core (the x2 CPU-test harness uses the last)
        os.sched_setaffinity(0, {max(0, ncpu - 2) if cpu is None else int(cpu)})
    except (AttributeError, OSError):
        pass


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shm-name", required=True, help="e.g. /dsart-rxring-1")
    p.add_argument("--out-root", type=Path,
                   default=Path("/home/ubuntu/x2_scratch/recordings"))
    p.add_argument("--label", default="")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--dms", type=lambda s: [int(x) for x in s.split(",")], default=None,
                   help="coarse DMs to record (default: auto-detect active)")
    p.add_argument("--n-active-dms", type=int, default=None,
                   help="slots per corr per sample (production: 2; default = #dms)")
    p.add_argument("--fan-in-min", type=int, default=60)
    p.add_argument("--lag-rows", type=int, default=64)
    p.add_argument("--chunk-rows", type=int, default=64)
    p.add_argument("--poll-s", type=float, default=0.05)
    p.add_argument("--max-gib", type=float, default=40.0)
    p.add_argument("--max-disk-frac", type=float, default=0.80)
    p.add_argument("--stall-s", type=float, default=10.0)
    p.add_argument("--cpu", type=int, default=None,
                   help="core to pin (default: second-to-last)")
    p.add_argument("--sync-every-mb", type=float, default=256.0,
                   help="fdatasync + drop written pages from the cache every N MB")
    p.add_argument("--git-sha", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-run-s", type=float, default=10.0)
    a = p.parse_args(argv)
    _lower_priority(a.cpu)
    if a.dry_run:
        print(json.dumps(dry_run(a.shm_name, a.dry_run_s, a.n_active_dms, a.dms), indent=1))
        return 0
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = a.out_root / (utc + (f"_{a.label}" if a.label else ""))
    a.out_root.mkdir(parents=True, exist_ok=True)
    rec = Recorder(RecorderConfig(
        shm_name=a.shm_name, out_dir=out, duration_s=a.duration_s, dms=a.dms,
        n_active=a.n_active_dms, fan_in_min=a.fan_in_min, lag_rows=a.lag_rows,
        chunk_rows=a.chunk_rows, poll_s=a.poll_s, max_gib=a.max_gib,
        max_disk_frac=a.max_disk_frac, stall_s=a.stall_s, label=a.label,
        sync_every_mb=a.sync_every_mb,
        git_sha=a.git_sha))
    reason = rec.run()
    m = rec.manifest(final=True)
    print(json.dumps({"out": str(out), "stop_reason": reason,
                      "bytes_written": m["bytes_written"], "cpu_frac": m["cpu_frac"],
                      "per_dm": m["per_dm"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

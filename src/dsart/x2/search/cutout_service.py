"""On-demand cutout producer of one search half (SPEC §9, frozen v1 by WS-D).

Flow per request (``docs/x2/downstream_inventory.md`` §3.4):

1. **Validate** the 128-B ``DSCQ`` request (size, magic, version,
   ``plan_id`` -- 0 means "no check", ``n_t <= 4096``, raw window <= 4 s);
   a 64-B DSRT v1 packet is counted (``c2_trigger_v1_ignored``) and ignored.
2. **Enqueue** (depth 8, FAILED/QUEUE_FULL beyond); MANUAL requests are
   served after triggered ones; send ACCEPTED (64-B ``DSCS`` to C2:11501).
3. **Raw snapshot** (RAW_ARCHIVE): per owned coarse DM the ring-v4 rows of
   every stream covering ``[t_peak - raw_pre, t_peak + raw_post)`` plus the
   stage-2 extents, written verbatim (``raw_d<DDD>.npz``, atomic) -- a start
   that rotated out is TOO_LATE, a window not complete by ``deadline_s`` is
   zero-filled and flagged.
4. **Cutout**: every owned fine re-imaged over ``[t0, t0 + n_t)`` on the
   low-priority stream (``OwnerEngine.reimage_iter``) from the snapshot (or
   the ring), normalised with the Layer-1 sigma applied at detection;
   tier-B rows are EXACT duplicates on ext rows 2k and 2k+1 (one tier
   sample, one image, two references -- C3's apex lookup relies on it);
   assembled by WS-D's ``x2.cutout.schema.assemble_cutout`` (the reference)
   and written with ``write_cutout`` (npz, then the json manifest).
5. **Report** DONE / PARTIAL / FAILED.

Socket listener and status sender are thin wrappers (``CutoutListener``);
everything else is CPU-testable without sockets or a GPU.
"""
from __future__ import annotations

import heapq
import itertools
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "REQ_STRUCT", "STATUS_STRUCT", "CutoutRequest", "decode_request", "encode_status",
    "CutoutImages", "CutoutProducer", "CutoutListener",
    "ACCEPTED", "DONE", "PARTIAL", "FAILED",
]

REQ_STRUCT = struct.Struct("<IHH16sIHHqqIIIHHffffihhhHfdQH14s")
STATUS_STRUCT = struct.Struct("<IHBB16sIBBBBfIQIh10s")
assert REQ_STRUCT.size == 128 and STATUS_STRUCT.size == 64
MAGIC_REQ = int.from_bytes(b"DSCQ", "little")
MAGIC_STATUS = int.from_bytes(b"DSCS", "little")
FLAG_CUTOUT, FLAG_RAW, FLAG_INJECTION, FLAG_MANUAL, FLAG_RESEND = 1, 2, 4, 8, 16
ACCEPTED, DONE, PARTIAL, FAILED = 1, 2, 3, 4
R_OK, R_TOO_LATE, R_WINDOW_TIMEOUT, R_QUEUE_FULL, R_PLAN_MISMATCH, R_BAD_REQUEST = 0, 1, 2, 3, 4, 5
R_GPU, R_IO, R_UPLOAD, R_SHUTDOWN = 6, 7, 8, 9


@dataclass(frozen=True)
class CutoutRequest:
    flags: int
    event_name: str
    request_id: int
    trigger_class_id: int
    t_peak_ext: int
    t0_ext: int
    n_t: int
    raw_pre_ext: int
    raw_post_ext: int
    width_ext: int
    stamp_half_px: int
    dm_peak_pc_cc: float
    snr_peak: float
    l_rad: float
    m_rad: float
    fine_global_peak: int
    coarse_global_peak: int
    l_pix: int
    m_pix: int
    n_stamp_t: int
    stamp_dm_halfwidth_pc_cc: float
    mjd_peak: float
    plan_id: int
    deadline_s: int

    @property
    def is_manual(self) -> bool:
        return bool(self.flags & FLAG_MANUAL)

    @property
    def key(self) -> tuple:
        return (self.event_name, self.request_id)


def decode_request(buf: bytes) -> CutoutRequest:
    """128-B DSCQ -> :class:`CutoutRequest`; ``ValueError`` on anything else.
    (WS-D's ``coinc/wire.py`` codec replaces this once merged -- same struct.)"""
    if len(buf) != REQ_STRUCT.size:
        raise ValueError(f"size {len(buf)}")
    f = REQ_STRUCT.unpack(buf)
    if f[0] != MAGIC_REQ or f[1] != 1:
        raise ValueError("magic/version")
    return CutoutRequest(
        flags=f[2], event_name=f[3].rstrip(b"\0").decode("ascii", "replace"), request_id=f[4],
        trigger_class_id=f[5], t_peak_ext=f[7], t0_ext=f[8], n_t=f[9], raw_pre_ext=f[10],
        raw_post_ext=f[11], width_ext=f[12], stamp_half_px=f[13], dm_peak_pc_cc=f[14],
        snr_peak=f[15], l_rad=f[16], m_rad=f[17], fine_global_peak=f[18],
        coarse_global_peak=f[19], l_pix=f[20], m_pix=f[21], n_stamp_t=f[22],
        stamp_dm_halfwidth_pc_cc=f[23], mjd_peak=f[24], plan_id=f[25], deadline_s=f[26])


def encode_request(r: CutoutRequest) -> bytes:
    """Test helper (C2 owns the sender)."""
    return REQ_STRUCT.pack(MAGIC_REQ, 1, r.flags, r.event_name.encode()[:16], r.request_id,
                           r.trigger_class_id, 0, r.t_peak_ext, r.t0_ext, r.n_t, r.raw_pre_ext,
                           r.raw_post_ext, r.width_ext, r.stamp_half_px, r.dm_peak_pc_cc,
                           r.snr_peak, r.l_rad, r.m_rad, r.fine_global_peak,
                           r.coarse_global_peak, r.l_pix, r.m_pix, r.n_stamp_t,
                           r.stamp_dm_halfwidth_pc_cc, r.mjd_peak, r.plan_id, r.deadline_s,
                           b"\0" * 14)


def encode_status(req: CutoutRequest, state: int, reason: int, *, sid: int, g: int, n_dm: int,
                  n_ok: int, valid_frac: float = 0.0, cutout_bytes: int = 0,
                  raw_bytes: int = 0, latency_ms: int = 0, upload_rc: int = -1) -> bytes:
    return STATUS_STRUCT.pack(MAGIC_STATUS, 1, state, reason, req.event_name.encode()[:16],
                              req.request_id, sid, g, n_dm, n_ok, float(valid_frac),
                              int(cutout_bytes), int(raw_bytes), int(latency_ms),
                              int(upload_rc), b"\0" * 10)


def validate_request(req: CutoutRequest, *, plan_id: int, t_ext_us: float) -> int:
    """Reason code (R_OK if acceptable)."""
    if int(req.plan_id) != 0 and int(req.plan_id) != int(plan_id):
        return R_PLAN_MISMATCH
    if req.n_t <= 0 or req.n_t > 4096 or req.n_t % 2 or req.t0_ext % 2:
        return R_BAD_REQUEST
    if (req.raw_pre_ext + req.raw_post_ext) * t_ext_us * 1e-6 > 4.0 + 1e-9:
        return R_BAD_REQUEST
    return R_OK


class CutoutImages:
    """Per-fine image source over EXTERNAL rows ``[t0_ext, t0_ext + n_t)``.

    One owned DM at a time is re-imaged (all its fines, tier rows covering
    the window) and cached (LRU of ``cache_dms``); a tier-B fine's ext rows
    2k, 2k+1 reference the SAME tier image, so they are bit-identical.
    """

    def __init__(self, engine, *, t0_ext: int, n_t: int,
                 raw: Optional[Dict[int, tuple]] = None,
                 cache_dms: int = 2, batch_rows: int = 32) -> None:
        self.eng = engine
        self.t0 = int(t0_ext)
        self.n_t = int(n_t)
        self.raw = raw or {}
        self.cache: "Dict[int, dict]" = {}
        self.order: List[int] = []
        self.cache_dms = int(cache_dms)
        self.batch_rows = int(batch_rows)
        self.specs = [s for s in sorted(engine.plan.dms, key=lambda s: s.d)]
        self.offsets = np.concatenate([[0], np.cumsum([s.n_fdm for s in self.specs])])

    def _dm(self, k: int) -> dict:
        spec = self.specs[k]
        if spec.d in self.cache:
            return self.cache[spec.d]
        r = self.eng.plan.ext_factor(spec)
        e_rows = np.arange(self.t0, self.t0 + self.n_t)
        t_rows = e_rows // r
        tlo, thi = int(t_rows[0]), int(t_rows[-1]) + 1
        snap = self.raw.get(spec.d)
        kw = {} if snap is None else dict(raw=snap[0], valid=snap[1], raw_row0=int(snap[2]))
        imgs, sig, vf = [], [], []
        for rows, img, v in self.eng.reimage_iter(spec.d, tlo, thi, batch_rows=self.batch_rows,
                                                  normalise=True, **kw):
            imgs.append(img.astype(np.float16))
            s, _found = self.eng.sigma_at(spec.d, rows)
            sig.append(s)
            vf.append((len(rows), v))
        tier_img = np.concatenate(imgs, axis=0)                  # [n_tier, F_d, N, N]
        idx = t_rows - tlo                                       # ext row -> tier row
        ent = dict(img=tier_img, idx=idx, sigma=np.concatenate(sig, axis=0)[idx],
                   valid_frac=float(sum(n * v for n, v in vf) / max(1, sum(n for n, _ in vf))))
        self.cache[spec.d] = ent
        self.order.append(spec.d)
        while len(self.order) > self.cache_dms:
            self.cache.pop(self.order.pop(0), None)
        return ent

    def block_of(self, f: int) -> tuple:
        k = int(np.searchsorted(self.offsets, f, side="right") - 1)
        return k, int(f - self.offsets[k])

    def __call__(self, f: int) -> np.ndarray:
        """``[n_t, N, N]`` float16 Layer-1-normalised images of plane fine ``f``."""
        k, fl = self.block_of(f)
        ent = self._dm(k)
        return ent["img"][ent["idx"], fl]                       # tier-B: duplicated rows

    def sigma_applied(self) -> np.ndarray:
        out = np.zeros((self.n_t, int(self.offsets[-1])), dtype=np.float32)
        for k in range(len(self.specs)):
            ent = self._dm(k)
            out[:, self.offsets[k]:self.offsets[k + 1]] = ent["sigma"]
        return out

    def valid(self) -> tuple:
        """``(row_valid [n_t, D] uint8, input_valid_frac [F] float32)``."""
        D = len(self.specs)
        rv = np.ones((self.n_t, D), dtype=np.uint8)
        ivf = np.zeros(int(self.offsets[-1]), dtype=np.float32)
        for k in range(D):
            ent = self._dm(k)
            ivf[self.offsets[k]:self.offsets[k + 1]] = ent["valid_frac"]
            if ent["valid_frac"] < 1.0:
                rv[:, k] = 0
        return rv, ivf


class CutoutProducer:
    """Request -> raw snapshot -> cutout arrays (WS-D schema) -> files."""

    def __init__(self, engine, *, search_node_id: int, gpu_half: int, out_root: Path,
                 plan=None, plan_id: int = 0, code_version: str = "", queue_depth: int = 8,
                 geometry: Optional[dict] = None, clock_mjd=None) -> None:
        self.eng = engine
        self.sid = int(search_node_id)
        self.g = int(gpu_half)
        self.out_root = Path(out_root)
        self.plan = plan
        self.plan_id = int(plan_id)
        self.code_version = code_version
        self.queue_depth = int(queue_depth)
        self.geometry = dict(geometry or dict(n_grid=engine.cfg.n_grid))
        self.clock_mjd = clock_mjd
        self._q: list = []
        self._seq = itertools.count()
        self._seen: Dict[tuple, int] = {}
        self.counters = dict(requests=0, accepted=0, done=0, partial=0, failed=0,
                             failed_queue_full=0, failed_plan_mismatch=0, failed_bad_request=0,
                             failed_too_late=0, c2_trigger_v1_ignored=0)

    @property
    def t_ext_us(self) -> float:
        return float(self.eng.plan.ext_period_us)

    def offer(self, buf: bytes) -> Optional[tuple]:
        """Packet in -> ``(request, state, reason)`` to report, or None."""
        if len(buf) == 64:
            self.counters["c2_trigger_v1_ignored"] += 1
            return None
        try:
            req = decode_request(buf)
        except ValueError:
            return None
        self.counters["requests"] += 1
        if req.key in self._seen:                                   # duplicate / RESEND
            return req, self._seen[req.key], R_OK
        why = validate_request(req, plan_id=self.plan_id, t_ext_us=self.t_ext_us)
        if why != R_OK:
            self.counters["failed"] += 1
            self.counters["failed_plan_mismatch" if why == R_PLAN_MISMATCH
                          else "failed_bad_request"] += 1
            self._seen[req.key] = FAILED
            return req, FAILED, why
        if len(self._q) >= self.queue_depth:
            self.counters["failed"] += 1
            self.counters["failed_queue_full"] += 1
            self._seen[req.key] = FAILED
            return req, FAILED, R_QUEUE_FULL
        heapq.heappush(self._q, (1 if req.is_manual else 0, next(self._seq), req))
        self._seen[req.key] = ACCEPTED
        self.counters["accepted"] += 1
        return req, ACCEPTED, R_OK

    def next_request(self) -> Optional[CutoutRequest]:
        return heapq.heappop(self._q)[2] if self._q else None

    # ------------------------------------------------------------------
    def raw_snapshot(self, req: CutoutRequest, out_dir: Optional[Path] = None) -> tuple:
        """Per owned DM ``(raw, valid, row0)`` covering the raw window plus
        stage-2 extents; written verbatim (atomic) when ``out_dir``."""
        snaps, too_late, raw_bytes = {}, [], 0
        lo_e = req.t_peak_ext - req.raw_pre_ext
        hi_e = req.t_peak_ext + req.raw_post_ext
        for spec in self.eng.plan.dms:
            r = self.eng.plan.ext_factor(spec)
            t_lo, t_hi = lo_e // r, -(-hi_e // r)
            s_lo, s_hi = t_lo - spec.max_pos, t_hi + spec.max_neg
            try:
                raw, valid = self.eng.ring.wait_window(spec.d, s_lo, s_hi - s_lo,
                                                       timeout_s=float(req.deadline_s))
            except TimeoutError:
                too_late.append(spec.d)
                continue
            snaps[spec.d] = (raw, valid, s_lo)
            raw_bytes += raw.nbytes
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp = out_dir / f".raw_d{spec.d:03d}.npz.tmp"
                with open(tmp, "wb") as fh:
                    np.savez(fh, codes=raw, valid=valid, row0=np.int64(s_lo),
                             t_native=np.int32(spec.t_native), d=np.int32(spec.d),
                             shifts=spec.shifts, fine_global=spec.fine_global)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, out_dir / f"raw_d{spec.d:03d}.npz")
        return snaps, too_late, raw_bytes

    def build_arrays(self, req: CutoutRequest, snaps: Optional[dict] = None) -> tuple:
        """``(arrays dict, status)`` via WS-D's ``assemble_cutout``."""
        from ..cutout.schema import assemble_cutout, fine_axis_for_half

        t0 = int(req.t0_ext)
        src = CutoutImages(self.eng, t0_ext=t0, n_t=int(req.n_t), raw=snaps or {})
        axis = fine_axis_for_half(self.plan, self.eng.plan.half,
                                  t_ext_native=self.eng.plan.ext_native)
        rv, ivf = src.valid()
        status = "complete" if rv.all() else "partial"
        mjd = self.clock_mjd(t0) if self.clock_mjd is not None else 0.0
        arrays = assemble_cutout(
            axis=axis, images=src, n_t=int(req.n_t), sigma_applied=src.sigma_applied(),
            row_valid=rv, input_valid_frac=ivf, request=req,
            identity=dict(search_node_id=self.sid, gpu_half=self.g, half_index=self.eng.plan.half,
                          opoint=self.eng.plan.opoint, plan_id=self.plan_id,
                          pattern_id=0, code_version=self.code_version),
            clock=dict(t_ext_us=self.t_ext_us, native_per_ext=self.eng.plan.ext_native,
                       mjd_t0=mjd, mjd_clock="capture_epoch+native" if self.clock_mjd else ""),
            geometry=self.geometry, status=status,
            kernel_ids=self.eng.bank.kernel_ids(),
            sigma_k=np.stack([self.eng.states[s.d].layer2._s_k.detach().cpu().numpy()
                              for s in sorted(self.eng.plan.dms, key=lambda s: s.d)]))
        return arrays, status

    def serve(self, req: CutoutRequest, *, event_dir: Path) -> tuple:
        """Run one job; returns ``(state, reason, info)``."""
        from ..cutout.schema import write_cutout

        t_start = time.monotonic()
        snaps, too_late, raw_bytes = ({}, [], 0)
        if req.flags & FLAG_RAW:
            snaps, too_late, raw_bytes = self.raw_snapshot(
                req, event_dir / "raw" / f"raw_s{self.sid}_g{self.g}")
        try:
            arrays, status = self.build_arrays(req, snaps)
            path = write_cutout(event_dir / "cutouts", arrays)
        except Exception as e:                                  # noqa: BLE001
            self.counters["failed"] += 1
            return FAILED, R_IO, dict(error=repr(e))
        state = DONE if (status == "complete" and not too_late) else PARTIAL
        self.counters["done" if state == DONE else "partial"] += 1
        self._seen[req.key] = state
        return state, (R_TOO_LATE if too_late else R_OK), dict(
            path=str(path), raw_bytes=raw_bytes, too_late=too_late,
            latency_ms=int(1e3 * (time.monotonic() - t_start)))


class CutoutListener:
    """UDP listener on ``base_port + g`` + status sender to C2 (thin)."""

    def __init__(self, producer: CutoutProducer, *, bind_host: str, base_port: int,
                 c2_host: str, status_port: int = 11501, event_root: Path) -> None:
        self.p = producer
        self.addr = (bind_host, int(base_port) + producer.g)
        self.c2 = (c2_host, int(status_port))
        self.event_root = Path(event_root)
        self._stop = threading.Event()
        self._sock: Optional[socket.socket] = None

    def _send(self, payload: bytes) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(payload, self.c2)
        except OSError:
            pass                                            # lossy by design

    def _status(self, req, state, reason, **kw) -> None:
        n = len(self.p.eng.plan.dms)
        self._send(encode_status(req, state, reason, sid=self.p.sid, g=self.p.g, n_dm=n,
                                 n_ok=n if state == DONE else 0, **kw))

    def run(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(self.addr)
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                buf, _a = self._sock.recvfrom(256)
                got = self.p.offer(buf)
                if got is not None:
                    self._status(got[0], got[1], got[2])
            except socket.timeout:
                pass
            req = self.p.next_request()
            if req is not None:
                state, reason, info = self.p.serve(req, event_dir=self.event_root / req.event_name)
                self._status(req, state, reason, raw_bytes=info.get("raw_bytes", 0),
                             latency_ms=info.get("latency_ms", 0))
        self._sock.close()

    def stop(self) -> None:
        self._stop.set()

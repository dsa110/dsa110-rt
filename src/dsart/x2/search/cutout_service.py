"""On-demand cutout producer of one search half (SPEC §9, frozen v1 by WS-D).

Flow per request (``docs/x2/downstream_inventory.md`` §3.4):

1. **Validate** the 128-B ``DSCQ`` request (size, magic, version,
   ``plan_id`` -- 0 means "no check", ``n_t <= 4096``, raw window <= 4 s);
   a 64-B DSRT v1 packet is counted (``c2_trigger_v1_ignored``) and ignored.
2. **Enqueue** (depth 8, FAILED/QUEUE_FULL beyond); MANUAL requests are
   served after triggered ones; send ACCEPTED (64-B ``DSCS`` to C2:11501).
3. **Copy at accept** (``capture``, on the socket thread): per owned coarse
   DM the ring-v4 rows of every stream covering the cutout window and, with
   RAW_ARCHIVE, ``[t_peak - raw_pre, t_peak + raw_post)``, plus the stage-2
   extents -- before they rotate out of the 20 s ring (a window not released
   by ``deadline_s``, or already rotated out, is TOO_LATE). A builder thread
   serves the queue: the raw window is written verbatim with WS-D's D4
   ``raw_archive`` (``write_raw_slab`` per DM, ``write_raw_meta``,
   ``mark_raw_ok`` last; nothing when no DM has rows).
4. **Cutout**: every owned fine re-imaged ONCE over ``[t0, t0 + n_t)`` on
   the low-priority stream (``OwnerEngine.reimage_iter``) from the copy (or
   the ring), with the per-fine reductions on the GPU, normalised with the
   Layer-1 sigma applied at detection; tier-B rows are EXACT duplicates on ext rows 2k and 2k+1 (one tier
   sample, one image, two references -- C3's apex lookup relies on it);
   assembled by WS-D's ``x2.cutout.schema.assemble_cutout`` (the reference)
   and written with ``write_cutout`` (npz, then the json manifest).
5. **Upload** (``make_uploaders``): the cutout first, then the raw window;
   each half purges only its own files. **Report** DONE / PARTIAL / FAILED.

Socket listener and status sender are thin wrappers (``CutoutListener``);
everything else is CPU-testable without sockets or a GPU.
"""
from __future__ import annotations

import heapq
import itertools
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

__all__ = [
    "REQ_STRUCT", "STATUS_STRUCT", "CutoutRequest", "decode_request", "encode_status",
    "CutoutImages", "CutoutProducer", "CutoutListener", "make_uploaders", "fine_axis_for_owner",
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


def fine_axis_for_owner(oplan, plan=None, *, t_ext_native: int) -> Dict[str, np.ndarray]:
    """The cutout's fine axis + D-axis CSR (WS-D ``schema.fine_axis_for_half``
    keys and dtypes) for the coarse DMs the ENGINE owns -- ``oplan.dms`` in
    ascending ``d``, each DM's fines in plan order: the same order
    :class:`CutoutImages` lays out its planes in, so the two can never
    disagree when an engine owns only some of its half's DMs.
    ``fine_dm`` / ``fine_local_idx`` / ``tier`` come from the tiered
    ``plan`` when given (checked against the owner plan), else from the
    DmSpecs (tier A -> 0, B -> 1; local index = position in the DM)."""
    from ..cutout.schema import CutoutError

    specs = sorted(oplan.dms, key=lambda s: s.d)
    fines = np.concatenate([np.asarray(s.fine_global, np.int64) for s in specs]
                           or [np.zeros(0, np.int64)])
    dms = np.array([s.d for s in specs], dtype=np.int64)
    f2c = np.concatenate([np.full(s.n_fdm, s.d, np.int64) for s in specs]
                         or [np.zeros(0, np.int64)])
    t_nat = np.concatenate([np.full(s.n_fdm, s.t_native, np.int64) for s in specs]
                           or [np.zeros(0, np.int64)])
    if np.any(t_nat % int(t_ext_native)):
        raise CutoutError(f"coarse sample {sorted(set(t_nat.tolist()))} not a multiple of "
                          f"the external sample {t_ext_native}")
    if plan is not None:
        if not np.array_equal(np.asarray(plan.fine_to_coarse)[fines], f2c):
            raise CutoutError("owner plan fines are not the tiered plan's fines of their DMs")
        fine_dm = np.asarray(plan.fine_dm)[fines]
        local = np.asarray(plan.fine_local_idx)[fines]
        tier = np.asarray(plan.fine_tier)[fines]
    else:
        fine_dm = np.concatenate([np.asarray(s.fine_dm) for s in specs] or [np.zeros(0)])
        local = np.concatenate([np.arange(s.n_fdm) for s in specs] or [np.zeros(0, np.int64)])
        tier = np.concatenate([np.full(s.n_fdm, 1 if s.tier == "B" else 0) for s in specs]
                              or [np.zeros(0, np.int64)])
    offset = np.zeros(len(specs) + 1, dtype=np.int32)
    offset[1:] = np.cumsum([s.n_fdm for s in specs])
    return {
        "fine_dm_pc_cc": np.asarray(fine_dm).astype(np.float32),
        "fine_global_idx": fines.astype(np.int32),
        "coarse_global_idx": f2c.astype(np.int16),
        "fine_local_idx": np.asarray(local).astype(np.int16),
        "tier": np.asarray(tier).astype(np.uint8),
        "t_step_ext": (t_nat // int(t_ext_native)).astype(np.uint8),
        "coarse_owned": dms.astype(np.int16),
        "fine_offset": offset,
    }


class CutoutImages:
    """Per-fine image source over EXTERNAL rows ``[t0_ext, t0_ext + n_t)``.

    Each owned DM is re-imaged ONCE (all its fines, tier rows covering the
    window) and kept for the request as float16, with the ``reduce_images``
    reductions at pixel ``pix`` computed alongside (on the GPU in
    production); a tier-B fine's ext rows 2k, 2k+1 reference the SAME tier
    image, so they are bit-identical. (G2 2026-11-09: an LRU of 2 DMs over a
    half's 3 re-imaged each DM ~12 times per request -- cutouts took
    150-190 s against C2's 120 s wait.)
    """

    def __init__(self, engine, *, t0_ext: int, n_t: int,
                 raw: Optional[Dict[int, tuple]] = None, pix: tuple = (-1, -1),
                 batch_rows: int = 32, batch_pause_s: float = 0.0) -> None:
        self.eng = engine
        self.t0 = int(t0_ext)
        self.n_t = int(n_t)
        self.raw = raw or {}
        self.pix = (int(pix[0]), int(pix[1]))
        self.cache: "Dict[int, dict]" = {}
        self.batch_rows = int(batch_rows)
        self.batch_pause_s = float(batch_pause_s)
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
        N = self.eng.cfg.n_grid
        F = spec.n_fdm
        tier_img = np.empty((thi - tlo, F, N, N), dtype=np.float16)
        peak = np.empty((thi - tlo, F), dtype=np.float32)
        ppix = np.empty((thi - tlo, F), dtype=np.int32)
        lc = np.empty((thi - tlo if self.pix[0] >= 0 and self.pix[1] >= 0 else 0, F),
                      dtype=np.float32)
        sig, vf = [], []
        for rows, img, v, red in self.eng.reimage_iter(
                spec.d, tlo, thi, batch_rows=self.batch_rows, normalise=True,
                reduce_pix=self.pix, **kw):
            a, b = int(rows[0]) - tlo, int(rows[-1]) + 1 - tlo
            tier_img[a:b] = img
            peak[a:b], ppix[a:b] = red[0], red[1]
            if lc.shape[0]:
                lc[a:b] = red[2]
            s, _found = self.eng.sigma_at(spec.d, rows)
            sig.append(s)
            vf.append((len(rows), v))
            if self.batch_pause_s > 0:                           # let the live search breathe
                time.sleep(self.batch_pause_s)
        idx = t_rows - tlo                                       # ext row -> tier row
        ent = dict(img=tier_img, idx=idx, sigma=np.concatenate(sig, axis=0)[idx],
                   peak=peak[idx], pix=ppix[idx], lc=lc[idx] if lc.shape[0] else lc,
                   valid_frac=float(sum(n * v for n, v in vf) / max(1, sum(n for n, _ in vf))))
        self.cache[spec.d] = ent
        return ent

    def block_of(self, f: int) -> tuple:
        k = int(np.searchsorted(self.offsets, f, side="right") - 1)
        return k, int(f - self.offsets[k])

    def __call__(self, f: int) -> np.ndarray:
        """``[n_t, N, N]`` float16 Layer-1-normalised images of plane fine ``f``."""
        k, fl = self.block_of(f)
        ent = self._dm(k)
        return ent["img"][ent["idx"], fl]                       # tier-B: duplicated rows

    def reduced(self, f: int) -> tuple:
        """``reduce_images`` of plane fine ``f`` over the window rows:
        ``(peak [n_t], pix [n_t], lc [n_t] or [0])`` -- what
        ``assemble_cutout`` would compute from ``self(f)``, without it."""
        k, fl = self.block_of(f)
        ent = self._dm(k)
        return ent["peak"][:, fl], ent["pix"][:, fl], ent["lc"][:, fl]

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
                 geometry: Optional[dict] = None, clock_mjd=None,
                 batch_pause_s: float = 0.0) -> None:
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
        self.batch_pause_s = float(batch_pause_s)
        self.uploaders: Optional[tuple] = None
        self.last_latency_ms = 0
        self._q: list = []
        self._seq = itertools.count()
        self._seen: Dict[tuple, int] = {}
        self._lock = threading.Lock()
        self._held: Dict[tuple, CutoutRequest] = {}            # accepted, rows not copied yet
        self._snaps: Dict[tuple, tuple] = {}                   # key -> copy_rows() result
        self.last_capture_ms = 0
        self.counters = dict(requests=0, accepted=0, done=0, partial=0, failed=0,
                             failed_queue_full=0, failed_plan_mismatch=0, failed_bad_request=0,
                             failed_too_late=0, c2_trigger_v1_ignored=0)

    @property
    def t_ext_us(self) -> float:
        return float(self.eng.plan.ext_period_us)

    def offer(self, buf: bytes, *, hold: bool = False) -> Optional[tuple]:
        """Packet in -> ``(request, state, reason)`` to report, or None.
        With ``hold`` an accepted request is queued only by :meth:`capture`
        (the listener copies its rows first)."""
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
        with self._lock:
            if len(self._q) + len(self._held) >= self.queue_depth:
                self.counters["failed"] += 1
                self.counters["failed_queue_full"] += 1
                self._seen[req.key] = FAILED
                return req, FAILED, R_QUEUE_FULL
            if hold:
                self._held[req.key] = req
            else:
                heapq.heappush(self._q, (1 if req.is_manual else 0, next(self._seq), req))
            self._seen[req.key] = ACCEPTED
        self.counters["accepted"] += 1
        return req, ACCEPTED, R_OK

    def next_request(self) -> Optional[CutoutRequest]:
        with self._lock:
            return heapq.heappop(self._q)[2] if self._q else None

    def capture(self, req: CutoutRequest) -> bool:
        """At ACCEPT: copy the request's rows out of the ring before they
        rotate out (retention 20 s; C2's request arrives ~8 s after the
        burst), then queue it. False if ``req`` was not held (duplicate).
        (G2 2026-11-09: copying only when a request was served let a
        backlog of slow builds lap every later window -- empty cutouts and
        ~9 GB raw windows of invalid rows.)"""
        with self._lock:
            if self._held.pop(req.key, None) is None:
                return False
        t = time.monotonic()
        try:
            got = self.copy_rows(req)
        except Exception:                                       # noqa: BLE001
            got = None                                          # serve() copies (and fails) itself
        self.last_capture_ms = int(1e3 * (time.monotonic() - t))
        with self._lock:
            if got is not None:
                self._snaps[req.key] = got
            heapq.heappush(self._q, (1 if req.is_manual else 0, next(self._seq), req))
        return True

    # ------------------------------------------------------------------
    def _windows(self, req: CutoutRequest, spec) -> tuple:
        """``((s_lo, n) raw window or None, (s_lo, n) cutout window)`` in tier rows."""
        from ..cutout.raw_archive import stream_window

        kw = dict(ext_factor=self.eng.plan.ext_factor(spec), max_pos=spec.max_pos,
                  max_neg=spec.max_neg)
        cut = stream_window(req.t0_ext, 0, req.n_t, **kw)
        raw = (stream_window(req.t_peak_ext, req.raw_pre_ext, req.raw_post_ext, **kw)
               if req.flags & FLAG_RAW else None)
        return raw, cut

    def copy_rows(self, req: CutoutRequest) -> tuple:
        """Per owned DM ``(raw, valid, row0)`` covering the cutout window and,
        with RAW_ARCHIVE, the raw window (``raw_archive.stream_window``) ->
        ``(snaps, too_late, raw_bytes)``. A window not released by
        ``deadline_s``, or whose rows have all rotated out, is TOO_LATE."""
        snaps, too_late, raw_bytes = {}, [], 0
        for spec in self.eng.plan.dms:
            rw, cw = self._windows(req, spec)
            lo, hi = cw[0], cw[0] + cw[1]
            if rw is not None:
                lo, hi = min(lo, rw[0]), max(hi, rw[0] + rw[1])
            try:
                raw, valid = self.eng.ring.wait_window(spec.d, lo, hi - lo,
                                                       timeout_s=float(req.deadline_s))
            except TimeoutError:
                too_late.append(spec.d)
                continue
            if not np.asarray(valid).any():                        # lapped: nothing to keep
                too_late.append(spec.d)
                continue
            snaps[spec.d] = (raw, valid, lo)
            raw_bytes += raw.nbytes
        return snaps, too_late, raw_bytes

    def write_raw(self, req: CutoutRequest, snaps: dict, too_late: list, out_dir: Path) -> int:
        """The raw window of every snapshot DM with WS-D's D4 ``write_raw_slab``,
        then ``write_raw_meta``, then ``mark_raw_ok`` LAST (readers trust only
        RAW_OK dirs). Nothing is written when no DM has rows. Returns bytes."""
        from ..cutout.raw_archive import mark_raw_ok, write_raw_meta, write_raw_slab
        from .codec import as_stream_formats

        if not snaps:
            return 0
        n_bytes = 0
        for spec in self.eng.plan.dms:
            if spec.d not in snaps:
                continue
            raw, valid, row0 = snaps[spec.d]
            s_lo, n = self._windows(req, spec)[0]
            a = s_lo - int(row0)
            codes = np.ascontiguousarray(raw[:, a:a + n], dtype=np.uint8)
            inp = self.eng.states[spec.d].inp
            fm = as_stream_formats(inp.fmt, spec.n_stream)
            extra = dict(n_values=np.array([f.n_values for f in fm.per_stream], np.int32),
                         n_heavy=np.array([getattr(f, "n_heavy", 0) for f in fm.per_stream],
                                          np.int32),
                         bits_main=np.int32(2 * getattr(fm.per_stream[0], "bits", 8)),
                         n_grid=np.int32(self.eng.cfg.n_grid))
            if inp.pattern_id is not None:
                extra["pattern_id"] = np.uint64(int(inp.pattern_id))
            write_raw_slab(out_dir, codes=codes,
                           valid=np.asarray(valid[:, a:a + n]).astype(np.uint8), row0=int(s_lo),
                           t_native=int(spec.t_native), d=int(spec.d), shifts=spec.shifts,
                           fine_global=spec.fine_global, **extra)
            n_bytes += codes.nbytes
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_raw_meta(out_dir, req, identity=dict(
            search_node_id=self.sid, gpu_half=self.g, half_index=self.eng.plan.half,
            opoint=self.eng.plan.opoint, plan_id=self.plan_id), too_late=too_late)
        mark_raw_ok(out_dir)
        return n_bytes

    def raw_snapshot(self, req: CutoutRequest, out_dir: Optional[Path] = None) -> tuple:
        """:meth:`copy_rows` now, then (``out_dir``, RAW_ARCHIVE) :meth:`write_raw`."""
        snaps, too_late, raw_bytes = self.copy_rows(req)
        if out_dir is not None and req.flags & FLAG_RAW:
            self.write_raw(req, snaps, too_late, out_dir)
        return snaps, too_late, raw_bytes

    def build_arrays(self, req: CutoutRequest, snaps: Optional[dict] = None) -> tuple:
        """``(arrays dict, status)`` via WS-D's ``assemble_cutout``."""
        from ..cutout.schema import assemble_cutout

        t0 = int(req.t0_ext)
        src = CutoutImages(self.eng, t0_ext=t0, n_t=int(req.n_t), raw=snaps or {},
                           pix=(int(req.l_pix), int(req.m_pix)), batch_pause_s=self.batch_pause_s)
        # the engine's own DMs (not the whole half's: an engine may own a subset)
        axis = fine_axis_for_owner(self.eng.plan, self.plan,
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
        """Run one job; returns ``(state, reason, info)``. Rows copied by
        :meth:`capture` are used, else copied now. Uploads (when ``uploaders``
        are set): the cutout first, then the raw window (D4)."""
        from ..cutout.schema import write_cutout

        t_start = time.monotonic()
        raw_dir = event_dir / "raw" / f"raw_s{self.sid}_g{self.g}"
        with self._lock:
            got = self._snaps.pop(req.key, None)
        raw_bytes = 0
        try:
            snaps, too_late, _n = got if got is not None else self.copy_rows(req)
            if req.flags & FLAG_RAW:
                raw_bytes = self.write_raw(req, snaps, too_late, raw_dir)
            arrays, status = self.build_arrays(req, snaps)
            path, _manifest = write_cutout(event_dir / "cutouts", arrays)
        except Exception as e:                                  # noqa: BLE001
            self.counters["failed"] += 1
            with self._lock:
                self._seen[req.key] = FAILED
            return FAILED, R_IO, dict(error=repr(e))
        if self.uploaders is not None:
            cut_up, raw_up = self.uploaders
            cut_up.submit(req.event_name, event_dir / "cutouts")
            if raw_bytes:
                raw_up.submit(req.event_name, raw_dir)
        self.last_latency_ms = int(1e3 * (time.monotonic() - t_start))
        state = DONE if (status == "complete" and not too_late) else PARTIAL
        self.counters["done" if state == DONE else "partial"] += 1
        with self._lock:
            self._seen[req.key] = state
        return state, (R_TOO_LATE if too_late else R_OK), dict(
            path=str(path), raw_bytes=raw_bytes, too_late=too_late,
            latency_ms=int(1e3 * (time.monotonic() - t_start)))


def make_uploaders(*, sid: int, g: int, dest_host: str, dest_root: str,
                   raw_bwlimit_kbps: int = 50_000) -> tuple:
    """WS-D D4: two bounded uploaders -- (a) the cutout (``cutouts``, purge
    only this half's ``cutout_s<sid>_g<g>.*``), (b) the raw window
    (``raw/raw_s<sid>_g<g>``, purge ``*``, bandwidth-limited)."""
    from ...coinc.cube_uploader import BoundedCubeUploader

    cut = BoundedCubeUploader(dest_host=dest_host, dest_root=dest_root, dest_subdir="cutouts",
                              purge_pattern=f"cutout_s{sid}_g{g}.*",
                              thread_name=f"cutout-upload-g{g}")
    raw = BoundedCubeUploader(dest_host=dest_host, dest_root=dest_root,
                              dest_subdir=f"raw/raw_s{sid}_g{g}", purge_pattern="*",
                              bandwidth_limit_kbps=int(raw_bwlimit_kbps),
                              thread_name=f"raw-upload-g{g}")
    return cut, raw


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
        """Socket thread: validate, ACK, copy the rows (``capture``); a builder
        thread serves the queue, so a slow build never delays a copy."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(self.addr)
        self._sock.settimeout(0.2)
        ready = threading.Event()
        builder = threading.Thread(target=self._build_loop, args=(ready,), daemon=True,
                                   name=f"x2-cutout-build-g{self.p.g}")
        builder.start()
        while not self._stop.is_set():
            try:
                buf, _a = self._sock.recvfrom(256)
            except socket.timeout:
                continue
            got = self.p.offer(buf, hold=True)
            if got is None:
                continue
            self._status(got[0], got[1], got[2])
            if got[1] == ACCEPTED and self.p.capture(got[0]):
                ready.set()
        self._sock.close()
        builder.join(timeout=5.0)

    def _build_loop(self, ready: threading.Event) -> None:
        while not self._stop.is_set():
            req = self.p.next_request()
            if req is None:
                ready.wait(0.2)
                ready.clear()
                continue
            state, reason, info = self.p.serve(req, event_dir=self.event_root / req.event_name)
            self._status(req, state, reason, raw_bytes=info.get("raw_bytes", 0),
                         latency_ms=info.get("latency_ms", 0))

    def stop(self) -> None:
        self._stop.set()

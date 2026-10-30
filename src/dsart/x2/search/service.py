"""2x search half as a service: scheduler + engine + per-cycle C1 + cutouts + mon.

``python -m dsart.services.search_compute --opoint 2x-<kind>-ns<n>-cint<b> ...``
dispatches here (the 1x path never imports this module). One process = one
search half (node, GPU), as today.

* :class:`CycleScheduler` -- cycle ``k`` covers native samples
  ``[k*6144, (k+1)*6144)``; a cycle runs when every owned DM's ring
  frontier covers its rows (+ the negative stage-2 extent); if the newest
  complete cycle is more than ``max_lag_cycles`` ahead the scheduler seeks
  forward (counted -- today's lag-recovery seek, per cycle).
* :class:`OwnerService` -- per cycle: ``engine.process_cycle(k)``, the
  half's candidates pooled ONCE through :class:`~dsart.x2.search.c1_cycle.
  C1CyclePool`, one C1 batch per cycle (``C1BatchHeader`` in external
  samples: ``sample_period_specnum`` = 8, ``sample_period_us`` = 524.288 in
  2x; ``dm_idx_global`` = the plan's global fine index, ``fine_dm_idx`` =
  the index within its coarse bucket), monitoring accumulators.
* :func:`c1_geometry` / :class:`MjdClock` / :func:`c1_batch_header` /
  :func:`make_c1_emit` -- the C1 path of :func:`main`: rows in radians
  (``1 / (N cell_lambda)`` per pixel, the 1x convention), one batch per
  cycle with the 2x header (``event_specnum_start = k * 384`` external
  samples, ``sample_period_specnum`` = 8 x 65.536 us, ``mjd_start`` from
  the capture arm record), shipped by the production ``C1TcpEmitter``;
  cutouts by :class:`~dsart.x2.search.cutout_service.CutoutListener` on
  ``cutout_base_port + gpu_half``.
* :func:`mon_payloads` -- the keys WS-D's D11 monitoring reads:
  ``/mon/search_rt/<cn>/compute/<g>``, ``/mon/search_rt/<cn>/dm/<d>``,
  ``/mon/search_rt/<cn>/cutout/<g>`` (published at <= 2 Hz from the host
  thread; GPU duty from the engine's own CUDA-event timing, no extra sync).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .geometry import CYCLE_NATIVE, SEARCH_NODES

__all__ = ["CycleScheduler", "NoCycleYet", "OwnerService", "mon_payloads", "c1_rows_for_cycle",
           "c1_geometry", "MjdClock", "MjdEpochError", "c1_batch_header", "make_c1_emit", "main"]

_LOG = logging.getLogger("dsart.x2.search.service")
NATIVE_S = 32.768e-6
CYCLE_S = CYCLE_NATIVE * NATIVE_S


class NoCycleYet(TimeoutError):
    """No complete cycle within ``next_cycle``'s timeout (no data: before the arm, or an
    outage upstream); raised before any processing, so the caller may simply wait on."""


class CycleScheduler:
    def __init__(self, plan, ring, *, max_lag_cycles: int = 8) -> None:
        self.plan = plan
        self.ring = ring
        self.max_lag = int(max_lag_cycles)
        self.k: Optional[int] = None
        self.n_seeks = 0
        self.n_skipped = 0
        self.n_startup_skipped = 0
        self._armed = True                      # disarmed while the first cycle runs

    def _ready_cycle(self, pos_name: str = "write_pos") -> Optional[int]:
        """Newest cycle whose rows are released on every owned DM."""
        wp = getattr(self.ring, pos_name, None)
        if wp is None:
            return None
        return int(min((wp(s.d) - s.max_neg) // s.hop for s in self.plan.dms)) - 1

    def next_cycle(self, *, poll_s: float = 0.002, timeout_s: Optional[float] = None) -> int:
        t_end = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            ready = self._ready_cycle()
            if ready is None:                                    # no frontier API: block in K0
                self.k = 0 if self.k is None else self.k + 1
                return self.k
            if self.k is None:
                # start at the newest cycle the fan-in QUORUM has (the ring
                # starts empty and its flows start one by one: the deadline
                # edge would hand out a cycle the data has not reached yet)
                q = self._ready_cycle("quorum_pos")
                first = ready if q is None else min(ready, q)
                if first >= 0:
                    self.k, self._armed = first, False
                    return self.k
            else:
                nxt = self.k + 1
                if not self._armed:
                    # the lag clock arms after the first cycle (kernel first
                    # use, history imaging): its backlog is a start-up
                    # re-anchor to the frontier, not a lag-recovery seek
                    self._armed = True
                    if ready - nxt > self.max_lag:
                        self.n_startup_skipped += ready - nxt
                        _LOG.info("start-up re-anchor: cycle %d -> %d", nxt, ready)
                        nxt = ready
                if ready - nxt > self.max_lag:
                    self.n_seeks += 1
                    self.n_skipped += ready - nxt
                    _LOG.warning("lag-recovery seek: cycle %d -> %d", nxt, ready)
                    nxt = ready
                if nxt <= ready:
                    self.k = nxt
                    return self.k
            if t_end is not None and time.monotonic() >= t_end:
                raise NoCycleYet("no complete cycle")
            time.sleep(poll_s)

    def lag_ms(self, k: int) -> Dict[int, float]:
        wp = getattr(self.ring, "write_pos", None)
        if wp is None:
            return {}
        return {s.d: max(0.0, (wp(s.d) - (k + 1) * s.hop) * s.t_native * 32.768e-3)
                for s in self.plan.dms}


def c1_rows_for_cycle(cands, *, geom: dict):
    """2x C1 rows (``services.c1_emit.C1CandidateRow``) for one cycle."""
    from ...services.c1_emit import C1CandidateRow, centred_pix_offset

    N = int(geom["n_grid"])
    rows = []
    for c in cands:
        rows.append(C1CandidateRow(
            snr=float(c.snr),
            l_rad=float(geom.get("l0_rad", 0.0)) + float(geom["cell_l_rad"])
            * centred_pix_offset(float(c.m), N),
            m_rad=float(geom.get("m0_rad", 0.0)) + float(geom["cell_m_rad"])
            * centred_pix_offset(float(c.l), N),
            l_pix=int(c.l), m_pix=int(c.m), dm_pc_cc=float(c.dm),
            dm_idx_global=int(c.fine_global), fine_dm_idx=int(c.fdm_local),
            event_specnum=int(c.event_specnum), width_samples=int(c.width_ext),
            kernel_id=f"unit:d1:{c.filt}", flags=int(c.flags)))
    return rows


def c1_geometry(cell_lambda: float, n_grid: int) -> dict:
    """C1 / cutout pixel geometry: ``cell_{l,m}_rad = 1 / (N * cell_lambda)``
    -- the 1x ``image_cell_rad`` (``services/search_compute.py``) from the
    value map's own ``cell_lambda`` (the corr grids with the same one);
    ``l0 = m0 = 0``: offsets from the phase centre at pixel ``N // 2``
    (``centred_pix_offset``)."""
    if not (np.isfinite(cell_lambda) and cell_lambda > 0):
        raise ValueError(f"cell_lambda={cell_lambda} must be > 0")
    cell = 1.0 / (float(n_grid) * float(cell_lambda))
    return dict(n_grid=int(n_grid), cell_l_rad=cell, cell_m_rad=cell, l0_rad=0.0, m0_rad=0.0)


class MjdEpochError(RuntimeError):
    """The capture epoch needed to label candidates is unavailable or inconsistent."""


def _read_arm_record():
    """``(ArmRecord | None, reason)`` from ``/mon/snap/1/armed_mjd`` (``prod_inputs``)."""
    from ..prod_inputs import read_arm_record
    return read_arm_record()


class MjdClock:
    """MJD of native sample ``n`` = ``mjd_at_native_0 + n * 32.768 us``.

    Native indices are absolute (SPEC §9: 2 x SNAP specnum; corr_fast ``--x2-epoch
    capture`` starts the first page after an arm at specnum ``u`` at native
    ``prod_inputs.capture_native0(u)``). The anchor is resolved once, at the first query:

    * an explicit ``mjd_at_native_0`` (> 0: replays, tests), else
    * the capture arm record (``read_arm`` -> ``(ArmRecord, reason)``):
      ``armed_mjd - capture_native0(arm_seq) x 32.768 us`` -- the record 1x labels its
      candidates from (``SearchComputeService._read_capture_anchor_mjd``), accepted when
      the first query's data time lags the wall clock by -2..120 s (the 1x rule);
    * otherwise ``strict`` (production) raises :class:`MjdEpochError`; non-strict
      (development) falls back to a wall latch at that query (late by the pipeline
      latency, logged).
    """

    def __init__(self, mjd_at_native_0: float = 0.0, *, read_arm=_read_arm_record,
                 wall=time.time, strict: bool = True) -> None:
        self.anchor: Optional[float] = float(mjd_at_native_0) if mjd_at_native_0 > 0 else None
        self.source = "explicit" if self.anchor is not None else ""
        self._read_arm = read_arm
        self._wall = wall
        self.strict = bool(strict)
        self.record = None

    def _latch(self, n_now: int) -> float:
        wall_anchor = 40587.0 + self._wall() / 86400.0 - float(n_now) * NATIVE_S / 86400.0
        rec, why = self._read_arm() if self._read_arm is not None else (None, "no arm reader")
        if rec is not None:
            anchor = float(rec.mjd_at_native_0)
            lag_s = (wall_anchor - anchor) * 86400.0
            if -2.0 <= lag_s <= 120.0:
                self.anchor, self.source, self.record = anchor, f"arm:{rec.source}", rec
                _LOG.info("x2 mjd_at_native_0 = %.9f from the arm record (armed_mjd %.9f, "
                          "arm_seq %d, %s); native %d lags the wall clock by %.2f s",
                          anchor, rec.armed_mjd, rec.arm_seq, rec.source or "?", n_now, lag_s)
                return self.anchor
            why = (f"the arm record (armed_mjd {rec.armed_mjd:.9f}, arm_seq {rec.arm_seq}) puts "
                   f"native sample {n_now} {lag_s:+.1f} s behind the wall clock (want -2..120 s): "
                   "not this data's arm, or corr_fast is not on --x2-epoch capture")
        if self.strict:
            _LOG.error("x2 MJD epoch unavailable: %s", why)
            raise MjdEpochError(f"x2 MJD epoch unavailable: {why}")
        _LOG.warning("x2 MJD epoch: %s; using the wall latch (labels late by the pipeline "
                     "latency)", why)
        self.anchor, self.source = wall_anchor, "wall"
        _LOG.info("x2 mjd_at_native_0 = %.9f (%s)", self.anchor, self.source)
        return self.anchor

    def mjd_of_native(self, n: int, *, latch_n: Optional[int] = None) -> float:
        """``latch_n``: the newest native sample in hand (the wall latch's
        "now"), default ``n``."""
        anchor = self.anchor if self.anchor is not None else self._latch(
            int(n if latch_n is None else latch_n))
        return float(anchor + float(n) * NATIVE_S / 86400.0)


def c1_batch_header(k: int, *, plan, n_rows: int, search_node_id: int, gpu_half: int,
                    n_grid: int, clock: MjdClock):
    """The C1 batch header of cycle ``k`` of one half (``coinc.wire``): the
    cycle's first external sample, its MJD, ``sample_period_specnum`` in
    65.536-us spec-nums (8 in 2x, 16 in 1x-compat), the half's fine count."""
    from ...coinc import wire
    from ...common.constants import SPECNUM_PERIOD_US

    ext = int(plan.ext_native)
    e0 = int(k) * (CYCLE_NATIVE // ext)
    period_us = float(plan.ext_period_us)
    return wire.build_header(
        cube_id=int(k), event_specnum_start=e0,
        mjd_start=clock.mjd_of_native(e0 * ext, latch_n=(int(k) + 1) * CYCLE_NATIVE),
        sample_period_specnum=int(round(period_us / SPECNUM_PERIOD_US)),
        sample_period_us=period_us, n_grid=int(n_grid),
        n_fdm_in_cube=int(sum(s.n_fdm for s in plan.dms)),
        search_node_id=int(search_node_id), gpu_half=int(gpu_half), n_candidates=int(n_rows))


def make_c1_emit(sender, *, plan, search_node_id: int, gpu_half: int, n_grid: int,
                 clock: MjdClock) -> Callable:
    """:class:`OwnerService` ``emit`` callback: one batch per cycle (also an
    empty one -- C2's per-half liveness) through ``sender.submit(header,
    candidates, rows=...)`` (the production ``services.c1_emit.C1TcpEmitter``)."""
    def emit(k: int, kept, rows) -> bool:
        hdr = c1_batch_header(k, plan=plan, n_rows=len(rows), search_node_id=search_node_id,
                              gpu_half=gpu_half, n_grid=n_grid, clock=clock)
        return bool(sender.submit(hdr, kept, rows=rows))
    return emit


@dataclass
class _Mon:
    cycles: int = 0
    wall_ms: deque = field(default_factory=lambda: deque(maxlen=64))
    gpu_ms: deque = field(default_factory=lambda: deque(maxlen=64))
    lag_ms: Dict[int, float] = field(default_factory=dict)
    cubes_total: Dict[int, int] = field(default_factory=dict)
    fanin: Dict[int, int] = field(default_factory=dict)
    cands: Dict[int, deque] = field(default_factory=dict)
    t_first: Optional[float] = None
    c1_rows: int = 0
    c1_batches: int = 0


class OwnerService:
    def __init__(self, engine, *, search_node_id: int, gpu_half: int, c1_pool=None,
                 emit: Optional[Callable] = None, cutout=None, geom: Optional[dict] = None,
                 max_lag_cycles: int = 8) -> None:
        self.eng = engine
        self.sid = int(search_node_id)
        self.g = int(gpu_half)
        self.sched = CycleScheduler(engine.plan, engine.ring, max_lag_cycles=max_lag_cycles)
        self.c1 = c1_pool
        self.emit = emit
        self.cutout = cutout
        self.geom = dict(geom or dict(n_grid=engine.cfg.n_grid, cell_l_rad=1.0, cell_m_rad=1.0))
        self.mon = _Mon()
        self._stop = threading.Event()

    def step(self, *, timeout_s: Optional[float] = None):
        k = self.sched.next_cycle(timeout_s=timeout_s)
        t_a = time.monotonic()
        res = self.eng.process_cycle(k)
        gms = self.eng._gpu.pop_gpu_ms() if self.eng._gpu is not None else {}
        m = self.mon
        m.cycles += 1
        m.t_first = m.t_first or t_a
        m.wall_ms.append(1e3 * (time.monotonic() - t_a))
        m.gpu_ms.append(float(sum(gms.values())))
        m.lag_ms = self.sched.lag_ms(k)
        for d, r in res.per_dm.items():
            m.cubes_total[d] = m.cubes_total.get(d, 0) + 1
            st = self.eng.states[d]
            rows = (r.t0 + np.arange(st.spec.hop))[:, None] - st.spec.shifts.min(axis=0)[None]
            ok = st.streams.valid[np.arange(st.spec.n_stream)[None, :],
                                  st.streams.slots(rows)].all(axis=0)
            m.fanin[d] = int(ok.sum())
            m.cands.setdefault(d, deque(maxlen=64)).append(len(r.candidates))
        kept, legacy = res.candidates, []
        if self.c1 is not None:
            kept, legacy = self.c1.filter(res.candidates, now_s=time.monotonic(),
                                          search_node_id=self.sid, gpu_half=self.g)
        if self.emit is not None:
            self.emit(k, kept, c1_rows_for_cycle(kept, geom=self.geom))
            m.c1_batches += 1
            m.c1_rows += len(kept)
        return k, res, kept

    def run(self) -> None:
        while not self._stop.is_set():
            self.step(timeout_s=5.0)

    def stop(self) -> None:
        self._stop.set()


def _ring_fanin(ring, d: int) -> dict:
    """Ring v4 fan-in of DM ``d`` for its D11 key (empty for other rings)."""
    fan = getattr(ring, "fan_in", None)
    if fan is None:
        return {}
    f = fan(d)
    w = f["windows"]
    return dict(fanin_last_status=f["last_status"], fanin_last_n_final=int(f["last_n_final"]),
                fanin_last_wait_ms=float(f["last_waited_ms"]),
                fanin_windows_complete_total=int(w.get("complete", 0)),
                fanin_windows_partial_total=int(w.get("fan_in", 0)),
                fanin_windows_deadline_total=int(w.get("deadline", 0)),
                fanin_rows_missing_total=int(f["rows_missing"]),
                fanin_rows_lapped_total=int(f["rows_lapped"]))


def mon_payloads(svc: OwnerService, *, opoint: str) -> Dict[str, dict]:
    """``{etcd key: payload}`` for WS-D's D11 keys (``cn`` = search_node_id)."""
    eng, m = svc.eng, svc.mon
    plan = eng.plan
    now = time.time()
    wall = np.array(m.wall_ms) if m.wall_ms else np.zeros(1)
    gpu = np.array(m.gpu_ms) if m.gpu_ms else np.zeros(1)
    dt = max(1e-9, time.monotonic() - (m.t_first or time.monotonic()))
    n_dm = len(plan.dms)
    out = {}
    base = f"/mon/search_rt/{svc.sid}"
    out[f"{base}/compute/{svc.g}"] = dict(
        search_node_id=svc.sid, gpu_half=svc.g, opoint=opoint,
        coarse_dms=",".join(str(s.d) for s in plan.dms), n_dm_owned=n_dm,
        gpu_duty=float(gpu.mean() / (1e3 * CYCLE_S)),
        cubes_per_s=float(sum(m.cubes_total.values()) / dt) if m.t_first else 0.0,
        cubes_per_s_nominal=float(n_dm / CYCLE_S),
        cycle_ms_p50=float(np.percentile(wall, 50)), cycle_ms_p99=float(np.percentile(wall, 99)),
        host_ms_p99=float(np.percentile(np.maximum(wall - gpu, 0.0), 99)),
        ring_lag_ms_max=float(max(m.lag_ms.values(), default=0.0)),
        lag_recovery_seeks_total=svc.sched.n_seeks, cycles_skipped_total=svc.sched.n_skipped,
        c1_batches_sent_total=m.c1_batches, c1_rows_total=m.c1_rows,
        c1_cands_dropped_color_total=(svc.c1.counters["dropped_color"] if svc.c1 else 0),
        c1_cands_dropped_dmfloor_total=(svc.c1.counters["dropped_dmfloor"] if svc.c1 else 0),
        c1_cands_dropped_meter_total=(svc.c1.counters["dropped_meter"] if svc.c1 else 0),
        pattern_build_ok=int(all(v.get("ok") for v in eng.build_status.values())),
        pattern_id=str(next(iter(eng.build_status.values()), {}).get("pattern_id", "")),
        plan_id=plan.plan_id, ts_wall_unix=now)
    for s in plan.dms:
        st = eng.states[s.d]
        sk = st.layer2._s_k.detach().cpu().numpy()
        sig = st.sigma_prev if st.sigma_prev is not None else np.zeros(1)
        out[f"{base}/dm/{s.d}"] = dict(
            d=s.d, tier=s.tier, gpu_half=svc.g, n_fdm=s.n_fdm,
            cubes_total=int(m.cubes_total.get(s.d, 0)),
            ring_lag_ms=float(m.lag_ms.get(s.d, 0.0)),
            fanin_streams_present=int(m.fanin.get(s.d, 0)), fanin_streams_expected=s.n_stream,
            **_ring_fanin(eng.ring, s.d),
            s_k_median=float(np.median(sk)), s_k_max=float(np.max(sk)),
            layer1_sigma_median=float(np.median(sig)), layer1_sigma_min=float(np.min(sig)),
            c1_cands_per_cycle_mean=float(np.mean(m.cands.get(s.d, [0]))), ts_wall_unix=now)
    cut = dict(search_node_id=svc.sid, gpu_half=svc.g, ts_wall_unix=now)
    wp = getattr(eng.ring, "write_pos", None)
    if wp is not None:
        cut["raw_ring_newest_ext"] = int(min((wp(s.d) - s.max_neg) * plan.ext_factor(s) - 1
                                             for s in plan.dms))
    if svc.cutout is not None:
        c = svc.cutout.counters
        cut.update(cutout_requests_total=c["requests"], cutout_accepted_total=c["accepted"],
                   cutout_done_total=c["done"], cutout_partial_total=c["partial"],
                   cutout_failed_total=c["failed"],
                   cutout_failed_queue_full_total=c["failed_queue_full"],
                   cutout_failed_plan_mismatch_total=c["failed_plan_mismatch"],
                   c2_trigger_v1_ignored=c["c2_trigger_v1_ignored"],
                   cutout_queue_depth=len(svc.cutout._q),
                   cutout_last_latency_ms=getattr(svc.cutout, "last_latency_ms", 0))
    out[f"{base}/cutout/{svc.g}"] = cut
    return out


class MonThread(threading.Thread):
    """Publishes :func:`mon_payloads` every ``interval_s`` (>= 0.5 s)."""

    def __init__(self, svc: OwnerService, *, opoint: str, put: Callable[[str, dict], None],
                 interval_s: float = 2.0) -> None:
        super().__init__(daemon=True, name="x2-mon")
        self.svc, self.opoint, self.put = svc, opoint, put
        self.interval = max(0.5, float(interval_s))
        self.stop_ev = threading.Event()

    def run(self) -> None:
        while not self.stop_ev.wait(self.interval):
            try:
                for k, v in mon_payloads(self.svc, opoint=self.opoint).items():
                    self.put(k, v)
            except Exception:                                    # noqa: BLE001
                _LOG.warning("x2 mon publish failed", exc_info=True)


def _etcd_put() -> Callable[[str, dict], None]:
    store = None

    def put(key: str, payload: dict) -> None:
        nonlocal store
        if store is None:
            from dsautils.dsa_store import DsaStore
            store = DsaStore()
        store.put_dict(key, payload)
    return put


def main(argv: Optional[List[str]] = None) -> int:
    """2x search half. Arguments mirror the 1x service where they apply."""
    p = argparse.ArgumentParser(description="dsart 2x search half (owner engine)")
    p.add_argument("--opoint", required=True)
    p.add_argument("--plan-path", required=True, help="TieredDmPlan npz (SPEC §2)")
    p.add_argument("--plan-id", default=None,
                   help="refuse a plan whose plan_id differs (the set's ${plan_id}, SPEC §10)")
    p.add_argument("--t-int-search-us", type=float, default=None,
                   help="must equal the op-point's T_ext (SPEC §10 explicit time units)")
    p.add_argument("--chan-sum-factor", type=int, default=None,
                   help="must equal the op-point's chan_sum_factor (SPEC §10)")
    p.add_argument("--search-node-id", type=int, required=True, help="cn: 1, 2, 9, 13")
    p.add_argument("--gpu-half", type=int, required=True)
    p.add_argument("--shm-name", required=True, help="ring v4 (WS-T)")
    p.add_argument("--antpos-path", required=True,
                   help="cal blob (antennas.out) or an antpos npz (value map, SPEC §3)")
    p.add_argument("--obs-dec-deg", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fft-mode", default="fused", choices=("fused", "warp", "pruned", "irfft2"))
    p.add_argument("--bank", default="b3x", choices=("b3x", "box"))
    p.add_argument("--c1-host", default="", help="C2 host (C1 batches, cutout status); '' = off")
    p.add_argument("--c1-port", type=int, default=11500)
    p.add_argument("--c1-queue-depth", type=int, default=16)
    p.add_argument("--mjd-at-native-0", type=float, default=0.0,
                   help="MJD of native sample 0 (replays); 0 = from the capture arm record")
    p.add_argument("--mjd-epoch", default="capture", choices=("capture", "capture-or-wall"),
                   help="capture: the arm record or refuse to label (production); "
                        "capture-or-wall: fall back to a wall latch (development)")
    p.add_argument("--cutout-bind-host", default="", help="'' = no cutout listener")
    p.add_argument("--cutout-base-port", type=int, default=11227)
    p.add_argument("--cutout-status-port", type=int, default=11501)
    p.add_argument("--cutout-root", default="/home/ubuntu/data/c2/cutouts")
    p.add_argument("--upload-dest-host", default="", help="cutout/raw uploads (rsync); '' = off")
    p.add_argument("--upload-dest-root", default="/dataz/dsa110/candidates")
    p.add_argument("--raw-bwlimit-kbps", type=int, default=50_000)
    # age/size backstop over --cutout-root (today's c1.retention over c1.dump_root; the
    # sweeper sweeps the x2 cutouts / raw_s product dirs per half)
    p.add_argument("--cutout-retention-max-age-h", type=float, default=96.0)
    p.add_argument("--cutout-retention-max-total-gb", type=float, default=150.0,
                   help="0 = no retention sweeper")
    p.add_argument("--cutout-retention-low-water-gb", type=float, default=120.0)
    p.add_argument("--mon-interval-s", type=float, default=2.0)
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--ring-access", default="pread", choices=("pread", "mmap"),
                   help="ring v4 read path (SPEC §6; pread maps nothing -- the n13 incident)")
    p.add_argument("--ring-attach-timeout-s", type=float, default=60.0)
    p.add_argument("--k3b", default="reg", choices=("v1", "fused", "cuda", "reg"))   # G0 timing pick
    p.add_argument("--k1", default="staged", choices=("triton", "staged"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)                  # strict: a stray (e.g. 1x) argument fails loudly
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO))
    from ...x2.opoint import opoint_from_id
    from ..prod_inputs import load_plan, node_index_of_cn
    from .c1_cycle import C1CycleConfig, C1CyclePool
    from .engine import EngineConfig, OwnerEngine
    from .geometry import owner_plan_from_tiered
    from .valuemap import dm_inputs_from_value_map

    op = opoint_from_id(a.opoint)
    if a.t_int_search_us is not None and abs(a.t_int_search_us - op.t_ext_us) > 1e-6:
        p.error(f"--t-int-search-us {a.t_int_search_us} != {op.opoint_id} T_ext {op.t_ext_us}")
    if a.chan_sum_factor is not None and a.chan_sum_factor != op.chan_sum_factor:
        p.error(f"--chan-sum-factor {a.chan_sum_factor} != {op.opoint_id} "
                f"{op.chan_sum_factor}")
    if a.gpu_half not in (0, 1):
        p.error(f"--gpu-half {a.gpu_half}: 0 or 1")
    half = node_index_of_cn(a.search_node_id) * 2 + int(a.gpu_half)
    node = SEARCH_NODES[half // 2]
    plan = load_plan(a.plan_path, op, a.plan_id)
    oplan = owner_plan_from_tiered(plan, half=half)
    vm = _value_map_for(op, a.antpos_path, a.obs_dec_deg)
    s_main = {8: 0.3534, 10: 0.1932, 12: 0.1055}[int(op.bits_main)]
    s_heavy = {2: 0.0773, 4: 0.0547}[int(op.n_sub)]
    inp = dm_inputs_from_value_map(vm, s_main=s_main, s_heavy=s_heavy, bits_main=op.bits_main)
    from .ring import RingV4Source                           # WS-T reader, SPEC §6
    ring = RingV4Source.open(a.shm_name, owned=[s.d for s in oplan.dms], access=a.ring_access,
                             attach_timeout_s=a.ring_attach_timeout_s)
    inputs = {s.d: inp for s in oplan.dms}
    ring.validate(oplan, inputs, vmap=vm)                    # fail loudly (SPEC §7)
    cfg = EngineConfig(backend="gpu", fft_mode=a.fft_mode,
                       families=("b3x",) if a.bank == "b3x" else ("b",), wait_timeout_s=10.0,
                       k3b_impl=a.k3b, k1_impl=a.k1)
    eng = OwnerEngine(oplan, inputs, ring, cfg)
    _LOG.info("x2 half %d (%s g%d) status: %s", half, node, a.gpu_half, eng.status())
    pool = C1CyclePool(C1CycleConfig(chan_sum_factor=int(op.chan_sum_factor),
                                     ext_period_us=float(op.t_ext_us)))
    geom = c1_geometry(vm.cell_lambda, cfg.n_grid)
    clock = MjdClock(a.mjd_at_native_0, strict=a.mjd_epoch == "capture")
    emitter = emit = None
    if a.c1_host:
        from ...services.c1_emit import C1EmitConfig, C1TcpEmitter
        emitter = C1TcpEmitter(config=C1EmitConfig(
            host=a.c1_host, port=int(a.c1_port), search_node_id=int(a.search_node_id),
            gpu_half=int(a.gpu_half), queue_depth=int(a.c1_queue_depth)))
        emitter.start()
        emit = make_c1_emit(emitter, plan=oplan, search_node_id=a.search_node_id,
                            gpu_half=a.gpu_half, n_grid=cfg.n_grid, clock=clock)
    producer = listener = retention = None
    if a.cutout_bind_host:
        from pathlib import Path

        from .cutout_service import CutoutListener, CutoutProducer, make_uploaders
        ext = int(oplan.ext_native)
        producer = CutoutProducer(eng, search_node_id=a.search_node_id, gpu_half=a.gpu_half,
                                  out_root=Path(a.cutout_root), plan=plan,
                                  plan_id=int(plan.plan_id), geometry=geom,
                                  clock_mjd=lambda t0e: clock.mjd_of_native(int(t0e) * ext))
        if a.upload_dest_host:
            producer.uploaders = make_uploaders(sid=a.search_node_id, g=a.gpu_half,
                                                dest_host=a.upload_dest_host,
                                                dest_root=a.upload_dest_root,
                                                raw_bwlimit_kbps=a.raw_bwlimit_kbps)
            for up in producer.uploaders:
                up.start()
        listener = CutoutListener(producer, bind_host=a.cutout_bind_host,
                                  base_port=a.cutout_base_port, c2_host=a.c1_host or "h23",
                                  status_port=a.cutout_status_port, event_root=Path(a.cutout_root))
        threading.Thread(target=listener.run, daemon=True, name="x2-cutout").start()
        if a.cutout_retention_max_total_gb > 0:
            from ...dump.cube_retention import CubeRetentionConfig, CubeRetentionSweeper
            Path(a.cutout_root).mkdir(parents=True, exist_ok=True)
            retention = CubeRetentionSweeper(CubeRetentionConfig(
                dump_root=Path(a.cutout_root), max_age_h=a.cutout_retention_max_age_h,
                max_total_gb=a.cutout_retention_max_total_gb,
                low_water_gb=a.cutout_retention_low_water_gb),
                thread_name=f"cutout-retention-g{a.gpu_half}")
            retention.start()
    svc = OwnerService(eng, search_node_id=a.search_node_id, gpu_half=a.gpu_half, c1_pool=pool,
                       emit=emit, cutout=producer, geom=geom)
    mon = MonThread(svc, opoint=a.opoint, put=_etcd_put(), interval_s=a.mon_interval_s)
    mon.start()
    n = 0
    idle_s = 0.0
    try:
        while a.max_cycles <= 0 or n < a.max_cycles:
            try:
                svc.step(timeout_s=30.0)
            except NoCycleYet:
                # no data yet (dsart_rt starts the halves before the arm) or a corr /
                # search_rx outage: keep waiting, the scheduler re-anchors on the data
                idle_s += 30.0
                _LOG.warning("x2 half %d: no complete cycle for %.0f s (not armed yet, or no "
                             "frames reach the ring); waiting", half, idle_s)
                continue
            idle_s = 0.0
            n += 1
    finally:
        mon.stop_ev.set()
        if listener is not None:
            listener.stop()
        if emitter is not None:
            emitter.stop()
        if retention is not None:
            retention.stop()
    return 0


def _value_map_for(op, antpos_path: str, dec_deg: float):
    """The value map corr_fast and search_rx build for the same inputs
    (``dsart.x2.prod_inputs``: cal blob or antpos npz, the canonical DEC)."""
    from ..prod_inputs import value_map_for
    return value_map_for(op, antpos_path, dec_deg)

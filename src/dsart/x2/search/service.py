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

__all__ = ["CycleScheduler", "OwnerService", "mon_payloads", "c1_rows_for_cycle", "main"]

_LOG = logging.getLogger("dsart.x2.search.service")
CYCLE_S = CYCLE_NATIVE * 32.768e-6


class CycleScheduler:
    def __init__(self, plan, ring, *, max_lag_cycles: int = 8) -> None:
        self.plan = plan
        self.ring = ring
        self.max_lag = int(max_lag_cycles)
        self.k: Optional[int] = None
        self.n_seeks = 0
        self.n_skipped = 0

    def _ready_cycle(self) -> Optional[int]:
        """Newest cycle whose rows are complete on every owned DM."""
        wp = getattr(self.ring, "write_pos", None)
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
                if ready >= 0:
                    self.k = ready
                    return self.k
            else:
                nxt = self.k + 1
                if ready - nxt > self.max_lag:
                    self.n_seeks += 1
                    self.n_skipped += ready - nxt
                    _LOG.warning("lag-recovery seek: cycle %d -> %d", nxt, ready)
                    nxt = ready
                if nxt <= ready:
                    self.k = nxt
                    return self.k
            if t_end is not None and time.monotonic() >= t_end:
                raise TimeoutError("no complete cycle")
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
    p.add_argument("--search-node-id", type=int, required=True, help="cn: 1, 2, 9, 13")
    p.add_argument("--gpu-half", type=int, required=True)
    p.add_argument("--shm-name", required=True, help="ring v4 (WS-T)")
    p.add_argument("--antpos-path", required=True,
                   help="cal blob (antennas.out) or an antpos npz (value map, SPEC §3)")
    p.add_argument("--obs-dec-deg", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fft-mode", default="warp", choices=("warp", "pruned", "irfft2"))
    p.add_argument("--bank", default="b3x", choices=("b3x", "box"))
    p.add_argument("--c1-host", default="")
    p.add_argument("--c1-port", type=int, default=11500)
    p.add_argument("--cutout-bind-host", default="")
    p.add_argument("--cutout-base-port", type=int, default=11227)
    p.add_argument("--cutout-root", default="/home/ubuntu/data/c2/cutouts")
    p.add_argument("--mon-interval-s", type=float, default=2.0)
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--ring-access", default="pread", choices=("pread", "mmap"),
                   help="ring v4 read path (SPEC §6; pread maps nothing -- the n13 incident)")
    p.add_argument("--ring-attach-timeout-s", type=float, default=60.0)
    p.add_argument("--k3b", default="v1", choices=("v1", "fused", "cuda", "reg"))   # CP1 picks
    p.add_argument("--k1", default="triton", choices=("triton", "staged"))
    p.add_argument("--log-level", default="INFO")
    a, _unknown = p.parse_known_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO))
    from ...x2.opoint import opoint_from_id
    from ...x2.plan import TieredDmPlan
    from .c1_cycle import C1CycleConfig, C1CyclePool
    from .engine import EngineConfig, OwnerEngine
    from .geometry import owner_plan_from_tiered
    from .valuemap import dm_inputs_from_value_map

    op = opoint_from_id(a.opoint)
    node = {1: "n01", 2: "n02", 9: "n09", 13: "n13"}[int(a.search_node_id)]
    half = SEARCH_NODES.index(node) * 2 + int(a.gpu_half)
    plan = TieredDmPlan.from_npz(a.plan_path, verify=True)
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
    svc = OwnerService(eng, search_node_id=a.search_node_id, gpu_half=a.gpu_half, c1_pool=pool)
    mon = MonThread(svc, opoint=a.opoint, put=_etcd_put(), interval_s=a.mon_interval_s)
    mon.start()
    n = 0
    while a.max_cycles <= 0 or n < a.max_cycles:
        svc.step(timeout_s=30.0)
        n += 1
    mon.stop_ev.set()
    return 0


def _value_map_for(op, antpos_path: str, dec_deg: float):
    from pathlib import Path

    from ...grid.baseline_grid import build_value_map_for_opoint
    from ...grid.sparsity_pattern import core_baseline_mask_from_station_numbers

    if antpos_path.endswith(".npz"):                            # e.g. the WS-P fixture
        z = np.load(antpos_path)
        e, n = z["antpos_e"], z["antpos_n"]
        mask = core_baseline_mask_from_station_numbers(z["station"])
    else:                                                        # production cal blob
        from ...services.corr_fast_integration import load_antpos_from_cal_blob
        e, n, mask = load_antpos_from_cal_blob(Path(antpos_path))
    return build_value_map_for_opoint(op, e, n, is_core_baseline_mask=mask,
                                      dec_deg=float(dec_deg))

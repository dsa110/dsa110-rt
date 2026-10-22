"""2x search half as a service (x2/search/service.py), CPU path.

* the cycle scheduler aligns to absolute cycles, runs only complete cycles
  and seeks forward past ``max_lag_cycles`` (counted);
* one pooled C1 batch per cycle; rows carry ``dm_idx_global`` = the plan's
  global fine index and ``fine_dm_idx`` = the index within its bucket;
* WS-D's D11 monitoring keys and fields;
* ``search_compute --opoint 2x-*`` dispatches to the x2 service, any other
  invocation reaches the 1x parser unchanged.
"""
from __future__ import annotations

import numpy as np

from dsart.x2.search.c1_cycle import C1CycleConfig, C1CyclePool
from dsart.x2.search.codec import RowFormat
from dsart.x2.search.engine import EngineConfig, OwnerEngine
from dsart.x2.search.geometry import owner_plan_from_tiers
from dsart.x2.search.ring import MemoryRing
from dsart.x2.search.service import CycleScheduler, OwnerService, mon_payloads
from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

N = 32
V = 20


def _svc(bursts=(), last_cycle=3):
    fa = np.array([50.0, 53.0, 56.0])
    fb = np.array([100.0, 106.0])
    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0]), fine=fa, f2c=np.zeros(3, int)),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=fb, f2c=np.zeros(2, int))]
    plan = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs = {}, {}
    for s in plan.dms:
        cells[s.d] = random_value_cells(16, V, N, radius=6, seed=s.d + 9, n_unused=2)
        inputs[s.d] = make_dm_inputs(cells[s.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                     step_heavy=0.1)
    ring = MemoryRing({s.d: fmt for s in plan.dms}, 16, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells=cells, formats={s.d: fmt for s in plan.dms}, n_grid=N,
                       bursts=list(bursts))
    for s in plan.dms:
        sky.fill(ring, s, 0, (last_cycle + 1) * s.hop)
    eng = OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",), widths=(1, 2, 4),
                                                       fft_mode="pruned", n_tap_pixels=32,
                                                       layer2_n_burnin=0))
    sent = []
    svc = OwnerService(eng, search_node_id=9, gpu_half=1, c1_pool=C1CyclePool(C1CycleConfig()),
                       emit=lambda k, kept, rows: sent.append((k, kept, rows)),
                       geom=dict(n_grid=N, cell_l_rad=2e-4, cell_m_rad=2e-4), max_lag_cycles=1)
    return svc, sky, ring, sent


def test_scheduler_runs_complete_cycles_and_seeks():
    svc, sky, ring, sent = _svc(last_cycle=3)
    k, _res, _kept = svc.step(timeout_s=1.0)
    assert k == 3                                    # newest complete cycle (frontier = 4 hops)
    sch = CycleScheduler(svc.eng.plan, ring, max_lag_cycles=1)
    sch.k = 0
    assert sch.next_cycle(timeout_s=1.0) == 3 and sch.n_seeks == 1 and sch.n_skipped == 2


def test_one_pooled_c1_batch_per_cycle_with_global_indices():
    bursts = [Burst(d=0, f=2, t=3 * 384 + 150, l=11, m=21, amp=1.8)]
    svc, _sky, _ring, sent = _svc(bursts=bursts, last_cycle=3)
    k, res, kept = svc.step(timeout_s=1.0)
    assert len(sent) == 1 and sent[0][0] == k
    rows = sent[0][2]
    hit = [r for r in rows if (r.l_pix, r.m_pix) == (11, 21)]
    assert hit and hit[0].dm_idx_global == 2 and hit[0].fine_dm_idx == 2
    assert hit[0].event_specnum == 3 * 384 + 150 and hit[0].kernel_id.startswith("unit:d1:b")
    assert len(rows) <= 8


def test_mon_payload_keys_for_ws_d():
    svc, *_ = _svc()
    svc.step(timeout_s=1.0)
    p = mon_payloads(svc, opoint="2x-gridded-ns2-cint4")
    comp = p["/mon/search_rt/9/compute/1"]
    for f in ("gpu_duty", "cubes_per_s", "cubes_per_s_nominal", "coarse_dms", "opoint"):
        assert f in comp
    assert comp["coarse_dms"] == "0,1"
    for d in (0, 1):
        dm = p[f"/mon/search_rt/9/dm/{d}"]
        for f in ("cubes_total", "ring_lag_ms", "fanin_streams_present", "gpu_half", "tier"):
            assert f in dm
        assert dm["cubes_total"] == 1 and dm["fanin_streams_present"] == 16
    cut = p["/mon/search_rt/9/cutout/1"]
    assert cut["raw_ring_newest_ext"] == 4 * 384 - 1          # T_ext units, tier B x2 aligned


def test_search_compute_dispatches_2x_and_leaves_1x_alone(monkeypatch):
    import dsart.services.search_compute as sc
    import dsart.x2.search.service as xs

    seen = {}
    monkeypatch.setattr(xs, "main", lambda argv: seen.setdefault("x2", argv) and 7)
    assert sc.main(["--opoint", "2x-gridded-ns2-cint4", "--plan-path", "p"]) == 7
    assert seen["x2"][:2] == ["--opoint", "2x-gridded-ns2-cint4"]
    try:                                              # 1x: --opoint stripped, 1x parser runs
        sc.main(["--opoint", "1x"])
    except SystemExit as e:                           # required 1x args missing -> argparse exit
        assert e.code == 2
    assert "x2" in seen and len(seen) == 1

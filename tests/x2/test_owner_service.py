"""2x search half as a service (x2/search/service.py), CPU path.

* the cycle scheduler aligns to absolute cycles, runs only complete cycles
  and seeks forward past ``max_lag_cycles`` (counted);
* one pooled C1 batch per cycle; rows carry ``dm_idx_global`` = the plan's
  global fine index and ``fine_dm_idx`` = the index within its bucket;
* WS-D's D11 monitoring keys and fields;
* the C1 path of ``main()``: rows in radians (``1 / (N cell_lambda)`` per
  pixel from the centre), one batch per cycle with the 2x header through the
  production ``C1TcpEmitter`` (no socket: its queue), the MJD anchor rule;
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
from dsart.x2.search.service import (CycleScheduler, MjdClock, OwnerService, c1_batch_header,
                                    c1_geometry, c1_rows_for_cycle, make_c1_emit, mon_payloads)
from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

N = 32
V = 20


def _svc(bursts=(), last_cycle=3, emit=None):
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
                       emit=emit or (lambda k, kept, rows: sent.append((k, kept, rows))),
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
    assert seen["x2"] == ["--opoint", "2x-gridded-ns2-cint4", "--plan-path", "p"]
    seen.clear()                                      # the --opoint=<id> form too
    assert sc.main(["--plan-path", "p", "--opoint=2x-baselines-ns4-cint5"]) == 7
    assert seen["x2"] == ["--opoint", "2x-baselines-ns4-cint5", "--plan-path", "p"]
    for form in (["--opoint", "1x"], ["--opoint=1x"]):
        try:                                          # 1x: --opoint stripped, 1x parser runs
            sc.main(form)
        except SystemExit as e:                       # required 1x args missing -> argparse exit
            assert e.code == 2
    assert "x2" in seen and len(seen) == 1


def test_c1_rows_are_in_radians_from_the_value_map_cell():
    from types import SimpleNamespace as NS

    cl = 31.4922                                      # production tee45 cell_lambda (1x note)
    g = c1_geometry(cl, 256)
    cell = 1.0 / (256 * cl)
    assert g["cell_l_rad"] == g["cell_m_rad"] == cell and g["l0_rad"] == g["m0_rad"] == 0.0
    assert abs(cell - 1.24039e-4) < 1e-9
    c = NS(snr=12.0, l=128 - 2, m=128 + 1, dm=300.0, fine_global=17, fdm_local=3,
           event_specnum=1000, width_ext=4, filt="q4", flags=0)
    r = c1_rows_for_cycle([c], geom=g)[0]
    # row axis (Candidate.l) = sky m, column axis (Candidate.m) = sky l; centre pixel N // 2
    assert r.l_rad == cell and r.m_rad == -2 * cell and (r.l_pix, r.m_pix) == (126, 129)


def test_mjd_clock_anchor_rule():
    import pytest

    from dsart.x2.prod_inputs import ArmRecord, capture_native0
    from dsart.x2.search.service import MjdEpochError

    arm_seq = 17_179_869_184 + 16 * 1234                # absolute SNAP specnum of the arm
    n0 = capture_native0(arm_seq)                     # = 2 x arm_seq (a multiple of 16)
    n = n0 + 1_000_000                                # absolute native index of the data in hand
    wall_s = 1.8e9
    wall_mjd = 40587.0 + wall_s / 86400.0
    armed = wall_mjd - (1_000_000 * 32.768e-6 + 5.0) / 86400.0  # 5 s pipeline latency
    rec = ArmRecord(armed_mjd=armed, arm_seq=arm_seq, source="pps_epoch")
    c = MjdClock(read_arm=lambda: (rec, ""), wall=lambda: wall_s)
    assert abs((c.mjd_of_native(n0, latch_n=n) - armed) * 86400.0) < 1e-5
    assert c.source == "arm:pps_epoch" and c.record == rec
    assert c.anchor == armed - n0 * 32.768e-6 / 86400.0
    stale = ArmRecord(armed_mjd=armed - 3600.0 / 86400.0, arm_seq=arm_seq)  # not this run's arm
    with pytest.raises(MjdEpochError, match="behind the wall clock"):
        MjdClock(read_arm=lambda: (stale, ""), wall=lambda: wall_s).mjd_of_native(n)
    with pytest.raises(MjdEpochError, match="no record"):          # production: fail loudly
        MjdClock(read_arm=lambda: (None, "no record"), wall=lambda: wall_s).mjd_of_native(n)
    c = MjdClock(read_arm=lambda: (stale, ""), wall=lambda: wall_s, strict=False)
    got = c.mjd_of_native(n)                          # development: the wall latch
    assert c.source == "wall" and abs((got - wall_mjd) * 86400.0) < 1e-3
    c = MjdClock(61341.25, read_arm=lambda: 1 / 0)    # explicit: etcd never read
    assert c.mjd_of_native(6144) == 61341.25 + 6144 * 32.768e-6 / 86400.0


def test_one_c1_batch_per_cycle_through_the_production_emitter():
    from dsart.coinc import wire
    from dsart.services.c1_emit import C1EmitConfig, C1TcpEmitter

    tx = C1TcpEmitter(config=C1EmitConfig(host="127.0.0.1", search_node_id=9, gpu_half=1))
    clock = MjdClock(61341.0, read_arm=None)
    bursts = [Burst(d=0, f=2, t=3 * 384 + 150, l=11, m=21, amp=1.8)]
    holder = {}
    svc, *_ = _svc(bursts=bursts, last_cycle=3,
                   emit=lambda k, kept, rows: holder["emit"](k, kept, rows))
    holder["emit"] = make_c1_emit(tx, plan=svc.eng.plan, search_node_id=9, gpu_half=1, n_grid=N,
                                  clock=clock)
    k, _res, kept = svc.step(timeout_s=1.0)
    assert tx._queue.qsize() == 1 and svc.mon.c1_batches == 1
    b = tx._queue.get_nowait()
    h = b.header
    assert (h.cube_id, h.event_specnum_start, h.sample_period_specnum) == (k, k * 384, 8)
    assert abs(h.sample_period_us - 524.288) < 1e-9 and h.n_grid == N and h.n_fdm_in_cube == 5
    assert (h.search_node_id, h.gpu_half, h.n_candidates) == (9, 1, len(b.rows))
    assert h.mjd_start == 61341.0 + k * 6144 * 32.768e-6 / 86400.0
    hit = [r for r in b.rows if (r.l_pix, r.m_pix) == (11, 21)]
    assert hit and hit[0].event_specnum == 3 * 384 + 150
    # C2's parser takes it; a candidate's MJD is its external sample's
    blob = wire.C1BatchEncoder.encode(h, b.rows).decode()
    back = wire.parse_c1_batch(blob.splitlines())
    assert back.header.event_specnum_start == k * 384 and len(back.candidates) == len(b.rows)
    assert abs(back.header.candidate_mjd(3 * 384 + 150) - clock.mjd_of_native(
        (3 * 384 + 150) * 16)) * 86400.0 < 1e-5
    # an empty cycle still sends its batch (C2's per-half liveness)
    hdr0 = c1_batch_header(k + 1, plan=svc.eng.plan, n_rows=0, search_node_id=9, gpu_half=1,
                           n_grid=N, clock=clock)
    assert hdr0.n_candidates == 0 and hdr0.event_specnum_start == (k + 1) * 384

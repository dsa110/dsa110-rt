"""Datagram loss and corr-node outages never cost a lag-recovery seek (G1).

G1 fleet-load (2026-10-29): the engine lag-seeked on the lossy nodes (n01
7.3e-4, n02 3.6e-4 datagram loss; 15 and 8 seeks in 5 min) and once at start
everywhere. Causes: the per-cycle valid fraction fell to a dense [F, hop, S]
gather whenever a row was missing (~40 ms of host time per cycle at ns4), and
the scheduler started from the deadline edge of a ring whose flows were still
starting. CPU: the sparse valid fraction equals the dense definition; the
scheduler starts at the fan-in quorum and its one start-up catch-up is not a
seek; a stopped corr node costs no wall-clock wait per window. GPU (x2_gpu,
both halves on one node, full rate): 1e-3 datagram loss plus one corr node
out for 10 s -> zero seeks, the engine's valid fraction == the ring's mask on
every DM-cycle, every candidate of a partial cube flagged.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_search_ring_v4 import _setup, _src  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_sparse_valid_fraction_equals_the_dense_definition():
    from dsart.x2.search.engine import EngineConfig, OwnerEngine

    plan, inputs, mr = _setup()
    eng = OwnerEngine(plan, inputs, mr, EngineConfig(n_grid=32, backend="cpu"))
    rng = np.random.default_rng(3)
    for st in eng.states.values():
        sh = st.spec.shifts
        for trial in range(12):
            p = (0.0, 1e-3, 0.02, 0.3)[trial % 4]
            st.streams.valid[...] = rng.random(st.streams.valid.shape) >= p
            if trial % 3 == 2:
                st.streams.valid[3] = False                      # a stream out entirely
            t0 = 1000 + 37 * trial
            rows = t0 + np.arange(st.spec.hop)
            sl = np.mod(rows[None, :, None] - sh[:, None, :], st.R_s)
            dense = float(st.streams.valid[np.arange(st.spec.n_stream)[None, None, :],
                                           sl].all(axis=2).mean())
            assert eng._valid_fraction(st, t0) == dense


def test_scheduler_starts_at_the_fan_in_quorum():
    """Flows start one by one: no cycle until >= min_streams streams have it
    (the deadline edge alone would hand out a cycle with no data)."""
    from dsart.x2.search.service import CycleScheduler

    plan, _inputs, mr = _setup()
    fake, src = _src(plan, mr, grace_s=0.01, deadline_s=0.02)
    for d in fake.ct:
        fake.ct[d][:] = 0
        fake.ct[d][:13] = 30 * plan.by_d(d).hop                 # 13 of 16 flowing (< 15)
    sched = CycleScheduler(plan, src)
    with pytest.raises(TimeoutError):
        sched.next_cycle(timeout_s=0.05)
    for d in fake.ct:
        fake.ct[d][:15] = 30 * plan.by_d(d).hop                 # quorum reached
    k = sched.next_cycle(timeout_s=0.05)
    assert all((k + 1) * s.hop + s.max_neg <= src.quorum_pos(s.d) for s in plan.dms)
    assert sched.n_seeks == 0


class _Edge:
    def __init__(self, hop):
        self.pos = 0
        self.hop = hop

    def write_pos(self, d):
        return self.pos


def test_start_up_catch_up_is_not_a_seek_but_a_later_backlog_is():
    from dsart.x2.search.service import CycleScheduler

    plan = SimpleNamespace(dms=[SimpleNamespace(d=0, hop=384, max_neg=10)])
    ring = _Edge(384)
    sched = CycleScheduler(plan, ring, max_lag_cycles=8)
    ring.pos = 5 * 384 + 10
    assert sched.next_cycle(timeout_s=0.01) == 4
    ring.pos += 20 * 384                                         # cycle 0 was slow
    assert sched.next_cycle(timeout_s=0.01) == 24
    assert (sched.n_seeks, sched.n_startup_skipped) == (0, 19)
    ring.pos += 20 * 384                                         # a real stall later
    assert sched.next_cycle(timeout_s=0.01) == 44
    assert sched.n_seeks == 1


def test_a_stopped_corr_node_costs_no_wall_clock_wait_per_window():
    """One corr node's streams stopped (here 1 of 16, as 4 of 64 in production:
    the quorum still holds), the rest flowing: every window near the live edge
    is released at once (data-time grace only), its stopped stream zero-filled
    and flagged."""
    plan, _inputs, mr = _setup(last_cycle=10)
    d = plan.dms[0].d
    fake, src = _src(plan, mr)                                   # production grace / deadline
    fake.ct[d][:] = 4000
    fake.ct[d][4] = 100                                          # corr node out from row 100
    t = time.monotonic()
    for k in range(20):
        _raw, valid = src.wait_window(d, 200 + 64 * k, 64, timeout_s=0.0)
        assert not valid[4].any() and valid[:4].all() and valid[5:].all()
    assert time.monotonic() - t < 0.2
    assert src.fan_in(d)["windows"] == dict(fan_in=20)


@pytest.mark.x2_gpu
def test_gpu_full_rate_loss_and_corr_node_outage_zero_seeks(tmp_path):
    """Both halves of S1 (2x-gridded-ns4-cint4) at full rate on GPUs 0, 1:
    1e-3 datagram loss (4 rows each) and streams 0-3 out for 10 s."""
    import torch
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs (harness -g 0,1)")
    out = tmp_path / "eng"
    cmd = [sys.executable, str(ROOT / "bench" / "timeres" / "b8_owner2x.py"), "engine",
           "--gpus", "0,1", "--input", "synthetic-ring", "--opoint", "2x-gridded-ns4-cint4",
           "--minutes", "1.0", "--loss", "1e-3", "--loss-rows-per-dg", "4",
           "--dropout-streams", "0-3", "--dropout-at-s", "20", "--dropout-s", "10",
           "--out", str(out)]
    plan = os.environ.get("DSART_X2_PLAN_NS4", "/home/ubuntu/x2_scratch/plans/dm_plan_2x_ns4.npz")
    if Path(plan).exists():
        cmd += ["--plan", plan]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    res = json.loads((out / "b8_engine.json").read_text())
    halves = [r for r in res["results"] if r["check"] == "engine"]
    assert len(halves) == 2, (res, p.stderr[-3000:])
    for r in halves:
        fc = r["fanin_check"]
        assert r["lag_recovery_seeks"] == 0, r
        assert r["status"] == "ok", r
        assert fc["valid_fraction_mismatches"] == 0 and fc["unflagged_partial_candidates"] == 0
        assert fc["partial_dm_cycles"] > 0 and fc["rows_missing"] > 0
        assert fc["min_valid_fraction"] == 0.0                   # the outage: every image short

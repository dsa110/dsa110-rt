"""Ring v4 binding (x2/search/ring.RingV4Source over WS-T's SPEC §6 reader).

A fake reader implementing the frozen read API (dms, fmt, streams, heads,
copy_window, counters) over a MemoryRing checks, on CPU:

* the DATA-time fan-in rule: complete / fan_in (>= ceil(15/16 S) final and
  the lead stream ``grace_s`` of data past the window) / deadline (lead
  ``deadline_s`` past, any count) / timeout, and that ``write_pos`` is the
  same release edge;
* a consumer reading the PAST with a stream missing is released at once (the
  reader's own wall-clock grace would cost 0.25 s per DM window);
* non-PRESENT rows are zero-filled and flagged invalid; ``wait_window_into``
  fills the caller's buffer;
* ``validate`` lists every ring / engine mismatch (tier clock, counts,
  steps, pattern ids) and passes a consistent pair;
* the owner engine over the binding equals the engine over MemoryRing, also
  with a lagging stream (== that stream flagged invalid).

The same binding against WS-T's real C writer/reader runs in
test_search_ring_v4_wst.py (skips until recv_ring has ring v4).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from dsart.x2.search.codec import RowFormat
from dsart.x2.search.engine import EngineConfig, OwnerEngine
from dsart.x2.search.geometry import owner_plan_from_tiers
from dsart.x2.search.ring import (RV_PRESENT, MemoryRing, RingTimeout, RingV4Source)
from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

N, V, S = 32, 20, 16
RV_MISSING = 8


class FakeV4Reader:
    """SPEC §6 read API over a MemoryRing; ``ct[d]`` = per-stream complete_through."""

    def __init__(self, mr: MemoryRing, tif: dict, *, s_main=0.3534, s_heavy=0.1, pids=None,
                 n_values=V, n_heavy=2):
        self.mr = mr
        self.access = "fake"
        self.shm_name = "/fake"
        self._tif = dict(tif)
        rb0 = max(f.row_bytes for f in mr._fmt.values())
        self.fmt = SimpleNamespace(n_stream=S, row_bytes=(rb0 + 7) // 8 * 8, bits_main=8,
                                   s_main=float(np.float32(s_main)),
                                   s_heavy=float(np.float32(s_heavy)), n_sub=1)
        self._streams = [SimpleNamespace(stream=s, n_values=n_values, n_heavy=n_heavy,
                                         bytes_per_sample=rb0,
                                         pattern_id=None if pids is None else pids[s])
                         for s in range(S)]
        self.ct = {d: np.zeros(S, np.int64) for d in mr._fmt}
        self.n_copy = 0

    def dms(self):
        return [SimpleNamespace(dm=d, local=i, t_int_factor=self._tif[d], t_buf=self.mr._T,
                                reorder_rows=8) for i, d in enumerate(sorted(self.mr._fmt))]

    def streams(self):
        return list(self._streams)

    def heads(self, dm):
        return {"complete_through": self.ct[dm].astype(np.uint64)}

    def copy_window(self, dm, t_first, n_rows, *, out=None, with_steps=False):
        self.n_copy += 1
        rows = np.arange(t_first, t_first + n_rows)
        if out is None:
            out = np.empty((S, n_rows, self.fmt.row_bytes), np.uint8)
        out[...] = 0
        rb = self.mr._fmt[dm].row_bytes
        out[:, :, :rb] = self.mr._buf[dm][:, np.mod(rows, self.mr._T)]
        present = rows[None, :] < self.ct[dm][:, None]
        out[~present] = 0
        v = np.where(present, RV_PRESENT, RV_MISSING).astype(np.uint8)
        return SimpleNamespace(view=out, validity=v, status="copied", n_final_streams=0)

    def counters(self):
        return dict(writer={}, per_dm={})

    def close(self):
        pass


def _setup(bursts=(), last_cycle=3):
    fa = np.array([50.0, 53.0, 56.0])
    fb = np.array([100.0, 106.0])
    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0]), fine=fa, f2c=np.zeros(3, int)),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=fb, f2c=np.zeros(2, int))]
    plan = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs = {}, {}
    for s in plan.dms:
        cells[s.d] = random_value_cells(S, V, N, radius=6, seed=s.d + 9, n_unused=2)
        inputs[s.d] = make_dm_inputs(cells[s.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                     step_heavy=0.1)
    mr = MemoryRing({s.d: fmt for s in plan.dms}, S, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells=cells, formats={s.d: fmt for s in plan.dms}, n_grid=N,
                       bursts=list(bursts))
    for s in plan.dms:
        sky.fill(mr, s, 0, (last_cycle + 1) * s.hop + s.max_neg + 8)
    return plan, inputs, mr


def _src(plan, mr, **kw):
    fake = FakeV4Reader(mr, {s.d: s.t_native for s in plan.dms})
    return fake, RingV4Source(fake, **kw)


def test_release_rule_complete_fan_in_deadline_timeout():
    plan, _inputs, mr = _setup()
    d = plan.dms[0].d                                   # tier A: 16 native per row
    fake, src = _src(plan, mr, grace_s=0.01, deadline_s=0.05)
    g, dl = src._grace[d], src._dead[d]
    assert src.need == 15 and g == 20 and dl == 96      # ceil(0.01 / 524.288 us), 0.05 s
    t0, n = 100, 50
    end = t0 + n
    fake.ct[d][:] = end                                 # every stream final: complete
    raw, valid = src.wait_window(d, t0, n, timeout_s=0.0)
    assert valid.all() and src.fan_in(d)["last_status"] == "complete"
    assert src.write_pos(d) == end
    # one stream behind: released only once the lead is grace rows past the end
    fake.ct[d][:] = end + g - 1
    fake.ct[d][3] = t0 + 10
    assert src.write_pos(d) == end - 1
    with pytest.raises(RingTimeout):
        src.wait_window(d, t0, n, timeout_s=0.0)
    fake.ct[d][:] = end + g
    fake.ct[d][3] = t0 + 10
    raw, valid = src.wait_window(d, t0, n, timeout_s=0.0)
    assert src.fan_in(d)["last_status"] == "fan_in" and src.fan_in(d)["last_n_final"] == 15
    assert valid[3, :10].all() and not valid[3, 10:].any() and valid[[0, 15]].all()
    assert not raw[3, 10:].any() and raw[3, :10].any()          # missing rows zero-filled
    # three streams behind (< need): only the deadline releases
    fake.ct[d][:] = end + dl - 1
    fake.ct[d][[1, 2, 3]] = t0
    with pytest.raises(RingTimeout):
        src.wait_window(d, t0, n, timeout_s=0.0)
    fake.ct[d][:] = end + dl
    fake.ct[d][[1, 2, 3]] = t0
    _raw, valid = src.wait_window(d, t0, n, timeout_s=0.0)
    assert src.fan_in(d)["last_status"] == "deadline" and not valid[[1, 2, 3]].any()
    f = src.fan_in(d)
    assert f["windows"] == dict(complete=1, fan_in=1, deadline=1) and f["timeouts"] == 2
    assert f["rows_missing"] == (n - 10) + 3 * n


def test_lagging_consumer_is_not_charged_wall_clock_grace():
    """A window in the past with a stream missing (one corr node out) is
    released immediately: no per-call grace (the reader's wall-clock rule
    would charge 0.25 s per DM window)."""
    plan, _inputs, mr = _setup()
    d = plan.dms[0].d
    fake, src = _src(plan, mr)                          # production grace 0.25 s, deadline 1 s
    fake.ct[d][:] = 3000
    fake.ct[d][5] = 0                                   # stream 5 never arrived
    t = time.monotonic()
    for k in range(5):
        _raw, valid = src.wait_window(d, 100 + 64 * k, 64, timeout_s=0.0)
        assert not valid[5].any() and valid[:5].all()
    assert time.monotonic() - t < 0.2
    assert src.fan_in(d)["windows"] == dict(fan_in=5)


def test_wait_window_into_fills_the_callers_buffer():
    plan, _inputs, mr = _setup()
    d = plan.dms[0].d
    fake, src = _src(plan, mr)
    fake.ct[d][:] = 500
    buf = np.full((S, 40, src.row_bytes), 7, np.uint8)
    raw, valid = src.wait_window_into(d, 200, 40, out=buf, timeout_s=0.0)
    assert raw is buf and valid.all()
    rb = mr._fmt[d].row_bytes
    assert np.array_equal(buf[:, :, :rb], mr._buf[d][:, 200:240]) and not buf[:, :, rb:].any()
    a, _ = src.wait_window(d, 200, 40, timeout_s=0.0)          # fresh array per call
    b, _ = src.wait_window(d, 200, 40, timeout_s=0.0)
    assert a is not b and np.array_equal(a, b)


def test_validate_lists_every_mismatch():
    plan, inputs, mr = _setup()
    tif = {s.d: s.t_native for s in plan.dms}
    assert RingV4Source(FakeV4Reader(mr, tif)).validate(plan, inputs) == []
    pids = list(range(100, 100 + S))
    vmap = SimpleNamespace(stream_pattern_ids=pids)
    assert RingV4Source(FakeV4Reader(mr, tif, pids=pids)).validate(plan, inputs, vmap) == []
    bad_tif = dict(tif)
    bad_tif[plan.dms[1].d] = 16
    cases = [
        (FakeV4Reader(mr, bad_tif), None, "t_int_factor"),
        (FakeV4Reader(mr, tif, s_main=0.1932), None, "s_main"),
        (FakeV4Reader(mr, tif, s_heavy=0.0773), None, "s_heavy"),
        (FakeV4Reader(mr, tif, n_values=V + 1), None, "n_values"),
        (FakeV4Reader(mr, tif, pids=pids[:3] + [7] + pids[4:]), vmap, "PATTERN_MISMATCH"),
    ]
    for fake, vm, what in cases:
        with pytest.raises(ValueError, match=what):
            RingV4Source(fake).validate(plan, inputs, vm)
    with pytest.raises(ValueError, match="does not hold"):
        RingV4Source(FakeV4Reader(mr, tif), owned=[99])


def _engine(plan, inputs, ring):
    return OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",),
                                                        widths=(1, 2, 4), fft_mode="pruned",
                                                        n_tap_pixels=32, layer2_n_burnin=0))


def _cands(res):
    return sorted((c.fine_global, c.t_tier, c.l, c.m, c.filt, round(float(c.snr), 4))
                  for c in res.candidates)


def test_engine_over_binding_equals_memory_ring_incl_a_lagging_stream():
    bursts = [Burst(d=0, f=1, t=3 * 384 + 120, l=9, m=20, amp=2.0)]
    plan, inputs, mr = _setup(bursts=bursts)
    k = 3
    want = _engine(plan, inputs, mr).process_cycle(k, keep_maps=True)
    fake, src = _src(plan, mr)
    for s in plan.dms:
        fake.ct[s.d][:] = mr.write_pos(s.d)
    got = _engine(plan, inputs, src).process_cycle(k, keep_maps=True)
    assert _cands(got) == _cands(want) and _cands(want)
    for d in want.per_dm:
        assert np.array_equal(got.per_dm[d].best, want.per_dm[d].best)
    # stream 4 of every DM stops early: == the MemoryRing with stream 4 flagged
    lag = {}
    for s in plan.dms:
        lag[s.d] = k * s.hop - 40
        fake.ct[s.d][4] = lag[s.d]
    src2 = RingV4Source(fake, grace_s=1e-3)
    got2 = _engine(plan, inputs, src2).process_cycle(k, keep_maps=True)
    assert src2.fan_in(plan.dms[0].d)["windows"].get("fan_in", 0) >= 1
    mr2 = MemoryRing(mr._fmt, S, t_buf=mr._T)
    for s in plan.dms:
        hi = mr.write_pos(s.d)
        valid = np.ones((S, hi), bool)
        valid[4, lag[s.d]:] = False
        mr2.push(s.d, 0, mr._buf[s.d][:, :hi], valid)
    want2 = _engine(plan, inputs, mr2).process_cycle(k, keep_maps=True)
    assert _cands(got2) == _cands(want2)
    for d in want2.per_dm:
        assert np.array_equal(got2.per_dm[d].best, want2.per_dm[d].best)
    assert not np.array_equal(want2.per_dm[plan.dms[0].d].best, want.per_dm[plan.dms[0].d].best)

"""Replay driver (x2/search/replay.py) on a synthetic recording with WS-T's
``Recording`` interface: today's chain and the 1x-compat engine agree cycle
by cycle (production candidates all matched within t/DM/l/m +- 1,
dS/N <= 0.1); recorder-flagged slots become invalid ring rows."""
from __future__ import annotations

import numpy as np

from dsart.x2.search.codec import RowFormat
from dsart.x2.search.geometry import owner_plan_1x
from dsart.x2.search.replay import F_TORN, F_WRITTEN, VF_DATA_PRESENT, ring_from_recording, \
    replay_1x
from dsart.x2.search.ring import MemoryRing
from dsart.x2.search.synth import Burst, SyntheticSky, random_value_cells

N = 32


class FakeRecording:
    def __init__(self, payload, t0, flags=None, vf=None):
        self.p = payload                       # [T, n_corr, 2 nfm] int8
        self.t0 = t0
        T, C, _ = payload.shape
        self.cf = np.full((T, C), F_WRITTEN, np.uint8) if flags is None else flags
        self.vf = np.full((T, C), VF_DATA_PRESENT, np.uint16) if vf is None else vf

    def t_range(self, dm):
        return self.t0, self.t0 + self.p.shape[0]

    def slots(self, dm, a, b):
        i, j = a - self.t0, b - self.t0
        return dict(payload=self.p[i:j], vf=self.vf[i:j], corr_flags=self.cf[i:j])


def _recording(n_val=30):
    plan = owner_plan_1x(coarse_dm=np.array([102.0]), fine_dm=np.array([96.0, 102.0, 108.0]),
                         fine_to_coarse=np.zeros(3, int), owned=[0], n_sub=1, csf=8)
    spec = plan.dms[0]
    cells = random_value_cells(16, n_val, N, radius=6, seed=21, n_unused=4)
    fmt = RowFormat("cint8", n_values=n_val)
    tmp = MemoryRing({0: fmt}, 16, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells={0: cells}, formats={0: fmt}, n_grid=N, seed=3,
                       bursts=[Burst(d=0, f=1, t=4 * 192 + 70, l=12, m=19, amp=1.2)])
    sky.fill(tmp, spec, 100, 6 * 192)
    raw, _ = tmp.wait_window(0, 100, 6 * 192)
    payload = np.ascontiguousarray(raw.transpose(1, 0, 2)).view(np.int8)
    lut = np.stack([np.where(c >= 0, c, 0) for c in cells]).astype(np.int32)
    nf = np.array([(c >= 0).sum() for c in cells], np.int32)
    return plan, payload, lut, nf


def test_ring_from_recording_marks_bad_slots_invalid():
    _plan, payload, _lut, _nf = _recording()
    cf = np.full(payload.shape[:2], F_WRITTEN, np.uint8)
    cf[5, 3] |= F_TORN
    vf = np.full(payload.shape[:2], VF_DATA_PRESENT, np.uint16)
    vf[7, 1] = 0
    ring = ring_from_recording(FakeRecording(payload, 100, cf, vf), 0, n_values=30)
    raw, valid = ring.wait_window(0, 100, 10)
    assert not valid[3, 5] and not valid[1, 7] and valid.sum() == valid.size - 2
    assert np.array_equal(raw[0, 0].view(np.int8), payload[0, 0])


def test_replay_1x_engine_matches_production_cycle_by_cycle():
    plan, payload, lut, nf = _recording()
    out = replay_1x(FakeRecording(payload, 100), 0, plan=plan, lut=lut, n_filled=nf, n_grid=N)
    assert out["ok"] and len(out["cycles"]) >= 2, out
    assert any(c["n_prod"] > 0 for c in out["cycles"])

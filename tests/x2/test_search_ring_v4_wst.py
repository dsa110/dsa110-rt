"""Ring v4 binding against WS-T's REAL writer + reader (SPEC §5/§6).

A small private test ring (unique /dev/shm name, never a production one)
is filled through ``RxRingV4Writer.ingest`` with frame-v2 datagrams whose
payload is packed by THIS engine's packer (and checked byte-equal to
WS-T's ``prod_frame2.pack_samples``); the owner engine then reads it
through ``RingV4Source`` (both ``access="mmap"`` and ``"pread"``) and must
equal the engine over a MemoryRing holding the same rows -- complete, and
with one stream stopped early (fan_in: that stream zero-filled/invalid).

Skips until ``dsart.transport.recv_ring`` carries ring v4 (WS-T branch).
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

rr = pytest.importorskip("dsart.transport.recv_ring")
if not hasattr(rr, "RxRingV4Writer") or not hasattr(rr._get_lib(), "rx4_create"):
    pytest.skip("recv_ring without ring v4 (WS-T)", allow_module_level=True)
f2 = pytest.importorskip("dsart.transport.prod_frame2")

from dsart.x2.search.codec import RowFormat, unpack_frame2_rows  # noqa: E402
from dsart.x2.search.engine import EngineConfig, OwnerEngine  # noqa: E402
from dsart.x2.search.geometry import owner_plan_from_tiers  # noqa: E402
from dsart.x2.search.ring import MemoryRing, RingV4Source  # noqa: E402
from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells  # noqa: E402,E501

N, V, S, NH = 32, 20, 16, 2
S_MAIN, S_HEAVY = float(np.float32(0.3534)), float(np.float32(0.1))
PIDS = [0x1000 + s for s in range(S)]


def _setup(last_cycle=3):
    fa = np.array([50.0, 53.0, 56.0])
    fb = np.array([100.0, 106.0])
    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0]), fine=fa, f2c=np.zeros(3, int)),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=fb, f2c=np.zeros(2, int))]
    plan = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=NH, bits=4)
    cells, inputs = {}, {}
    for s in plan.dms:
        cells[s.d] = random_value_cells(S, V, N, radius=6, seed=s.d + 9, n_unused=2)
        inputs[s.d] = make_dm_inputs(cells[s.d], n_grid=N, fmt=fmt, step_main=S_MAIN,
                                     step_heavy=S_HEAVY)
    mr = MemoryRing({s.d: fmt for s in plan.dms}, S, t_buf=4096)
    sky = SyntheticSky(plan=plan, cells=cells, formats={s.d: fmt for s in plan.dms}, n_grid=N,
                       bursts=[Burst(d=0, f=1, t=3 * 384 + 120, l=9, m=20, amp=2.0)])
    for s in plan.dms:
        sky.fill(mr, s, 0, (last_cycle + 1) * s.hop + s.max_neg + 8)
    return plan, inputs, mr, fmt


def _spec(plan, fmt):
    return rr.RingV4Spec(dms=tuple(s.d for s in plan.dms),
                         t_int_factor=tuple(s.t_native for s in plan.dms),
                         t_buf=tuple(4096 for _ in plan.dms), n_stream=S, n_sub=1, bits_main=8,
                         bits_heavy=16, n_grid=N, s_main=S_MAIN, s_heavy=S_HEAVY,
                         stream_n_values=(V,) * S, stream_n_heavy=(NH,) * S, min_retention_s=0.0,
                         reorder_rows=8, hole_timeout_s=0.05)


def _ingest(w, mr, plan, fmt, *, stop=None, k=16):
    rb = fmt.row_bytes
    bps = f2.bytes_per_sample(V, NH, 8, 16)
    assert bps == rb
    rcs = []
    for spec in plan.dms:
        d, tif = spec.d, spec.t_native
        hi = mr.write_pos(d)
        for s in range(S):
            end = hi if stop is None or s not in stop else stop[s][d]
            for seq in range(0, end, k):
                kk = min(k, end - seq)
                payload = np.ascontiguousarray(mr._buf[d][s, seq:seq + kk, :rb])
                h = f2.Frame2Header(
                    seq_first=seq, t0_native=seq * tif, stream_id=s, dm_idx=d, k=kk, n_sub=1,
                    bits_main=8, bits_heavy=16, n_grid=N, t_int_factor=tif, value_kind=0,
                    n_values=V, pattern_id=PIDS[s], n_heavy=NH, step_main=0.5, step_heavy=0.25,
                    payload_bytes=kk * bps, s_main=S_MAIN, s_heavy=S_HEAVY,
                    flags=f2.FLAG_QUANTIZED | f2.FLAG_HEAVY_PRESENT)
                rcs.append(w.ingest(f2.pack_frame(h, payload.tobytes())))
    return rcs


def test_my_frame2_payload_is_ws_t_pack_samples():
    rng = np.random.default_rng(3)
    fmt = RowFormat("frame2", n_values=37, n_heavy=5, bits=5)
    from dsart.x2.search.codec import pack_frame2_rows
    re = rng.integers(-16, 16, size=(7, 37))
    im = rng.integers(-16, 16, size=(7, 37))
    re[:, :5] = rng.integers(-128, 128, size=(7, 5))
    im[:, :5] = rng.integers(-128, 128, size=(7, 5))
    mine = pack_frame2_rows(re, im, fmt)
    theirs = f2.pack_samples(np.stack([re[:, :5], im[:, :5]], -1),
                             np.stack([re[:, 5:], im[:, 5:]], -1), 10)
    assert np.array_equal(mine, theirs)


def _engine(plan, inputs, ring):
    return OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",),
                                                        widths=(1, 2, 4), fft_mode="pruned",
                                                        n_tap_pixels=32, layer2_n_burnin=0))


def _cands(res):
    return sorted((c.fine_global, c.t_tier, c.l, c.m, c.filt, round(float(c.snr), 4))
                  for c in res.candidates)


@pytest.mark.parametrize("access", ["pread", "mmap"])
def test_engine_over_real_ring_v4_equals_memory_ring(access):
    plan, inputs, mr, fmt = _setup()
    k = 3
    want = _engine(plan, inputs, mr).process_cycle(k, keep_maps=True)
    name = f"/x2s-test-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, _spec(plan, fmt))
    try:
        for s in range(S):
            w.set_expected_pattern_id(s, PIDS[s])
        assert set(_ingest(w, mr, plan, fmt)) == {rr.RX4_OK}
        src = RingV4Source.open(name, owned=[s.d for s in plan.dms], access=access)
        try:
            from types import SimpleNamespace
            assert src.validate(plan, inputs, SimpleNamespace(stream_pattern_ids=PIDS)) == []
            d0 = plan.dms[0].d
            assert src.write_pos(d0) == mr.write_pos(d0)
            raw, valid = src.wait_window(d0, 100, 30, timeout_s=0.0)
            assert valid.all() and raw.shape == (S, 30, src.row_bytes)
            for s in (0, 7):
                got = unpack_frame2_rows(raw[s], fmt)
                ref = unpack_frame2_rows(mr._buf[d0][s, 100:130], fmt)
                assert all(np.array_equal(a, b) for a, b in zip(got, ref))
            res = _engine(plan, inputs, src).process_cycle(k, keep_maps=True)
            assert _cands(res) == _cands(want) and _cands(want)
            for d in want.per_dm:
                assert np.array_equal(res.per_dm[d].best, want.per_dm[d].best)
            assert all(f["windows"].get("complete", 0) >= 1 for f in
                       (src.fan_in(s.d) for s in plan.dms))
        finally:
            src.close()
    finally:
        w.close()
        w.unlink()


def test_stream_stopped_early_is_fan_in_and_zero_filled():
    plan, inputs, mr, fmt = _setup()
    k = 3
    stop = {4: {s.d: k * s.hop - 40 for s in plan.dms}}
    name = f"/x2s-test-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, _spec(plan, fmt))
    try:
        assert set(_ingest(w, mr, plan, fmt, stop=stop)) == {rr.RX4_OK}
        src = RingV4Source.open(name, access="pread", grace_s=1e-3)
        try:
            got = _engine(plan, inputs, src).process_cycle(k, keep_maps=True)
            assert src.fan_in(plan.dms[0].d)["windows"].get("fan_in", 0) >= 1
        finally:
            src.close()
    finally:
        w.close()
        w.unlink()
    mr2 = MemoryRing(mr._fmt, S, t_buf=mr._T)
    for s in plan.dms:
        hi = mr.write_pos(s.d)
        valid = np.ones((S, hi), bool)
        valid[4, stop[4][s.d]:] = False
        mr2.push(s.d, 0, mr._buf[s.d][:, :hi], valid)
    want = _engine(plan, inputs, mr2).process_cycle(k, keep_maps=True)
    assert _cands(got) == _cands(want)
    for d in want.per_dm:
        assert np.array_equal(got.per_dm[d].best, want.per_dm[d].best)

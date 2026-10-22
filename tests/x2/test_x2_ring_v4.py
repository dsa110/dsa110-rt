"""Ring v4 (SPEC §6): C writer (rx4_ingest) + read-only Python reader, no sockets."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from dsart.transport import prod_frame, prod_frame2 as f2
from dsart.x2 import quant as q

rr = pytest.importorskip("dsart.transport.recv_ring")
if not hasattr(rr._get_lib(), "rx4_create"):
    pytest.skip("_recv_ring.so built without ring v4", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec)
sys.modules["wst_util"] = wu
_spec.loader.exec_module(wu)

C = {name: i for i, name in enumerate(rr.RX4_COUNTERS)}
S_MAIN = float(np.float32(0.3534))
S_HEAVY = float(np.float32(0.1))
BPS = 4 + 18            # n_values 20, n_heavy 2, cint4


def small_spec(**kw) -> rr.RingV4Spec:
    base = dict(dms=(3, 11), t_int_factor=(16, 32), t_buf=(64, 32), n_stream=4, n_sub=2,
                bits_main=8, bits_heavy=16, n_grid=256, s_main=S_MAIN, s_heavy=S_HEAVY,
                stream_n_values=(20,) * 4, stream_n_heavy=(2,) * 4, min_retention_s=0.0,
                reorder_rows=8, hole_timeout_s=0.05)
    base.update(kw)
    return rr.RingV4Spec(**base)


@pytest.fixture(params=["mmap", "pread"])
def ring(request):
    name = f"/x2r4-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, small_spec())
    r = rr.RxRingV4Reader.open(name, access=request.param)
    yield w, r
    r.close()
    w.close()
    w.unlink()


def send(w, *, streams=range(4), dm=3, t0=0, t1=16, k=2, tif=None, skip=(), now_ns=None, **kw):
    tif = tif or (16 if dm == 3 else 32)
    rcs = []
    for s in streams:
        for seq in range(t0, t1, k):
            if (s, seq) in skip:
                continue
            rcs.append(w.ingest(wu.synthetic_frame(stream=s, dm=dm, seq_first=seq,
                                                   k=min(k, t1 - seq), t_int_factor=tif, **kw),
                                now_ns=now_ns))
    return rcs


def rows_of(view, s):
    return view[s, :, :BPS]


def test_spec_validation():
    assert small_spec().row_bytes_eff == 24
    with pytest.raises(ValueError):
        small_spec(min_retention_s=20.0)                     # 64 rows * 0.52 ms << 20 s
    with pytest.raises(ValueError):
        small_spec(dms=(3, 3))
    with pytest.raises(ValueError):
        small_spec(n_stream=65, stream_n_values=(), stream_n_heavy=(), row_bytes=24)
    s = small_spec()
    assert rr.RingV4Spec.from_dict(s.to_dict()) == s


def test_round_trip_small(ring):
    w, r = ring
    assert set(send(w, t0=100, t1=116)) == {rr.RX4_OK}
    win = r.wait_window(3, 100, 16, timeout_s=1.0)
    assert win.status == "complete" and win.n_final_streams == 4
    assert win.zero_copy == (r.access == "mmap")
    view, valid = win
    assert (valid == rr.RV_PRESENT).all()
    for s in range(4):
        np.testing.assert_array_equal(rows_of(view, s), np.repeat(np.arange(100, 116)[:, None] & 0xFF, BPS, 1))
        assert not view[s, :, BPS:].any()                     # row padding zeroed
    cw = r.copy_window(3, 100, 16, with_steps=True)
    np.testing.assert_array_equal(cw.view, view)
    assert (cw.steps[..., 0] == 0.5).all() and (cw.steps[..., 1] == 0.25).all()
    assert [d.dm for d in r.dms()] == [3, 11]
    assert r.dm_info(11).t_int_factor == 32 and r.fmt.row_bytes == 24
    assert r.streams()[0].bytes_per_sample == BPS


def test_holes_reorder_late_dup(ring):
    w, r = ring
    send(w, streams=[0], t0=0, t1=40, skip={(0, 10)}, now_ns=1_000_000_000)
    h = r.heads(3)
    assert int(h["complete_through"][0]) == 40           # rows 10, 11 declared holes
    v = r.window_validity(3, 8, 6)[0]
    assert v.tolist() == [1, 1, 2, 2, 1, 1]
    assert not r.copy_window(3, 10, 2).view[0].any()        # hole rows zeroed
    send(w, streams=[0], t0=10, t1=12)                       # late: dropped
    fl = r.counters()["per_dm"][3]["writer"]
    assert fl["rows_hole"] == 2 and fl["rows_late"] == 2
    # tail gap: only the sweep (gap timeout) can close it
    send(w, streams=[0], t0=40, t1=42, now_ns=2_000_000_000)
    send(w, streams=[0], t0=44, t1=46, now_ns=2_000_000_000)
    assert int(r.heads(3)["complete_through"][0]) == 42
    w.sweep(now_ns=2_010_000_000)                            # 10 ms: not yet
    assert int(r.heads(3)["complete_through"][0]) == 42
    w.sweep(now_ns=2_100_000_000)                            # 100 ms > 50 ms timeout
    assert int(r.heads(3)["complete_through"][0]) == 46
    # duplicate ahead of a gap
    send(w, streams=[0], t0=50, t1=52)
    send(w, streams=[0], t0=50, t1=52)
    assert r.counters()["per_dm"][3]["writer"]["rows_dup"] == 2


def test_validation_counters(ring):
    w, r = ring
    ok = wu.synthetic_frame(stream=1, dm=3, seq_first=0)
    v1 = prod_frame.pack_frame(prod_frame.ProdFrameHeader(
        seq=0, specnum=0, chgroup=0, dm_idx=3, frag_idx=0, n_frags=1, n_grid=256, n_filled=1,
        pattern_id=7, bits_per_cell=16, t_int_factor=16, scale=1.0, offset=0.0,
        payload_bytes_in_frag=2, flags=prod_frame.FLAG_QUANTIZED), b"\x00\x00")
    bad_t0 = bytearray(ok)
    bad_t0[16] ^= 1
    bad_magic = bytearray(ok)
    bad_magic[0] ^= 1
    cases = [
        (v1, "bad_version"),
        (bytes(bad_t0), "bad_time"),
        (bytes(bad_magic), "bad_magic"),
        (ok[:50], "bad_length"),
        (ok + b"\x00", "bad_length"),
        (wu.synthetic_frame(dm=5), "not_owned"),
        (wu.synthetic_frame(dm=3, t_int_factor=32), "tier_mismatch"),
        (wu.synthetic_frame(dm=3, bits_main=10), "format_mismatch"),
        (wu.synthetic_frame(dm=3, s_main=0.35), "format_mismatch"),
        (wu.synthetic_frame(dm=3, n_sub=4), "format_mismatch"),
        (wu.synthetic_frame(dm=3, n_values=21), "stream_mismatch"),
        (wu.synthetic_frame(dm=3, stream=5), "bad_field"),
    ]
    for dg, name in cases:
        assert rr.RX4_COUNTERS[w.ingest(dg)] == name, name
    assert w.ingest(wu.synthetic_frame(stream=3, dm=3), port_chgroup=0) == C["port_mismatch"]
    assert w.ingest(wu.synthetic_frame(stream=3, dm=3), port_chgroup=1) == rr.RX4_OK
    c = r.counters()["writer"]
    assert c["v1_frames"] == 1 and c["format_mismatch"] == 3 and c["ok"] == 1
    assert c["datagrams"] == len(cases) + 2
    # pattern mismatch: rows are final, zeroed and flagged
    w.set_expected_pattern_id(1, 99)
    assert w.ingest(ok) == C["pattern_mismatch"]
    assert r.streams()[1].pattern_id == 99
    assert (r.window_validity(3, 0, 2)[1] == rr.RV_PATTERN_MISMATCH).all()
    assert not r.copy_window(3, 0, 2).view[1].any()
    w.clear_expected_pattern_id(1)
    assert w.ingest(wu.synthetic_frame(stream=1, dm=3, seq_first=2)) == rr.RX4_OK


def test_warmup_flags_reach_validity(ring):
    w, r = ring
    send(w, streams=[2], t0=0, t1=2, flags=f2.FLAG_RFI_WARMING_UP | f2.FLAG_NOISE_WARMUP)
    v = r.window_validity(3, 0, 2)[2]
    assert (v == rr.RV_PRESENT | rr.RV_RFI_WARMING_UP | rr.RV_NOISE_WARMUP).all()
    send(w, streams=[1], t0=0, t1=1, flags=f2.FLAG_ROW_GUARD, step_main=1.5)
    cw = r.copy_window(3, 0, 1, with_steps=True)
    assert cw.validity[1, 0] == rr.RV_PRESENT | rr.RV_ROW_GUARD
    assert cw.steps[1, 0, 0] == 1.5


def test_fan_in_deadline_timeout(ring):
    w, r = ring
    send(w, streams=[0, 1, 2], t0=0, t1=8)
    win = r.wait_window(3, 0, 8, min_streams=3, grace_s=0.05, deadline_s=5.0)
    assert win.status == "fan_in" and win.waited_s >= 0.05 and win.n_final_streams == 3
    assert (win.validity[3] == rr.RV_MISSING).all() and (win.validity[:3] == rr.RV_PRESENT).all()
    cw = r.wait_window(3, 0, 8, min_streams=3, grace_s=0.0, copy=True)
    assert not cw.view[3].any() and cw.status == "fan_in"          # missing stream zero-filled
    send(w, streams=[0], t0=8, t1=16)
    win = r.wait_window(3, 8, 8, min_streams=3, grace_s=0.0, deadline_s=0.1)
    assert win.status == "deadline" and win.n_final_streams == 1
    win = r.wait_window(3, 30, 8, timeout_s=0.1)
    assert win.status == "timeout" and (win.validity == rr.RV_MISSING).all()
    assert r.counters()["per_dm"][3]["reader"]["windows"] == {
        "fan_in": 2, "deadline": 1, "timeout": 1}
    assert r.wait_window(3, 0, 8, min_streams=4, timeout_s=0.05).status == "timeout"


def test_lapped_and_recheck(ring):
    w, r = ring
    send(w, t0=0, t1=16)
    win = r.wait_window(3, 0, 8)
    assert win.status == "complete" and r.recheck(win) == 0
    send(w, t0=16, t1=72)                                   # rows 64..71 overwrite 0..7
    assert r.recheck(win) == 4 * 8
    assert (win.validity == rr.RV_LAPPED).all()
    again = r.wait_window(3, 0, 8)
    assert again.status == "lapped" and (again.validity == rr.RV_LAPPED).all()
    assert r.counters()["per_dm"][3]["reader"]["overruns"] == 2
    assert r.oldest(3) == 72 - 64


def test_window_wrapping_ring_end(ring):
    w, r = ring
    send(w, t0=60, t1=68)
    win = r.wait_window(3, 60, 8)
    assert win.status == "complete" and not win.zero_copy
    for s in range(4):
        np.testing.assert_array_equal(rows_of(win.view, s)[:, 0], np.arange(60, 68))
    assert r.view(3, 60, 8) is None


def test_resync_on_jump_beyond_retention(ring):
    w, r = ring
    send(w, streams=[0], t0=0, t1=4)
    send(w, streams=[0], t0=1000, t1=1004)
    h = r.heads(3)
    assert int(h["complete_through"][0]) == 1004
    assert r.counters()["per_dm"][3]["writer"]["resyncs"] == 1


def test_reader_never_writes(ring):
    w, r = ring
    send(w, t0=0, t1=16)
    send(w, dm=11, t0=0, t1=8)
    w.sweep()
    seg = Path("/dev/shm") / w.name.lstrip("/")
    before = hashlib.sha256(seg.read_bytes()).hexdigest()
    r.wait_window(3, 0, 16)
    r.copy_window(3, 0, 16, with_steps=True)
    r.wait_window(3, 20, 4, timeout_s=0.02)
    r.snapshot(0, 256, deadline_s=0.0)
    r.frontier(256)
    r.counters()
    r.stall_report(0.0)
    r.live_edge(3)
    assert hashlib.sha256(seg.read_bytes()).hexdigest() == before
    if r.access == "pread":
        assert r.view(3, 0, 4) is None and r._mm is None          # no mapping at all
        return
    with pytest.raises(ValueError):
        r.view(3, 0, 4)[0, 0, 0] = 1
    r2 = rr.RxRingV4Reader.open(w.name, data_writable=True)
    with pytest.raises(ValueError):
        r2.view(3, 0, 4)[0, 0, 0] = 1                       # numpy-read-only even then
    np.testing.assert_array_equal(r2.view(3, 0, 16), r.view(3, 0, 16))
    r2.close()


def test_snapshot_and_frontier(ring):
    w, r = ring
    send(w, dm=3, t0=100, t1=160)                           # tier A rows (native 1600..2560)
    send(w, dm=11, t0=48, t1=78)                            # tier B rows
    fr = r.frontier(2400)
    assert fr[3]["seq_end"] == 150 and fr[3]["coverage"] == 1.0
    assert fr[11]["seq_end"] == 75 and fr[11]["coverage"] == 1.0
    snap = r.snapshot(1600, 2400, pad_rows={3: (2, 2)}, deadline_s=0.5)
    a, b = snap[3], snap[11]
    assert (a.seq0, a.n_rows, a.status, a.coverage) == (98, 54, "complete", 1.0)
    assert (b.seq0, b.n_rows, b.status) == (50, 25, "complete")
    assert (a.row0_t0_native == 98 * 16).all() and (b.row0_t0_native == 50 * 32).all()
    assert (a.valid[:, :2] == rr.RV_MISSING).all() and not a.codes[:, :2].any()
    assert (a.valid[:, 2:] == rr.RV_PRESENT).all()
    np.testing.assert_array_equal(a.codes[1, 2:, 0], np.arange(100, 152) & 0xFF)
    assert (a.steps[:, 2:, 0] == 0.5).all() and (a.steps[:, :2] == 0).all()
    assert (a.bytes_per_sample == BPS).all()
    # not yet arrived -> deadline; already overwritten -> too_late
    assert r.snapshot(2400, 2700, dms=[3], deadline_s=0.05)[3].status == "deadline"
    assert r.snapshot(1500, 1700, dms=[3], deadline_s=0.0)[3].status == "too_late"
    assert r.counters()["per_dm"][3]["reader"]["snapshots"] == {
        "complete": 1, "deadline": 1, "too_late": 1}


def test_stall_report_heartbeat_and_live_edge(ring):
    w, r = ring
    send(w, streams=[0, 1, 2], t0=0, t1=8)
    w.sweep()
    assert r.heartbeat_age_s() < 5.0 and r.writer_state == "ready"
    rep = r.stall_report(stall_s=0.0)
    assert rep["dms"][3]["never_started"] == [3]
    assert sorted(rep["dms"][3]["stalled"]) == [0, 1, 2]
    assert rep["dms"][11]["never_started"] == [0, 1, 2, 3]
    assert r.live_edge(3, min_streams=3) == 8 and r.live_edge(3, min_streams=4) == 0


def test_writer_close_marks_state():
    name = f"/x2r4-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, small_spec(), prefault=True)
    r = rr.RxRingV4Reader.open(name, attach_timeout_s=1.0)
    gen = r.generation
    w.close()
    assert r.writer_state == "closed" and gen != 0
    r.close()
    w.unlink()
    with pytest.raises(FileNotFoundError):
        rr.RxRingV4Reader.open(name, attach_timeout_s=0.1)


# ---------------------------------------------------------------------------
# With WS-P's real value maps and plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", [("gridded", 2, 8), ("baselines", 4, 10)])
def test_real_value_map_round_trip(fmt):
    vm = wu.value_map(*fmt)
    rule = q.RowRule.for_format(fmt[2], fmt[1], warmup_blocks=1)
    pl = wu.plan(fmt[1])
    dA = 0
    dB = int(next(d for d in range(len(pl.owner_half))
                  if int(pl.coarse_t_native[d]) == 32 and int(pl.owner_half[d]) == 0))
    tif = (int(pl.coarse_t_native[dA]), int(pl.coarse_t_native[dB]))
    assert tif == (16, 32)
    spec = rr.RingV4Spec(
        dms=(dA, dB), t_int_factor=tif, t_buf=(64, 32), reorder_rows=8,
        n_stream=vm.n_streams, n_sub=vm.n_sub,
        bits_main=rule.bits_main, value_kind=0 if vm.kind == "gridded" else 1, n_grid=vm.n_grid,
        s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32),
        stream_n_values=tuple(int(x) for x in vm.n_values),
        stream_n_heavy=tuple(int(x) for x in vm.n_heavy), min_retention_s=0.0)
    name = f"/x2r4-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, spec)
    r = rr.RxRingV4Reader.open(name)
    try:
        for s in range(vm.n_streams):
            w.set_expected_pattern_id(s, int(vm.stream_pattern_ids[s]))
        S = 12
        blocks = wu.stream_blocks(vm, range(vm.n_streams), rule=rule, n_samples=S, seed=3)
        for d, f, seq0 in ((dA, 16, 4096), (dB, 32, 2048)):
            for s, bc in blocks.items():
                for dg in wu.datagrams_for(vm, bc, stream=s, dm=d, seq_first=seq0,
                                           t_int_factor=f, rule=rule):
                    assert w.ingest(dg) == rr.RX4_OK
            win = r.wait_window(d, seq0, S, timeout_s=1.0)
            assert win.status == "complete" and (win.validity == rr.RV_PRESENT).all()
            for s, bc in blocks.items():
                sv = vm.streams[s]
                bps = f2.bytes_per_sample(sv.n_values, sv.n_heavy, rule.bits_main, 16)
                ch, cm = f2.unpack_samples(np.ascontiguousarray(win.view[s, :, :bps]), S,
                                           sv.n_values, sv.n_heavy, rule.bits_main, 16)
                np.testing.assert_array_equal(ch, bc.codes_heavy)
                np.testing.assert_array_equal(cm, bc.codes_main)
        assert r.counters()["writer"]["pattern_mismatch"] == 0
    finally:
        r.close()
        w.close()
        w.unlink()


def test_spec_for_node_from_plan_and_value_map():
    vm = wu.value_map("gridded", 2, 8)
    op = wu.opoint("gridded", 2, 8)
    pl = wu.plan(2)
    rule = q.RowRule.for_format(8, 2)
    seen = []
    for node in range(4):
        spec = rr.RingV4Spec.for_node(pl, vm, op, node_index=node, rule=rule)
        assert 1 <= len(spec.dms) <= 16
        assert all(int(pl.owner_half[d]) in (2 * node, 2 * node + 1) for d in spec.dms)
        assert spec.t_int_factor == tuple(int(pl.coarse_t_native[d]) for d in spec.dms)
        assert all(spec.retention_s(i) >= 20.0 for i in range(len(spec.dms)))
        assert all(t % (384 * 16 // f) == 0 for t, f in zip(spec.t_buf, spec.t_int_factor))
        bps = max(f2.bytes_per_sample(int(a), int(b), 8, 16) for a, b in zip(vm.n_values, vm.n_heavy))
        assert spec.row_bytes_eff >= bps and spec.row_bytes_eff - bps < 8
        assert 15e9 < spec.size_bytes < 40e9            # plan: 22-33 GB / node
        seen += list(spec.dms)
    assert sorted(seen) == list(range(len(pl.owner_half)))


def test_mmap_and_pread_access_return_identical_bytes():
    """The two reader copy paths (ledger E5) agree byte for byte on every row
    state: present, warm-up / guard flags, hole, mismatch, missing, lapped,
    wrapped windows, steps, heads, counters and snapshots."""
    name = f"/x2r4-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, small_spec())
    try:
        w.set_expected_pattern_id(2, 99)
        send(w, streams=[0, 1, 3], t0=0, t1=70, skip={(0, 20), (1, 66)}, now_ns=10**9)
        send(w, streams=[2], t0=0, t1=8)                                  # mismatch rows
        send(w, streams=[1], t0=70, t1=74, flags=f2.FLAG_ROW_GUARD | f2.FLAG_RFI_WARMING_UP,
             step_main=2.0)
        send(w, dm=11, streams=[0, 1, 2, 3], t0=5, t1=45)                 # wraps t_buf 32
        a = rr.RxRingV4Reader.open(name, access="mmap")
        b = rr.RxRingV4Reader.open(name, access="pread")
        for dm, t0, n in ((3, 0, 8), (3, 8, 40), (3, 60, 16), (3, 70, 4), (11, 14, 31),
                          (11, 0, 8), (3, 100, 8)):
            np.testing.assert_array_equal(a.window_validity(dm, t0, n), b.window_validity(dm, t0, n))
            ca, cb = a.copy_window(dm, t0, n, with_steps=True), b.copy_window(dm, t0, n, with_steps=True)
            np.testing.assert_array_equal(ca.validity, cb.validity)
            np.testing.assert_array_equal(ca.view, cb.view)
            np.testing.assert_array_equal(ca.steps, cb.steps)
            for k in ("complete_through", "max_end", "last_progress_ns"):
                np.testing.assert_array_equal(a.heads(dm)[k], b.heads(dm)[k])
        va = a.window_validity(3, 0, 64)
        names = {rr.RV_PRESENT, rr.RV_HOLE, rr.RV_PATTERN_MISMATCH, rr.RV_MISSING, rr.RV_LAPPED}
        assert names <= set(np.unique(va & 0x1F).tolist()) | {rr.RV_LAPPED}
        assert (a.window_validity(3, 70, 4)[1] == rr.RV_PRESENT | rr.RV_ROW_GUARD
                | rr.RV_RFI_WARMING_UP).all()
        ca, cb = a.counters(), b.counters()
        assert ca["writer"] == cb["writer"]
        assert ca["per_dm"][3]["writer"] == cb["per_dm"][3]["writer"]
        sa, sb = a.snapshot(16 * 40, 16 * 60, deadline_s=0.0), b.snapshot(16 * 40, 16 * 60, deadline_s=0.0)
        for dm in (3, 11):
            np.testing.assert_array_equal(sa[dm].codes, sb[dm].codes)
            np.testing.assert_array_equal(sa[dm].valid, sb[dm].valid)
            assert sa[dm].status == sb[dm].status
        pinned = np.zeros((4, 8, 24), np.uint8)                          # caller buffer
        wb = b.wait_window(3, 40, 8, out=pinned, min_streams=3, grace_s=0.0)
        assert wb.view is pinned and wb.status == "fan_in"
        a.close()
        b.close()
    finally:
        w.close()
        w.unlink()

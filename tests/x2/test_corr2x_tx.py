"""x2 WS-C: frame-v2 framing + the corr TX stage (SPEC §5, §6).

* ``framer2x.frame_stream_block`` == ``prod_frame2.iter_block_datagrams``
  byte for byte, flow by flow, in send order: regular blocks, guarded rows
  (alone and in equal-step runs), step changes, NO_DATA blocks (sigma 0),
  every k up to ``max_k``, RFI_WARMING_UP; every datagram <= 8972 B;
* device framing of regular blocks == the reference (CPU tensors; CUDA in x2_gpu);
* ``FramedTx2x`` over a packer result: the drain sink counts every
  datagram and sends nothing; its counters feed ``TxRateMeter`` (the
  corr_fast heartbeat keys);
* the C sendmmsg sender over loopback (127.0.0.1-4 standing in for the four
  search nodes, port = base + chgroup): every datagram lands on node
  ``owner_half[dm] // 2``, byte-equal and in row order per flow;
* corr_fast wiring: ``--transport-tx-mode v2|drain`` are 2x-only, ``v2``
  needs the four search-node hosts (the CP1 live argv, a 2x drain, is
  checked in ``test_corr2x_bench_cli``).
"""
from __future__ import annotations

import os
import socket
import time

import numpy as np
import pytest
import torch

import dsart.services.corr_fast_integration as cfi
from dsart.transport import prod_frame2 as f2
from dsart.x2 import quant
from dsart.x2.corr.framer2x import StreamFormat, frame_stream_block
from dsart.x2.corr.packer2x import RowRulePacker
from dsart.x2.corr.tx2x import DrainSink, FramedTx2x, block_tx_budget, corr_routes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _fmt(stream_id=5, n_values=300, n_heavy=4, bits_main=10, n_sub=2):
    return StreamFormat(stream_id=stream_id, n_values=n_values, n_heavy=n_heavy, bits_main=bits_main,
                        n_sub=n_sub, n_grid=256, value_kind=f2.VALUE_KIND_CELLS,
                        pattern_id=0x1234_5678_9ABC_DEF0, s_main=float(np.float32(0.1932)),
                        s_heavy=float(np.float32(0.0773)))


def _codes(rng, fmt, n_d, S):
    cb = fmt.bits_main // 2
    ch = rng.integers(-128, 128, size=(n_d, S, fmt.n_heavy, 2)).astype(np.int8)
    cm = rng.integers(-(1 << (cb - 1)), 1 << (cb - 1), size=(n_d, S, fmt.n_values - fmt.n_heavy, 2))
    return ch, cm.astype(np.int8)


def _reference(fmt, ch, cm, sm, sh, gd, dms, seq0, tif, k=None, extra=0):
    out = []
    for i, d in enumerate(dms):
        out += list(f2.iter_block_datagrams(
            codes_heavy=ch[i], codes_main=cm[i], seq_first=seq0, t_int_factor=tif,
            stream_id=fmt.stream_id, dm_idx=int(d), n_sub=fmt.n_sub, bits_main=fmt.bits_main,
            n_grid=fmt.n_grid, value_kind=fmt.value_kind, pattern_id=fmt.pattern_id,
            step_main=sm[i], step_heavy=sh[i], guarded=gd[i], s_main=fmt.s_main,
            s_heavy=fmt.s_heavy, extra_flags=extra, k=k))
    return out


@pytest.mark.parametrize("case", ["regular", "guards", "no_data", "k3", "tiny", "heavyless"])
def test_framer_equals_reference(case) -> None:
    rng = np.random.default_rng(len(case))
    fmt = _fmt(n_heavy=0 if case == "heavyless" else 4,
               n_values=40 if case == "tiny" else 1500 if case != "k3" else 300)
    n_d, S = 4, 256 if case != "k3" else 128
    ch, cm = _codes(rng, fmt, n_d, S)
    sig = rng.uniform(0.5, 2.0, size=n_d).astype(np.float32)
    sr = np.repeat(sig[:, None], S, axis=1)
    gd = np.zeros((n_d, S), bool)
    if case == "guards":
        gd[0, [0, 9, 10, 11, 255]] = True                 # 10, 11: an equal-step guarded pair
        sr[0, [0, 9, 255]] = [3.0, 4.0, 5.0]
        sr[0, [10, 11]] = 6.0
        gd[2, 100:140] = True                             # a long surge, every row its own scale
        sr[2, 100:140] = rng.uniform(3, 9, size=40)
        sr[3, 50] = np.nextafter(sr[3, 50], np.float32(9))  # a step change without a guard
    if case == "no_data":
        sr[:] = 0.0
    sm = (np.float32(fmt.s_main) * sr).astype(np.float32)
    sh = (np.float32(fmt.s_heavy) * sr).astype(np.float32)
    dms = [3, 11, 19, 46]
    k = 3 if case == "k3" else None
    extra = f2.FLAG_RFI_WARMING_UP if case == "k3" else 0
    pay = np.stack([f2.pack_samples(ch[i], cm[i], fmt.bits_main) for i in range(n_d)])
    fb = frame_stream_block(pay, sm, sh, gd, fmt=fmt, dms=dms, seq_first=7 * S, t_int_factor=16,
                            extra_flags=extra, k=k)
    want = _reference(fmt, ch, cm, sm, sh, gd, dms, 7 * S, 16, k=k, extra=extra)
    got = fb.datagrams()
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a == b
    # device assembly (torch payload; here on the CPU: the same torch ops as on CUDA)
    fb_t = frame_stream_block(torch.from_numpy(pay), sm, sh, gd, fmt=fmt, dms=dms, seq_first=7 * S,
                              t_int_factor=16, extra_flags=extra, k=k)
    assert fb_t.datagrams() == want
    assert int(fb.lengths.max()) <= f2.HEADER_BYTES + f2.MAX_PAYLOAD_BYTES
    # the buffer holds every datagram exactly once
    spans = sorted(zip(fb.offsets.tolist(), fb.lengths.tolist()))
    assert all(o0 + n0 == o1 for (o0, n0), (o1, _) in zip(spans, spans[1:]))
    assert spans[-1][0] + spans[-1][1] == fb.buf.size
    if case == "regular":
        assert len(fb) == n_d * -(-S // fmt.max_k)


@pytest.mark.parametrize("n_values,n_heavy,S", [(1500, 4, 256), (300, 0, 128), (40, 4, 256), (2600, 7, 128)])
def test_device_framing_equals_reference(n_values, n_heavy, S) -> None:
    """Regular blocks (one step per flow, no guard): the device framer (here on CPU tensors:
    the same torch ops) == the reference framing, byte for byte, in send order."""
    from dsart.x2.corr.tx2x import frame_regular_device
    rng = np.random.default_rng(n_values)
    fmt = _fmt(n_values=n_values, n_heavy=n_heavy)
    n_d = 5
    ch, cm = _codes(rng, fmt, n_d, S)
    sig = rng.uniform(0.5, 2.0, size=n_d).astype(np.float32)
    sig[2] = 0.0                                               # a flow still in warm-up
    sm = (np.float32(fmt.s_main) * sig).astype(np.float32)
    sh = (np.float32(fmt.s_heavy) * sig).astype(np.float32)
    pay = np.stack([f2.pack_samples(ch[i], cm[i], fmt.bits_main) for i in range(n_d)])
    dms = [1, 9, 17, 25, 33]
    fb = frame_regular_device(torch.from_numpy(pay), sm, sh, fmt=fmt, dms=dms, seq_first=3 * S,
                              t_int_factor=32, extra_flags=f2.FLAG_RFI_WARMING_UP)
    want = _reference(fmt, ch, cm, np.repeat(sm[:, None], S, 1), np.repeat(sh[:, None], S, 1),
                      np.zeros((n_d, S), bool), dms, 3 * S, 32, extra=f2.FLAG_RFI_WARMING_UP)
    assert fb.datagrams() == want


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
def test_framed_tx_device_path_gpu() -> None:
    """CUDA packets: regular streams framed on the device, guarded ones on the host; every
    datagram equal to the reference."""
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    res, voff, nh, _ = _packed_block(rule, n_d=3, surge=slice(30, 70))     # stream 1 only
    fmts = _formats(rule, voff, nh)
    assert not res[0]["guarded"].any() and res[1]["guarded"].any()

    class _Keep(DrainSink):
        def __init__(self):
            super().__init__()
            self.dgs = []

        def submit(self, d, **kw):
            buf, offs, lens = d
            self.dgs += [buf[int(o):int(o) + int(n)].tobytes() for o, n in zip(offs, lens)]
            return super().submit(d)

    sink = _Keep()
    tx = FramedTx2x(sink, fmts)
    dms = np.array([0, 8, 16])
    for blk in range(2):                                         # buffers reused across blocks
        gpu = [{k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in r.items()} for r in res]
        sink.dgs.clear()
        tx.transmit_tier(block_n=blk, tier="A", dms=dms, t0_native=(9 + blk) * 4096, T_native=16,
                         stream_ids=[2, 3], packets=gpu, rfi_warming_up=False)
        want = []
        for s_, r in enumerate(res):
            want += _reference(fmts[s_], r["codes_heavy"].numpy(), r["codes_main"].numpy(),
                               r["step_main"].numpy(), r["step_heavy"].numpy(), r["guarded"].numpy(),
                               dms, (9 + blk) * 4096 // 16, 16)
        assert sink.dgs == want
    assert tx.n_device_framed >= 1                               # stream 0 has no guarded row


def _packed_block(rule, n_d=3, S=32, seed=0, surge=slice(None)):
    rng = np.random.default_rng(seed)
    voff, nh = [0, 30, 70], [2, 3]
    n = rng.integers(1, 500, size=70)
    pk = RowRulePacker(n_d=n_d, value_offsets=voff, n_contrib=n, n_heavy=nh, rule=rule)
    for _ in range(int(rule.warmup_blocks) + 1):
        x = ((rng.normal(size=(n_d, S, 70)) + 1j * rng.normal(size=(n_d, S, 70))) * np.sqrt(n))
        x[1, 5, surge] *= 6.0                              # one guarded row
        res = pk.pack(torch.from_numpy(x.astype(np.complex64)))
    return res, voff, nh, n


def _formats(rule, voff, nh, g=1, n_sub=2):
    return [StreamFormat(stream_id=g * n_sub + s, n_values=voff[s + 1] - voff[s], n_heavy=nh[s],
                         bits_main=rule.bits_main, n_sub=n_sub, n_grid=256,
                         value_kind=f2.VALUE_KIND_CELLS, pattern_id=100 + s,
                         s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32))
            for s in range(n_sub)]


def test_framed_tx_drain_counts_and_heartbeat_keys() -> None:
    from dsart.transport.send_mmsg import TxRateMeter
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    res, voff, nh, _ = _packed_block(rule)
    fmts = _formats(rule, voff, nh)
    tx = FramedTx2x(DrainSink(), fmts)
    meter = TxRateMeter()
    assert cfi._x2_tx_mon(meter, tx) == {}                         # first call primes the meter
    dms = np.array([0, 8, 16])
    n = tx.transmit_tier(block_n=9, tier="A", dms=dms, t0_native=9 * 4096, T_native=16,
                         stream_ids=[2, 3], packets=res, rfi_warming_up=True)
    want = []
    for s, r in enumerate(res):
        bc_h, bc_m = r["codes_heavy"].numpy(), r["codes_main"].numpy()
        want += _reference(fmts[s], bc_h, bc_m, r["step_main"].numpy(), r["step_heavy"].numpy(),
                           r["guarded"].numpy(), dms, 9 * 4096 // 16, 16, extra=f2.FLAG_RFI_WARMING_UP)
    assert n == len(want) == tx.n_datagrams and tx.n_bytes == sum(map(len, want))
    assert any(f2.unpack_header(d).row_guard for d in want)
    c = tx.counters()
    assert c["per_dest"][0]["sent"] == n and c["per_dest"][0]["bytes"] == tx.n_bytes
    time.sleep(0.01)
    mon = cfi._x2_tx_mon(meter, tx)
    assert set(mon) == {"tx_gbps", "tx_datagrams_per_s", "tx_drops_per_s", "tx_queue_depth_max"}
    assert mon["tx_datagrams_per_s"] > 0 and mon["tx_drops_per_s"] == 0


def test_routes_and_budget_follow_the_owner_map() -> None:
    owner_half = np.arange(47) % 8
    routes = corr_routes(owner_half, range(47), n_dest=4)
    assert all(routes[d] == (d % 8) // 2 for d in range(47))
    with pytest.raises(ValueError):
        corr_routes(owner_half, range(47), n_dest=3)
    fmts = _formats(quant.RowRule.for_format(8, 2), [0, 3000, 5800], [20, 25])
    nbytes, ndg = block_tx_budget(fmts, {0: 16, 1: 32}, {0: 0, 1: 3}, 4)
    for f in fmts:
        assert f.max_k * f.bytes_per_sample <= f2.MAX_PAYLOAD_BYTES
    assert ndg[0] == sum(-(-256 // f.max_k) for f in fmts) and ndg[3] == sum(-(-128 // f.max_k) for f in fmts)
    assert nbytes[0] == ndg[0] * 72 + 256 * sum(f.bytes_per_sample for f in fmts)
    assert nbytes[3] == ndg[3] * 72 + 128 * sum(f.bytes_per_sample for f in fmts)
    assert nbytes[1] == nbytes[2] == ndg[1] == ndg[2] == 0


def _have_sender() -> bool:
    try:
        from dsart.transport.send_mmsg import _lib
        _lib()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _have_sender(), reason="_send_mmsg not built (harness -b)")
def test_mmsg_sender_routes_every_dm_to_its_owner_node() -> None:
    """Four loopback 'search nodes' on port base + chgroup: every datagram of
    DM d arrives at node owner_half[d] // 2, byte-equal and in row order."""
    from dsart.x2.corr.tx2x import build_mmsg_sender
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    res, voff, nh, _ = _packed_block(rule, n_d=4)
    g = 1
    fmts = _formats(rule, voff, nh, g=g)
    hosts = ["127.0.0.1", "127.0.0.2", "127.0.0.3", "127.0.0.4"]
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    base = probe.getsockname()[1]
    probe.close()
    socks = []
    for h in hosts:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        s.bind((h, base))
        s.settimeout(2.0)
        socks.append(s)
    owner_half = np.arange(47) % 8
    dms = np.array([2, 5, 9, 14])                       # nodes 1, 2, 0, 3
    sender = build_mmsg_sender(formats=fmts, dm_t_native={int(d): 16 for d in dms},
                               owner_half=owner_half, hosts=hosts, port=base, start=True)
    tx = FramedTx2x(sender, fmts)
    try:
        n = tx.transmit_tier(block_n=3, tier="A", dms=dms, t0_native=3 * 4096, T_native=16,
                             stream_ids=[g * 2, g * 2 + 1], packets=res, rfi_warming_up=False)
        assert sender.flush(5.0)
        got = {i: [] for i in range(4)}
        for i, s in enumerate(socks):
            while True:
                try:
                    got[i].append(s.recv(9100))
                except socket.timeout:
                    break
        want = {i: [] for i in range(4)}
        for s_, r in enumerate(res):
            for j, d in enumerate(dms):
                want[(int(d) % 8) // 2] += _reference(
                    fmts[s_], r["codes_heavy"].numpy()[j:j + 1], r["codes_main"].numpy()[j:j + 1],
                    r["step_main"].numpy()[j:j + 1], r["step_heavy"].numpy()[j:j + 1],
                    r["guarded"].numpy()[j:j + 1], [d], 3 * 4096 // 16, 16)
        assert n == sum(len(v) for v in want.values())
        for i in range(4):
            assert got[i] == want[i], i                  # per-destination FIFO = framer order
        c = sender.counters()
        assert c["unrouted"] == 0 and all(p["full_drops"] == 0 for p in c["per_dest"])
    finally:
        tx.close()
        for s in socks:
            s.close()


def test_x2_tx_modes_are_2x_only_and_v2_needs_four_hosts(monkeypatch) -> None:
    class _Orch:
        def stream_formats(self):
            return _formats(quant.RowRule.for_format(8, 2), [0, 30, 70], [2, 3])

        def tier_dms(self):
            return {0: 16, 1: 16}

    tx = cfi._build_x2_tx(_Orch(), owner_half=np.arange(47) % 8, mode="drain", hosts="",
                          port_base=6625, chgroup=1)
    assert isinstance(tx.sink, DrainSink)
    with pytest.raises(ValueError, match="search-node"):
        cfi._build_x2_tx(_Orch(), owner_half=np.arange(47) % 8, mode="v2", hosts="10.41.0.205",
                         port_base=6625, chgroup=1)
    cfg = cfi.FastIntegrationConfig(chgroup=1, obs_dec_rad=1.25)
    for opoint, mode in (("1x", "v2"), ("1x", "drain"), ("2x-gridded-ns2-cint4", "prod"),
                         ("2x-gridded-ns2-cint4", "chunk8")):
        with pytest.raises(ValueError, match="transport-tx-mode"):
            cfi.run(0, "/tmp/dsart-x2-ws-corr-never", torch.device("cpu"), cfg, opoint=opoint,
                    transport_tx_mode=mode)

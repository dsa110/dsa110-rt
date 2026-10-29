"""x2 WS-C: corr side end to end on the CPU, through WS-T's ring v4.

    corr2x reference blocks (float64 stage 1 + decimation + static sky, from
    random GEMM output on WS-P's fixture value map)
      -> RowRulePacker (the 2x corr packer)
      -> FramedTx2x -> frame-v2 datagrams (the corr TX stage; its sink here
         ingests into an in-process RxRingV4Writer -- no sockets)
      -> RxRingV4Reader.copy_window (codes, steps, validity)
      -> unpack_samples + quant.decode_row x sigma_row x sqrt(n)

for gridded n_sub 2 (cint4) and per-baseline n_sub 4 (cint5), one tier-A and
one tier-B coarse DM, 12 corr blocks: the 8 warm-up blocks arrive as PRESENT
rows of NO_DATA codes (decode 0, step 0); afterwards every decoded value is
the reference value within the quantisation model (half a step per
component, or saturated with the right sign), the ring's codes are
bit-identical to ``quant.RowQuantiser`` on the same values, the per-row steps
travel, a surge row arrives as ROW_GUARD at its own scale, and flagged
(exactly-zero) values decode to exactly 0. Values are a subset of each
stream (the lightest heavy values + light main values) so the float64
reference stays fast; the ring is told the subset sizes.
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import torch

from dsart.transport import prod_frame2 as f2
from dsart.x2 import quant
from dsart.x2.corr.framer2x import StreamFormat
from dsart.x2.corr.packer2x import RowRulePacker
from dsart.x2.corr.plan2x import proposed_corr_plan
from dsart.x2.corr.tx2x import FramedTx2x
from dsart.x2.ref import corr2x_ref as ref

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NQ = 128                       # q1 samples per (test) corr block: S = 64 tier A, 32 tier B
N_BLOCKS = 12                  # pushes; 11 emitted, the first 8 are the row-rule warm-up
SURGE_BLOCK = 9


def _rr():
    try:
        from dsart.transport import recv_ring as rr
        rr._rx4_lib()
        return rr
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ring v4 C extension not built (harness -b): {exc!r}")


class _RingSink:
    """TX sink that ingests every datagram into a ring v4 writer."""

    def __init__(self, writer, rr) -> None:
        self.w, self.rr = writer, rr
        self.n = self.bad = 0
        self.flags = []

    def submit(self, dgs, **_kw) -> int:
        buf, offs, lens = dgs
        ok = 0
        for o, n in zip(offs.tolist(), lens.tolist()):
            d = buf[o:o + n].tobytes()
            self.flags.append(f2.unpack_header(d).flags)
            if self.w.ingest(d) == self.rr.RX4_OK:
                ok += 1
            else:
                self.bad += 1
        self.n += ok
        return ok

    def counters(self) -> dict:
        return {"submitted": self.n, "accepted": self.n, "unrouted": 0, "per_dest": []}

    def close(self) -> None:
        pass


def _fixture_vm(kind, n_sub, bits_main):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_2x
    z = np.load(os.path.join(ROOT, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    e, n = z["antpos_e"], z["antpos_n"]
    mask = core_baseline_mask_from_station_numbers(z["station"])
    vm = build_value_map_for_opoint(opoint_2x(kind, n_sub, bits_main), e, n,
                                    is_core_baseline_mask=mask, dec_deg=71.63)
    return vm, e, n


def _select(st, rng, n_h=2, n_m=18):
    """The lightest heavy values + light main values (random among the 4 n_m
    lightest), sorted, heavy first."""
    nc, nh = np.asarray(st.n_contrib), int(st.n_heavy)
    heavy = np.sort(np.argsort(nc[:nh], kind="stable")[:n_h])
    light = np.argsort(nc[nh:], kind="stable")[: 4 * n_m] + nh
    main = np.sort(rng.choice(light, size=n_m, replace=False))
    return np.concatenate([heavy, main]), int(heavy.size)


@pytest.mark.parametrize("kind,n_sub,bits_main", [("gridded", 2, 8), ("baselines", 4, 10)])
def test_reference_to_ring_v4_decodes_within_the_quantisation_model(kind, n_sub, bits_main) -> None:
    rr = _rr()
    rng = np.random.default_rng(n_sub)
    vm, e, n = _fixture_vm(kind, n_sub, bits_main)
    plan = proposed_corr_plan(n_sub)
    rule = quant.RowRule.for_format(bits_main, n_sub)              # Phase-0 rules, warm-up 8
    assert rule.warmup_blocks == 8 and N_BLOCKS - 1 > rule.warmup_blocks
    per = plan.per
    # chgroup + one DM per tier whose stage-1 shifts fit the 2-block window
    G, dms = None, None
    for g in range(15, -1, -1):
        pick = []
        for ti in range(2):
            s1 = plan.stage1_bins(ti)[g]
            ok = [c for c in range(s1.shape[1]) if int(s1[:, c].max()) <= NQ - 4]
            if ok:
                pick.append(plan.tiers[ti].d0 + ok[-1])
        if len(pick) == 2:
            G, dms = g, pick
            break
    assert G is not None
    sids = [G * n_sub + s for s in range(n_sub)]

    # ---- value subsets and the reference geometry on the rows they need
    sel, nh, dest_full, conj_full = [], [], [], []
    for sid in sids:
        st = vm.streams[sid]
        v, h = _select(st, rng)
        sel.append(v)
        nh.append(h)
        dest, conj = ref.value_dest_from_vm(vm, sid, e, n)
        newi = np.full(st.n_values, -1, np.int64)
        newi[v] = np.arange(v.size)
        dest_full.append(np.where(dest >= 0, newi[np.maximum(dest, 0)], -1))
        conj_full.append(conj)
    rows = np.flatnonzero(np.any([np.any(d >= 0, axis=1) for d in dest_full], axis=0))
    dests = [d[rows] for d in dest_full]
    conjs = [c[rows] for c in conj_full]
    R = rows.size
    nvs = [v.size for v in sel]
    voff = np.concatenate([[0], np.cumsum(nvs)])
    n_contrib = np.concatenate([np.asarray(vm.streams[sid].n_contrib)[v] for sid, v in zip(sids, sel)])
    for s in range(n_sub):                                          # geometry == the map's counts
        assert np.array_equal(np.bincount(dests[s][dests[s] >= 0], minlength=nvs[s]),
                              np.asarray(vm.streams[sids[s]].n_contrib)[sel[s]])
    flag_rows = rng.choice(R, size=max(1, R // 20), replace=False)   # flagged baselines (zeros)

    # ---- packers (one per tier, as Corr2xPipeline), TX stage, ring v4
    tiers = [plan.tier_of(d) for d in dms]
    packers = [RowRulePacker(n_d=1, value_offsets=voff, n_contrib=n_contrib, n_heavy=nh, rule=rule)
               for _ in dms]
    fmts = [StreamFormat.from_value_map(vm, sid, rule, n_values=nvs[s], n_heavy=nh[s])
            for s, sid in enumerate(sids)]
    t_nat = [plan.tiers[ti].T_native for ti, _ in tiers]
    n_values_all = tuple(nvs[s % n_sub] for s in range(vm.n_streams))
    n_heavy_all = tuple(nh[s % n_sub] for s in range(vm.n_streams))
    spec = rr.RingV4Spec(
        dms=tuple(dms), t_int_factor=tuple(t_nat),
        t_buf=tuple(2 * N_BLOCKS * NQ * 8 // t for t in t_nat), reorder_rows=8,
        n_stream=vm.n_streams, n_sub=n_sub, bits_main=rule.bits_main,
        value_kind=0 if kind == "gridded" else 1, n_grid=vm.n_grid,
        s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32),
        stream_n_values=n_values_all, stream_n_heavy=n_heavy_all, min_retention_s=0.0)
    name = f"/dsart-x2-corr-e2e-{uuid.uuid4().hex[:10]}"
    w = rr.RxRingV4Writer.create(name, spec)
    rd = rr.RxRingV4Reader.open(name)
    try:
        for sid in sids:
            w.set_expected_pattern_id(sid, int(vm.stream_pattern_ids[sid]))
        sink = _RingSink(w, rr)
        tx = FramedTx2x(sink, fmts)
        quants = {(d, s): quant.RowQuantiser(rule=rule, n_contrib=n_contrib[voff[s]:voff[s + 1]],
                                             n_heavy=nh[s]) for d in dms for s in range(n_sub)}
        sky = ref.StaticSkyRef(window_blocks=8, warmup_cubes=8)
        b0 = 1000
        prev = None
        sent = []                                    # (block, d, t0_native, S, x (S, V) c64)
        for k in range(N_BLOCKS):
            V = ((rng.normal(size=(NQ, plan.nsum, R)) + 1j * rng.normal(size=(NQ, plan.nsum, R)))
                 .astype(np.complex64))
            V[:, :, flag_rows] = 0
            if prev is None:
                prev = V
                continue
            joined = np.concatenate([prev, V], axis=0)
            blk = b0 + k - 1
            for j, (d, (ti, c)) in enumerate(zip(dms, tiers)):
                tier = plan.tiers[ti]
                parts = []
                for s in range(n_sub):
                    ks = np.arange(s * per, (s + 1) * per)
                    x = ref.stage1_values(joined, dests[s], conjs[s], ks=ks,
                                          s1=plan.stage1_bins(ti)[G][ks, c], n_values=nvs[s],
                                          t0=0, n_out=NQ)
                    parts.append(sky.apply((d, s), ref.decimate(x, tier.decim)))
                xt = np.concatenate(parts, axis=1).astype(np.complex64)      # (S, V)
                if k == SURGE_BLOCK:
                    xt[5] *= 8.0                                             # broadband surge row
                t0 = blk * NQ * tier.q1_native
                res = packers[j].pack(torch.from_numpy(xt[None]))
                tx.transmit_tier(block_n=blk, tier=tier.name, dms=np.array([d]), t0_native=t0,
                                 T_native=tier.T_native, stream_ids=sids, packets=res,
                                 rfi_warming_up=False)
                sent.append((k - 1, d, t0, tier.T_native, xt))
            prev = V
        assert sink.bad == 0 and sink.n == tx.n_datagrams > 0
        assert any(fl & f2.FLAG_ROW_GUARD for fl in sink.flags)
        c = rd.counters()["writer"]
        assert c["pattern_mismatch"] == 0 and c["format_mismatch"] == 0 and c["ok"] == sink.n

        # ---- read back, decode, compare
        n_checked = n_guard = n_sat = 0
        for e_idx, d, t0, T, xt in sent:
            S = xt.shape[0]
            win = rd.copy_window(d, t0 // T, S, with_steps=True)
            for s, sid in enumerate(sids):
                assert (win.validity[sid] & rr.RV_PRESENT).all()
                bc = quants[(d, s)].process_block(xt[:, voff[s]:voff[s + 1]])
                bps = fmts[s].bytes_per_sample
                ch, cm = f2.unpack_samples(np.ascontiguousarray(win.view[sid, :, :bps]), S,
                                           nvs[s], nh[s], rule.bits_main, 16 if nh[s] else 0)
                np.testing.assert_array_equal(ch, bc.codes_heavy)              # bit-exact chain
                np.testing.assert_array_equal(cm, bc.codes_main)
                step_m = win.steps[sid, :, 0]
                assert step_m.tobytes() == bc.step_main.tobytes()
                assert win.steps[sid, :, 1].tobytes() == bc.step_heavy.tobytes()
                guard = (win.validity[sid] & rr.RV_ROW_GUARD) != 0
                np.testing.assert_array_equal(guard, bc.guarded)
                v = quant.decode_row(ch, cm, rule)                             # sigma_hat units
                if e_idx < rule.warmup_blocks:
                    assert not np.any(v) and not np.any(step_m)
                    continue
                n_v = n_contrib[voff[s]:voff[s + 1]]
                sig_row = step_m / rule.s_main_f32                             # per row
                xhat = quant.natural_weight(v * sig_row[:, None], n_v)
                y = xt[:, voff[s]:voff[s + 1]]
                sv = np.concatenate([np.full(nh[s], rule.s_heavy_f32), np.full(nvs[s] - nh[s],
                                                                            rule.s_main_f32)])
                half = 0.5 * sv[None, :] * sig_row[:, None] * np.sqrt(n_v)[None, :]
                cmax = np.concatenate([np.full(nh[s], 127), np.full(nvs[s] - nh[s],
                                                                    rule.main_range[1])])
                codes = np.concatenate([ch, cm], axis=1)                       # (S, V, 2)
                for comp, (a, b) in enumerate(((xhat.real, y.real), (xhat.imag, y.imag))):
                    sat = np.abs(codes[..., comp]) == cmax[None, :]
                    tol = half * (1 + 1e-5) + 1e-6 * np.abs(b)
                    assert np.all(np.abs(a - b)[~sat] <= tol[~sat]), (kind, d, s, e_idx, comp)
                    assert np.all(np.sign(a[sat]) == np.sign(b[sat]))
                    assert np.all(np.abs(b[sat]) >= (np.abs(a[sat]) - half[sat]) * (1 - 1e-5))
                    n_sat += int(sat.sum())
                zero = np.all(y == 0, axis=0)                                  # flagged values
                assert not np.any(v[:, zero])
                n_guard += int(guard.sum())
                n_checked += y.size
        assert n_checked > 0 and n_guard >= 2 * n_sub
    finally:
        rd.close()
        w.close()
        w.unlink()

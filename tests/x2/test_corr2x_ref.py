"""x2 WS-C deliverable 2: the numpy reference of the 2x corr chain.

* stage-1 tables and coarse DMs == the approved op-point oracle
  (``docs/timeres_study/snr2x/schemes.py::proposed``), both n_sub;
* a dispersed impulse lands on the predicted decimated sample for EVERY
  coarse DM of both tiers (vis domain, three chgroups), and end to end
  from raw int4 bytes through unpack / cal / float64 GEMM / gridding;
* the interim value map reproduces today's sub-band patterns exactly
  (SPEC §3 1x-compat) and builds consistent redundancy classes;
* row rule, bit packing and frame v2 round-trip; the v1 cint8 rule equals
  ``TransportTx._encode_rows`` byte for byte.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from dsart.common.constants import NANTS, NATIVE_SAMPLE_US, NCHAN_PER_CHGROUP
from dsart.x2.corr.plan2x import proposed_corr_plan, stage1_a1_us
from dsart.x2.corr.value_map import build_value_map, row_csr_for_values
from dsart.x2.ref import corr2x_ref as ref

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _oracle(n_sub: int):
    for sub in ("snr2x", "model"):
        p = os.path.join(ROOT, "docs", "timeres_study", sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import schemes  # noqa: PLC0415
    return schemes.proposed(q1=schemes.T4, n_sub=n_sub)


# ---------------------------------------------------------------------------
# plan vs oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_sub,counts", [(2, (32, 15)), (4, (16, 8))])
def test_stage1_tables_equal_oracle(n_sub, counts) -> None:
    plan = proposed_corr_plan(n_sub)
    orc = _oracle(n_sub)
    assert tuple(t.n_coarse for t in plan.tiers) == counts
    assert np.array_equal(plan.a1_us, orc.a1)
    for ti, (t, ot) in enumerate(zip(plan.tiers, orc.tiers)):
        assert np.array_equal(t.coarse_dm, ot["coarse"])
        assert t.T_us == ot["q2"] and t.q1_native * NATIVE_SAMPLE_US == ot["q1"]
        tab = plan.stage1_bins(ti)
        for c in range(t.n_coarse):
            assert np.array_equal(tab[:, :, c], orc.stage1_bins(ti, c).astype(np.int64)), (ti, c)
    assert plan.n_coarse_total == sum(counts)
    assert [plan.owner_half(d) for d in range(10)] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


def test_tiered_plan_adapter_when_available() -> None:
    plan_mod = pytest.importorskip("dsart.x2.plan", reason="WS-P TieredDmPlan not merged yet")
    builder = getattr(plan_mod, "proposed_tiered_plan", None) or getattr(plan_mod, "build_proposed", None)
    if builder is None:
        pytest.skip("TieredDmPlan has no proposed() builder under the expected names")
    from dsart.x2.corr.plan2x import CorrPlan2x
    for n_sub in (2, 4):
        mine = proposed_corr_plan(n_sub)
        theirs = CorrPlan2x.from_tiered_plan(builder(n_sub=n_sub))
        assert np.array_equal(mine.stage1_bins_global(), theirs.stage1_bins_global())


# ---------------------------------------------------------------------------
# dispersed impulse -> predicted decimated sample (vis domain, every DM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_sub", [2, 4])
def test_impulse_lands_on_predicted_sample_every_dm(n_sub) -> None:
    plan = proposed_corr_plan(n_sub)
    a1 = stage1_a1_us(plan.csf, n_sub)
    R = 3
    for g in (0, 7, 15):
        for ti, tier in enumerate(plan.tiers):
            s1_tab = plan.stage1_bins(ti)[g]
            D = tier.decim
            for c in range(tier.n_coarse):
                dm = tier.coarse_dm[c]
                nat = np.rint(a1[g] * dm / NATIVE_SAMPLE_US).astype(np.int64)
                m = 3
                t0_q = D * m + 1                      # not on a decimation boundary
                t0 = 8 * t0_q
                n_q = t0_q + int(nat.max()) // 8 + 4 * D + 8
                n_out = (t0_q // D + 2) * D
                for s in range(n_sub):
                    ks = np.arange(s * plan.per, (s + 1) * plan.per)
                    V = np.zeros((n_q, plan.nsum, R), np.complex128)
                    for k in ks:
                        V[(t0 + nat[k]) // 8, k, :] += 1.0
                    dest = np.zeros((R, plan.per), np.int64)
                    conj = np.zeros((R, plan.per), bool)
                    x = ref.stage1_values(V, dest, conj, ks=ks, s1=s1_tab[ks, c],
                                          n_values=1, t0=0, n_out=n_out)
                    xd = ref.decimate(x, D)[:, 0].real
                    j = int(np.argmax(xd))
                    assert j == t0_q // D, (g, tier.name, c, s, j)
                    assert xd[j] == R * plan.per and xd.sum() == R * plan.per


# ---------------------------------------------------------------------------
# end to end from raw int4 bytes (toy array, one DM per tier)
# ---------------------------------------------------------------------------


def _toy_array(n_core: int = 6):
    rng = np.random.default_rng(3)
    e = rng.uniform(-40, 40, n_core)
    n = rng.uniform(-40, 40, n_core)
    return e, n


@pytest.mark.parametrize("kind", ["cells", "baselines"])
def test_voltage_impulse_end_to_end(kind) -> None:
    n_sub, g = 2, 15
    plan = proposed_corr_plan(n_sub)
    a1 = stage1_a1_us(plan.csf, n_sub)
    e, n = _toy_array()
    nants = e.size
    vm = build_value_map(kind, chgroup=g, antpos_e=e, antpos_n=n, is_core_ant=np.ones(nants, bool),
                         csf=plan.csf, n_sub=n_sub, n_grid=64, cell_lambda=16.0)
    dests = ref.value_dest(vm, e, n)
    for ti, tier in enumerate(plan.tiers):
        c = tier.n_coarse - 1
        dm = tier.coarse_dm[c]
        D = tier.decim
        t0_q = D * 5 + 1
        t0 = 8 * t0_q
        nat_k = np.rint(a1[g] * dm / NATIVE_SAMPLE_US).astype(np.int64)          # (nsum,)
        n_native = 8 * (t0_q + int(nat_k.max()) // 8 + 16)
        n_pkt = n_native // 2
        raw = np.zeros((n_pkt, nants, NCHAN_PER_CHGROUP, 2, 2), np.uint8)
        for ch in range(NCHAN_PER_CHGROUP):
            tn = t0 + nat_k[ch // plan.csf]
            raw[tn // 2, :, ch, tn % 2, :] = 3                                    # re = +3, im = 0
        E = ref.unpack_int4_onwire(raw.reshape(-1), n_packets=n_pkt, nants=nants,
                                   nchan=NCHAN_PER_CHGROUP)
        gains = np.exp(1j * np.zeros((nants, NCHAN_PER_CHGROUP, 2)))
        E = ref.apply_cal(ref.apply_masks(E), gains)
        V = ref.gemm_core(E, vm.rows.a, vm.rows.b, t_int=8, csf=plan.csf)
        # point source at the phase centre: every baseline real, = csf*2pol*(3*0.05)^2
        unit = plan.csf * 2 * (3 * 0.05) ** 2
        s1 = plan.stage1_bins(ti)[g][:, c]
        n_out = (t0_q // D + 2) * D
        for s, sv in enumerate(vm.streams):
            ks = np.arange(s * plan.per, (s + 1) * plan.per)
            dest, conj = dests[s]
            x = ref.stage1_values(V, dest, conj, ks=ks, s1=s1[ks], n_values=sv.n_values,
                                  t0=0, n_out=n_out)
            xd = ref.decimate(x, D)
            tot = xd.sum(axis=1)
            j = int(np.argmax(np.abs(tot)))
            assert j == t0_q // D, (kind, tier.name, s, j)
            np.testing.assert_allclose(tot[j].real, vm.rows.n * plan.per * unit, rtol=1e-12)
            assert abs(tot[j].imag) < 1e-9
            assert np.allclose(np.delete(tot, j), 0.0)
            # every value got exactly its static contribution count
            np.testing.assert_allclose(xd[j].real, sv.n * unit, rtol=1e-12)


# ---------------------------------------------------------------------------
# value maps
# ---------------------------------------------------------------------------


def _synth_antpos(seed: int = 42):
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


@pytest.mark.parametrize("csf,n_sub", [(8, 4), (2, 2)])
def test_cells_map_equals_todays_subband_patterns(csf, n_sub) -> None:
    from dsart.grid import FastVisGridder, build_pattern, cell_lambda_for_pixel_arcsec
    from dsart.grid.subband import build_subband_layout, build_subband_patterns
    from dsart.grid import IMAGE_PIXEL_ARCSEC_TEE
    from dsart.services.corr_fast_integration import _build_core_baseline_mask

    e, n = _synth_antpos()
    is_core = np.zeros(NANTS, bool)
    is_core[:82] = True
    core = _build_core_baseline_mask(n_core=82)
    cl = cell_lambda_for_pixel_arcsec(IMAGE_PIXEL_ARCSEC_TEE, 256)
    g = 9
    pats = build_subband_patterns(e, n, chgroup=g, n_sub=n_sub, dec_deg=71.6, n_grid=256,
                                  kernel_support=1, chan_sum_factor=csf, cell_lambda=cl,
                                  is_core_baseline_mask=core)
    whole = build_pattern(e, n, chgroup=g, dec_deg=71.6, n_grid=256, kernel_support=1,
                          chan_sum_factor=csf, cell_lambda=cl, is_core_baseline_mask=core)
    gr = FastVisGridder.from_pattern(whole, e, n, is_core_baseline_mask=core, device="cpu")
    lay = build_subband_layout(pats, gr, device="cpu")
    vm = build_value_map("cells", chgroup=g, antpos_e=e, antpos_n=n, is_core_ant=is_core,
                         csf=csf, n_sub=n_sub, n_grid=256, cell_lambda=cl)
    nsum = NCHAN_PER_CHGROUP // csf
    cim = lay.cell_index_map.numpy().reshape(-1, nsum)                  # (NBASE, nsum)
    for s, (p, sv) in enumerate(zip(pats, vm.streams)):
        assert np.array_equal(p.ix_row.astype(np.int64), sv.cell_row)
        assert np.array_equal(p.ix_col.astype(np.int64), sv.cell_col)
        lo = int(lay.offsets[s])
        flat = cim.reshape(-1)
        sel = flat[(flat >= lo) & (flat < lo + p.n_filled)] - lo
        assert np.array_equal(np.bincount(sel, minlength=p.n_filled), sv.n)
        # same (baseline, channel) -> cell assignment
        for v in (0, sv.n_values // 2, sv.n_values - 1):
            a, b = sv.src_ptr[v], sv.src_ptr[v + 1]
            bls = vm.rows.bls[sv.src_row[a:b]]
            assert np.all(cim[bls, sv.src_k[a:b]] == lo + v)
    csr = row_csr_for_values(vm)
    assert csr["ptr"][-1] == csr["row"].size == int(csr["n"].sum())
    assert int(csr["n"].sum()) == vm.rows.n * nsum - int(np.sum(cim[vm.rows.bls] >= lay.n_filled_total))


def _redundant_array():
    """DSA-like Tee: 30 E-W at 5.75 m, 20 N-S at 5.75 m (sharing the
    corner), heights 0 -- lots of exact redundancy."""
    ew = [(5.75 * i, 0.0) for i in range(30)]
    ns = [(0.0, 5.75 * j) for j in range(1, 21)]
    pts = np.array(ew + ns)
    return pts[:, 0], pts[:, 1]


def test_baselines_map_redundancy_and_counts() -> None:
    e, n = _redundant_array()
    nants = e.size
    vm = build_value_map("baselines", chgroup=4, antpos_e=e, antpos_n=n,
                         is_core_ant=np.ones(nants, bool), csf=2, n_sub=2, n_grid=256,
                         cell_lambda=2.0, heavy_threshold=40 * 96)
    R = vm.rows.n
    assert R == nants * (nants - 1) // 2
    assert vm.n_class < R // 3                     # strongly redundant
    d = np.stack([e[vm.rows.a] - e[vm.rows.b], n[vm.rows.a] - n[vm.rows.b]], 1)
    for c in range(vm.n_class):
        mem = np.flatnonzero(vm.class_of_row == c)
        sgn = np.where(vm.conj_of_row[mem], -1.0, 1.0)[:, None]
        assert np.allclose(d[mem] * sgn, d[mem[0]], atol=0.05)
        assert not vm.conj_of_row[mem[0]]
    for sv in vm.streams:
        assert int(sv.n.sum()) == R * vm.per
        assert np.all(sv.n[: sv.n_heavy] >= 40 * 96) and np.all(sv.n[sv.n_heavy:] < 40 * 96)
        assert sv.n_heavy > 0
    csr = row_csr_for_values(vm, rows="classes")
    assert csr["row"].max() < vm.n_class and not csr["conj"].any()


# ---------------------------------------------------------------------------
# quantisation, packing, frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 5])
def test_row_rule_roundtrip(bits) -> None:
    rng = np.random.default_rng(11)
    n = rng.integers(1, 3000, size=200)
    n[:5] = 5000
    x = (rng.normal(size=(64, 200)) + 1j * rng.normal(size=(64, 200))) * np.sqrt(n)[None, :]
    st = ref.RowRuleState()
    sig = st.step("k", x / np.sqrt(n)[None, :])
    assert 0.9 < sig < 1.1
    h, m, sm, sh = ref.quantise_row_rule(x, n, 5, sig, bits=bits)
    assert h.dtype == np.int8 and m.min() >= -(1 << (bits - 1)) and m.max() <= (1 << (bits - 1)) - 1
    dec = ref.decode_row_rule(h, m, n, 5, bits=bits) * sig      # back to data units / sqrt(n) * sqrt(n)
    y = x / np.sqrt(n)[None, :]
    yd = dec / np.sqrt(n)[None, :]
    half = (1 << (bits - 1)) * sm
    main_in = (np.abs(y.real[:, 5:]) < half) & (np.abs(y.imag[:, 5:]) < half)
    err = yd - y
    assert np.all(np.abs(err.real[:, 5:])[main_in] <= sm / 2 + 1e-12)
    assert np.all(np.abs(err.imag[:, 5:])[main_in] <= sm / 2 + 1e-12)
    assert np.all(np.abs(err.real[:, :5]) <= sh / 2 + 1e-12)
    assert main_in.mean() > 0.95
    # EMA moves towards a new level
    st2 = ref.RowRuleState(tau_s=1.0)
    st2.step("k", np.ones((10, 10)) * (1 + 1j))
    s_before = st2.sigma["k"]
    st2.step("k", 3 * (rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))))
    assert st2.sigma["k"] != s_before


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_bit_packing_roundtrip(bits) -> None:
    rng = np.random.default_rng(bits)
    c = rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=777)
    buf = ref.pack_bits_le(c, bits)
    assert len(buf) == math.ceil(777 * bits / 8)
    assert np.array_equal(ref.unpack_bits_le(buf, bits, 777), c)
    # little-endian bit order: first code in the low bits of byte 0
    assert ref.pack_bits_le(np.array([1, 0]), 4) == bytes([0x01])
    assert ref.pack_bits_le(np.array([0, 1]), 4) == bytes([0x10])


def test_frame2_pack_roundtrip() -> None:
    rng = np.random.default_rng(5)
    n_t, n_h, n_m = 300, 7, 900
    h = rng.integers(-128, 128, size=(n_t, n_h, 2)).astype(np.int8)
    m = rng.integers(-8, 8, size=(n_t, n_m, 2))
    frames = ref.pack_stream_block(h, m, bits_main=8, t0_native=4096 * 1000, T_native=16,
                                   stream_id=37, dm_idx=12, n_sub=2, n_grid=256, value_kind=0,
                                   pattern_id=(1 << 63) + 5, step_main=0.3, step_heavy=0.1)
    sb = ref.sample_bytes(n_h + n_m, n_h, 8)
    k = 8900 // sb
    assert len(frames) == math.ceil(n_t / k)
    t = 0
    for f in frames:
        hd, pl = ref.unpack_frame2(f)
        assert hd.magic == ref.MAGIC and hd.version == 2 and len(f) == 72 + hd.payload_bytes
        assert hd.payload_bytes <= 8900 and hd.flags & ref.FLAG_HEAVY_PRESENT
        assert hd.t0_native == 4096 * 1000 + 16 * t and hd.seq_first == hd.t0_native // 16
        for i in range(hd.k):
            row = pl[i * sb:(i + 1) * sb]
            assert np.array_equal(np.frombuffer(row[: 2 * n_h], np.int8).reshape(n_h, 2), h[t + i])
            got = ref.unpack_bits_le(row[2 * n_h:], 4, 2 * n_m).reshape(n_m, 2)
            assert np.array_equal(got, m[t + i])
        t += hd.k
    assert t == n_t


def test_v1_cint8_equals_transport_tx() -> None:
    from dsart.transport.prod_frame import BITS_CINT8_COMPLEX
    from dsart.transport.tx import TransportTx
    rng = np.random.default_rng(9)
    rows = (rng.normal(size=(33, 517)) + 1j * rng.normal(size=(33, 517))).astype(np.complex64) * 3.7
    rows[4] = 0
    q, sc = ref.quantise_v1_rows(rows)
    q2, sc2 = TransportTx._encode_rows(rows, BITS_CINT8_COMPLEX)
    assert np.array_equal(sc, sc2) and q.tobytes() == np.asarray(q2).tobytes()


def test_static_sky_ref_semantics() -> None:
    import torch
    from dsart.services.corr_fast_integration import StaticSkyMean
    rng = np.random.default_rng(2)
    a = StaticSkyMean(window_blocks=3, warmup_cubes=2)
    b = ref.StaticSkyRef(window_blocks=3, warmup_cubes=2)
    for _ in range(7):
        x = rng.normal(size=(16, 9)) + 1j * rng.normal(size=(16, 9))
        ya = a.apply(torch.from_numpy(x.astype(np.complex64))).numpy()
        yb = b.apply(0, x)
        np.testing.assert_allclose(ya, yb, rtol=1e-5, atol=1e-5)

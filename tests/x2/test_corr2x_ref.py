"""x2 WS-C deliverable 2: the numpy reference of the 2x corr chain.

* stage-1 tables and coarse DMs == the approved op-point oracle
  (``docs/timeres_study/snr2x/schemes.py::proposed``), both n_sub;
* a dispersed impulse lands on the predicted decimated sample for EVERY
  coarse DM of both tiers (vis domain, three chgroups), and end to end
  from raw int4 bytes through unpack / cal / float64 GEMM / gridding;
* the reference's GEOMETRIC source placement agrees with WS-P's value-map
  CSR (both kinds, production antenna fixture), so the CSR the kernels use
  is checked against an independent derivation;
* bit packing round-trips and the payload layout equals WS-T's frame-v2
  codec; the row rule is WS-T's reference, called through
  ``quantise_flow``; the v1 cint8 rule equals ``TransportTx._encode_rows``
  byte for byte.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz
from dsart.common.constants import NATIVE_SAMPLE_US, NCHAN_PER_CHGROUP
from dsart.x2.corr.plan2x import proposed_corr_plan, stage1_a1_us
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
    """WS-P's plan (through the corr view), the oracle and an independent
    restatement of a1 all give the same stage-1 table."""
    plan = proposed_corr_plan(n_sub)
    orc = _oracle(n_sub)
    a1 = stage1_a1_us(2, n_sub)
    assert tuple(t.n_coarse for t in plan.tiers) == counts
    assert np.array_equal(a1, orc.a1)
    for ti, (t, ot) in enumerate(zip(plan.tiers, orc.tiers)):
        assert np.array_equal(t.coarse_dm, ot["coarse"])
        assert t.T_us == ot["q2"] and t.q1_native * NATIVE_SAMPLE_US == ot["q1"]
        assert t.decim == (2 if t.name == "A" else 4)
        tab = plan.stage1_bins(ti)
        mine = np.rint(np.rint(a1[:, :, None] * t.coarse_dm[None, None, :] / NATIVE_SAMPLE_US)
                       / t.q1_native).astype(np.int64)
        assert np.array_equal(tab, mine)
        for c in range(t.n_coarse):
            assert np.array_equal(tab[:, :, c], orc.stage1_bins(ti, c).astype(np.int64)), (ti, c)
    assert plan.n_coarse_total == sum(counts)
    assert [plan.owner_half(d) for d in range(10)] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


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


def _toy_array():
    """6 antennas on an E-W line at 5.75 m plus one N offset: redundant."""
    e = np.array([0.0, 5.75, 11.5, 17.25, 23.0, 5.75])
    n = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 7.0])
    return e, n


@pytest.mark.parametrize("kind", ["gridded", "baselines"])
def test_voltage_impulse_end_to_end(kind) -> None:
    from dsart.grid.baseline_grid import kept_core_baselines, redundancy_classes
    n_sub, g = 2, 15
    plan = proposed_corr_plan(n_sub)
    a1 = stage1_a1_us(plan.csf, n_sub)
    e, n = _toy_array()
    nants = e.size
    _, ka, kb = kept_core_baselines(None, nants)
    R = ka.size
    per = plan.per
    nus = build_summed_chgroup_freq_table_GHz(plan.csf)[g]
    streams = []
    for s_ in range(n_sub):
        chans = np.arange(s_ * per, (s_ + 1) * per)
        if kind == "gridded":
            lam = 299792458.0 / (nus[chans] * 1e9)
            col = np.rint(-((e[ka] - e[kb])[:, None] / lam) / 16.0).astype(np.int64) + 32
            row = np.rint(-((n[ka] - n[kb])[:, None] / lam) / 16.0).astype(np.int64) + 32
            keys = np.unique((row << 16) | col)
            dest = ref.value_dest_cells(e, n, ka, kb, nus[chans], cell_lambda=16.0, n_grid=64,
                                        cell_row=keys >> 16, cell_col=keys & 0xFFFF)
            conj = np.zeros(dest.shape, bool)
            nv = keys.size
        else:
            rc = redundancy_classes(e, n, is_core_baseline_mask=None)
            assert rc.n_class < R
            dest, conj = ref.value_dest_classes(rc.bl_class, rc.bl_conj, np.arange(rc.n_class), per)
            nv = rc.n_class
        assert (dest >= 0).all()
        streams.append((dest, conj, nv))
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
        V = ref.gemm_core(E, ka, kb, t_int=8, csf=plan.csf)
        # point source at the phase centre: every baseline real, = csf*2pol*(3*0.05)^2
        unit = plan.csf * 2 * (3 * 0.05) ** 2
        s1 = plan.stage1_bins(ti)[g][:, c]
        n_out = (t0_q // D + 2) * D
        for s_, (dest, conj, nv) in enumerate(streams):
            ks = np.arange(s_ * per, (s_ + 1) * per)
            x = ref.stage1_values(V, dest, conj, ks=ks, s1=s1[ks], n_values=nv, t0=0, n_out=n_out)
            xd = ref.decimate(x, D)
            tot = xd.sum(axis=1)
            j = int(np.argmax(np.abs(tot)))
            assert j == t0_q // D, (kind, tier.name, s_, j)
            np.testing.assert_allclose(tot[j].real, R * per * unit, rtol=1e-12)
            assert abs(tot[j].imag) < 1e-9
            assert np.allclose(np.delete(tot, j), 0.0)
            n_v = np.bincount(dest.ravel(), minlength=nv)
            np.testing.assert_allclose(xd[j].real, n_v * unit, rtol=1e-12)


# ---------------------------------------------------------------------------
# reference geometry vs WS-P's value-map CSR (production antenna fixture)
# ---------------------------------------------------------------------------


def _fixture_vm(kind: str, n_sub: int):
    from dsart.grid.baseline_grid import build_value_map_for_opoint
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers
    from dsart.x2.opoint import opoint_2x
    z = np.load(os.path.join(ROOT, "tests", "fixtures", "x2", "dsa110_antpos.npz"))
    e, n = z["antpos_e"], z["antpos_n"]
    mask = core_baseline_mask_from_station_numbers(z["station"])
    vm = build_value_map_for_opoint(opoint_2x(kind, n_sub), e, n, is_core_baseline_mask=mask,
                                    dec_deg=71.63)
    return vm, e, n, mask


@pytest.mark.parametrize("kind,n_sub", [("gridded", 2), ("baselines", 4)])
def test_reference_geometry_equals_value_map_csr(kind, n_sub) -> None:
    vm, e, n, _ = _fixture_vm(kind, n_sub)
    kept = vm.kept_bl.astype(np.int64)
    per = vm.nchan_eff // n_sub
    for stream in (0, n_sub * 7 + 1, vm.n_streams - 1):
        dest, conj = ref.value_dest_from_vm(vm, stream, e, n)
        csr = vm.corr_csr(stream)
        st = vm.streams[stream]
        # rebuild the CSR from the reference placement, in the SPEC's fixed order
        r_idx, k_loc = np.nonzero(dest >= 0)
        v = dest[r_idx, k_loc]
        flat = kept[r_idx] * vm.nchan_eff + st.sub_band * per + k_loc
        order = np.lexsort((flat, v))
        assert np.array_equal(np.bincount(v, minlength=st.n_values), st.n_contrib)
        assert np.array_equal(flat[order] // vm.nchan_eff, csr.bl)
        assert np.array_equal(flat[order] % vm.nchan_eff, csr.ch)
        assert np.array_equal(conj[r_idx, k_loc][order], csr.conj)


# ---------------------------------------------------------------------------
# quantisation, packing, frames
# ---------------------------------------------------------------------------


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


def test_payload_row_matches_frame2_codec() -> None:
    """The reference payload layout == WS-T's ``prod_frame2.pack_samples``."""
    pf2 = pytest.importorskip("dsart.transport.prod_frame2", reason="WS-T frame v2 not merged yet")
    rng = np.random.default_rng(5)
    for comp_bits in (4, 5, 6):
        lim = (1 << (comp_bits - 1)) - 1
        h = rng.integers(-127, 128, size=(3, 7, 2)).astype(np.int8)
        m = rng.integers(-lim, lim + 1, size=(3, 301, 2)).astype(np.int8)
        want = pf2.pack_samples(h, m, 2 * comp_bits)
        for t in range(3):
            assert ref.payload_row(h[t], m[t], comp_bits) == want[t].tobytes()
        assert want.shape[1] == ref.sample_bytes(308, 7, 2 * comp_bits)


def test_quantise_flow_delegates_to_ws_t_reference() -> None:
    quant = pytest.importorskip("dsart.x2.quant", reason="WS-T row-rule reference not merged yet")
    rng = np.random.default_rng(6)
    n = rng.integers(1, 3000, size=50)
    rule = quant.RowRule(comp_bits=4)                         # frozen rule: 8 warm-up blocks
    nb = rule.warmup_blocks + 3
    blocks = [(rng.normal(size=(64, 50)) + 1j * rng.normal(size=(64, 50))) * np.sqrt(n) for _ in range(nb)]
    out = ref.quantise_flow(blocks, n, 5, rule)
    assert len(out) == nb and float(out[rule.warmup_blocks - 1].sigma) == 0.0
    assert 0.9 < float(out[-1].sigma) < 1.1
    assert out[0].codes_main.shape == (64, 45, 2) and out[0].codes_heavy.shape == (64, 5, 2)


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

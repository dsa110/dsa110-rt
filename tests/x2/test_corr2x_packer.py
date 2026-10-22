"""x2 WS-C deliverable 3: batched static sky + GPU packer (CPU twins).

* ``BatchedStaticSky`` == ``StaticSkyMean`` per DM (to the last fp32 bit
  of the time mean's reduction order) and == the float64 reference;
* bit packer: Triton kernel (interpreter) == torch twin == reference;
* row-rule packer: bit-equal to WS-T's ``dsart.x2.quant`` reference
  (codes, sigma, steps) and to its frame-v2 payload, for the reference
  defaults, the Phase-0 parameters, mid-rise and 5/6-bit codes, flagged
  values, a mostly-flagged block and a non-finite value; mid-tread keeps
  zeros at code 0 and never emits the no-data code; the warm-up hold, the
  50% skip and the per-row guard (Phase 0, not yet in quant.py);
* v1 cint8 mode == ``TransportTx._encode_rows`` byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.x2.corr.packer2x import PackerRule, RowRulePacker, pack_bits_torch, pack_v1
from dsart.x2.corr.static_sky2x import BatchedStaticSky
from dsart.x2.ref import corr2x_ref as ref

from ._interp import run_interpreted


def test_batched_static_sky_equals_per_dm() -> None:
    from dsart.services.corr_fast_integration import StaticSkyMean
    rng = np.random.default_rng(0)
    n_d, n_t, V = 5, 12, 33
    bs = BatchedStaticSky(n_d=n_d, n_values=V, window_blocks=3, warmup_cubes=2)
    per = StaticSkyMean(window_blocks=3, warmup_cubes=2, n_dm=n_d)
    rs = ref.StaticSkyRef(window_blocks=3, warmup_cubes=2)
    for blk in range(9):
        x = torch.from_numpy((rng.normal(size=(n_d, n_t, V)) + 1j * rng.normal(size=(n_d, n_t, V)) + 3.0)
                             .astype(np.complex64))
        want = torch.stack([per.apply(x[d].clone(), dm_slot=d) for d in range(n_d)])
        r64 = np.stack([rs.apply(d, x[d].numpy().astype(np.complex128)) for d in range(n_d)])
        got = bs.apply_(x.clone())
        # same semantics; the batched time-mean may reduce in a different
        # order than the per-DM one, so allow the last fp32 bit
        torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(got.numpy(), r64, rtol=1e-5, atol=1e-5)
    bs.reset()
    assert bs.running_mean() is None


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_pack_bits_torch_equals_reference(bits) -> None:
    rng = np.random.default_rng(bits)
    c = rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(7, 101)).astype(np.int8)
    got = pack_bits_torch(torch.from_numpy(c), bits).numpy()
    for r in range(7):
        assert got[r].tobytes() == ref.pack_bits_le(c[r], bits)


def test_pack_bits_triton_interpreted() -> None:
    code = r"""
import numpy as np, torch
from dsart.x2.corr.packer2x import pack_bits_triton, pack_bits_torch
rng = np.random.default_rng(1)
ok = []
for bits in (4, 5, 6):
    c = torch.from_numpy(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(5, 203)).astype(np.int8))
    ok.append(torch.equal(pack_bits_triton(c, bits, block=64), pack_bits_torch(c, bits)))
print("RESULT", *ok)
"""
    out = run_interpreted(code)
    assert "RESULT True True True" in out, out


def _flow_inputs(seed, n_d=3, n_t=64, voff=(0, 40, 95), nh=(4, 6), n_blocks=4):
    """Blocks with drifting noise, flagged (exact-zero) values, one mostly
    flagged block (must not update sigma) and a non-finite value."""
    rng = np.random.default_rng(seed)
    V = voff[-1]
    n = rng.integers(1, 2000, size=V)
    n[voff[0]:voff[0] + nh[0]] = 3000
    n[voff[1]:voff[1] + nh[1]] = 2500
    blocks = []
    for b in range(n_blocks):
        x = (rng.normal(size=(n_d, n_t, V)) + 1j * rng.normal(size=(n_d, n_t, V))) * np.sqrt(n) * (1 + 0.3 * b)
        x[:, :, 7] = 0                                   # a flagged value
        x[0, ::5, 50] = 0
        if b == 2:
            x[1, :, : int(0.8 * V)] = 0                  # mostly flagged flow
        if b == 3:
            x[2, 3, 60] = np.inf
        blocks.append(x.astype(np.complex64))
    return blocks, n, list(voff), list(nh)


_RULES = [
    dict(comp_bits=4),
    dict(comp_bits=5, s_heavy=0.0547),
    dict(comp_bits=4, tau_s=3.0, excl_sigma=float("inf"), min_frac=0.5, s_heavy=0.0773),
    dict(comp_bits=4, mode="midrise"),
    dict(comp_bits=6, est_stride=4),
]


@pytest.mark.parametrize("kw", _RULES)
def test_packer_bit_equal_to_ws_t_reference(kw) -> None:
    """The packer == ``dsart.x2.quant.RowQuantiser`` on the same inputs:
    codes, sigma, steps, and the payload == ``prod_frame2.pack_samples``."""
    quant = pytest.importorskip("dsart.x2.quant", reason="WS-T row-rule reference not merged yet")
    rule = quant.RowRule(**kw)
    blocks, n, voff, nh = _flow_inputs(sum(map(ord, repr(kw))))
    n_d = blocks[0].shape[0]
    pk = RowRulePacker(n_d=n_d, value_offsets=voff, n_contrib=n, n_heavy=nh,
                       rule=PackerRule.from_quant(rule))
    flows = {(d, s): quant.RowQuantiser(rule=rule, n_contrib=n[voff[s]:voff[s + 1]], n_heavy=nh[s])
             for d in range(n_d) for s in range(2)}
    try:
        from dsart.transport.prod_frame2 import pack_samples
    except ImportError:
        pack_samples = None
    for x in blocks:
        res = pk.pack(torch.from_numpy(x))
        for (d, s), q in flows.items():
            bc = q.process_block(x[d, :, voff[s]:voff[s + 1]])
            r = res[s]
            assert np.array_equal(r["codes_heavy"][d].numpy(), bc.codes_heavy)
            assert np.array_equal(r["codes_main"][d].numpy(), bc.codes_main)
            assert r["sigma"][d].numpy().tobytes() == np.float32(bc.sigma).tobytes()
            assert r["step_main"][d].numpy().tobytes() == np.float32(bc.step_main).tobytes()
            assert r["step_heavy"][d].numpy().tobytes() == np.float32(bc.step_heavy).tobytes()
            pay = r["payload"][d].numpy()
            if pack_samples is not None:
                assert np.array_equal(pay, pack_samples(bc.codes_heavy, bc.codes_main, rule.bits_main))
            else:
                for t in (0, 17):
                    assert pay[t].tobytes() == ref.payload_row(bc.codes_heavy[t], bc.codes_main[t],
                                                               rule.comp_bits)


def test_packer_midtread_never_emits_no_data_code_and_keeps_zero() -> None:
    blocks, n, voff, nh = _flow_inputs(1)
    pk = RowRulePacker(n_d=3, value_offsets=voff, n_contrib=n, n_heavy=nh,
                       rule=PackerRule(comp_bits=4))
    for x in blocks:
        for s, r in enumerate(pk.pack(torch.from_numpy(x))):
            m, h = r["codes_main"].numpy(), r["codes_heavy"].numpy()
            assert m.min() >= -7 and m.max() <= 7 and h.min() >= -127
            v7 = 7 - voff[s] - nh[s]
            if 0 <= v7 < m.shape[2]:
                assert not m[:, :, v7].any()                   # flagged value -> code 0


def test_packer_skip_and_hold_rules() -> None:
    blocks, n, voff, nh = _flow_inputs(2)
    rule = PackerRule(comp_bits=4, min_frac=0.5, excl_sigma=float("inf"), hold_blocks=2)
    pk = RowRulePacker(n_d=3, value_offsets=voff, n_contrib=n, n_heavy=nh, rule=rule)
    r0 = pk.pack(torch.from_numpy(blocks[0]))
    assert float(pk.sigma.abs().max()) == 0.0                   # held: sigma not initialised
    assert not r0[0]["codes_main"].numpy().any()                # unset sigma -> code 0
    pk.pack(torch.from_numpy(blocks[1]))                        # still held
    assert float(pk.sigma.abs().max()) == 0.0
    pk.pack(torch.from_numpy(blocks[1]))                        # 3rd call: first update
    s_before = pk.sigma.clone()
    assert float(s_before.min()) > 0.0
    pk.pack(torch.from_numpy(blocks[2]))
    # flow d=1 of block 2 is 80% zero: below min_frac -> its stream-0 sigma
    # is unchanged, while a normal flow moves
    assert float(pk.sigma[1, 0]) == float(s_before[1, 0])
    assert float(pk.sigma[0, 0]) != float(s_before[0, 0])


def test_packer_row_guard() -> None:
    rng = np.random.default_rng(3)
    n = np.full(60, 100)
    x = (rng.normal(size=(1, 32, 60)) + 1j * rng.normal(size=(1, 32, 60))) * 10.0
    rule = PackerRule(comp_bits=4, est_stride=1, guard_g=1.5)
    pk = RowRulePacker(n_d=1, value_offsets=[0, 60], n_contrib=n, n_heavy=[0], rule=rule)
    pk.pack(torch.from_numpy(x.astype(np.complex64)))           # initialise sigma ~ 1
    x2 = x.copy()
    x2[0, 9] *= 5.0                                             # a broadband surge in one row
    r = pk.pack(torch.from_numpy(x2.astype(np.complex64)))[0]
    rs = r["row_sigma"][0].numpy()
    sig = float(r["sigma"][0])
    assert rs[9] > 4.0 * sig and np.all(rs[np.arange(32) != 9] == np.float32(sig))
    # coded at its own scale: saturation stays at the normal-noise level
    assert np.mean(np.abs(r["codes_main"][0, 9].numpy()) == 7) < 0.1


def test_v1_mode_equals_transport_tx() -> None:
    from dsart.transport.prod_frame import BITS_CINT8_COMPLEX
    from dsart.transport.tx import TransportTx
    rng = np.random.default_rng(4)
    x = torch.from_numpy((rng.normal(size=(2, 9, 301)) + 1j * rng.normal(size=(2, 9, 301)))
                         .astype(np.complex64) * 5.0)
    x[1, 3] = 0
    q, sc = pack_v1(x)
    for d in range(2):
        q2, sc2 = TransportTx._encode_rows(x[d].numpy(), BITS_CINT8_COMPLEX)
        assert np.array_equal(sc[d].numpy(), sc2)
        assert q[d].numpy().tobytes() == np.asarray(q2).tobytes()

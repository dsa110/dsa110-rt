"""x2 WS-C deliverable 3: batched static sky + GPU packer (CPU twins).

* ``BatchedStaticSky`` == ``StaticSkyMean`` per DM (bit for bit on CPU:
  same ops per DM) and == the float64 reference;
* bit packer: Triton kernel (interpreter) == torch twin == reference;
* row-rule packer: codes == the float64 reference codes except at
  rounding boundaries (< 1e-4 of components, never more than one step),
  sigma == reference sigma to fp32 precision, payload bytes == reference
  datagram payload bytes when the codes agree, EMA behaviour;
* v1 cint8 mode == ``TransportTx._encode_rows`` byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.x2.corr.packer2x import RowRulePacker, pack_bits_torch, pack_v1
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
        assert torch.equal(got, want), blk
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


@pytest.mark.parametrize("bits_main", [8, 10])
def test_row_rule_packer_vs_reference(bits_main) -> None:
    rng = np.random.default_rng(bits_main)
    n_d, n_t = 3, 64
    voff = [0, 40, 95]
    nh = [4, 6]
    V = voff[-1]
    n = rng.integers(1, 2000, size=V)
    n[0:4] = 3000
    n[40:46] = 2500
    pk = RowRulePacker(n_d=n_d, value_offsets=voff, n_contrib=n, n_heavy=nh, bits_main=bits_main)
    st = ref.RowRuleState()
    b = bits_main // 2
    for blk in range(3):
        scale = 1.0 + 0.5 * blk
        x = (rng.normal(size=(n_d, n_t, V)) + 1j * rng.normal(size=(n_d, n_t, V))) * np.sqrt(n) * scale
        x64 = x.astype(np.complex64).astype(np.complex128)
        res = pk.pack(torch.from_numpy(x64.astype(np.complex64)))
        for s in range(2):
            lo, hi = voff[s], voff[s + 1]
            for d in range(n_d):
                sig = st.step((s, d), x64[d, :, lo:hi] / np.sqrt(n[lo:hi])[None, :])
                assert abs(float(res[s]["sigma"][d]) - sig) <= 1e-5 * sig
                h, m, sm, sh = ref.quantise_row_rule(x64[d, :, lo:hi], n[lo:hi], nh[s], sig, bits=b)
                gh = res[s]["codes_heavy"][d].numpy()
                gm = res[s]["codes_main"][d].numpy()
                mism = np.mean(gm != m) + np.mean(gh != h)
                assert mism < 1e-3, mism
                assert np.abs(gm.astype(int) - m).max() <= 1 and np.abs(gh.astype(int) - h).max() <= 1
                # payload bytes for rows whose codes agree
                pay = res[s]["payload"][d].numpy()
                assert pay.shape[1] == ref.sample_bytes(hi - lo, nh[s], bits_main)
                same = np.all(gm == m, axis=(1, 2)) & np.all(gh == h, axis=(1, 2))
                for t in np.flatnonzero(same)[:5]:
                    want = h[t].tobytes() + ref.pack_bits_le(m[t], b)
                    assert pay[t].tobytes() == want
    # the EMA moved towards the louder blocks
    assert float(pk.sigma.mean()) > 1.0


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

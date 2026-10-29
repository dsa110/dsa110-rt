"""x2 WS-C: batched static sky + the row-rule packer (SPEC §4, frozen).

* ``BatchedStaticSky`` == ``StaticSkyMean`` per DM and == the float64 reference;
* the row-rule packer is bit-equal to WS-T's reference ``dsart.x2.quant``
  (``RowQuantiser.process_block``) on the same inputs, with Phase 0's rules on:
  the 8-block warm-up (every code NO_DATA, sigma untouched), the 50% skip,
  tau 3 s, mid-tread with NO_DATA, flagged zeros at code 0, non-finite
  values, and the per-row guard (exact lower medians; more guarded rows than
  the on-device capacity go through ``resolve_overflow``) -- codes, sigma,
  per-row sigma / steps / guard flags, and the payload == ``pack_samples``;
* the exact guard threshold (smallest float32 t with f32(1.4826 t) > G);
* the kernels: row-stats / codes / bit packer under the Triton interpreter
  == their torch twins, and the whole packer with the Triton kernels
  (interpreted) == quant.py; every interpreted check also runs a MUTANT
  kernel that must fail;
* v1 cint8 == ``TransportTx._encode_rows`` byte for byte;
* x2_gpu (CP1): the CUDA packer == the CPU twin, byte for byte.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.transport.prod_frame2 import pack_samples
from dsart.x2 import quant
from dsart.x2.corr.packer2x import (
    RowRulePacker,
    codes_torch,
    guard_threshold,
    pack_bits_torch,
    pack_v1,
    row_stats_torch,
)
from dsart.x2.corr.static_sky2x import BatchedStaticSky
from dsart.x2.ref import corr2x_ref as ref

from ._interp import run_interpreted


def _last(out: str) -> str:
    """One-line failure message: the snippet's RESULT line (or its tail)."""
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    return lines[-1] if lines else out.strip()[-300:].replace("\n", " | ")


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


def test_batched_static_sky_keeps_excised_samples_zero() -> None:
    """2x: an excised sample (exactly 0 + 0i) stays exactly 0 after the subtraction (an
    all-excised block becomes all zeros, not -sky); every other sample is unchanged versus
    StaticSkyMean; the window still takes the block's mean."""
    from dsart.services.corr_fast_integration import StaticSkyMean
    rng = np.random.default_rng(1)
    n_d, n_t, V = 3, 10, 21
    bs = BatchedStaticSky(n_d=n_d, n_values=V, window_blocks=3, warmup_cubes=1)
    per = StaticSkyMean(window_blocks=3, warmup_cubes=1, n_dm=n_d)
    rs = ref.StaticSkyRef(window_blocks=3, warmup_cubes=1)
    for blk in range(8):
        x = (rng.normal(size=(n_d, n_t, V)) + 1j * rng.normal(size=(n_d, n_t, V)) + 3.0).astype(np.complex64)
        x[0, 2:5] = 0                                       # excised rows
        x[1, :, 7] = 0                                      # an excised value
        if blk == 5:
            x[:] = 0                                        # a whole excised block
        xt = torch.from_numpy(x)
        want = torch.stack([per.apply(xt[d].clone(), dm_slot=d) for d in range(n_d)])
        got = bs.apply_(xt.clone())
        r64 = np.stack([rs.apply(d, x[d].astype(np.complex128)) for d in range(n_d)])
        zero = torch.from_numpy(x == 0)
        assert (got[zero] == 0).all()
        if blk >= 1 and blk != 5:
            assert (got[~zero] != 0).all()
        torch.testing.assert_close(got[~zero], want[~zero], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(got.numpy(), r64, rtol=1e-5, atol=1e-5)
        if blk == 5:
            assert (got == 0).all()


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_pack_bits_torch_equals_reference(bits) -> None:
    rng = np.random.default_rng(bits)
    c = rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(7, 101)).astype(np.int8)
    got = pack_bits_torch(torch.from_numpy(c), bits).numpy()
    for r in range(7):
        assert got[r].tobytes() == ref.pack_bits_le(c[r], bits)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

VOFF = (0, 40, 95)
NH = (4, 6)


def _blocks(seed, rule, *, n_d=3, S=64, n_after=4):
    """``warmup_blocks + n_after`` blocks of drifting noise with: flagged
    (exact-zero) values; after the warm-up a broadband surge on 2 rows of
    flow 0 and on 11 rows of flow 2 (more than the guard capacity 8), a
    mostly-flagged flow (no sigma update, never guarded), and non-finite
    values."""
    rng = np.random.default_rng(seed)
    V = VOFF[-1]
    n = rng.integers(1, 2000, size=V)
    n[VOFF[0]:VOFF[0] + NH[0]] = 3000
    n[VOFF[1]:VOFF[1] + NH[1]] = 2500
    w = int(rule.warmup_blocks)
    out = []
    for b in range(w + n_after):
        x = (rng.normal(size=(n_d, S, V)) + 1j * rng.normal(size=(n_d, S, V))) * np.sqrt(n) * (1 + 0.3 * b)
        x[:, :, 7] = 0                                   # a flagged value
        x[0, ::5, 50] = 0
        if b == w + 1:
            x[0, 9:11] *= 5.0                            # surge: guarded, within capacity
            x[2, 20:31] *= 4.0                           # surge: 11 rows > capacity 8
            x[1, 33, : int(0.6 * V)] = 0                 # a surge row: stream 0 all zero (< 50%
            x[1, 33] *= 9.0                              # used: never guarded), stream 1 guarded
        if b == w + 2:
            x[1, :, : int(0.8 * V)] = 0                  # mostly flagged flow: no update
        if b == w + 3:
            x[2, 3, 60] = np.inf
            x[2, 4, 61] = np.nan + 1j
            x[0, 5, 2] = np.inf                          # a heavy value
        out.append(x.astype(np.complex64))
    return out, n


def _flows(rule, n, n_d):
    return {(d, s): quant.RowQuantiser(rule=rule, n_contrib=n[VOFF[s]:VOFF[s + 1]], n_heavy=NH[s])
            for d in range(n_d) for s in range(2)}


def _assert_block_equal(res, flows, x, rule) -> int:
    """res (packer) == quant on every flow; returns guarded rows seen."""
    n_guard = 0
    for (d, s), q in flows.items():
        bc = q.process_block(x[d, :, VOFF[s]:VOFF[s + 1]])
        r = res[s]
        np.testing.assert_array_equal(r["codes_heavy"][d].numpy(), bc.codes_heavy)
        np.testing.assert_array_equal(r["codes_main"][d].numpy(), bc.codes_main)
        assert r["sigma"][d].numpy().tobytes() == np.float32(bc.sigma).tobytes()
        assert r["sigma_row"][d].numpy().tobytes() == bc.sigma_row.astype(np.float32).tobytes()
        assert r["step_main"][d].numpy().tobytes() == bc.step_main.tobytes()
        assert r["step_heavy"][d].numpy().tobytes() == bc.step_heavy.tobytes()
        np.testing.assert_array_equal(r["guarded"][d].numpy(), bc.guarded)
        assert not bool(r["guard_overflow"][d])
        np.testing.assert_array_equal(r["payload"][d].numpy(),
                                      pack_samples(bc.codes_heavy, bc.codes_main, rule.bits_main))
        n_guard += int(bc.guarded.sum())
    return n_guard


_RULES = [
    dict(bits_main=8, n_sub=2),
    dict(bits_main=10, n_sub=4),
    dict(bits_main=12, n_sub=2, est_stride=4, warmup_blocks=2),
    dict(bits_main=8, n_sub=4, warmup_blocks=0),
]


@pytest.mark.parametrize("kw", _RULES)
def test_packer_bit_equal_to_quant(kw) -> None:
    """Every block, every flow: codes, sigma, per-row sigma / steps / guard
    and the frame-v2 payload == ``quant.RowQuantiser`` + ``pack_samples``."""
    kw = dict(kw)
    rule = quant.RowRule.for_format(kw.pop("bits_main"), kw.pop("n_sub"), **kw)
    blocks, n = _blocks(sum(map(ord, repr(sorted(kw.items())))) + rule.comp_bits, rule)
    n_d = blocks[0].shape[0]
    pk = RowRulePacker(n_d=n_d, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule)
    flows = _flows(rule, n, n_d)
    w = int(rule.warmup_blocks)
    n_guard = 0
    for b, x in enumerate(blocks):
        res = pk.pack(torch.from_numpy(x))
        n_guard += _assert_block_equal(res, flows, x, rule)
        if b < w:                                         # warm-up: sigma 0, all NO_DATA
            assert float(pk.sigma.abs().max()) == 0.0
            assert (res[0]["codes_main"].numpy() == quant.no_data_code(rule.comp_bits)).all()
            assert (res[1]["codes_heavy"].numpy() == -128).all()
        if b == w + 2:                                    # mostly flagged flow: not updated
            assert np.isnan(float(res[0]["est"][1])) and np.isnan(float(res[1]["est"][1]))
    assert n_guard >= 13                                  # 2 + 11 surge rows tripped the guard
    assert pk.n_overflow_resolved >= 1                    # flow 2 went through resolve_overflow


def test_mid_tread_zero_and_no_data_codes() -> None:
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    blocks, n = _blocks(5, rule, n_after=4)
    pk = RowRulePacker(n_d=3, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule)
    for b, x in enumerate(blocks):
        res = pk.pack(torch.from_numpy(x))
        if b < 1:
            continue
        m = res[0]["codes_main"].numpy()
        assert m.min() >= -8 and m.max() <= 7
        assert not m[:, :, 7 - NH[0]].any()              # flagged value -> code 0 (mid-tread)
        if b == 4:                                       # non-finite -> NO_DATA, row neighbours fine
            assert res[1]["codes_main"][2, 3, 60 - VOFF[1] - NH[1], 0].item() == -8    # inf + 0i
            assert res[1]["codes_main"][2, 3, 60 - VOFF[1] - NH[1], 1].item() == 0
            assert res[1]["codes_main"][2, 4, 61 - VOFF[1] - NH[1], 0].item() == -8    # nan + 1i
            assert res[0]["codes_heavy"][0, 5, 2, 0].item() == -128
            assert res[0]["codes_heavy"][0, 5, 2, 1].item() != -128


def test_codes_twin_equals_quant_quantise_incl_halfway() -> None:
    """Exact halfway points round to even; clipping; NO_DATA for non-finite q
    and for rows whose step is not finite > 0."""
    rng = np.random.default_rng(9)
    S, V, nh = 6, 23, 3
    base = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 3.5, 7.5, 8.5, -7.5, 126.5, 127.5, -200.0, 1e30,
                     np.inf, np.nan, 0.0, -0.0, 6.4999995, 6.5], np.float32)
    y = rng.choice(base, size=(1, S, V, 2)).astype(np.float32)
    r = np.ones(V, np.float32)
    steps = np.array([1.0, 2.0, 0.0, np.inf, 1e-45, 0.75], np.float32)
    for cb in (4, 5, 6):
        st = steps[None, :]
        ok = np.isfinite(st) & (st > 0)
        inv = (np.float32(1) / np.where(ok, st, np.float32(1))).astype(np.float32)
        got = codes_torch(torch.from_numpy(y), torch.from_numpy(r), torch.from_numpy(inv),
                          torch.from_numpy(inv), torch.from_numpy(ok), torch.from_numpy(ok),
                          n_heavy=nh, comp_bits=cb).numpy()
        rule = quant.RowRule(comp_bits=cb)
        want_h = quant.quantise(y[0, :, :nh, 0], y[0, :, :nh, 1], st[0][:, None], rule.heavy_range, -128)
        want_m = quant.quantise(y[0, :, nh:, 0], y[0, :, nh:, 1], st[0][:, None], rule.main_range,
                                quant.no_data_code(cb))
        np.testing.assert_array_equal(got[0, :, :nh], want_h)
        np.testing.assert_array_equal(got[0, :, nh:], want_m)


def test_row_stats_twin_counts() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=(2, 5, 30, 2)).astype(np.float32)
    y[0, 1, :20] = 0.0
    y[1, 2, 3, 0] = np.nan
    y[1, 3, 4, 1] = np.inf
    t = np.array([0.5, 1.0], np.float32)
    m, c = row_stats_torch(torch.from_numpy(y), torch.ones(30), torch.from_numpy(t))
    for d in range(2):
        for j in range(5):
            u = quant._used(y[d, j, :, 0], y[d, j, :, 1])
            a = np.concatenate([np.abs(y[d, j, u, 0]), np.abs(y[d, j, u, 1])])
            assert int(m[d, j]) == int(u.sum()) and int(c[d, j]) == int((a < t[d]).sum())


def test_guard_threshold_is_the_exact_boundary() -> None:
    rng = np.random.default_rng(4)
    k = np.float32(1.4826)
    G = np.concatenate([
        (10.0 ** rng.uniform(-29, 37, size=200000)).astype(np.float32),
        np.float32(2.0) ** np.arange(-90, 120, dtype=np.float32),
        np.nextafter((np.float32(2.0) ** np.arange(-90, 120, dtype=np.float32)), np.float32(0)),
        np.array([1.5, 3.0, 0.75 * 1.4826, 1e-30], np.float32),
    ]).astype(np.float32)
    t, valid = guard_threshold(torch.from_numpy(G), float(k))
    t, valid = t.numpy(), valid.numpy()
    assert valid.all()
    assert ((k * t).astype(np.float32) > G).all()
    prev = np.nextafter(t, np.float32(0))
    assert not ((k * prev).astype(np.float32) > G).any()
    tt, vv = guard_threshold(torch.tensor([np.inf, 0.0], dtype=torch.float32), float(k))
    assert not bool(vv[0])


@pytest.mark.parametrize("capacity", [0, 1])
def test_guard_overflow_is_deferred_and_resolved(capacity) -> None:
    """With ``resolve_overflow=False`` a flow with more guarded rows than the
    capacity is flagged (capacity 0, the CUDA pipeline's: every guarded flow);
    ``resolve_overflow`` then makes it exact."""
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    blocks, n = _blocks(8, rule, n_after=3)
    pk = RowRulePacker(n_d=3, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule,
                       guard_capacity=capacity)
    flows = _flows(rule, n, 3)
    for b, x in enumerate(blocks):
        xt = torch.from_numpy(x)
        res = pk.pack(xt, resolve_overflow=False)
        if b == 2:                                         # the surge block
            ov = [r["guard_overflow"].numpy().copy() for r in res]
            assert ov[0][0] and ov[1][0] and ov[0][2] and ov[1][2]
        pk.resolve_overflow(res, xt)
        _assert_block_equal(res, flows, x, rule)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_lowmed_rows_selection_equals_the_sort(dev) -> None:
    """resolve_overflow's selection (unused values split -inf / +inf, one topk with k = V + 1)
    == the sort-based lower median, bit for bit, for rows with unused, zero-component,
    non-finite and duplicate values and any used count m >= 1."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-only")
    rng = np.random.default_rng(5)
    y = rng.normal(size=(64, 37, 2)).astype(np.float32)
    y[3, :10] = 0                                      # unused values
    y[5, 5, 0] = np.inf
    y[6, 2, 1] = np.nan
    y[7, :36] = 0                                      # m = 1
    y[8, ::2, 0] = 0                                   # a zero component: still used
    y[9] = 1.5                                         # all magnitudes tied
    y[10, :20] = np.float32(0.25)                      # ties around the median
    y[11, 1:] = 0                                      # m = 1 at the front
    t = torch.from_numpy(y).to(dev)
    med, m = RowRulePacker._lowmed(None, t)
    assert int(m.min()) >= 1
    got = RowRulePacker._lowmed_rows(t)
    assert torch.equal(got, med), (got - med).abs().max()


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_excision_release_and_step_up_bit_equal_to_quant(dev) -> None:
    """quant v1.1 on the packer (CPU twin; CUDA with capacity 0 + resolve_overflow), every
    block == quant.RowQuantiser: a long excision whose residue is ~1e-9 of the noise leaves
    sigma_hat exactly where it was, the release guards no row, and a x3 level step resets
    sigma_hat in one block (no guarded rows either)."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-only")
    rule = quant.RowRule.for_format(8, 2, warmup_blocks=1)
    rng = np.random.default_rng(21)
    V, n_d, S = VOFF[-1], 3, 64
    n = rng.integers(1, 2000, size=V)
    pk = RowRulePacker(n_d=n_d, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule, device=dev,
                       guard_capacity=0 if dev == "cuda" else 8)
    flows = _flows(rule, n, n_d)
    v10 = _flows(quant.RowRule.for_format(8, 2, warmup_blocks=1, used_floor=0.0, reset_factor=0.0),
                 n, n_d)
    amp = np.sqrt(n)
    plan = [1.0] * 6 + [1e-9] * 40 + [1.0] * 4 + [3.0] * 4
    sig0 = None
    g_release = g_step = g10 = 0
    for i, scale in enumerate(plan):
        x = ((rng.normal(size=(n_d, S, V)) + 1j * rng.normal(size=(n_d, S, V))) * amp * scale)
        x = x.astype(np.complex64)
        xt = torch.from_numpy(x).to(dev)
        if dev == "cuda":
            res = pk.pack(xt, resolve_overflow=False)
            pk.resolve_overflow(res, xt)
            res = [{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in r.items()} for r in res]
        else:
            res = pk.pack(xt)
        _assert_block_equal(res, flows, x, rule)
        g = sum(int(q.process_block(x[d, :, VOFF[s_]:VOFF[s_ + 1]]).guarded.sum()) for (d, s_), q in v10.items())
        if i >= 46:
            g10 += g
        if i == 5:
            sig0 = pk.sigma.cpu().clone()
        if 6 <= i < 46:
            assert torch.equal(pk.sigma.cpu(), sig0)            # held through the excision
        n_g = sum(int(r["guarded"].sum()) for r in res)
        if 46 <= i < 50:
            g_release += n_g
        if i >= 50:
            g_step += n_g
    # 64-row blocks of 30-40 values: a row's robust scale has ~15% scatter, so an odd row may
    # cross 1.5 x on noise; v1.0 guards the whole release and the step (~2 x 3 x 64 rows a block)
    assert g_release + g_step <= 4 and g10 > 500, (g_release, g_step, g10)
    assert float((pk.sigma.cpu() / sig0).min()) > 2.5           # reset onto the x3 level


def test_packer_rejects_a_non_quant_rule() -> None:
    with pytest.raises(TypeError):
        RowRulePacker(n_d=1, value_offsets=[0, 4], n_contrib=np.ones(4), n_heavy=[0], rule=object())


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


# ---------------------------------------------------------------------------
# Triton kernels under the interpreter (with mutants)
# ---------------------------------------------------------------------------


def test_pack_bits_triton_interpreted() -> None:
    code = r"""
import numpy as np, torch
from dsart.x2.corr.packer2x import pack_bits_triton, pack_bits_torch
rng = np.random.default_rng(1)
ok = []
for bits in (4, 5, 6):
    c = torch.from_numpy(rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(5, 203)).astype(np.int8))
    ok.append(torch.equal(pack_bits_triton(c, bits, block=64), pack_bits_torch(c, bits)))
# strided rows in and out (the packer writes straight into payload rows)
big = torch.from_numpy(rng.integers(-16, 16, size=(5, 260)).astype(np.int8))
pay = torch.zeros((5, 300), dtype=torch.uint8)
pack_bits_triton(big[:, 10:213], 5, block=64, out=pay[:, 7:7 + 127])
ok.append(torch.equal(pay[:, 7:134], pack_bits_torch(big[:, 10:213].contiguous(), 5))
          and not pay[:, :7].any() and not pay[:, 134:].any())
import dsart.x2.corr.packer2x as pk
pk._pack_bits_kernel = _mutant("dsart.x2.corr.packer2x", "_pack_bits_kernel",
                               "(v1 << (B - sh))", "(v1 << (B - sh + 1))")
ok.append(not torch.equal(pk.pack_bits_triton(c, 5, block=64), pack_bits_torch(c, 5)))
print("RESULT", *ok)
"""
    out = run_interpreted(code)
    assert "RESULT True True True True True" in out, _last(out)


def test_row_stats_and_codes_triton_interpreted() -> None:
    code = r"""
import numpy as np, torch
import dsart.x2.corr.packer2x as pk
rng = np.random.default_rng(3)
n_d, S, V, Vtot, lo = 2, 5, 45, 60, 9
x = (rng.normal(size=(n_d, S, Vtot, 2)) * 3).astype(np.float32)
x[0, 1, lo:lo + 30] = 0.0
x[1, 2, lo + 3, 0] = np.nan
x[1, 3, lo + 4, 1] = np.inf
x[0, 4, lo + 5] = [2.5, -3.5]                          # halfway at inv 1, r 1
xs = torch.from_numpy(x)[:, :, lo:lo + V]               # strided view, as the packer passes it
r = torch.from_numpy((1.0 / np.sqrt(rng.integers(1, 50, size=V))).astype(np.float32))
r[5] = 1.0
t = torch.tensor([0.7, 1.9])
ok = []
a = pk.row_stats_triton(xs, r, t, block_v=16)
b = pk.row_stats_torch(xs, r, t)
ok.append(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
u = torch.tensor([0.3, 1e-6])                            # quant v1.1 used thresholds
x2 = x.copy(); x2[0, 2, lo:lo + 20] = 1e-3               # sub-threshold values in flow 0
xs2 = torch.from_numpy(x2)[:, :, lo:lo + V]
a2 = pk.row_stats_triton(xs2, r, t, u, block_v=16)
b2 = pk.row_stats_torch(xs2, r, t, u)
ok.append(torch.equal(a2[0], b2[0]) and torch.equal(a2[1], b2[1]) and not torch.equal(a2[0], a[0]))
inv_m = torch.from_numpy(rng.uniform(0.2, 3, size=(n_d, S)).astype(np.float32))
inv_h = torch.from_numpy(rng.uniform(0.05, 1, size=(n_d, S)).astype(np.float32))
inv_m[0, 4] = 1.0
ok_m = torch.ones((n_d, S), dtype=torch.bool); ok_m[1, 1] = False
ok_h = torch.ones((n_d, S), dtype=torch.bool); ok_h[0, 2] = False
for cb in (4, 5, 6):
    c1 = pk.codes_triton(xs, r, inv_m, inv_h, ok_m, ok_h, n_heavy=3, comp_bits=cb, block_v=16)
    c2 = pk.codes_torch(xs, r, inv_m, inv_h, ok_m, ok_h, n_heavy=3, comp_bits=cb)
    ok.append(torch.equal(c1, c2))
ok.append(int(c1[0, 4, 5, 0]) == 2 and int(c1[0, 4, 5, 1]) == -4)   # half-to-even
pk._row_stats_kernel = _mutant("dsart.x2.corr.packer2x", "_row_stats_kernel",
                               "(used & (are < t))", "(used & (are > t))")
a = pk.row_stats_triton(xs, r, t, block_v=16)
ok.append(not torch.equal(a[1], b[1]))
pk._codes_kernel = _mutant("dsart.x2.corr.packer2x", "_codes_kernel",
                           "rre = (sre + _RINT_MAGIC) - _RINT_MAGIC", "rre = sre")
c1 = pk.codes_triton(xs, r, inv_m, inv_h, ok_m, ok_h, n_heavy=3, comp_bits=6, block_v=16)
ok.append(not torch.equal(c1, c2))
print("RESULT", *ok)
"""
    out = run_interpreted(code)
    assert "RESULT " + " ".join(["True"] * 8) in out, _last(out)


def test_packer_with_triton_kernels_interpreted_equals_quant() -> None:
    """The whole packer through the Triton kernels (interpreted) is
    bit-equal to quant.py and to the torch twin; a mutant codes kernel is not."""
    code = r"""
import numpy as np, torch
from dsart.x2 import quant
import dsart.x2.corr.packer2x as pk
from dsart.transport.prod_frame2 import pack_samples
rng = np.random.default_rng(5)
voff, nh = [0, 21, 50], [2, 3]
n = rng.integers(1, 900, size=50); n[:2] = 2000; n[21:24] = 1800
rule = quant.RowRule.for_format(10, 2, warmup_blocks=1, est_stride=4)
mk = lambda **kw: pk.RowRulePacker(n_d=2, value_offsets=voff, n_contrib=n, n_heavy=nh, rule=rule, **kw)
tri, twin = mk(use_triton=True, guard_capacity=2), mk(use_triton=False, guard_capacity=2)
flows = {(d, s): quant.RowQuantiser(rule=rule, n_contrib=n[voff[s]:voff[s + 1]], n_heavy=nh[s])
         for d in range(2) for s in range(2)}
ok = True
blocks = []
for b in range(4):
    x = ((rng.normal(size=(2, 16, 50)) + 1j * rng.normal(size=(2, 16, 50))) * np.sqrt(n)).astype(np.complex64)
    x[:, :, 4] = 0
    if b == 2:
        x[0, 3:8] *= 6.0                     # 5 guarded rows > capacity 2
        x[1, 9] *= 4.0
    if b == 3:
        x[1, 2, 30] = np.inf
    blocks.append(x)
    ra, rb = tri.pack(torch.from_numpy(x)), twin.pack(torch.from_numpy(x))
    for s in range(2):
        for key in ("payload", "codes_main", "codes_heavy", "sigma", "sigma_row", "guarded"):
            ok &= torch.equal(ra[s][key], rb[s][key])
    for (d, s), q in flows.items():
        bc = q.process_block(x[d, :, voff[s]:voff[s + 1]])
        ok &= np.array_equal(ra[s]["payload"][d].numpy(), pack_samples(bc.codes_heavy, bc.codes_main, 10))
        ok &= np.array_equal(ra[s]["guarded"][d].numpy(), bc.guarded)
        ok &= ra[s]["sigma_row"][d].numpy().tobytes() == bc.sigma_row.tobytes()
pk._codes_kernel = _mutant("dsart.x2.corr.packer2x", "_codes_kernel",
                           "cim = tl.where(fim & ok, cim, nod)", "cim = tl.where(fim, cim, nod)")
bad = mk(use_triton=True)
diff = False
twin2 = mk(use_triton=False)
for x in blocks:
    ra, rb = bad.pack(torch.from_numpy(x)), twin2.pack(torch.from_numpy(x))
    diff |= not all(torch.equal(ra[s]["payload"], rb[s]["payload"]) for s in range(2))
print("RESULT", bool(ok), tri.n_overflow_resolved > 0, diff)
"""
    out = run_interpreted(code)
    assert "RESULT True True True" in out, _last(out)


# ---------------------------------------------------------------------------
# CP1 (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
@pytest.mark.parametrize("cap", [8, 0])
@pytest.mark.parametrize("kw", _RULES)
def test_packer_gpu_equals_cpu(kw, cap) -> None:
    """CUDA packer (Triton row-stats / codes / bit packer, deferred overflow)
    == the CPU twin == quant.py, byte for byte. ``cap=0`` is the 2x pipeline's
    setting: every flow with a guarded row overflows on the device and is
    recoded exactly by ``resolve_overflow`` (the surge rows force it). During
    the warm-up sigma_hat is 0: no row is guarded, nothing overflows."""
    kw = dict(kw)
    rule = quant.RowRule.for_format(kw.pop("bits_main"), kw.pop("n_sub"), **kw)
    blocks, n = _blocks(11, rule)
    cpu = RowRulePacker(n_d=3, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule)
    gpu = RowRulePacker(n_d=3, value_offsets=VOFF, n_contrib=n, n_heavy=NH, rule=rule, device="cuda",
                        guard_capacity=cap)
    w = int(rule.warmup_blocks)
    n_over = 0
    for i, x in enumerate(blocks):
        a = cpu.pack(torch.from_numpy(x))
        xg = torch.from_numpy(x).cuda()
        b = gpu.pack(xg, resolve_overflow=False)
        ov = sum(int(r["guard_overflow"].sum()) for r in b)
        if i < w:
            assert ov == 0 and not any(bool(r["guarded"].any()) for r in b)
        n_over += ov
        gpu.resolve_overflow(b, xg)
        for ra, rb in zip(a, b):
            assert not bool(rb["guard_overflow"].any())
            for k in ("payload", "codes_main", "codes_heavy", "sigma", "sigma_row", "step_main",
                      "step_heavy", "guarded"):
                assert torch.equal(ra[k], rb[k].cpu()), k
    torch.cuda.synchronize()
    assert n_over >= (3 if cap == 0 else 1) and gpu.n_overflow_resolved == n_over


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
def test_static_sky_and_v1_gpu_equals_cpu() -> None:
    rng = np.random.default_rng(12)
    x = torch.from_numpy((rng.normal(size=(6, 128, 2500)) + 1j * rng.normal(size=(6, 128, 2500)))
                         .astype(np.complex64))
    a = BatchedStaticSky(n_d=6, n_values=2500, window_blocks=3, warmup_cubes=1)
    b = BatchedStaticSky(n_d=6, n_values=2500, window_blocks=3, warmup_cubes=1, device="cuda")
    for _ in range(4):
        torch.testing.assert_close(a.apply_(x.clone()), b.apply_(x.cuda()).cpu(), rtol=1e-6, atol=1e-6)
    q1, s1 = pack_v1(x)
    q2, s2 = pack_v1(x.cuda())
    assert torch.equal(q1, q2.cpu()) and torch.equal(s1, s2.cpu())


@pytest.mark.x2_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only (CP1)")
def test_sky_subtraction_kernel_equals_the_torch_twin() -> None:
    """The fused CUDA kernel (x -= sky except where x == 0 + 0i) == the torch twin bit for
    bit, including excised rows / values / components of signed zero and a partial last block."""
    from dsart.x2.corr.static_sky2x import subtract_sky_keep_zeros_
    rng = np.random.default_rng(9)
    n_d, S, V = 4, 37, 1201
    x = (rng.normal(size=(n_d, S, V)) + 1j * rng.normal(size=(n_d, S, V))).astype(np.complex64)
    x[1, 3:9] = 0
    x[2, :, 17] = 0
    x[0, 5, 6] = complex(0.0, 1.5)                         # re 0, im non-zero: subtracted
    x[3, 7, 8] = complex(-0.0, -0.0)                       # signed zero: stays
    sky = (rng.normal(size=(n_d, V)) + 1j * rng.normal(size=(n_d, V))).astype(np.complex64)
    a = subtract_sky_keep_zeros_(torch.from_numpy(x.copy()), torch.from_numpy(sky))
    b = subtract_sky_keep_zeros_(torch.from_numpy(x.copy()).cuda(), torch.from_numpy(sky).cuda()).cpu()
    assert torch.equal(torch.view_as_real(a), torch.view_as_real(b))
    assert (b[1, 3:9] == 0).all() and (b[2, :, 17] == 0).all()

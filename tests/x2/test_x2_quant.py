"""Row rule reference (SPEC §4): arithmetic, statistics, zero handling."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from dsart.x2 import quant as q

F32 = np.float32


def _noise(rng, shape, sigma=1.0):
    return (rng.normal(0, sigma, shape) + 1j * rng.normal(0, sigma, shape)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Independent scalar implementation (element loops, float32 scalars only).
# Guards the vectorised reference against reordering / dtype promotion.
# ---------------------------------------------------------------------------

def _scalar_block(x, n_contrib, n_heavy, sigma_prev, rule):
    S, V = x.shape
    r = [F32(1.0 / math.sqrt(float(n))) for n in n_contrib]
    re = [[F32(F32(x[j, v].real) * r[v]) for v in range(V)] for j in range(S)]
    im = [[F32(F32(x[j, v].imag) * r[v]) for v in range(V)] for j in range(S)]
    comps, m, M = [], 0, 0
    thr = F32(F32(rule.excl_sigma) * F32(sigma_prev)) if sigma_prev > 0 else None
    for j in range(0, S, rule.est_stride):
        for v in range(V):
            M += 1
            a, b = re[j][v], im[j][v]
            if a == 0 and b == 0:
                continue
            if thr is not None and (abs(a) > thr or abs(b) > thr):
                continue
            m += 1
            comps += [abs(a), abs(b)]
    sigma = F32(sigma_prev)
    if m >= max(rule.min_count, math.ceil(rule.min_frac * M)):
        comps.sort()
        est = F32(F32(rule.robust_k) * comps[(len(comps) - 1) // 2])
        if sigma > 0:
            sigma = F32(sigma + F32(rule.alpha_f32 * F32(est - sigma)))
        else:
            sigma = est
    out = np.zeros((S, V, 2), np.int64)
    for v in range(V):
        s = rule.s_heavy_f32 if v < n_heavy else rule.s_main_f32
        lo, hi = rule.heavy_range if v < n_heavy else rule.main_range
        step = F32(s * sigma)
        if not step > 0:
            continue
        inv = F32(F32(1.0) / step)
        for j in range(S):
            for c, val in enumerate((re[j][v], im[j][v])):
                qq = F32(val * inv)
                code = math.floor(qq) if rule.midrise else round(float(qq))  # round(): half-even
                out[j, v, c] = min(hi, max(lo, code))
    return out, sigma


@pytest.mark.parametrize("mode", [q.MODE_MIDTREAD, q.MODE_MIDRISE])
@pytest.mark.parametrize("bits", [4, 5])
def test_vectorised_matches_scalar_reference(mode, bits):
    rule = q.RowRule(comp_bits=bits, mode=mode, est_stride=4)
    rng = np.random.default_rng(11 + bits)
    V, nh, S = 37, 5, 12
    n = rng.integers(1, 4000, size=V)
    rq = q.RowQuantiser(rule, n, nh)
    for blk in range(3):
        x = _noise(rng, (S, V), 1.0) * np.sqrt(n)[None, :].astype(np.float32) * F32(3.0)
        x[2, 3] = 0                                          # flagged value
        if blk == 2:
            x[4, 7] *= 200                                   # hot value
        sigma_prev = rq.sigma
        bc = rq.process_block(x)
        ref, ref_sigma = _scalar_block(x, n, nh, sigma_prev, rule)
        assert bc.sigma == ref_sigma
        np.testing.assert_array_equal(bc.codes_heavy, ref[:, :nh])
        np.testing.assert_array_equal(bc.codes_main, ref[:, nh:])


def test_golden_vector_is_stable():
    """Regression guard for the packer/decoder implementations: a fixed
    input must keep producing the same codes (change only with the SPEC)."""
    rule = q.RowRule(comp_bits=4)
    rng = np.random.default_rng(20261022)
    V, nh, S = 64, 4, 256
    n = rng.integers(1, 3936, size=V)
    rq = q.RowQuantiser(rule, n, nh)
    h = hashlib.sha256()
    for _ in range(4):
        x = _noise(rng, (S, V), 2.5) * np.sqrt(n)[None, :].astype(np.float32)
        bc = rq.process_block(x)
        h.update(bc.codes_heavy.tobytes())
        h.update(bc.codes_main.tobytes())
        h.update(np.float32(bc.sigma).tobytes())
    assert h.hexdigest()[:16] == GOLDEN_4BIT


GOLDEN_4BIT = "f0949d5d9053efad"   # numpy Generator(PCG64).normal stream; re-derive only with a SPEC change


def test_midtread_zero_preserving_and_symmetric():
    rule = q.RowRule(comp_bits=4)
    lo, hi = rule.main_range
    assert (lo, hi) == (-7, 7)
    c = q.quantise(np.array([0.0, -0.0, 100.0, -100.0], np.float32),
                   np.zeros(4, np.float32), F32(1.0), (lo, hi), False)
    assert c[:, 0].tolist() == [0, 0, 7, -7]
    v = q.decode(c, rule.s_main_f32, False)
    assert v[0] == 0 and v[1] == 0
    assert v[2] == -v[3]
    assert rule.heavy_range == (-127, 127)


def test_midrise_ranges_and_half_offset():
    rule = q.RowRule(comp_bits=5, mode=q.MODE_MIDRISE)
    assert rule.main_range == (-16, 15)
    assert rule.heavy_range == (-128, 127)
    c = q.quantise(np.array([0.0, -1e-9, 1e9, -1e9], np.float32),
                   np.zeros(4, np.float32), F32(0.5), rule.main_range, True)
    assert c[:, 0].tolist() == [0, -1, 15, -16]
    v = q.decode(c, F32(0.2), True)
    assert v[0].real == F32(0.5) * F32(0.2)               # zero is NOT preserved
    assert v[1].real == -v[0].real


def test_rint_is_half_to_even():
    c = q.quantise(np.array([0.5, 1.5, 2.5, -0.5, -2.5], np.float32),
                   np.zeros(5, np.float32), F32(1.0), (-7, 7), False)
    assert c[:, 0].tolist() == [0, 2, 2, 0, -2]


def test_nonfinite_and_uninitialised_give_zero_codes():
    c = q.quantise(np.array([np.nan, np.inf, -np.inf, 3.0], np.float32),
                   np.zeros(4, np.float32), F32(1.0), (-7, 7), False)
    assert c[:, 0].tolist() == [0, 0, 0, 3]
    c0 = q.quantise(np.ones(3, np.float32), np.ones(3, np.float32), F32(0.0), (-7, 7), False)
    assert not c0.any()


def test_robust_estimate_unbiased_and_ignores_zeros():
    rng = np.random.default_rng(1)
    rule = q.RowRule()
    x = _noise(rng, (64, 2000), 1.7)
    re, im = x.real.astype(np.float32), x.imag.astype(np.float32)
    est, m = q.robust_estimate(re, im, F32(0), rule)
    assert abs(est / 1.7 - 1) < 0.01 and m == re.size
    # 60% of values flagged (exact zeros): estimate unchanged within noise
    re2, im2 = re.copy(), im.copy()
    re2[:, :1200] = 0
    im2[:, :1200] = 0
    est2, m2 = q.robust_estimate(re2, im2, F32(0), rule)
    assert m2 == 64 * 800 and abs(est2 / 1.7 - 1) < 0.02


def test_robust_estimate_excludes_hot_values_and_min_count():
    rng = np.random.default_rng(2)
    rule = q.RowRule()
    x = _noise(rng, (8, 500), 1.0)
    re, im = x.real.astype(np.float32), x.imag.astype(np.float32)
    est0, _ = q.robust_estimate(re, im, F32(1.0), rule)
    re[:, :100] = 50.0                                    # 20% hot, > 6 sigma_prev
    est1, m1 = q.robust_estimate(re, im, F32(1.0), rule)
    assert m1 == 8 * 400
    assert abs(est1 / est0 - 1) < 0.05
    # too few usable values: no estimate
    z = np.zeros((8, 500), np.float32)
    z[0, :10] = 1.0
    assert q.robust_estimate(z, z.copy(), F32(0), rule)[0] is None


def test_ema_initialises_then_tracks_with_alpha():
    rule = q.RowRule()
    alpha = rule.alpha_f32
    assert abs(float(alpha) - (1 - math.exp(-0.134217728 / 10.0))) < 1e-7
    s = q.ema_update(F32(0), F32(2.0), rule)
    assert s == F32(2.0)
    s1 = q.ema_update(s, F32(3.0), rule)
    assert s1 == F32(F32(2.0) + F32(alpha * F32(1.0)))
    assert q.ema_update(s1, None, rule) == s1
    # ~tau: after 10 s of blocks at a new level, ~63% of the step
    s = F32(1.0)
    for _ in range(int(round(10.0 / q.BLOCK_S))):
        s = q.ema_update(s, F32(2.0), rule)
    assert 1.6 < float(s) < 1.66


def test_equal_noise_per_value_and_natural_weighting():
    """sqrt(n) normalisation: codes of n=1 and n=3936 values have the same
    distribution; decode * sqrt(n) recovers x / sigma (efficiency ~0.9935)."""
    rng = np.random.default_rng(5)
    rule = q.RowRule(comp_bits=4)
    V = 400
    n = np.where(np.arange(V) < 200, 1, 3936)
    rq = q.RowQuantiser(rule, n, n_heavy=0)
    x = None
    for _ in range(3):
        x = _noise(rng, (256, V), 0.8) * np.sqrt(n)[None, :].astype(np.float32)
        bc = rq.process_block(x)
    lo = np.abs(bc.codes_main[:, :200].astype(float)).mean()
    hi = np.abs(bc.codes_main[:, 200:].astype(float)).mean()
    assert abs(lo / hi - 1) < 0.03
    v = q.natural_weight(q.decode(bc.codes_main, rule.s_main_f32, False), n)
    ref = x / bc.sigma
    rho = np.corrcoef(np.concatenate([v.real.ravel(), v.imag.ravel()]),
                      np.concatenate([ref.real.ravel(), ref.imag.ravel()]))[0, 1]
    assert abs(rho - 0.9935) < 0.002
    # natural-weight scale: ratio of decoded to true ~1 in sigma units
    ratio = np.vdot(ref, v).real / np.vdot(ref, ref).real
    assert abs(ratio - 1) < 0.03


def test_5bit_efficiency():
    rng = np.random.default_rng(6)
    rule = q.RowRule(comp_bits=5)
    rq = q.RowQuantiser(rule, np.ones(1000, int), 0)
    for _ in range(2):
        x = _noise(rng, (256, 1000), 1.0)
        bc = rq.process_block(x)
    v = q.decode(bc.codes_main, rule.s_main_f32, False)
    rho = np.corrcoef(np.r_[v.real.ravel(), v.imag.ravel()],
                      np.r_[x.real.ravel(), x.imag.ravel()])[0, 1]
    assert abs(rho - 0.99815) < 0.0007


def test_zero_block_decodes_to_zero_and_keeps_sigma():
    rule = q.RowRule()
    rng = np.random.default_rng(8)
    rq = q.RowQuantiser(rule, np.full(300, 50), n_heavy=10)
    rq.process_block(_noise(rng, (256, 300), 7.0))
    s0 = rq.sigma
    bc = rq.process_block(np.zeros((256, 300), np.complex64))     # array-burst excision
    assert rq.sigma == s0
    assert not bc.codes_main.any() and not bc.codes_heavy.any()
    assert not q.decode_row(bc.codes_heavy, bc.codes_main, rule).any()


def test_hot_value_does_not_compress_the_row():
    """Burst-independent scale: a 50-sigma value changes only its own codes."""
    rule = q.RowRule()
    rng = np.random.default_rng(9)
    n = np.full(500, 20)
    a = q.RowQuantiser(rule, n, 8)
    b = q.RowQuantiser(rule, n, 8)
    x0 = _noise(rng, (256, 500), 1.0) * np.float32(np.sqrt(20))
    a.process_block(x0)
    b.process_block(x0)
    x1 = _noise(rng, (256, 500), 1.0) * np.float32(np.sqrt(20))
    x2 = x1.copy()
    x2[17, 100] = 50 * np.sqrt(20) * a.sigma               # j=17: not in the estimate subsample
    ca, cb = a.process_block(x1), b.process_block(x2)
    assert ca.sigma == cb.sigma
    diff = np.argwhere(ca.codes_main != cb.codes_main)
    assert {(int(i), int(j)) for i, j, _ in diff} == {(17, 100 - 8)}
    assert cb.codes_main[17, 92, 0] == 7                    # clipped at the top


def test_heavy_clip_level():
    rule = q.RowRule()
    rq = q.RowQuantiser(rule, np.array([4000, 1]), n_heavy=1)
    rng = np.random.default_rng(10)
    x = _noise(rng, (256, 2), 1.0) * np.sqrt([4000, 1]).astype(np.float32)
    rq.process_block(x)
    sig = rq.sigma
    y = np.zeros((256, 2), np.complex64)
    y[0, 0] = 10.0 * sig * np.sqrt(4000)            # 10 sigma: below the 12.7 clip
    y[1, 0] = 20.0 * sig * np.sqrt(4000)            # 20 sigma: clipped
    y[2:, 1] = 1.0                                  # keep the estimate alive
    bc = rq.process_block(y)
    assert bc.sigma == sig                          # too few values: no update
    assert bc.codes_heavy[0, 0, 0] == 100
    assert bc.codes_heavy[1, 0, 0] == 127


def test_v1_rowmax_matches_tx():
    torch = pytest.importorskip("torch")  # noqa: F841
    from dsart.transport.tx import TransportTx
    rng = np.random.default_rng(12)
    rows = _noise(rng, (6, 333), 3.0)
    rows[2] = 0
    q_ref, s_ref = TransportTx._encode_rows(rows, 16)
    q_new, s_new = q.v1_rowmax_cint8(rows)
    np.testing.assert_array_equal(q_new, q_ref)
    np.testing.assert_array_equal(s_new, s_ref)

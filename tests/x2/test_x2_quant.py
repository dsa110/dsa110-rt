"""Row rule reference (SPEC §4, frozen with Phase 0): arithmetic, statistics,
zero / no-data handling, warm-up, row guard; on WS-P's real value maps."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dsart.transport import prod_frame2 as f2
from dsart.x2 import quant as q

_spec = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec)
sys.modules["wst_util"] = wu
_spec.loader.exec_module(wu)

F32 = np.float32
RULE0 = q.RowRule(warmup_blocks=0)            # most tests skip the warm-up


def _noise(rng, shape, sigma=1.0):
    return (rng.normal(0, sigma, shape) + 1j * rng.normal(0, sigma, shape)).astype(np.complex64)


# ---------------------------------------------------------------------------
# Independent scalar implementation (element loops, float32 scalars only).
# ---------------------------------------------------------------------------

def _lowmed(vals):
    vals = sorted(vals)
    return vals[(len(vals) - 1) // 2]


def _scalar_block(x, n_contrib, n_heavy, sigma_prev, blocks_seen, rule):
    S, V = x.shape
    r = [F32(1.0 / math.sqrt(float(n))) for n in n_contrib]
    re = [[F32(F32(x[j, v].real) * r[v]) for v in range(V)] for j in range(S)]
    im = [[F32(F32(x[j, v].imag) * r[v]) for v in range(V)] for j in range(S)]
    k = F32(rule.robust_k)
    sigma = F32(sigma_prev)
    thr = F32(F32(rule.used_floor) * sigma)                 # v1.1 used threshold

    def unused(a, b):
        return abs(a) <= thr and abs(b) <= thr

    if blocks_seen >= rule.warmup_blocks:
        comps, m, M = [], 0, 0
        for j in range(0, S, rule.est_stride):
            for v in range(V):
                M += 1
                if unused(re[j][v], im[j][v]):
                    continue
                m += 1
                comps += [abs(re[j][v]), abs(im[j][v])]
        if m >= max(rule.min_count, math.ceil(0.5 * M)):
            est = max(F32(k * _lowmed(comps)), F32(1e-30))
            if sigma == 0:
                sigma = est
            elif rule.reset_factor > 0 and est > F32(F32(rule.reset_factor) * sigma):
                sigma = est                                    # v1.1 reset on a step up
            else:
                sigma = F32(sigma + F32(rule.alpha_f32 * F32(est - sigma)))
    out = np.zeros((S, V, 2), np.int64)
    guard = np.zeros(S, bool)
    need = max(rule.min_count, math.ceil(0.5 * V))
    for j in range(S):
        sig = sigma
        if sigma > 0:
            comps = [c for v in range(V) if not unused(re[j][v], im[j][v])
                     for c in (abs(re[j][v]), abs(im[j][v]))]
            if len(comps) // 2 >= need:
                srow = F32(k * _lowmed(comps))
                if srow > F32(F32(rule.guard_factor) * sigma):
                    sig, guard[j] = srow, True
        for v in range(V):
            heavy = v < n_heavy
            s = rule.s_heavy_f32 if heavy else rule.s_main_f32
            lo, hi = rule.heavy_range if heavy else rule.main_range
            nd = -128 if heavy else -(1 << (rule.comp_bits - 1))
            step = F32(s * sig)
            for c, val in enumerate((re[j][v], im[j][v])):
                if not step > 0:
                    out[j, v, c] = nd
                    continue
                qq = F32(val * F32(F32(1.0) / step))
                out[j, v, c] = min(hi, max(lo, round(float(qq))))          # round(): half-even
    return out, sigma, guard


@pytest.mark.parametrize("bits", [4, 5])
def test_vectorised_matches_scalar_reference(bits):
    rule = q.RowRule(comp_bits=bits, est_stride=4, warmup_blocks=1)
    rng = np.random.default_rng(11 + bits)
    V, nh, S = 37, 5, 12
    n = rng.integers(1, 4000, size=V)
    rq = q.RowQuantiser(rule, n, nh)
    for blk in range(4):
        x = _noise(rng, (S, V), 1.0) * np.sqrt(n)[None, :].astype(np.float32) * F32(3.0)
        x[2, 3] = 0                                          # flagged value
        if blk >= 2:
            x[4, 7] *= 200                                   # hot value
            x[6] *= 4                                        # noise surge -> row guard
        sigma_prev, seen = rq.sigma, rq.blocks_seen
        bc = rq.process_block(x)
        ref, ref_sigma, ref_guard = _scalar_block(x, n, nh, sigma_prev, seen, rule)
        assert bc.sigma == ref_sigma
        np.testing.assert_array_equal(bc.guarded, ref_guard)
        np.testing.assert_array_equal(bc.codes_heavy, ref[:, :nh])
        np.testing.assert_array_equal(bc.codes_main, ref[:, nh:])
        if blk == 0:
            assert not bc.sigma and (bc.codes_main == -(1 << (bits - 1))).all()   # warm-up
        if blk >= 2:
            assert bc.guarded[6] and bc.guarded.sum() == 1


def test_golden_vector_is_stable():
    """Regression guard for the packer/decoder implementations: a fixed
    input must keep producing the same codes (change only with the SPEC)."""
    rule = q.RowRule(comp_bits=4, warmup_blocks=1)
    rng = np.random.default_rng(20261022)
    V, nh, S = 64, 4, 256
    n = rng.integers(1, 3936, size=V)
    rq = q.RowQuantiser(rule, n, nh)
    h = hashlib.sha256()
    for b in range(4):
        x = _noise(rng, (S, V), 2.5) * np.sqrt(n)[None, :].astype(np.float32)
        if b == 3:
            x[100:104] *= 3
        bc = rq.process_block(x)
        for a in (bc.codes_heavy, bc.codes_main, np.float32(bc.sigma), bc.sigma_row, bc.guarded):
            h.update(np.asarray(a).tobytes())
    assert h.hexdigest()[:16] == GOLDEN_4BIT


GOLDEN_4BIT = "ea340eea9b8a7c0b"   # numpy Generator(PCG64).normal stream; re-derive only with a SPEC change


def test_ranges_no_data_and_zero():
    for bits, cmax in ((4, 7), (5, 15), (6, 31)):
        rule = q.RowRule(comp_bits=bits)
        assert rule.main_range == (-cmax, cmax) and q.no_data_code(bits) == -cmax - 1
    rule = q.RowRule()
    c = q.quantise(np.array([0.0, -0.0, 100.0, -100.0, np.nan, np.inf], np.float32),
                   np.zeros(6, np.float32), F32(1.0), rule.main_range, -8)
    assert c[:, 0].tolist() == [0, 0, 7, -7, -8, -8]
    v = q.decode(c, rule.s_main_f32, 4)
    assert v[0] == 0 and v[1] == 0 and v[4] == 0 and v[5] == 0       # NO_DATA -> 0
    assert v[2] == -v[3]
    assert q.decode(np.array([[-128, 127]]), rule.s_heavy_f32, 8)[0] == 1j * F32(127) * rule.s_heavy_f32
    assert rule.heavy_range == (-127, 127)
    nd = q.quantise(np.ones(3, np.float32), np.ones(3, np.float32), F32(0.0), (-7, 7), -8)
    assert (nd == -8).all()                                           # sigma 0 -> NO_DATA


def test_rint_is_half_to_even():
    c = q.quantise(np.array([0.5, 1.5, 2.5, -0.5, -2.5], np.float32),
                   np.zeros(5, np.float32), F32(1.0), (-7, 7), -8)
    assert c[:, 0].tolist() == [0, 2, 2, 0, -2]


def test_steps_and_heavy_threshold_constants():
    assert q.RowRule(comp_bits=4).s_main_f32 == F32(0.3534)
    assert q.RowRule(comp_bits=5).s_main_f32 == F32(0.1932)
    assert abs(float(q.s_heavy_for(3936, 10202112)) - 0.0773) < 5e-5        # n_sub 2
    assert abs(float(q.s_heavy_for(1968, 10202112)) - 0.0547) < 5e-5        # n_sub 4
    assert q.RowRule.for_format(10, 4).s_heavy_f32 == F32(0.0547)
    assert q.RowRule.for_format(8, 2).comp_bits == 4
    alpha = float(q.RowRule().alpha_f32)
    assert abs(alpha - (1 - math.exp(-0.134217728 / 3.0))) < 1e-7


@pytest.mark.parametrize("fmt", [("gridded", 2, 8), ("gridded", 4, 10), ("baselines", 4, 8)])
def test_real_maps_heavy_threshold_and_steps(fmt):
    """WS-P's maps: heavy = n >= 3e-5 N_image, heaviest value clips at 2x
    its S/N for a 250-sigma image with the n_sub default s_heavy."""
    vm = wu.value_map(*fmt)
    rule = q.RowRule.for_format(fmt[2], fmt[1])
    n_max = max(int(s.n_contrib.max()) for s in vm.streams)
    for s in vm.streams:
        if s.n_heavy:
            assert s.n_contrib[: s.n_heavy].min() >= 3e-5 * vm.n_image
        assert (s.n_contrib[s.n_heavy:] < 3e-5 * vm.n_image).all()
    clip = 127 * float(rule.s_heavy_f32)
    snr_at_rail = 250 * math.sqrt(n_max / vm.n_image)
    assert clip >= 2 * snr_at_rail * 0.999


def test_robust_estimate_unbiased_and_50pct_skip():
    rng = np.random.default_rng(1)
    rule = q.RowRule()
    x = _noise(rng, (64, 2000), 1.7)
    re, im = x.real.astype(np.float32), x.imag.astype(np.float32)
    est, m = q.robust_estimate(re, im, rule)
    assert abs(est / 1.7 - 1) < 0.01 and m == re.size
    re2, im2 = re.copy(), im.copy()
    re2[:, :800] = 0                                   # 40% flagged: unchanged within noise
    im2[:, :800] = 0
    est2, m2 = q.robust_estimate(re2, im2, rule)
    assert m2 == 64 * 1200 and abs(est2 / 1.7 - 1) < 0.02
    re2[:, :1100] = 0                                  # 55% flagged: skipped
    im2[:, :1100] = 0
    assert q.robust_estimate(re2, im2, rule) == (None, 64 * 900)


def test_ema_initialises_then_tracks_with_tau_3s():
    rule = q.RowRule()
    s = q.ema_update(F32(0), F32(2.0), rule)
    assert s == F32(2.0)
    s1 = q.ema_update(s, F32(3.0), rule)                # 3.0 = 1.5 x 2.0: not above, so EMA
    assert s1 == F32(F32(2.0) + F32(rule.alpha_f32 * F32(1.0)))
    assert q.ema_update(s1, None, rule) == s1
    s = F32(1.0)
    for _ in range(int(round(3.0 / q.BLOCK_S))):       # ~tau: 63% of a x1.4 step
        s = q.ema_update(s, F32(1.4), rule)
    assert 1.24 < float(s) < 1.264
    v10 = q.RowRule(reset_factor=0.0)                   # v1.0: a x2 step is tracked by the EMA
    s = F32(1.0)
    for _ in range(int(round(3.0 / q.BLOCK_S))):
        s = q.ema_update(s, F32(2.0), v10)
    assert 1.6 < float(s) < 1.66


def test_v11_step_up_resets_sigma_hat():
    """v1.1: est > f32(1.5) * sigma_hat sets sigma_hat = est in one block; a step down and a
    step up to exactly 1.5 x keep the EMA."""
    rule = q.RowRule()
    assert q.ema_update(F32(1.0), F32(2.0), rule) == F32(2.0)
    assert q.ema_update(F32(1.0), F32(1.5), rule) == F32(F32(1.0) + F32(rule.alpha_f32 * F32(0.5)))
    assert q.ema_update(F32(2.0), F32(0.5), rule) == F32(F32(2.0) + F32(rule.alpha_f32 * F32(-1.5)))
    assert q.ema_update(F32(1.0), F32(2.0), q.RowRule(reset_factor=0.0)) < F32(1.1)


def test_v11_used_threshold():
    """v1.1: |re| and |im| both <= f32(1e-6 * sigma_hat_prev) -> unused; 0 threshold = v1.0."""
    rule = q.RowRule()
    thr = rule.used_threshold(F32(2.0))
    assert thr == F32(F32(1e-6) * F32(2.0))
    re = np.array([0, 1e-7, 3e-6, 0, np.inf, -0.0], np.float32)
    im = np.array([0, 1e-7, 0, 1e-5, 1.0, 0.0], np.float32)
    np.testing.assert_array_equal(q._used(re, im, thr), [False, False, True, True, False, False])
    np.testing.assert_array_equal(q._used(re, im), [False, True, True, True, False, False])


@pytest.mark.parametrize("v11", [True, False])
def test_excision_then_release(v11):
    """A long whole-cube excision whose static-sky residue is ~1e-9 of the noise: v1.1 holds
    sigma_hat through it (residue unused -> no update) and the release guards no row; v1.0
    lets sigma_hat collapse onto the residue and the release guards every row."""
    kw = {} if v11 else dict(used_floor=0.0, reset_factor=0.0)
    rule = q.RowRule(comp_bits=4, warmup_blocks=1, **kw)
    rng = np.random.default_rng(3)
    S, V = 64, 200
    n = rng.integers(10, 2000, size=V)
    rq = q.RowQuantiser(rule, n, 4)
    amp = np.sqrt(n)[None, :].astype(np.float32)
    for _ in range(10):
        rq.process_block(_noise(rng, (S, V), 1.0) * amp)
    s0 = rq.sigma
    for _ in range(600):                                    # excised: residue only
        rq.process_block(_noise(rng, (S, V), 1e-9) * amp)
    s_exc = rq.sigma
    guarded = [int(rq.process_block(_noise(rng, (S, V), 1.0) * amp).guarded.sum()) for _ in range(5)]
    if v11:
        assert s_exc == s0
        assert sum(guarded) == 0
    else:
        assert s_exc < F32(1e-6) * s0
        assert guarded[0] == S


def test_warmup_then_initialise_from_first_valid_block():
    rule = q.RowRule(warmup_blocks=8)
    rng = np.random.default_rng(4)
    rq = q.RowQuantiser(rule, np.full(200, 30), n_heavy=5)
    for b in range(8):
        bc = rq.process_block(_noise(rng, (128, 200), 50.0))        # warm-up: 5x noise
        assert bc.sigma == 0 and bc.est is None
        assert (bc.codes_main == -8).all() and (bc.codes_heavy == -128).all()
        assert not q.decode_row(bc.codes_heavy, bc.codes_main, rule).any()
    bc = rq.process_block(np.zeros((128, 200), np.complex64))      # all flagged: skipped
    assert bc.sigma == 0 and (bc.codes_main == -8).all()
    x = _noise(rng, (128, 200), 10.0) * F32(np.sqrt(30))
    bc = rq.process_block(x)
    assert bc.est is not None and bc.sigma == bc.est and abs(bc.sigma / 10 - 1) < 0.05


def test_zero_block_decodes_to_zero_and_keeps_sigma():
    rng = np.random.default_rng(8)
    rq = q.RowQuantiser(RULE0, np.full(300, 50), n_heavy=10)
    rq.process_block(_noise(rng, (256, 300), 7.0))
    s0 = rq.sigma
    bc = rq.process_block(np.zeros((256, 300), np.complex64))     # array-burst excision
    assert rq.sigma == s0 and bc.est is None
    assert not bc.codes_main.any() and not bc.codes_heavy.any()     # code 0, not NO_DATA
    assert not q.decode_row(bc.codes_heavy, bc.codes_main, RULE0).any()


def test_hot_value_does_not_compress_the_row():
    rng = np.random.default_rng(9)
    n = np.full(500, 20)
    a = q.RowQuantiser(RULE0, n, 8)
    b = q.RowQuantiser(RULE0, n, 8)
    x0 = _noise(rng, (256, 500), 1.0) * np.float32(np.sqrt(20))
    a.process_block(x0)
    b.process_block(x0)
    x1 = _noise(rng, (256, 500), 1.0) * np.float32(np.sqrt(20))
    x2 = x1.copy()
    x2[17, 100] = 50 * np.sqrt(20) * a.sigma               # j=17: not in the estimate subsample
    ca, cb = a.process_block(x1), b.process_block(x2)
    assert ca.sigma == cb.sigma and not cb.guarded.any()
    diff = np.argwhere(ca.codes_main != cb.codes_main)
    assert {(int(i), int(j)) for i, j, _ in diff} == {(17, 100 - 8)}
    assert cb.codes_main[17, 92, 0] == 7                    # clipped at the top level


def test_heavy_clip_level():
    rq = q.RowQuantiser(RULE0, np.array([4000] + [1] * 63), n_heavy=1)
    rng = np.random.default_rng(10)
    nn = np.sqrt([4000] + [1] * 63).astype(np.float32)
    rq.process_block(_noise(rng, (256, 64), 1.0) * nn)
    sig = rq.sigma
    y = _noise(rng, (256, 64), 1.0) * nn
    y[16, 0] = 9.0 * sig * np.sqrt(4000)            # 9 sigma: below the 9.8 clip
    y[17, 0] = 20.0 * sig * np.sqrt(4000)           # 20 sigma: clipped
    bc = rq.process_block(y)
    sh = float(RULE0.s_heavy_f32 * bc.sigma)
    assert bc.codes_heavy[16, 0, 0] == round(9.0 * float(sig) / sh)
    assert bc.codes_heavy[17, 0, 0] == 127


def test_equal_noise_per_value_and_natural_weighting_on_real_map():
    """sqrt(n) normalisation on WS-P's gridded n_sub 2 stream: codes of small
    and large n have the same distribution; decode * sqrt(n) recovers x/sigma
    (efficiency 0.9935, 4-bit mid-tread)."""
    sv = wu.value_map("gridded", 2, 8).streams[5]
    rule = q.RowRule.for_format(8, 2, warmup_blocks=0)
    rq = q.RowQuantiser(rule, sv.n_contrib, sv.n_heavy)
    rng = np.random.default_rng(5)
    nn = np.sqrt(sv.n_contrib).astype(np.float32)
    x = None
    for _ in range(3):
        x = _noise(rng, (256, sv.n_values), 0.8) * nn[None, :]
        bc = rq.process_block(x)
    assert not bc.guarded.any()
    cm = np.abs(bc.codes_main.astype(float))
    nm = sv.n_contrib[sv.n_heavy:]
    lo, hi = cm[:, nm <= np.median(nm)].mean(), cm[:, nm > np.median(nm)].mean()
    assert abs(lo / hi - 1) < 0.03
    v = q.natural_weight(q.decode_row(bc.codes_heavy, bc.codes_main, rule), sv.n_contrib)
    ref = x / bc.sigma
    vm_, rm_ = v[:, sv.n_heavy:], ref[:, sv.n_heavy:]
    rho = np.corrcoef(np.r_[vm_.real.ravel(), vm_.imag.ravel()],
                      np.r_[rm_.real.ravel(), rm_.imag.ravel()])[0, 1]
    assert abs(rho - 0.9935) < 0.002
    ratio = np.vdot(ref, v).real / np.vdot(ref, ref).real
    assert abs(ratio - 1) < 0.03


def test_5bit_efficiency():
    rng = np.random.default_rng(6)
    rule = q.RowRule(comp_bits=5, warmup_blocks=0)
    rq = q.RowQuantiser(rule, np.ones(1000, int), 0)
    for _ in range(2):
        x = _noise(rng, (256, 1000), 1.0)
        bc = rq.process_block(x)
    v = q.decode(bc.codes_main, rule.s_main_f32, 5)
    rho = np.corrcoef(np.r_[v.real.ravel(), v.imag.ravel()],
                      np.r_[x.real.ravel(), x.imag.ravel()])[0, 1]
    assert abs(rho - 0.99815) < 0.0007


def test_row_guard_noise_surge_and_quiet_noise():
    """Never fires on normal noise; a x1.6+ broadband surge is coded at its own
    scale (saturated codes back to the normal ~2%); Phase 0 rowrule §6."""
    sv = wu.value_map("gridded", 2, 8).streams[0]
    rule = q.RowRule.for_format(8, 2, warmup_blocks=0)
    rq = q.RowQuantiser(rule, sv.n_contrib, sv.n_heavy)
    rng = np.random.default_rng(12)
    nn = np.sqrt(sv.n_contrib).astype(np.float32)[None, :]
    n_fired = 0
    for _ in range(6):                                        # 1536 quiet rows
        bc = rq.process_block(_noise(rng, (256, sv.n_values), 1.0) * nn)
        n_fired += int(bc.guarded.sum())
    assert n_fired == 0
    x = _noise(rng, (256, sv.n_values), 1.0) * nn
    x[10:20] *= 2.0
    x[30] *= 1.6
    x[40] *= 1.3                                              # below the 1.5 trip
    bc = rq.process_block(x)
    assert set(np.nonzero(bc.guarded)[0]) == set(range(10, 20)) | {30}
    sat = (np.abs(bc.codes_main) == 7).mean(axis=(1, 2))
    assert sat[10:20].max() < 0.04 and sat[40] > 0.04          # guarded rows look like noise
    assert (bc.sigma_row[10:20] > 1.9 * bc.sigma).all()
    np.testing.assert_allclose(bc.step_main[10], bc.sigma_row[10] * rule.s_main_f32, rtol=0)


def test_row_guard_point_source_threshold():
    """A point source at the phase centre trips the guard only above ~500
    sigma per sample (image S/N carried by one row, gridded n_sub 2)."""
    vm = wu.value_map("gridded", 2, 8)
    sv = vm.streams[0]
    rule = q.RowRule.for_format(8, 2, warmup_blocks=0)
    rq = q.RowQuantiser(rule, sv.n_contrib, sv.n_heavy)
    rng = np.random.default_rng(13)
    nn = np.sqrt(sv.n_contrib).astype(np.float32)
    rq.process_block(_noise(rng, (256, sv.n_values), 1.0) * nn[None, :])
    x = _noise(rng, (256, sv.n_values), 1.0) * nn[None, :]
    # per-value signal: n_v * a with a = rho / sqrt(N) per contribution (sigma 1)
    for j, rho in ((20, 300.0), (21, 400.0), (22, 1500.0)):
        x[j] += (sv.n_contrib * rho / np.sqrt(vm.n_image)).astype(np.float32)
    bc = rq.process_block(x)
    assert not bc.guarded[20] and not bc.guarded[21] and bc.guarded[22]


def test_codes_round_trip_through_frames_on_real_map():
    vm = wu.value_map("baselines", 4, 10)
    rule = q.RowRule.for_format(10, 4, warmup_blocks=0)
    blocks = wu.stream_blocks(vm, [0, 9, 63], rule=rule, n_samples=40, seed=2)
    for s, bc in blocks.items():
        dgs = wu.datagrams_for(vm, bc, stream=s, dm=3, seq_first=800, t_int_factor=16, rule=rule)
        got_h, got_m = [], []
        for d in dgs:
            h, p = f2.unpack_frame(d)
            assert h.s_main == float(rule.s_main_f32) and h.bits_main == 10
            a, b = f2.unpack_samples(p, h.k, h.n_values, h.n_heavy, 10, 16)
            got_h.append(a)
            got_m.append(b)
        np.testing.assert_array_equal(np.concatenate(got_h), bc.codes_heavy)
        np.testing.assert_array_equal(np.concatenate(got_m), bc.codes_main)


def test_v1_rowmax_matches_tx():
    pytest.importorskip("torch")
    from dsart.transport.tx import TransportTx
    rng = np.random.default_rng(12)
    rows = _noise(rng, (6, 333), 3.0)
    rows[2] = 0
    q_ref, s_ref = TransportTx._encode_rows(rows, 16)
    q_new, s_new = q.v1_rowmax_cint8(rows)
    np.testing.assert_array_equal(q_new, q_ref)
    np.testing.assert_array_equal(s_new, s_ref)

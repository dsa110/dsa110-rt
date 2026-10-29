"""Fast CPU kernels (x2/search/cpu_fast.py, ``EngineConfig.cpu_impl="fast"``), CPU.

* ``codec.unpack_frame2_rows`` (the fast unpack) == the bit-expansion
  reference for 3..8-bit components, heavy pairs and padding, bit for bit;
* ``k1_fast`` == ``combine.k1_twin`` BIT FOR BIT (many-to-one cells, a cell
  with > 14 entries per stream, conjugate entries, ring wrap, per-stream
  shifts, offset 0 and 0.5);
* ``k3b_fast`` == ``filters.k3b_twin`` to fp32 summation order (b3x and
  boxcars; ties only differ at fp32 near-ties);
* the engine's CPU path with ``cpu_impl="fast"`` finds the twin path's
  candidates (same pixel / fine / centre / filter, S/N to 1e-5).
"""
from __future__ import annotations

import numpy as np
import pytest

from dsart.x2.search.codec import RowFormat, pack_frame2_rows, unpack_frame2_rows, \
    unpack_frame2_rows_ref
from dsart.x2.search.combine import k1_twin
from dsart.x2.search.cpu_fast import k1_fast, k3b_fast
from dsart.x2.search.csr import csr_from_value_cells
from dsart.x2.search.filters import FilterBank, k3b_twin


@pytest.mark.parametrize("b", [3, 4, 5, 6, 8])
def test_fast_unpack_equals_the_reference(b):
    rng = np.random.default_rng(b)
    fmt = RowFormat("frame2", n_values=301, n_heavy=17, bits=b)
    re = rng.integers(-(1 << (b - 1)), 1 << (b - 1), size=(9, 301))
    im = rng.integers(-(1 << (b - 1)), 1 << (b - 1), size=(9, 301))
    re[:, :17] = rng.integers(-128, 128, size=(9, 17))
    im[:, :17] = rng.integers(-128, 128, size=(9, 17))
    raw = pack_frame2_rows(re, im, fmt)
    pad = np.concatenate([raw, rng.integers(0, 256, size=(9, 5), dtype=np.uint8)], axis=1)
    for got, want in zip(unpack_frame2_rows(pad, fmt), unpack_frame2_rows_ref(pad, fmt)):
        assert got.dtype == want.dtype == np.int8
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(unpack_frame2_rows(raw, fmt)[0], re)


def _k1_case(seed=0, N=16, S=5, V=30, R=96, F=6):
    rng = np.random.default_rng(seed)
    cells = []
    for s in range(S):
        c = rng.choice(N * N, size=V, replace=False)
        c[:6] = c[6:12]                                  # many-to-one within a stream
        c[12:26] = 37                                    # one cell carried 14 x per stream
        c[-2:] = -1
        cells.append(c.astype(np.int64))
    csr = csr_from_value_cells(cells, n_grid=N)
    codes = rng.integers(-8, 8, size=(S, V, R, 2)).astype(np.int8)
    cv = rng.uniform(0.05, 0.4, size=(S, V)).astype(np.float32)
    wv = np.sqrt(rng.integers(1, 4000, size=(S, V)).astype(np.float64)).astype(np.float32)
    base = rng.integers(0, 60, size=S)
    shifts = (base[None, :] + rng.integers(0, 7, size=(F, S))).astype(np.int32)
    return csr, codes, cv, wv, shifts


@pytest.mark.parametrize("offset", [0.0, 0.5])
@pytest.mark.parametrize("t_lo,n_t", [(90, 50), (3, 17)])
def test_k1_fast_equals_k1_twin_bit_for_bit(offset, t_lo, n_t):
    csr, codes, cv, wv, shifts = _k1_case()
    assert csr.max_entries > 64 and np.asarray(csr.ent_conj).any()
    kw = dict(c_v=cv, offset=offset, csr=csr, shifts=shifts, t_lo=t_lo, n_t=n_t)
    for w in (wv, None):
        want = k1_twin(codes, w_v=w, **kw)
        got = k1_fast(codes, w_v=w, **kw)
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))   # incl. signed zeros


@pytest.mark.parametrize("bank", [FilterBank.b3x(), FilterBank.build((1, 2, 4, 8, 16, 32, 64),
                                                                      ("b",))])
def test_k3b_fast_equals_the_twin_to_fp32_summation_order(bank):
    rng = np.random.default_rng(5)
    R, F, P = 400, 3, 50
    ring = rng.normal(size=(R, F, P)).astype(np.float16)
    ring[210:213, 1, 17] += np.float16(30.0)
    ring[100, 2, 5] = np.float16(600.0)                  # clipped at 250
    kw = dict(ring_row0=7, c_first=207, n_c=128, sigma_inv=np.array([1.0, 0.9, 1.1]),
              s_inv=1.0 / bank.seeds, bank=bank, clip=250.0, chunk=64,
              tap_pixels=np.array([3, 17, 40]))
    bw, tw, pw = k3b_twin(ring, **kw)
    bf, tf, pf = k3b_fast(ring, pix_block=16, **kw)
    np.testing.assert_allclose(bf, bw, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(pf / bank.seeds[:, None, None, None],
                               pw / bank.seeds[:, None, None, None], rtol=1e-5, atol=1e-3)
    assert np.mean(tf == tw) > 0.995                      # fp32 near-ties only
    i = np.unravel_index(np.argmax(bw), bw.shape)
    assert tf[i] == tw[i] and bf[i] == pytest.approx(bw[i], rel=1e-5)   # the burst itself


def test_engine_cpu_fast_finds_the_twin_candidates():
    from dsart.x2.search.engine import EngineConfig, OwnerEngine
    from dsart.x2.search.geometry import owner_plan_from_tiers
    from dsart.x2.search.ring import MemoryRing
    from dsart.x2.search.synth import Burst, SyntheticSky, make_dm_inputs, random_value_cells

    N, V = 32, 20
    tiers = [dict(name="A", t_native=16, coarse=np.array([53.0]), fine=np.array([50.0, 53.0, 56.0]),
                  f2c=np.zeros(3, int)),
             dict(name="B", t_native=32, coarse=np.array([103.0]), fine=np.array([100.0, 106.0]),
                  f2c=np.zeros(2, int))]
    plan = owner_plan_from_tiers(tiers, half=0, n_sub=1, csf=2, n_halves=1)
    fmt = RowFormat("frame2", n_values=V, n_heavy=2, bits=4)
    cells, inputs = {}, {}
    for s in plan.dms:
        cells[s.d] = random_value_cells(16, V, N, radius=6, seed=s.d + 9, n_unused=2)
        inputs[s.d] = make_dm_inputs(cells[s.d], n_grid=N, fmt=fmt, step_main=0.3534,
                                     step_heavy=0.1)
    out = {}
    for impl in ("twin", "fast"):
        ring = MemoryRing({s.d: fmt for s in plan.dms}, 16, t_buf=4096)
        sky = SyntheticSky(plan=plan, cells=cells, formats={s.d: fmt for s in plan.dms},
                           n_grid=N, bursts=[Burst(d=0, f=2, t=3 * 384 + 150, l=11, m=21,
                                                   amp=1.8)])
        for s in plan.dms:
            sky.fill(ring, s, 0, 4 * s.hop)
        eng = OwnerEngine(plan, inputs, ring, EngineConfig(
            n_grid=N, families=("b",), widths=(1, 2, 4), fft_mode="pruned", n_tap_pixels=32,
            layer2_n_burnin=0, cpu_impl=impl))
        eng.process_cycle(2)
        out[impl] = eng.process_cycle(3).candidates
    key = [(c.l, c.m, c.fine_global, c.event_specnum, c.filt) for c in out["twin"]]
    assert key == [(c.l, c.m, c.fine_global, c.event_specnum, c.filt) for c in out["fast"]]
    assert (11, 21) in [(c.l, c.m) for c in out["twin"]]
    np.testing.assert_allclose([c.snr for c in out["fast"]], [c.snr for c in out["twin"]],
                               rtol=1e-5)
    with pytest.raises(ValueError, match="cpu_impl"):
        OwnerEngine(plan, inputs, ring, EngineConfig(n_grid=N, families=("b",), widths=(1, 2, 4),
                                                     cpu_impl="numba")).process_cycle(3)

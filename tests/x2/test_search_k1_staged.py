"""K1 window staging (x2/search/k1_staged.py), CPU.

The numpy replay of the CUDA kernel's block / tile / entry-chunk / per-stream
window index arithmetic equals ``combine.k1_twin`` BIT FOR BIT across
geometries (cells per block, tile sizes that do not divide n_t, entry chunks
smaller than a cell, cells with > 64 entries, ring wrap, streams with very
different absolute shifts) -- and a window base off by one row is caught.
The CUDA kernel itself is compared with the twin on a GPU at CP1
(tests/x2/test_search_gpu.py).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dsart.x2.search.combine import k1_twin
from dsart.x2.search.csr import csr_from_value_cells
from dsart.x2.search.k1_staged import OPT, SMEM_CAP, k1_staged_twin, staged_geometry


def _case(seed=0, N=16, S=5, V=30, R=96, F=6, heavy_cell=True):
    rng = np.random.default_rng(seed)
    cells = []
    for s in range(S):
        c = rng.choice(N * N, size=V, replace=False)
        c[:6] = c[6:12]                                  # many-to-one within a stream
        if heavy_cell:
            c[12:26] = 37                                # one cell carried 14 x per stream
        c[-2:] = -1
        cells.append(c.astype(np.int64))
    csr = csr_from_value_cells(cells, n_grid=N)
    codes = rng.integers(-8, 8, size=(S, V, R, 2)).astype(np.int8)
    cv = rng.uniform(0.05, 0.4, size=(S, V)).astype(np.float32)
    wv = np.sqrt(rng.integers(1, 4000, size=(S, V)).astype(np.float64)).astype(np.float32)
    base = rng.integers(0, 60, size=S)                   # per-stream absolute offsets
    shifts = (base[None, :] + rng.integers(0, 7, size=(F, S))).astype(np.int32)
    return csr, codes, cv, wv, shifts


@pytest.mark.parametrize("geo", [
    dict(),                                              # defaults: one tile, one chunk
    dict(cpb=1, threads=32),                             # tile 42 < n_t
    dict(cpb=3, threads=16, smem_cap=600),               # tile 7, entry chunks of a few
    dict(cpb=4, threads=8, smem_cap=200),                # chunks of 1-2 entries
])
def test_staged_twin_equals_k1_twin_bit_for_bit(geo):
    csr, codes, cv, wv, shifts = _case()
    assert csr.max_entries > 64                          # no 64-entry cap
    n_t, t_lo = 50, 90                                   # rows wrap the 96-slot ring
    g = staged_geometry(shifts, n_t=n_t, **geo)
    assert g.smem <= geo.get("smem_cap", SMEM_CAP) and g.ec >= 1
    assert g.cpb * shifts.shape[0] * g.tile <= OPT * g.threads
    want = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=t_lo, n_t=n_t,
                   w_v=wv)
    got = k1_staged_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=t_lo,
                         n_t=n_t, w_v=wv, geom=g)
    assert np.array_equal(got.real, want.real) and np.array_equal(got.imag, want.imag)
    if geo.get("smem_cap", SMEM_CAP) < 1000:
        assert g.ec < csr.max_entries                    # chunking really exercised


def test_window_spans_the_per_stream_sweep_only():
    _csr, _codes, _cv, _wv, shifts = _case()
    g = staged_geometry(shifts, n_t=50)
    spread = int((shifts.max(0) - shifts.min(0)).max())
    assert g.spread == spread <= 6 and g.win_w == g.tile + spread
    assert int(shifts.max() - shifts.min()) > 20         # global range much wider: not staged


def test_wrong_window_base_is_caught():
    csr, codes, cv, wv, shifts = _case(seed=1)
    g = staged_geometry(shifts, n_t=50)
    bad = dataclasses.replace(g, smax=g.smax - 1)
    want = k1_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=90, n_t=50, w_v=wv)
    try:
        got = k1_staged_twin(codes, c_v=cv, offset=0.0, csr=csr, shifts=shifts, t_lo=90,
                             n_t=50, w_v=wv, geom=bad)
    except AssertionError:
        return                                           # index guard tripped
    assert not np.array_equal(got, want)

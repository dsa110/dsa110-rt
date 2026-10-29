"""Fast CPU kernels of the owner engine (``EngineConfig.cpu_impl="fast"``).

The numpy twins (``combine.k1_twin``, ``filters.k3b_twin``) are the
exactness references the CUDA kernels are tested against; they are Python
loops over entries x fine DMs / centres x lag states, built for N = 32 test
grids (at N = 256: K1 ~17 s per 64 rows, K3b ~128 s per tier-A cycle on
n02 -- I1's measurement). These are their vectorised equivalents for
offline replays and CPU end-to-end runs (promoted from I1's ``i1_util``):

* :func:`k1_fast` -- BIT-EXACT with ``k1_twin``: every stream's rows are
  decoded once with the twin's fp32 operations ``((code + off) * c_v) *
  w_v``; per fine DM each CSR entry's shifted window is a contiguous slice
  (a sliding-window view, gathered by (entry, offset)); the accumulation
  runs over entry position ``j`` in CSR order exactly as the twin (cells
  without a ``j``-th entry are skipped: the twin adds ``+0.0`` to an
  accumulator that can never be ``-0.0``, a no-op).
* :func:`k3b_fast` -- NOT bit-exact: each filter's box cascade as trailing
  prefix-sum differences in fp32 along time (the twin and the GPU kernels
  use running sums); same input ``clip(fl32(x16) * sigma_inv)``, same
  ``score * s_inv``, same tie order. Differs by fp32 summation order only
  (~1e-5 of ``||h||`` for q64); ``tk`` agrees except at fp32 near-ties.

The frame-v2 unpack's fast form is ``codec.unpack_frame2_rows`` itself (bit
for bit equal to the bit-expansion reference ``codec.unpack_frame2_rows_ref``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["k1_fast", "k3b_fast"]

_K1_TABLES: dict = {}


def _k1_tables(csr, V: int) -> list:
    """Per entry position ``j``: ``(cells, col, s, conj)`` of the cells
    that have a ``j``-th entry (CSR order), ``col = s * V + v``."""
    key = (id(csr), int(V))
    hit = _K1_TABLES.get(key)
    if hit is not None and hit[0] is csr:
        return hit[1]
    ptr = np.asarray(csr.cell_ptr, np.int64)
    ne = np.diff(ptr)
    out = []
    for j in range(int(ne.max(initial=0))):
        cells = np.nonzero(ne > j)[0]
        e = ptr[cells] + j
        s = np.asarray(csr.ent_stream, np.int64)[e]
        v = np.asarray(csr.ent_value, np.int64)[e]
        out.append((cells, s * int(V) + v, s, np.asarray(csr.ent_conj)[e].astype(bool)))
    _K1_TABLES[key] = (csr, out)
    return out


def k1_fast(codes: np.ndarray, *, c_v: np.ndarray, offset: float, csr, shifts: np.ndarray,
            t_lo: int, n_t: int, w_v: Optional[np.ndarray] = None) -> np.ndarray:
    """``combine.k1_twin`` (same arguments, same ``uv [F, n_t, n_act]``
    complex64), bit for bit."""
    S, V, R, _ = codes.shape
    F = int(shifts.shape[0])
    n_t = int(n_t)
    n_act = int(csr.n_active)
    cv = np.asarray(c_v, dtype=np.float32)
    wv = np.ones_like(cv) if w_v is None else np.asarray(w_v, dtype=np.float32)
    off = np.float32(offset)
    sh = np.asarray(shifts, np.int64)
    smax = sh.max(axis=0)
    n_u = n_t + int((smax - sh.min(axis=0)).max(initial=0))
    # decoded window per stream: absolute row t_lo - smax[s] + u, u < n_u
    xr = np.empty((S, V, n_u), np.float32)
    xi = np.empty((S, V, n_u), np.float32)
    for s in range(S):
        slots = np.mod(int(t_lo) - int(smax[s]) + np.arange(n_u), R)
        c = codes[s][:, slots]                                       # [V, n_u, 2]
        xr[s] = ((c[..., 0].astype(np.float32) + off) * cv[s][:, None]) * wv[s][:, None]
        xi[s] = ((c[..., 1].astype(np.float32) + off) * cv[s][:, None]) * wv[s][:, None]
    win_r = np.lib.stride_tricks.sliding_window_view(xr.reshape(S * V, n_u), n_t, axis=1)
    win_i = np.lib.stride_tricks.sliding_window_view(xi.reshape(S * V, n_u), n_t, axis=1)
    tabs = _k1_tables(csr, V)
    k = (np.float32(0.5) * csr.sign)[None, None, :]
    out = np.empty((F, n_t, n_act), np.complex64)
    for f in range(F):
        o = smax - sh[f]                                             # [S] window start
        acc_re = np.zeros((n_act, n_t), np.float32)
        acc_im = np.zeros((n_act, n_t), np.float32)
        for cells, col, s, cj in tabs:
            oe = o[s]
            acc_re[cells] += win_r[col, oe]
            vim = win_i[col, oe]
            vim[cj] = -vim[cj]
            acc_im[cells] += vim
        re = acc_re.T[None] * k
        im = acc_im.T[None] * k
        out[f] = (re + 1j * im).astype(np.complex64)[0]
    return out


def _box0(cs: np.ndarray, L: int) -> np.ndarray:
    """Trailing box sums of length ``L`` along axis 0 from the prefix sums
    ``cs`` of the series (zero before the start)."""
    out = cs.copy()
    out[L:] -= cs[:-L]
    return out


def k3b_fast(ring, *, ring_row0, c_first, n_c, sigma_inv, s_inv, bank, clip=250.0, chunk=64,
             tap_pixels=None, pix_block=128):
    """``filters.k3b_twin`` vectorised (same arguments and outputs; see the
    module docstring: fp32 summation order differs). The input is the
    twin's ``clip(fl32(x16) * sigma_inv)``; every filter's box cascade as
    trailing prefix-sum differences along time over a cache-sized pixel
    block (the first stage of every filter shares one prefix sum); windows
    of centres ``>= c_first`` never reach rows before the filter's first
    needed row, so no per-filter zeroing is needed; ``snr = score *
    s_inv[k]``; chunk maxima filter by filter with strict ``>`` (ties:
    earlier filter, then earlier centre -- the twin's order)."""
    R, F, P = ring.shape
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    n_ch = n_c // chunk
    K = len(bank)
    si = np.asarray(sigma_inv, np.float32)[None, :, None]
    sk = np.asarray(s_inv, np.float32)
    clip32 = np.float32(clip)
    lo = int(c_first) - bank.max_a
    hi = int(c_first) + int(n_c) + bank.max_d
    slots = np.mod(np.arange(lo, hi) - int(ring_row0), R)
    best = np.full((n_ch, F, P), -np.inf, np.float32)
    tk = np.zeros((n_ch, F, P), np.int32)
    taps_out = None
    if tap_pixels is not None:
        tap_pixels = np.asarray(tap_pixels, np.int64)
        taps_out = np.zeros((K, F, tap_pixels.size, n_c), np.float32)
    c0 = int(c_first) - lo
    for p0 in range(0, P, pix_block):
        p1 = min(P, p0 + pix_block)
        pb = p1 - p0
        x = ring[slots, :, p0:p1].astype(np.float32)                    # (T, F, pb)
        x *= si
        np.minimum(np.maximum(x, -clip32, out=x), clip32, out=x)
        cs1 = np.cumsum(x, axis=0, dtype=np.float32)
        sel = None
        if taps_out is not None:
            sel = np.nonzero((tap_pixels >= p0) & (tap_pixels < p1))[0]
        bb = best[:, :, p0:p1]
        tt = tk[:, :, p0:p1]
        for k, flt in enumerate(bank.filters):
            if not flt.stages:
                y = x
            else:
                y = _box0(cs1, int(flt.stages[0]))
                for L in flt.stages[1:]:
                    y = _box0(np.cumsum(y, axis=0, dtype=np.float32), int(L))
            e0 = c0 + flt.d
            score = y[e0:e0 + int(n_c)]                                   # (n_c, F, pb)
            if sel is not None and sel.size:
                taps_out[k][:, sel, :] = np.transpose(score[:, :, tap_pixels[sel] - p0],
                                                      (1, 2, 0))
            v = (score * sk[k]).reshape(n_ch, chunk, F, pb)
            vm = v.max(axis=1)
            ia = np.argmax(v == vm[:, None], axis=1).astype(np.int32)
            better = vm > bb
            np.copyto(bb, vm, where=better)
            np.copyto(tt, (ia << 8) | np.int32(k), where=better)
    return best, tk, taps_out

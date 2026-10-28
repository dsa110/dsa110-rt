"""K1 with shared-memory window reuse (CP1 ``--k1 staged``), CUDA + numpy twin.

Same contract and the same fp32 operation sequence as ``combine.k1_twin`` /
``kernels_triton.k1_combine`` (bit for bit): per (fine DM ``f``, output row
``t``, active cell), entries in CSR order ``(stream, conj, value)``, each
decoded ``((code + off) * c_v) * w_v`` (SPEC §4.5, two roundings), negated
imaginary part for a mirror entry, summed into one fp32 accumulator per
component starting at +0.0, then ``x 0.5 (-1)^(u+w)``.

What changes is the memory traffic, as in ``image.tiled_combine_cuda``'s
``sparse_combine_v4``: the Triton K1 re-reads every entry's codes once
per fine DM. Here one block owns ``CPB`` consecutive active cells (their
entries are one contiguous CSR range) and a tile of ``tile`` output rows,
and stages each entry's code WINDOW once in shared memory -- the rows
``t_lo + t0 - smax[s] .. + win_w`` with ``smax[s] = max_f shifts[f, s]``
and ``win_w = tile + max_s (smax[s] - smin[s])`` (the per-stream stage-2
sweep, ~50 rows in 2x; the base is PER STREAM, so streams with very
different absolute shifts cost nothing extra) -- then every fine DM of
the block reads it from shared memory at ``j = tl + smax[s] - shifts[f,
s]``. Entries are staged in chunks of ``ec`` (shared-memory bound), so
there is no cap on entries per cell; each thread keeps its ``OPT``
outputs' accumulators in registers across chunks, so the summation order
is the CSR order exactly. Threads map to outputs cell-fastest, so the
``CPB`` cells of a block write one 32-byte sector per (f, t).

FMA contraction cannot change the result: decode and accumulation use
``__fmul_rn`` / ``__fadd_rn``. :func:`k1_staged_twin` replays the exact
block / tile / chunk / window index arithmetic in numpy (CPU tests compare
it with ``k1_twin`` bit for bit); the CUDA kernel is compared with the twin
on a GPU at CP1 (``tests/x2/test_search_gpu.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["CUDA_SOURCE_K1S", "StagedGeometry", "staged_geometry", "k1_staged_twin",
           "k1_staged", "get_module"]

OPT = 8                       # outputs per thread (compile-time; registers)
SMEM_CAP = 48 * 1024          # default dynamic shared memory without opt-in
_META_B = 20                  # per staged entry: stream, smax, conj, c_v, w_v

CUDA_SOURCE_K1S = r"""
#define OPT 8
extern "C" __global__ void k1_staged(
    const short* __restrict__ codes,      /* [S, V, R] packed (re, im) int8 pairs */
    const float* __restrict__ c_v,        /* [S, V] decode step */
    const float* __restrict__ w_v,        /* [S, V] natural weight */
    const int* __restrict__ cell_ptr,     /* [n_act + 1] */
    const int* __restrict__ ent_s,
    const int* __restrict__ ent_v,
    const int* __restrict__ ent_c,
    const float* __restrict__ half_sign,  /* [n_act] 0.5 * (-1)^(u+w) */
    const int* __restrict__ shifts,       /* [F, S] */
    const int* __restrict__ smax,         /* [S] max_f shifts[f, s] */
    float2* __restrict__ out,             /* [F, n_t, n_act] */
    int n_act, int F, int S, int V, int R, int n_t, long long t_lo, float off,
    int cpb, int tile, int win_w, int ec, int meta_off)
{
    extern __shared__ unsigned char smem[];
    short* win = (short*)smem;                         /* [ec][win_w] */
    int* m_s = (int*)(smem + meta_off);                /* [ec] stream */
    int* m_off = m_s + ec;                             /* [ec] smax[s] */
    int* m_cj = m_off + ec;                            /* [ec] conj */
    float* m_c = (float*)(m_cj + ec);                  /* [ec] */
    float* m_w = m_c + ec;                             /* [ec] */

    const int a0 = blockIdx.x * cpb;
    const int ncell = min(cpb, n_act - a0);
    const int t0 = blockIdx.y * tile;
    const int nt = min(tile, n_t - t0);
    const int e_lo = cell_ptr[a0];
    const int e_hi = cell_ptr[a0 + ncell];
    const int n_out = ncell * F * nt;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float acc_re[OPT], acc_im[OPT];
    int my_e0[OPT], my_e1[OPT], my_f[OPT], my_tl[OPT];
#pragma unroll
    for (int k = 0; k < OPT; ++k) {
        acc_re[k] = 0.0f; acc_im[k] = 0.0f;
        const int idx = tid + k * nth;
        my_e0[k] = 0; my_e1[k] = 0; my_f[k] = 0; my_tl[k] = 0;
        if (idx < n_out) {
            const int c = idx % ncell;                 /* cell fastest: coalesced writes */
            const int ft = idx / ncell;
            my_f[k] = ft / nt;
            my_tl[k] = ft - my_f[k] * nt;
            my_e0[k] = cell_ptr[a0 + c];
            my_e1[k] = cell_ptr[a0 + c + 1];
        }
    }
    for (int c0 = e_lo; c0 < e_hi; c0 += ec) {
        const int nc = min(ec, e_hi - c0);
        __syncthreads();                               /* previous chunk consumed */
        for (int e = tid; e < nc; e += nth) {
            const int ent = c0 + e;
            const int s = ent_s[ent], v = ent_v[ent];
            m_s[e] = s;
            m_off[e] = smax[s];
            m_cj[e] = ent_c[ent];
            m_c[e] = c_v[(long long)s * V + v];
            m_w[e] = w_v[(long long)s * V + v];
        }
        for (int idx = tid; idx < nc * win_w; idx += nth) {
            const int e = idx / win_w;
            const int j = idx - e * win_w;
            const int ent = c0 + e;
            const int s = ent_s[ent], v = ent_v[ent];
            const long long row = t_lo + (long long)t0 - (long long)smax[s] + j;
            long long slot = row % R;
            if (slot < 0) slot += R;
            win[idx] = codes[((long long)s * V + v) * R + slot];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < OPT; ++k) {
            const int idx = tid + k * nth;
            if (idx < n_out) {
                const int lo = max(my_e0[k], c0), hi = min(my_e1[k], c0 + nc);
                float ar = acc_re[k], ai = acc_im[k];
                const int frow = my_f[k] * S;
                for (int ent = lo; ent < hi; ++ent) {
                    const int e = ent - c0;
                    const int j = my_tl[k] + m_off[e] - __ldg(&shifts[frow + m_s[e]]);
                    const short pv = win[e * win_w + j];
                    const float cre = (float)(signed char)(pv & 0xFF);
                    const float cim = (float)(signed char)((pv >> 8) & 0xFF);
                    const float vre = __fmul_rn(__fmul_rn(__fadd_rn(cre, off), m_c[e]), m_w[e]);
                    float vim = __fmul_rn(__fmul_rn(__fadd_rn(cim, off), m_c[e]), m_w[e]);
                    if (m_cj[e]) vim = -vim;
                    ar = __fadd_rn(ar, vre);
                    ai = __fadd_rn(ai, vim);
                }
                acc_re[k] = ar; acc_im[k] = ai;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < OPT; ++k) {
        const int idx = tid + k * nth;
        if (idx < n_out) {
            const int c = idx % ncell;
            const float hs = half_sign[a0 + c];
            out[((long long)my_f[k] * n_t + t0 + my_tl[k]) * n_act + a0 + c] =
                make_float2(__fmul_rn(acc_re[k], hs), __fmul_rn(acc_im[k], hs));
        }
    }
}
"""

_MOD = None


def get_module():
    """Compile (NVRTC) on first use -- GPU nodes at a checkpoint only."""
    global _MOD
    if _MOD is None:
        import cupy as cp
        _MOD = cp.RawModule(code=CUDA_SOURCE_K1S, backend="nvrtc", options=("--std=c++14",))
    return _MOD


@dataclass(frozen=True)
class StagedGeometry:
    cpb: int
    threads: int
    tile: int
    win_w: int
    ec: int
    meta_off: int
    smem: int
    smax: np.ndarray          # int32 [S]
    spread: int

    def grid(self, n_act: int, n_t: int) -> tuple:
        return (-(-n_act // self.cpb), -(-n_t // self.tile))


def staged_geometry(shifts: np.ndarray, *, n_t: int, cpb: int = 4, threads: int = 512,
                    smem_cap: int = SMEM_CAP, max_ec: int = 1024) -> StagedGeometry:
    """Tile / window / entry-chunk sizes (host-side, from the static shifts)."""
    sh = np.asarray(shifts, dtype=np.int64)
    F = int(sh.shape[0])
    smax = sh.max(axis=0)
    spread = int((smax - sh.min(axis=0)).max(initial=0))
    tile = int(max(1, min(int(n_t), (OPT * threads) // (cpb * F))))
    if cpb * F * tile > OPT * threads:
        raise ValueError(f"cpb={cpb} x F={F} > {OPT} x {threads} outputs per block")
    win_w = tile + spread
    ec = int(min(max_ec, (smem_cap - 16) // (2 * win_w + _META_B)))
    if ec < 1:
        raise ValueError(f"window of {win_w} rows does not fit {smem_cap} B of shared memory")
    meta_off = (2 * ec * win_w + 15) // 16 * 16
    smem = meta_off + _META_B * ec
    return StagedGeometry(cpb=cpb, threads=threads, tile=tile, win_w=win_w, ec=ec,
                          meta_off=meta_off, smem=smem, smax=smax.astype(np.int32), spread=spread)


def k1_staged_twin(codes: np.ndarray, *, c_v: np.ndarray, offset: float, csr, shifts: np.ndarray,
                   t_lo: int, n_t: int, w_v: Optional[np.ndarray] = None,
                   geom: Optional[StagedGeometry] = None) -> np.ndarray:
    """numpy replay of :data:`CUDA_SOURCE_K1S` (blocks, tiles, entry chunks,
    per-stream windows): ``uv [F, n_t, n_act]`` complex64."""
    S, V, R, _ = codes.shape
    sh = np.asarray(shifts, dtype=np.int64)
    F = int(sh.shape[0])
    g = geom or staged_geometry(sh, n_t=n_t)
    cv = np.asarray(c_v, dtype=np.float32)
    wv = np.ones_like(cv) if w_v is None else np.asarray(w_v, dtype=np.float32)
    off = np.float32(offset)
    ptr = csr.cell_ptr.astype(np.int64)
    es = csr.ent_stream.astype(np.int64)
    ev = csr.ent_value.astype(np.int64)
    ecj = csr.ent_conj.astype(bool)
    hs = (np.float32(0.5) * csr.sign).astype(np.float32)
    n_act = csr.n_active
    o_re = np.zeros((F, n_t, n_act), np.float32)
    o_im = np.zeros((F, n_t, n_act), np.float32)
    smax = g.smax.astype(np.int64)
    jj = np.arange(g.win_w)
    for a0 in range(0, n_act, g.cpb):
        ncell = min(g.cpb, n_act - a0)
        e_lo, e_hi = int(ptr[a0]), int(ptr[a0 + ncell])
        for t0 in range(0, n_t, g.tile):
            nt = min(g.tile, n_t - t0)
            tl = np.arange(nt)
            acc_re = np.zeros((ncell, F, nt), np.float32)
            acc_im = np.zeros((ncell, F, nt), np.float32)
            for c0 in range(e_lo, e_hi, g.ec):
                nc = min(g.ec, e_hi - c0)
                ents = np.arange(c0, c0 + nc)
                rows = int(t_lo) + t0 - smax[es[ents]][:, None] + jj[None, :]
                win = codes[es[ents][:, None], ev[ents][:, None], np.mod(rows, R)]  # [nc, W, 2]
                for c in range(ncell):
                    lo, hi = max(int(ptr[a0 + c]), c0), min(int(ptr[a0 + c + 1]), c0 + nc)
                    for ent in range(lo, hi):
                        e = ent - c0
                        s = es[ent]
                        j = tl[None, :] + smax[s] - sh[:, s][:, None]               # [F, nt]
                        if j.min() < 0 or j.max() >= g.win_w:
                            raise AssertionError("staged window index out of range")
                        cre = win[e, j, 0].astype(np.float32)
                        cim = win[e, j, 1].astype(np.float32)
                        c_ = cv[s, ev[ent]]
                        w_ = wv[s, ev[ent]]
                        vre = ((cre + off) * c_) * w_
                        vim = ((cim + off) * c_) * w_
                        if ecj[ent]:
                            vim = -vim
                        acc_re[c] = acc_re[c] + vre
                        acc_im[c] = acc_im[c] + vim
            for c in range(ncell):
                o_re[:, t0:t0 + nt, a0 + c] = acc_re[c] * hs[a0 + c]
                o_im[:, t0:t0 + nt, a0 + c] = acc_im[c] * hs[a0 + c]
    return (o_re + 1j * o_im).astype(np.complex64)


def k1_staged(codes, c_v, csr_t: dict, shifts, *, t_lo: int, n_t: int, offset: float = 0.0,
              w_v=None, out=None, geom: Optional[StagedGeometry] = None, stream=None):
    """CUDA launch (same call shape as ``kernels_triton.k1_combine``): ``uv
    [F, n_t, n_act, 2]`` float32. ``geom`` (from :func:`staged_geometry`
    with the DM's static shifts) is cached by the caller; ``csr_t`` needs
    ``smax`` (int32 [S] device) when ``geom`` is given."""
    import torch

    S, V, R, two = codes.shape
    F = int(shifts.shape[0])
    n_act = int(csr_t["n_act"])
    if geom is None:                                     # host sync: tests / benches only
        geom = staged_geometry(shifts.detach().cpu().numpy(), n_t=n_t)
    dev = codes.device
    smax = csr_t.get("smax")
    if smax is None:
        smax = torch.as_tensor(geom.smax, dtype=torch.int32, device=dev)
    if w_v is None:
        w_v = torch.ones_like(c_v)
    if out is None:
        out = torch.empty((F, n_t, n_act, 2), dtype=torch.float32, device=dev)
    import cupy as cp
    from ...image.tiled_combine_cuda import _ptr
    k = get_module().get_function("k1_staged")
    st = cp.cuda.ExternalStream((stream or torch.cuda.current_stream()).cuda_stream)
    with st:
        k(geom.grid(n_act, n_t), (geom.threads,),
          (_ptr(codes), _ptr(c_v), _ptr(w_v), _ptr(csr_t["cell_ptr"]), _ptr(csr_t["ent_stream"]),
           _ptr(csr_t["ent_value"]), _ptr(csr_t["ent_conj"]), _ptr(csr_t["half_sign"]),
           _ptr(shifts), _ptr(smax), _ptr(out), np.int32(n_act), np.int32(F), np.int32(S),
           np.int32(V), np.int32(R), np.int32(n_t), np.int64(t_lo), np.float32(offset),
           np.int32(geom.cpb), np.int32(geom.tile), np.int32(geom.win_w), np.int32(geom.ec),
           np.int32(geom.meta_off)), shared_mem=geom.smem)
    return out

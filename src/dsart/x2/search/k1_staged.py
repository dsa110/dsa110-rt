"""K1 with shared-memory window reuse (CP1 ``--k1 staged``), CUDA + numpy twin.

Same contract and the same fp32 operation sequence as ``combine.k1_twin`` /
``kernels_triton.k1_combine`` (bit for bit): per (fine DM ``f``, output row
``t``, active cell), entries in CSR order ``(stream, conj, value)``, each
decoded ``((code + off) * c_v) * w_v`` (SPEC §4.5, two roundings), negated
imaginary part for a mirror entry, summed into one fp32 accumulator per
component starting at +0.0, then ``x 0.5 (-1)^(u+w)``.

What changes is the memory traffic and the thread mapping. The Triton K1
re-reads every entry's codes from global memory once per (fine DM, row)
with a 2-D gather. Here one WARP owns one active cell, one LANE one output
row of a 32-row tile, and each thread accumulates EVERY fine DM in
registers (``FMAX`` compile-time): the entry loop is outer (each entry's
stream, value, conj and decode constants read once, a warp-uniform loop
count), the fine-DM loop inner (independent accumulators: ILP). For fine
DM ``f`` lane ``l`` reads the code at ring row ``t_lo + t0 + l - shifts[f,
s]``: 32 consecutive codes (64 B) per (entry, f), and an entry's window
(32 + the ~50-row stage-2 sweep) stays in L1 across the fine DMs, so the
codes leave DRAM / L2 about once. Window offsets come from a per-block
``[F][S]`` shared table. ``k1_staged_twin`` replays the (earlier) block /
tile / chunk / window index arithmetic; every variant accumulates each
output's entries in CSR order, so twin, Triton and CUDA agree bit for bit.

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

__all__ = ["CUDA_SOURCE_K1S", "k1_source", "StagedGeometry", "staged_geometry", "k1_staged_twin",
           "k1_staged", "get_module"]

_META_B = 16                  # per staged entry: stream, conj, c_v, w_v

CUDA_SOURCE_K1S = r"""
/* FMAX (fine DMs, compile-time: register accumulators) is prepended by k1_source() */
template <bool HAS_OFF>
__device__ __forceinline__ void k1w_body(
    const short* __restrict__ codes, const float* __restrict__ c_v, const float* __restrict__ w_v,
    const int* __restrict__ cell_ptr, const int* __restrict__ ent_s, const int* __restrict__ ent_v,
    const int* __restrict__ ent_c, const float* __restrict__ half_sign,
    const int* __restrict__ shifts, const int* __restrict__ smax, float2* __restrict__ out,
    int n_act, int F, int S, int V, int R, int n_t, long long t_lo, float off, int cpb)
{
    extern __shared__ int k1sm[];
    int* offS = k1sm;                                  /* [F][S] smax[s] - shifts[f, s] */
    float2* ost = (float2*)(k1sm + ((F * S + 1) & ~1)); /* [32 rows][cpb] output staging */
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < F * S; i += blockDim.x) offS[i] = smax[i % S] - shifts[i];
    __syncthreads();
    const int a0 = blockIdx.x * cpb;
    const int ncell = min(cpb, n_act - a0);
    const int a = a0 + warp;                           /* this warp's cell */
    const bool live = warp < ncell;
    const int t0 = blockIdx.y * 32;
    const int p0 = live ? cell_ptr[a] : 0, p1 = live ? cell_ptr[a + 1] : 0;
    float acc_re[FMAX], acc_im[FMAX];
#pragma unroll
    for (int f = 0; f < FMAX; ++f) { acc_re[f] = 0.0f; acc_im[f] = 0.0f; }
    for (int g0 = p0; g0 < p1; g0 += 32) {            /* 32 entries' metadata at once */
        const int ng = min(32, p1 - g0);
        int ms = 0, mb = 0, mcj = 0;
        float mcv = 0.0f, mwv = 0.0f;
        long long msrc = 0;
        if (lane < ng) {
            const int ent = g0 + lane;
            const int s = __ldg(ent_s + ent), v = __ldg(ent_v + ent);
            ms = s;
            mcj = __ldg(ent_c + ent);
            mcv = __ldg(c_v + (long long)s * V + v);
            mwv = __ldg(w_v + (long long)s * V + v);
            long long r0 = (t_lo + (long long)t0 - (long long)__ldg(smax + s)) % R;
            mb = (int)(r0 < 0 ? r0 + R : r0);
            msrc = ((long long)s * V + v) * R;
        }
        for (int e = 0; e < ng; ++e) {                 /* CSR order */
            const int s = __shfl_sync(0xffffffffu, ms, e);
            const bool cj = __shfl_sync(0xffffffffu, mcj, e) != 0;
            const float cv = __shfl_sync(0xffffffffu, mcv, e);
            const float wv = __shfl_sync(0xffffffffu, mwv, e);
            const int sl = __shfl_sync(0xffffffffu, mb, e) + lane;   /* < 2R */
            const short* src = codes + __shfl_sync(0xffffffffu, msrc, e);
            const int* os = offS + s;
#pragma unroll
            for (int f = 0; f < FMAX; ++f) {
                if (f < F) {
                    int q = sl + os[f * S]; if (q >= R) q -= R;
                    const short pv = __ldg(src + q);
                    float cre = (float)(signed char)(pv & 0xFF);
                    float cim = (float)(signed char)((pv >> 8) & 0xFF);
                    if (HAS_OFF) { cre = __fadd_rn(cre, off); cim = __fadd_rn(cim, off); }
                    const float vre = __fmul_rn(__fmul_rn(cre, cv), wv);
                    float vim = __fmul_rn(__fmul_rn(cim, cv), wv);
                    if (cj) vim = -vim;
                    acc_re[f] = __fadd_rn(acc_re[f], vre);
                    acc_im[f] = __fadd_rn(acc_im[f], vim);
                }
            }
        }
    }
    /* per fine DM: stage [32 rows][cpb cells] and write each row's run of
       consecutive cells together (coalesced) */
    const float hs = live ? half_sign[a] : 0.0f;
    const int nrow = min(32, n_t - t0);
#pragma unroll
    for (int f = 0; f < FMAX; ++f) {
        if (f < F) {
            __syncthreads();
            if (live) ost[lane * cpb + warp] = make_float2(__fmul_rn(acc_re[f], hs),
                                                           __fmul_rn(acc_im[f], hs));
            __syncthreads();
            for (int i = tid; i < nrow * ncell; i += blockDim.x) {
                const int r = i / ncell, c = i - r * ncell;
                out[((long long)f * n_t + t0 + r) * n_act + a0 + c] = ost[r * cpb + c];
            }
        }
    }
}

extern "C" __global__ void k1_staged(
    const short* __restrict__ codes, const float* __restrict__ c_v, const float* __restrict__ w_v,
    const int* __restrict__ cell_ptr, const int* __restrict__ ent_s, const int* __restrict__ ent_v,
    const int* __restrict__ ent_c, const float* __restrict__ half_sign,
    const int* __restrict__ shifts, const int* __restrict__ smax, float2* __restrict__ out,
    int n_act, int F, int S, int V, int R, int n_t, long long t_lo, float off, int cpb)
{
    /* off == +0.0 (mid-tread): code + 0.0 is the code itself, exactly */
    if (off != 0.0f)
        k1w_body<true>(codes, c_v, w_v, cell_ptr, ent_s, ent_v, ent_c, half_sign, shifts, smax,
                       out, n_act, F, S, V, R, n_t, t_lo, off, cpb);
    else
        k1w_body<false>(codes, c_v, w_v, cell_ptr, ent_s, ent_v, ent_c, half_sign, shifts, smax,
                        out, n_act, F, S, V, R, n_t, t_lo, off, cpb);
}
"""


def k1_source(fmax: int) -> str:
    return f"#define FMAX {int(fmax)}\n" + CUDA_SOURCE_K1S

_MODS: dict = {}


def get_module(fmax: int = 32):
    """Compile (NVRTC) on first use per FMAX -- GPU nodes only."""
    if fmax not in _MODS:
        import cupy as cp
        _MODS[fmax] = cp.RawModule(code=k1_source(fmax), backend="nvrtc",
                                   options=("--std=c++14",))
    return _MODS[fmax]


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


def staged_geometry(shifts: np.ndarray, *, n_t: int, cpb: int = 4, threads: int = 0,
                    smem_cap: int = 20 * 1024, max_ec: int = 1024) -> StagedGeometry:
    """Block / window / entry-chunk sizes (host-side, from the static shifts).

    One warp per cell (``cpb`` cells per block), one lane per output row (32-row
    tiles), every fine DM per thread. ``smem_cap`` bounds the staged entry chunk
    (small blocks -> several blocks per SM)."""
    sh = np.asarray(shifts, dtype=np.int64)
    F = int(sh.shape[0])
    smax = sh.max(axis=0)
    spread = int((smax - sh.min(axis=0)).max(initial=0))
    tile = 32
    win_w = tile + spread
    S = int(sh.shape[1])
    offs_b = 4 * F * S                                    # [F][S] window offsets
    ec = int(min(max_ec, (smem_cap - 16 - offs_b) // (2 * win_w + _META_B)))
    if ec < 1:
        raise ValueError(f"window of {win_w} rows does not fit {smem_cap} B of shared memory")
    meta_off = (2 * ec * win_w + 15) // 16 * 16
    smem = meta_off + _META_B * ec + offs_b
    return StagedGeometry(cpb=cpb, threads=32 * cpb, tile=tile, win_w=win_w, ec=ec,
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
    k = get_module(-(-F // 8) * 8).get_function("k1_staged")
    st = cp.cuda.ExternalStream((stream or torch.cuda.current_stream()).cuda_stream)
    with st:
        k(geom.grid(n_act, n_t), (geom.threads,),
          (_ptr(codes), _ptr(c_v), _ptr(w_v), _ptr(csr_t["cell_ptr"]), _ptr(csr_t["ent_stream"]),
           _ptr(csr_t["ent_value"]), _ptr(csr_t["ent_conj"]), _ptr(csr_t["half_sign"]),
           _ptr(shifts), _ptr(smax), _ptr(out), np.int32(n_act), np.int32(F), np.int32(S),
           np.int32(V), np.int32(R), np.int32(n_t), np.int64(t_lo), np.float32(offset),
           np.int32(geom.cpb)), shared_mem=4 * ((F * S + 1) & ~1) + 8 * 32 * geom.cpb)
    return out

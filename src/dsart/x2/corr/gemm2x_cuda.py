"""2x fast GEMM on tensor cores: CUDA ``mma.sync`` kernel + numpy twin of its index logic.

The production 2x corr GEMM (CP1/G0: 11.5 ms per block for gridded n_sub 2 on an RTX 2080 Ti,
vs 73 ms for the Triton tile kernel in :mod:`dsart.x2.corr.gemm2x`, which stays as the portable
reference kernel run under the Triton interpreter). Same arithmetic contract:

* core antennas through the 96-slot index table (``Gemm2xTables``: ``core_ants``, ``pair_dst``);
* cal + RFI cube mask (folded into the gains) and the array-burst time mask applied to each
  voltage on load in fp32 with separate roundings (``__fmul_rn`` / ``__fsub_rn``, no FMA), then
  rounded once to fp16 -- the operands are bit-identical to :func:`gemm2x_reference`'s;
* ``V(m, n) = sum_K conj(E_m) E_n`` over K = csf 2 x 2 pols x t_int 8 = 32, fp32 accumulation
  on the tensor cores (``mma.sync.m16n8k8.f32.f16``), one rounding to fp16 at the store: equal to
  the float64 twin within one fp16 ULP;
* output: the cfp16 ring, **k-major** ``ring[k][row][t mod T_ring]`` (half2 re, im), time-minor;
  members of redundant classes go to an f-major fp32 scratch ``[k][fv][member]`` that
  :func:`reduce_classes_cuda` sums in the fixed (ascending-member) order with ``bl_conj``.

Kernel layout: one block = (4 consecutive fast samples, summed channel k), 16 warps. Per fast
sample the 96 x 32 calibrated operands (re, im) are staged in shared memory (double-buffered,
next sample prefetched into registers), then 42 units = 21 upper-triangular 16x16 slot-tile
pairs x 2 n8 halves are spread over the warps (unit u -> warp u % 16); each lane owns 4 output
elements of its unit (the m16n8k8 C-fragment layout) and keeps the 4 samples in registers, so
every ring row gets ONE 16-byte store per block (4 consecutive time samples): the k-major,
time-minor ring is written at near the pattern's DRAM limit (measured 4-5 ms per block for the
1.3 GB of gridded output).

:func:`unit_element_map` is the kernel's (warp, unit, lane, register) -> (slot m, slot n) map in
numpy; :func:`gemm2x_mma_twin` replays the kernel's stores (ring k-major, scratch f-major) with
the reference arithmetic, so the index/tile/layout logic is tested on the CPU.
"""
from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("dsart.x2.corr.gemm2x_cuda")

__all__ = [
    "CUDA_SOURCE",
    "FVB",
    "NSLOT",
    "gemm2x_mma_into_ring",
    "gemm2x_mma_twin",
    "reduce_classes_cuda",
    "unit_element_map",
]

NSLOT = 96
FVB = 4
NTHR = 512
_TILES = 6
PAIR_TM = [i for i in range(_TILES) for j in range(i, _TILES)]
PAIR_TN = [j for i in range(_TILES) for j in range(i, _TILES)]

CUDA_SOURCE = r"""
#include <cuda_fp16.h>
typedef unsigned int u32;

#define NSLOT 96
#define KQ 32
#define KP 40
#define FVB 4
#define NTHR 512
#define NPER ((KQ * NSLOT) / NTHR)

__device__ __forceinline__ void mma1688(float* d, u32 a0, u32 a1, u32 b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ u32 pack_h2(float re, float im) {
    __half2 h = __floats2half2_rn(re, im);
    return *reinterpret_cast<u32*>(&h);
}

/* tile pairs (tm <= tn) of the 6 16-slot tiles, row-major */
__constant__ unsigned char PAIR_TM[21] = {0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5};
__constant__ unsigned char PAIR_TN[21] = {0,1,2,3,4,5,1,2,3,4,5,2,3,4,5,3,4,5,4,5,5};

/* grid (n_fv_chunk / FVB, NSUM), NTHR threads. Voltages fp16 (NCHAN, 2, NVP=2, NPKT, NANTS)
   with strides s_*; gains fp32 (NCHAN, 2, NANTS); tkeep fp32 (n_acc, NCHAN, 2) if has_tkeep.
   ring: half2 bits [NSUM][R][T_ring]; scratch: float2 [NSUM][n_fv_scr][n_scr]. */
extern "C" __global__ void __launch_bounds__(NTHR, 1) gemm2x_mma(
    const __half* __restrict__ vre, const __half* __restrict__ vim,
    const float* __restrict__ calr, const float* __restrict__ cali,
    const float* __restrict__ tkeep,
    const int* __restrict__ core_ants, const int* __restrict__ pair_dst,
    u32* __restrict__ ring, float2* __restrict__ scratch,
    int n_core, int R, int n_scr, int T_ring, int t_ring0, int fv0, int n_fv_scr,
    int nants, int nchan, int pk_per_acc, int has_tkeep,
    long long s_ch, long long s_t, long long s_p, long long s_pk)
{
    __shared__ __align__(16) __half ErB[2][NSLOT * KP];
    __shared__ __align__(16) __half EiB[2][NSLOT * KP];
    __shared__ float gr_s[4 * NSLOT];
    __shared__ float gi_s[4 * NSLOT];
    __shared__ int ant_s[NSLOT];
    const int k = blockIdx.y;
    const int fb = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;

    for (int i = tid; i < 4 * NSLOT; i += NTHR) {
        const int slot = i % NSLOT, cp = i / NSLOT, ci = cp >> 1, p = cp & 1;
        const bool ok = slot < n_core;
        const int ant = ok ? core_ants[slot] : 0;
        const int gix = ((2 * k + ci) * 2 + p) * nants + ant;
        gr_s[i] = ok ? calr[gix] : 0.f;
        gi_s[i] = ok ? cali[gix] : 0.f;
        if (cp == 0) ant_s[slot] = ant;
    }

    /* this warp's units u = warp + 16 j (u < 42) -> (pair u >> 1, n8 half u & 1);
       lane (g, c) owns C-fragment registers r: row g + 8 (r >> 1), col 2c + (r & 1) */
    int dst_r[3][4];
    u32 stage[3][4][FVB];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int u = warp + 16 * j;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            dst_r[j][r] = -1;
            if (u < 42) {
                const int pr = u >> 1, h = u & 1;
                const int sm = PAIR_TM[pr] * 16 + g + ((r >> 1) << 3);
                const int sn = PAIR_TN[pr] * 16 + h * 8 + 2 * c + (r & 1);
                dst_r[j][r] = pair_dst[sm * NSLOT + sn];
            }
#pragma unroll
            for (int f = 0; f < FVB; ++f) stage[j][r][f] = 0u;
        }
    }

    __half pre_r[NPER], pre_i[NPER];
#define LOAD_PRE(FV)                                                                          \
    _Pragma("unroll") for (int q = 0; q < NPER; ++q) {                                        \
        const int i = tid + q * NTHR, slot = i % NSLOT, kk = i / NSLOT;                       \
        const int ci = kk >> 4, p = (kk >> 3) & 1, t2 = (kk >> 2) & 1, pk = kk & 3;          \
        const long long pkt = (long long)(FV) * 4 + pk;                                       \
        const long long off = (2 * k + ci) * s_ch + t2 * s_t + p * s_p + pkt * s_pk + ant_s[slot]; \
        pre_r[q] = vre[off];                                                                  \
        pre_i[q] = vim[off];                                                                  \
    }
    __syncthreads();
    LOAD_PRE(fv0 + fb * FVB)
#pragma unroll
    for (int f = 0; f < FVB; ++f) {
        const int fv = fv0 + fb * FVB + f;
        __half* Er = ErB[f & 1];
        __half* Ei = EiB[f & 1];
#pragma unroll
        for (int q = 0; q < NPER; ++q) {
            const int i = tid + q * NTHR, slot = i % NSLOT, kk = i / NSLOT;
            const int ci = kk >> 4, p = (kk >> 3) & 1;
            const float vr = __half2float(pre_r[q]), vi = __half2float(pre_i[q]);
            const int gix = (ci * 2 + p) * NSLOT + slot;
            float g_r = gr_s[gix], g_i = gi_s[gix];
            if (has_tkeep) {
                const long long pkt = (long long)fv * 4 + (kk & 3);
                const float kp = tkeep[((pkt / pk_per_acc) * nchan + 2 * k + ci) * 2 + p];
                g_r = __fmul_rn(g_r, kp);
                g_i = __fmul_rn(g_i, kp);
            }
            const float xr = __fsub_rn(__fmul_rn(vr, g_r), __fmul_rn(vi, g_i));
            const float xi = __fadd_rn(__fmul_rn(vr, g_i), __fmul_rn(vi, g_r));
            Er[slot * KP + kk] = __float2half_rn(xr);
            Ei[slot * KP + kk] = __float2half_rn(xi);
        }
        __syncthreads();
        if (f + 1 < FVB) { LOAD_PRE(fv + 1) }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int u = warp + 16 * j;
            if (u >= 42) continue;
            const int pr = u >> 1, h = u & 1;
            const int m0 = PAIR_TM[pr] * 16, n0 = PAIR_TN[pr] * 16 + h * 8;
            float are[4] = {0.f, 0.f, 0.f, 0.f}, aim[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int kb = s * 8 + 2 * c;
                const u32 ar0 = *reinterpret_cast<const u32*>(&Er[(m0 + g) * KP + kb]);
                const u32 ar1 = *reinterpret_cast<const u32*>(&Er[(m0 + g + 8) * KP + kb]);
                const u32 ai0 = *reinterpret_cast<const u32*>(&Ei[(m0 + g) * KP + kb]);
                const u32 ai1 = *reinterpret_cast<const u32*>(&Ei[(m0 + g + 8) * KP + kb]);
                const u32 br = *reinterpret_cast<const u32*>(&Er[(n0 + g) * KP + kb]);
                const u32 bi = *reinterpret_cast<const u32*>(&Ei[(n0 + g) * KP + kb]);
                mma1688(are, ar0, ar1, br);                               /* Er.Er */
                mma1688(are, ai0, ai1, bi);                               /* + Ei.Ei */
                mma1688(aim, ar0, ar1, bi);                               /* Er.Ei */
                mma1688(aim, ai0 ^ 0x80008000u, ai1 ^ 0x80008000u, br);   /* - Ei.Er */
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                stage[j][r][f] = pack_h2(are[r], aim[r]);
                const int d = dst_r[j][r];
                if (d >= R)
                    scratch[((long long)k * n_fv_scr + (fb * FVB + f)) * n_scr + (d - R)] =
                        make_float2(are[r], aim[r]);
            }
        }
    }
    const long long t0 = (t_ring0 + fb * FVB) % T_ring;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int d = dst_r[j][r];
            if (d >= 0 && d < R)
                *reinterpret_cast<uint4*>(&ring[((long long)k * R + d) * T_ring + t0]) =
                    make_uint4(stage[j][r][0], stage[j][r][1], stage[j][r][2], stage[j][r][3]);
        }
    }
}

/* class reduce from the f-major scratch: thread = (class, FVB fast samples). Members in the
   fixed (ascending) order, conj applied, fp32, one rounding to half2; one 16-byte ring store. */
extern "C" __global__ void reduce_fmajor(
    const float2* __restrict__ scratch, u32* __restrict__ ring,
    const int* __restrict__ cptr, const int* __restrict__ csrc, const unsigned char* __restrict__ cconj,
    const int* __restrict__ crow, int n_multi, int n_scr, int R, int T_ring, int t_ring0,
    int n_fv_scr)
{
    const int cl = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y, fb = blockIdx.z;
    if (cl >= n_multi) return;
    const int s0 = cptr[cl], s1 = cptr[cl + 1];
    u32 o[FVB];
#pragma unroll
    for (int f = 0; f < FVB; ++f) {
        float ar = 0.f, ai = 0.f;
        const long long base = ((long long)k * n_fv_scr + fb * FVB + f) * n_scr;
        for (int s = s0; s < s1; ++s) {
            const float2 x = scratch[base + csrc[s]];
            ar = __fadd_rn(ar, x.x);
            ai = cconj[s] ? __fsub_rn(ai, x.y) : __fadd_rn(ai, x.y);
        }
        o[f] = pack_h2(ar, ai);
    }
    const long long t0 = (t_ring0 + fb * FVB) % T_ring;
    *reinterpret_cast<uint4*>(&ring[((long long)k * R + crow[cl]) * T_ring + t0]) =
        make_uint4(o[0], o[1], o[2], o[3]);
}
"""

_MOD = None


def _module():
    global _MOD
    if _MOD is None:
        import cupy as cp  # noqa: PLC0415
        LOG.info("compiling gemm2x_mma via NVRTC")
        _MOD = cp.RawModule(code=CUDA_SOURCE, backend="nvrtc", options=("--std=c++14",))
    return _MOD


def _cp(t):
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t))


def gemm2x_mma_into_ring(real_v, imag_v, cal_r, cal_i, tables, ring, *, fv0: int, n_fv: int,
                         t_ring0: int, scratch=None, time_keep=None) -> None:
    """Correlate fast samples ``[fv0, fv0 + n_fv)`` (t_int 8, csf 2) into the k-major ring
    ``(NSUM, n_ring_rows, T_ring, 2)`` fp16 at ``(t_ring0 + i) % T_ring``; class members into
    ``scratch`` fp32 ``(NSUM, n_fv, n_scratch_rows, 2)`` (f-major; then
    :func:`reduce_classes_cuda`). ``tables`` on the device; ``time_keep`` fp32 ``(n_acc, NCHAN,
    2)`` or None. Launches on the current torch stream."""
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    nchan, two, nvp, npkt, nants = real_v.shape
    if two != 2 or nvp != 2 or real_v.dtype != torch.float16 or not real_v.is_contiguous():
        raise ValueError("voltages must be contiguous fp16 (NCHAN, 2, 2, NPKT, NANTS)")
    if imag_v.shape != real_v.shape or imag_v.stride() != real_v.stride():
        raise ValueError("imag_v must match real_v")
    nsum = nchan // 2
    if tables.nc_pad != NSLOT:
        raise ValueError(f"the mma kernel is built for {NSLOT} core slots, tables have {tables.nc_pad}")
    if ring.dtype != torch.float16 or not ring.is_contiguous() or tuple(ring.shape[::3]) != (nsum, 2) \
            or ring.shape[1] != tables.n_ring_rows:
        raise ValueError(f"ring must be contiguous fp16 ({nsum}, {tables.n_ring_rows}, T_ring, 2)")
    T_ring = int(ring.shape[2])
    if n_fv % FVB or T_ring % FVB or t_ring0 % FVB or fv0 % FVB:
        raise ValueError(f"n_fv, fv0, t_ring0 and T_ring must be multiples of {FVB}")
    n_scr = int(tables.n_scratch_rows)
    if n_scr:
        if scratch is None or tuple(scratch.shape) != (nsum, n_fv, n_scr, 2) \
                or scratch.dtype != torch.float32 or not scratch.is_contiguous():
            raise ValueError(f"scratch must be contiguous fp32 ({nsum}, {n_fv}, {n_scr}, 2)")
    else:
        scratch = torch.empty((2,), dtype=torch.float32, device=real_v.device)
    has_t = time_keep is not None
    tk = time_keep.contiguous().float() if has_t else cal_r
    pk_per_acc = npkt // int(time_keep.shape[0]) if has_t else 1
    s = real_v.stride()
    kern = _module().get_function("gemm2x_mma")
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        kern((n_fv // FVB, nsum), (NTHR,),
             (_cp(real_v), _cp(imag_v), _cp(cal_r.contiguous()), _cp(cal_i.contiguous()), _cp(tk),
              _cp(tables.core_ants), _cp(tables.pair_dst), _cp(ring.view(torch.int32)), _cp(scratch),
              np.int32(tables.n_core), np.int32(tables.n_ring_rows), np.int32(max(1, n_scr)),
              np.int32(T_ring), np.int32(t_ring0 % T_ring), np.int32(fv0), np.int32(n_fv),
              np.int32(nants), np.int32(nchan), np.int32(pk_per_acc), np.int32(int(has_t)),
              np.int64(s[0]), np.int64(s[1]), np.int64(s[2]), np.int64(s[3])))


def reduce_classes_cuda(scratch, ring, tables, *, t_ring0: int, n_fv: int, block: int = 128) -> None:
    """Sum each redundant class's members (fixed order, conj) from the f-major ``scratch`` into
    its k-major ring row at ``(t_ring0 + i) % T_ring``."""
    import cupy as cp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    n_multi = int(tables.class_row.shape[0])
    if n_multi == 0:
        return
    nsum, R, T_ring = int(ring.shape[0]), int(ring.shape[1]), int(ring.shape[2])
    kern = _module().get_function("reduce_fmajor")
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        kern((-(-n_multi // block), nsum, n_fv // FVB), (block,),
             (_cp(scratch), _cp(ring.view(torch.int32)), _cp(tables.class_ptr), _cp(tables.class_src),
              _cp(tables.class_conj), _cp(tables.class_row), np.int32(n_multi),
              np.int32(int(scratch.shape[2])), np.int32(R), np.int32(T_ring),
              np.int32(t_ring0 % T_ring), np.int32(n_fv)))


# ---------------------------------------------------------------------------
# numpy twin of the kernel's index logic
# ---------------------------------------------------------------------------


def unit_element_map() -> np.ndarray:
    """``(n, 6)`` int rows ``(warp, j, lane, r, slot_m, slot_n)``: every output element the
    kernel's lanes own (unit u = warp + 16 j < 42 -> pair u >> 1, n8 half u & 1; lane
    (g, c) = (lane >> 2, lane & 3), register r -> row g + 8 (r >> 1), col 2c + (r & 1))."""
    out = []
    for warp in range(NTHR // 32):
        for j in range(3):
            u = warp + 16 * j
            if u >= 42:
                continue
            pr, h = u >> 1, u & 1
            for lane in range(32):
                g, c = lane >> 2, lane & 3
                for r in range(4):
                    sm = PAIR_TM[pr] * 16 + g + ((r >> 1) << 3)
                    sn = PAIR_TN[pr] * 16 + h * 8 + 2 * c + (r & 1)
                    out.append((warp, j, lane, r, sm, sn))
    return np.asarray(out, np.int64)


def gemm2x_mma_twin(real_v, imag_v, cal_r, cal_i, tables, *, fv0: int, n_fv: int, T_ring: int,
                    t_ring0: int, time_keep=None):
    """What the kernel stores, from the reference arithmetic (float64 accumulation of the
    fp16 operands, one rounding): returns ``(ring, scratch)`` as numpy -- ``ring`` complex64
    ``(NSUM, R, T_ring)`` k-major with NaN where the kernel writes nothing, ``scratch`` complex64
    ``(NSUM, n_fv, n_scr)`` f-major. Every element goes through :func:`unit_element_map` and the
    kernel's address arithmetic."""
    from dsart.x2.corr.gemm2x import gemm2x_reference  # noqa: PLC0415
    nchan = int(real_v.shape[0])
    nsum = nchan // 2
    if tables.nc_pad != NSLOT:
        raise ValueError(f"twin is for {NSLOT} slots")
    vis = gemm2x_reference(real_v, imag_v, cal_r, cal_i, tables, fv0=fv0, n_fv=n_fv, csf=2, t_int=8,
                           time_keep=time_keep).cpu().numpy()            # (n_fv, nsum, n_core_rows)
    sm_rows = tables.row_slot_m.cpu().numpy()
    sn_rows = tables.row_slot_n.cpu().numpy()
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(sm_rows, sn_rows))}
    R, n_scr = int(tables.n_ring_rows), int(tables.n_scratch_rows)
    pdst = tables.pair_dst.cpu().numpy()
    ring = np.full((nsum, R, T_ring), np.nan + 1j * np.nan, np.complex64)
    scr = np.full((nsum, n_fv, max(1, n_scr)), np.nan + 1j * np.nan, np.complex64)
    seen = set()
    for (_w, _j, _lane, _r, sm, sn) in unit_element_map():
        d = int(pdst[sm, sn])
        if d < 0:
            continue
        key = (int(sm), int(sn))
        if key in seen:
            raise AssertionError(f"slot pair {key} owned twice")
        seen.add(key)
        x = vis[:, :, row_of[key]].astype(np.complex64)                  # (n_fv, nsum)
        x = (x.real.astype(np.float16).astype(np.float32)
             + 1j * x.imag.astype(np.float16).astype(np.float32)) if d < R else x
        for fb in range(n_fv // FVB):
            t0 = (t_ring0 + fb * FVB) % T_ring
            for f in range(FVB):
                if d < R:
                    ring[:, d, t0 + f] = x[fb * FVB + f]
                else:
                    scr[:, fb * FVB + f, d - R] = vis[fb * FVB + f, :, row_of[key]]
    return ring, scr

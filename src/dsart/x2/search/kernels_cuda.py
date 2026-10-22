"""CUDA kernels of the owner engine (cupy RawModule, NVRTC): K0, K2, K3a.

* **K0** ``k0_unpack_frame2`` / ``k0_unpack_cint8``: one block per (row,
  stream); threads over values. Frame-v2 rows: heavy int8 pairs, then the
  main bit stream (value ``v`` at bit ``2 b v`` of the main area, re in the
  low ``b`` bits), a 3-byte little-endian gather guarded by the row end,
  two's complement; the "no data" sentinel (-2^(b-1); -128 heavy) becomes
  0 and is counted; a row with no data at all is flagged invalid. Writes
  the time-minor code ring ``int8 [S, V, R, 2]`` at slot ``row % R``.
  :func:`k0_twin_bitgather` is the numpy twin of the exact byte/bit index
  arithmetic (checked against ``codec.unpack_frame2_rows``).
* **K2** ``k2_colfft``: one warp per (image, FFT line -- a column in the
  col fold, a row in the row fold, ``csr.OwnerCsr.fold``): the line's
  active cells scattered into shared memory, lane ``l`` loads position
  ``l + 32 j`` (j < R1 = N/32), an R1-point inverse DFT in registers,
  twiddle ``W_N^{l y2}``, a 32-point radix-2 DIF across lanes with
  ``__shfl_xor_sync`` (span 16 -> 1), output ``y = y2 + R1 bitrev5(l)``.
  Unnormalised. Twin: ``fft.k2_twin``.
* **K3a** ``k3a_rowc2r``: one warp per (image, output coordinate ``y`` on
  the full axis): the half-spectrum assembled in shared memory (K2 lines,
  position-0-only lines as constants, zeros), the real-to-complex packing
  ``Z = (Y_k + conj Y_{M-k}) + i W_N^k (Y_k - conj Y_{M-k})`` (DC / Nyquist
  real parts), an M = N/2-point inverse DFT (R2 = M/32 in-lane x 32
  lanes), ``x[2n] = Re z[n], x[2n+1] = Im z[n]``, times the per-pixel
  ``fl32(mask * scale / N^2)``, rounded to half into the image ring
  ``[R_img, F, N*N]`` at slot ``t % R_img`` (``transpose`` = row fold
  writes pixel ``(n, y)``). Twin: ``fft.k3a_twin``.

Image ``b`` of a K1 batch is ``(f, t) = divmod(b, n_t)`` (K1 writes
``[F, n_t, n_act]``). Twiddles: ``fft.twiddles(N)`` (fp32, rounded once).
The CUDA FFTs are fp32 with possible FMA contraction, so they match their
twins to fp32 rounding (checked at CP1, ``x2_gpu`` tests), not bit for bit.
"""
from __future__ import annotations

import numpy as np

from .codec import RowFormat

__all__ = ["CUDA_SOURCE", "k0_twin_bitgather", "get_module"]

CUDA_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" __global__ void k0_unpack_frame2(
    const unsigned char* __restrict__ raw,    /* [S, n, row_bytes] */
    const unsigned char* __restrict__ valid,  /* [S, n] */
    const int* __restrict__ nv_s, const int* __restrict__ nh_s, const int* __restrict__ bits_s,
    signed char* __restrict__ codes,          /* [S, V, R, 2] */
    unsigned char* __restrict__ valid_out,    /* [S, R] */
    unsigned long long* __restrict__ nodata,  /* [S] */
    int S, int n, int row_bytes, int V, int R, long long row0)
{
    const int r = blockIdx.x, s = blockIdx.y;
    __shared__ int nd_row;
    if (threadIdx.x == 0) nd_row = 0;
    __syncthreads();
    long long slot = ((row0 + r) % R + R) % R;
    const unsigned char* src = raw + ((long long)s * n + r) * row_bytes;
    const int ok = valid[(long long)s * n + r];
    const int nv = nv_s[s], nh = nh_s[s], b = bits_s[s];
    const unsigned int m = (1u << b) - 1u;
    const int half = 1 << (b - 1), sent = -(1 << (b - 1));
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        int re = 0, im = 0, nd = 0;
        if (ok && v < nv) {
            if (v < nh) {
                re = (signed char)src[2 * v];
                im = (signed char)src[2 * v + 1];
                nd = (re == -128) || (im == -128);
            } else {
                long long bit = (long long)(v - nh) * 2 * b;
                long long byte = 2LL * nh + (bit >> 3);
                unsigned int w = src[byte];
                if (byte + 1 < row_bytes) w |= ((unsigned int)src[byte + 1]) << 8;
                if (byte + 2 < row_bytes) w |= ((unsigned int)src[byte + 2]) << 16;
                w >>= (unsigned int)(bit & 7);
                int ur = (int)(w & m), ui = (int)((w >> b) & m);
                re = ur >= half ? ur - (1 << b) : ur;
                im = ui >= half ? ui - (1 << b) : ui;
                nd = (re == sent) || (im == sent);
            }
            if (nd) { re = 0; im = 0; atomicAdd(&nd_row, 1); }
        }
        long long o = (((long long)s * V + v) * R + slot) * 2;
        codes[o] = (signed char)re;
        codes[o + 1] = (signed char)im;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        valid_out[(long long)s * R + slot] = (unsigned char)(ok && nd_row < nv);
        if (nd_row) atomicAdd(&nodata[s], (unsigned long long)nd_row);
    }
}

extern "C" __global__ void k0_unpack_cint8(
    const signed char* __restrict__ raw,      /* [S, n, 2 * V_row] */
    const unsigned char* __restrict__ valid,  /* [S, n] */
    signed char* __restrict__ codes,          /* [S, V, R, 2] */
    unsigned char* __restrict__ valid_out,    /* [S, R] */
    int S, int n, int row_bytes, int V, int R, long long row0)
{
    const int r = blockIdx.x, s = blockIdx.y;
    long long slot = ((row0 + r) % R + R) % R;
    const signed char* src = raw + ((long long)s * n + r) * row_bytes;
    const int ok = valid[(long long)s * n + r];
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        signed char re = 0, im = 0;
        if (ok && 2 * v + 1 < row_bytes) { re = src[2 * v]; im = src[2 * v + 1]; }
        long long o = (((long long)s * V + v) * R + slot) * 2;
        codes[o] = re;
        codes[o + 1] = im;
    }
    if (threadIdx.x == 0) valid_out[(long long)s * R + slot] = (unsigned char)ok;
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

template <int R1>
__device__ __forceinline__ void lane_dif32(float2 (&A)[R1], const float2* tw, int N, int lane) {
    #pragma unroll
    for (int h = 16; h >= 1; h >>= 1) {
        const int top = (lane & h) == 0;
        const float2 w = tw[((lane & (h - 1)) * (N / (2 * h))) % N];
        #pragma unroll
        for (int i = 0; i < R1; ++i) {
            float2 p;
            p.x = __shfl_xor_sync(0xffffffffu, A[i].x, h);
            p.y = __shfl_xor_sync(0xffffffffu, A[i].y, h);
            A[i] = top ? make_float2(A[i].x + p.x, A[i].y + p.y)
                       : cmul(make_float2(p.x - A[i].x, p.y - A[i].y), w);
        }
    }
}

/* K2: N = 32 * R1 (R1 = 8 at N = 256). blockDim = 32 * warps; shared N float2 / warp. */
template <int R1>
__device__ void k2_body(const float2* __restrict__ uv, const int* __restrict__ col_ptr,
                        const int* __restrict__ col_idx, const int* __restrict__ col_u,
                        const float2* __restrict__ tw, float2* __restrict__ Y,
                        int B, int n_act, int n_fc)
{
    const int N = 32 * R1;
    extern __shared__ float2 sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long job = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (job >= (long long)B * n_fc) return;
    const int b = (int)(job / n_fc), ci = (int)(job % n_fc);
    float2* col = sm + warp * N;
    for (int i = lane; i < N; i += 32) col[i] = make_float2(0.f, 0.f);
    __syncwarp();
    for (int e = col_ptr[ci] + lane; e < col_ptr[ci + 1]; e += 32)
        col[col_u[e]] = uv[(long long)b * n_act + col_idx[e]];
    __syncwarp();
    float2 x[R1], A[R1];
    #pragma unroll
    for (int j = 0; j < R1; ++j) x[j] = col[lane + 32 * j];
    #pragma unroll
    for (int y2 = 0; y2 < R1; ++y2) {
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int j = 0; j < R1; ++j) {
            float2 t = cmul(x[j], tw[(j * y2 * (N / R1)) % N]);
            acc.x += t.x; acc.y += t.y;
        }
        A[y2] = cmul(acc, tw[(lane * y2) % N]);
    }
    lane_dif32<R1>(A, tw, N, lane);
    const int y1 = __brev(lane) >> 27;
    float2* out = Y + ((long long)b * n_fc + ci) * N;
    #pragma unroll
    for (int y2 = 0; y2 < R1; ++y2) out[y2 + R1 * y1] = A[y2];
}

extern "C" __global__ void k2_colfft_256(const float2* uv, const int* col_ptr, const int* col_idx,
                                         const int* col_u, const float2* tw, float2* Y,
                                         int B, int n_act, int n_fc)
{ k2_body<8>(uv, col_ptr, col_idx, col_u, tw, Y, B, n_act, n_fc); }

/* K3a: N = 64 * R2 (R2 = 4 at N = 256); shared (N/2 + 1) float2 per warp. */
template <int R2>
__device__ void k3a_body(const float2* __restrict__ Y, const float2* __restrict__ uv,
                         const int* __restrict__ fc_w, const int* __restrict__ cc_idx,
                         const int* __restrict__ cc_w, const float2* __restrict__ tw,
                         const float* __restrict__ pix, __half* __restrict__ ring,
                         int B, int n_act, int n_fc, int n_cc, int n_t, int F,
                         long long t0, int R_img, int transpose)
{
    const int M = 32 * R2, N = 2 * M, NH = M + 1;
    extern __shared__ float2 sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long job = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (job >= (long long)B * N) return;
    const int b = (int)(job / N), y = (int)(job % N);   /* y: output coordinate on the full axis */
    float2* row = sm + warp * NH;
    for (int i = lane; i < NH; i += 32) row[i] = make_float2(0.f, 0.f);
    __syncwarp();
    for (int ci = lane; ci < n_fc; ci += 32)
        row[fc_w[ci]] = Y[((long long)b * n_fc + ci) * N + y];
    for (int c = lane; c < n_cc; c += 32)
        row[cc_w[c]] = uv[(long long)b * n_act + cc_idx[c]];
    __syncwarp();
    float2 A[R2];
    #pragma unroll
    for (int k2 = 0; k2 < R2; ++k2) {
        const int k = lane + 32 * k2;
        float2 yk = row[k], yr;
        if (k == 0) { yk = make_float2(row[0].x, 0.f); yr = make_float2(row[M].x, 0.f); }
        else { yr = make_float2(row[M - k].x, -row[M - k].y); }
        float2 fe = make_float2(yk.x + yr.x, yk.y + yr.y);
        float2 fo = cmul(make_float2(yk.x - yr.x, yk.y - yr.y), tw[k]);
        A[k2] = make_float2(fe.x - fo.y, fe.y + fo.x);
    }
    float2 Bv[R2];
    #pragma unroll
    for (int n2 = 0; n2 < R2; ++n2) {
        float2 acc = make_float2(0.f, 0.f);
        #pragma unroll
        for (int k2 = 0; k2 < R2; ++k2) {
            float2 t = cmul(A[k2], tw[(k2 * n2 * (N / R2)) % N]);
            acc.x += t.x; acc.y += t.y;
        }
        Bv[n2] = cmul(acc, tw[(2 * lane * n2) % N]);
    }
    lane_dif32<R2>(Bv, tw, N, lane);
    const int n1 = __brev(lane) >> 27;
    const int f = b / n_t, t = b % n_t;
    const long long slot = ((t0 + t) % R_img + R_img) % R_img;
    /* col fold: pixel (y, x = n); row fold (transpose): pixel (n, y) */
    __half* dst = ring + (slot * F + f) * (long long)N * N;
    #pragma unroll
    for (int n2 = 0; n2 < R2; ++n2) {
        const int nn = n2 + R2 * n1;
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int x = 2 * nn + q;
            const long long pidx = transpose ? (long long)x * N + y : (long long)y * N + x;
            const float val = q ? Bv[n2].y : Bv[n2].x;
            dst[pidx] = __float2half_rn(val * pix[pidx]);
        }
    }
}

extern "C" __global__ void k3a_rowc2r_256(const float2* Y, const float2* uv, const int* fc_w,
                                          const int* cc_idx, const int* cc_w, const float2* tw,
                                          const float* pix, __half* ring, int B, int n_act,
                                          int n_fc, int n_cc, int n_t, int F, long long t0,
                                          int R_img, int transpose)
{ k3a_body<4>(Y, uv, fc_w, cc_idx, cc_w, tw, pix, ring, B, n_act, n_fc, n_cc, n_t, F, t0, R_img,
              transpose); }
"""

_MOD = None


def get_module():
    """Compile (NVRTC) on first use -- GPU nodes at a checkpoint only."""
    global _MOD
    if _MOD is None:
        import cupy as cp
        _MOD = cp.RawModule(code=CUDA_SOURCE, backend="nvrtc", options=("--std=c++14",))
    return _MOD


def k0_twin_bitgather(raw: np.ndarray, fmt: RowFormat) -> tuple:
    """Numpy twin of ``k0_unpack_frame2``'s per-value index arithmetic for
    one stream: ``raw uint8 [n, row_bytes]`` -> ``(re, im, nodata)``
    ``[n, n_values]`` (sentinels NOT yet zeroed: the caller compares with
    ``codec.unpack_frame2_rows``)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    n, row_bytes = raw.shape
    nv, nh, b = fmt.n_values, fmt.n_heavy, fmt.bits
    re = np.zeros((n, nv), dtype=np.int64)
    im = np.zeros((n, nv), dtype=np.int64)
    nd = np.zeros((n, nv), dtype=bool)
    if nh:
        h = raw[:, :2 * nh].view(np.int8).astype(np.int64)
        re[:, :nh] = h[:, 0::2]
        im[:, :nh] = h[:, 1::2]
        nd[:, :nh] = (re[:, :nh] == -128) | (im[:, :nh] == -128)
    v = np.arange(nh, nv, dtype=np.int64)
    bit = (v - nh) * 2 * b
    byte = 2 * nh + (bit >> 3)
    w = raw[:, byte].astype(np.uint32)
    for k in (1, 2):
        ok = byte + k < row_bytes
        nxt = np.where(ok, byte + k, 0)
        w |= np.where(ok[None, :], raw[:, nxt].astype(np.uint32) << np.uint32(8 * k), 0)
    w >>= (bit & 7).astype(np.uint32)[None, :]
    m = np.uint32((1 << b) - 1)
    ur = (w & m).astype(np.int64)
    ui = ((w >> np.uint32(b)) & m).astype(np.int64)
    half = 1 << (b - 1)
    re[:, nh:] = np.where(ur >= half, ur - (1 << b), ur)
    im[:, nh:] = np.where(ui >= half, ui - (1 << b), ui)
    sent = -(1 << (b - 1))
    nd[:, nh:] = (re[:, nh:] == sent) | (im[:, nh:] == sent)
    return re, im, nd

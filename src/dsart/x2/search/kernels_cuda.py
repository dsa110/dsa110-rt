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

from typing import Optional

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

/* K0 2-D tiles: one block per (TR rows, VB values, stream). Only the byte range
   of the block's values is loaded (coalesced) per row; values are unpacked with
   k0_unpack_frame2's exact bit gather / NO_DATA rule into a [VB][TR] shared tile
   written as runs of TR consecutive ring slots per value. Per-row NO_DATA counts
   are summed across value tiles in ndc[s * n + r] (zeroed by the caller);
   k0_f2_valid then sets valid_out and nodata exactly as the per-row kernel. */
#define K0_TR 32
#define K0_VB 128
extern "C" __global__ void k0_f2_tile2(
    const unsigned char* __restrict__ raw, const unsigned char* __restrict__ valid,
    const int* __restrict__ nv_s, const int* __restrict__ nh_s, const int* __restrict__ bits_s,
    signed char* __restrict__ codes, int* __restrict__ ndc, int S, int n, int row_bytes, int V,
    int R, long long row0)
{
    __shared__ unsigned char rt[K0_TR * (2 * K0_VB + 8)];
    __shared__ unsigned short ot[K0_VB * K0_TR];
    __shared__ int nd_row[K0_TR];
    __shared__ int okr[K0_TR];
    const int r0 = blockIdx.x * K0_TR, v0 = blockIdx.y * K0_VB, s = blockIdx.z;
    const int nr = min(K0_TR, n - r0), nvb = min(K0_VB, V - v0);
    const int tid = threadIdx.x, nth = blockDim.x;
    const int nv = nv_s[s], nh = nh_s[s], b = bits_s[s];
    const int v1 = min(v0 + nvb, nv);                 /* values carrying data in this tile */
    int blo = 0, bhi = 0;
    if (v1 > v0) {
        blo = v0 < nh ? 2 * v0 : 2 * nh + (int)(((long long)(v0 - nh) * 2 * b) >> 3);
        bhi = v1 <= nh ? 2 * v1 : 2 * nh + (int)((((long long)(v1 - nh) * 2 * b) + 7) >> 3) + 2;
        bhi = min(bhi, row_bytes);
    }
    const int bw = bhi - blo;
    if (tid < K0_TR) { nd_row[tid] = 0; okr[tid] = tid < nr ? valid[(long long)s * n + r0 + tid] : 0; }
    for (int i = tid; i < nr * bw; i += nth) {
        const int r = i / bw, c = i - r * bw;
        rt[r * (2 * K0_VB + 8) + c] = raw[((long long)s * n + r0 + r) * row_bytes + blo + c];
    }
    __syncthreads();
    const unsigned int m = (1u << b) - 1u;
    const int half = 1 << (b - 1), sent = -(1 << (b - 1));
    for (int i = tid; i < nvb * K0_TR; i += nth) {
        const int vl = i / K0_TR, r = i - vl * K0_TR, v = v0 + vl;
        int re = 0, im = 0;
        if (r < nr && okr[r] && v < nv) {
            const unsigned char* row = rt + r * (2 * K0_VB + 8) - blo;   /* global byte index */
            int nd;
            if (v < nh) {
                re = (signed char)row[2 * v];
                im = (signed char)row[2 * v + 1];
                nd = (re == -128) || (im == -128);
            } else {
                const long long bit = (long long)(v - nh) * 2 * b;
                const long long byte = 2LL * nh + (bit >> 3);
                unsigned int w = row[byte];
                if (byte + 1 < row_bytes) w |= ((unsigned int)row[byte + 1]) << 8;
                if (byte + 2 < row_bytes) w |= ((unsigned int)row[byte + 2]) << 16;
                w >>= (unsigned int)(bit & 7);
                const int ur = (int)(w & m), ui = (int)((w >> b) & m);
                re = ur >= half ? ur - (1 << b) : ur;
                im = ui >= half ? ui - (1 << b) : ui;
                nd = (re == sent) || (im == sent);
            }
            if (nd) { re = 0; im = 0; atomicAdd(&nd_row[r], 1); }
        }
        ot[i] = (unsigned short)((unsigned int)(unsigned char)re
                                 | ((unsigned int)(unsigned char)im << 8));
    }
    __syncthreads();
    const long long sl0 = ((row0 + r0) % R + R) % R;
    unsigned short* cs = (unsigned short*)codes;
    for (int i = tid; i < nvb * K0_TR; i += nth) {
        const int vl = i / K0_TR, r = i - vl * K0_TR;
        if (r < nr) {
            long long sl = sl0 + r; if (sl >= R) sl -= R;
            cs[((long long)s * V + v0 + vl) * R + sl] = ot[i];
        }
    }
    if (tid < nr && nd_row[tid]) atomicAdd(&ndc[(long long)s * n + r0 + tid], nd_row[tid]);
}

extern "C" __global__ void k0_f2_valid(const unsigned char* __restrict__ valid,
    const int* __restrict__ nv_s, const int* __restrict__ ndc,
    unsigned char* __restrict__ valid_out, unsigned long long* __restrict__ nodata,
    int S, int n, int R, long long row0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * n) return;
    const int s = i / n, r = i - s * n;
    long long sl = ((row0 + r) % R + R) % R;
    const int nd = ndc[i];
    valid_out[(long long)s * R + sl] = (unsigned char)(valid[i] && nd < nv_s[s]);
    if (nd) atomicAdd(&nodata[s], (unsigned long long)nd);
}

/* Layer-1 sigma (noise_norm.layer1.sigma_clipped_std_batched in ONE launch per
   DM): block f gathers its M samples image[(t0 + l1_t[f, m]) % R_img, f,
   l1_p[f, m]] into shared memory, then: median = the LOWER median of the active
   (finite, then also in-band) samples by 4-pass radix select on order-preserving
   float keys (torch.nanmedian), sigma = sqrt(sum (s - med)^2 / count), and
   n_iter times: active &= med - n_sigma sigma <= s <= med + n_sigma sigma,
   recompute. A row with no active sample returns 0. M <= L1_MMAX. */
#define L1_MMAX 10240   /* samples per row; inactive samples are set to NaN */
__device__ __forceinline__ unsigned int l1_key(float v) {
    const unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float l1_unkey(unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
extern "C" __global__ void l1_clipstd(const __half* __restrict__ image, const int* __restrict__ l1_t,
    const int* __restrict__ l1_p, float* __restrict__ out, int F, int P, int R_img, long long t0,
    int M, float n_sigma, int n_iter)
{
    __shared__ float sv[L1_MMAX];
    __shared__ unsigned int hist[256];
    __shared__ float red[32];
    __shared__ int redn[32];
    __shared__ unsigned int s_pref, s_rank;
    __shared__ float s_med, s_sig;
    __shared__ int s_n;
    const int f = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nw = nth >> 5;
    for (int m = tid; m < M; m += nth) {
        long long sl = (t0 + l1_t[(long long)f * M + m]) % R_img; if (sl < 0) sl += R_img;
        const float v = __half2float(image[(sl * F + f) * (long long)P + l1_p[(long long)f * M + m]]);
        sv[m] = isfinite(v) ? v : __int_as_float(0x7fc00000);
    }
    __syncthreads();
    for (int it = 0; it <= n_iter; ++it) {
        if (it > 0) {
            const float med = s_med, sig = s_sig;
            const float lo = __fsub_rn(med, __fmul_rn(n_sigma, sig));
            const float hi = __fadd_rn(med, __fmul_rn(n_sigma, sig));
            for (int m = tid; m < M; m += nth)
                if (!(sv[m] >= lo && sv[m] <= hi)) sv[m] = __int_as_float(0x7fc00000);
            __syncthreads();
        }
        /* count active */
        int c = 0;
        for (int m = tid; m < M; m += nth) c += isnan(sv[m]) ? 0 : 1;
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
        if (lane == 0) redn[warp] = c;
        __syncthreads();
        if (tid == 0) {
            int n = 0;
            for (int w = 0; w < nw; ++w) n += redn[w];
            s_n = n;
            s_pref = 0u;
            s_rank = n > 0 ? (unsigned int)((n - 1) / 2) : 0u;   /* lower median */
        }
        __syncthreads();
        if (s_n == 0) { if (tid == 0) out[f] = 0.0f; return; }
        /* radix select, 8 bits per pass from the top */
        for (int sh = 24; sh >= 0; sh -= 8) {
            for (int i = tid; i < 256; i += nth) hist[i] = 0u;
            __syncthreads();
            const unsigned int pref = s_pref;
            const unsigned int hmask = sh == 24 ? 0u : (0xffffffffu << (sh + 8));
            for (int m = tid; m < M; m += nth) {
                if (isnan(sv[m])) continue;
                const unsigned int k = l1_key(sv[m]);
                if ((k & hmask) == pref) atomicAdd(&hist[(k >> sh) & 0xFFu], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int r = s_rank, cum = 0u, bin = 0u;
                for (bin = 0u; bin < 256u; ++bin) {
                    if (cum + hist[bin] > r) break;
                    cum += hist[bin];
                }
                s_rank = r - cum;
                s_pref = pref | (bin << sh);
            }
            __syncthreads();
        }
        const float med = l1_unkey(s_pref);
        /* sigma */
        float acc = 0.0f;
        for (int m = tid; m < M; m += nth)
            if (!isnan(sv[m])) { const float d = __fsub_rn(sv[m], med); acc = __fadd_rn(acc, __fmul_rn(d, d)); }
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nw; ++w) tot += red[w];
            s_med = med;
            s_sig = sqrtf(tot / (float)s_n);
        }
        __syncthreads();
    }
    if (tid == 0) out[f] = isnan(s_sig) ? 0.0f : s_sig;
}

/* The same clipped sigma for rows of any length held in global memory (Layer-2:
   K rows of up to ~1e5 tap scores): x [R, M] float32 is MUTATED (inactive
   samples -> NaN); one block per row. */
extern "C" __global__ void clipstd_rows(float* __restrict__ x, float* __restrict__ out, int M,
                                        float n_sigma, int n_iter)
{
    __shared__ unsigned int hist[256];
    __shared__ float red[32];
    __shared__ int redn[32];
    __shared__ unsigned int s_pref, s_rank;
    __shared__ float s_med, s_sig;
    __shared__ int s_n;
    float* sv = x + (long long)blockIdx.x * M;
    const int tid = threadIdx.x, nth = blockDim.x, lane = tid & 31, warp = tid >> 5, nw = nth >> 5;
    for (int m = tid; m < M; m += nth) if (!isfinite(sv[m])) sv[m] = __int_as_float(0x7fc00000);
    __syncthreads();
    for (int it = 0; it <= n_iter; ++it) {
        if (it > 0) {
            const float med = s_med, sig = s_sig;
            const float lo = __fsub_rn(med, __fmul_rn(n_sigma, sig));
            const float hi = __fadd_rn(med, __fmul_rn(n_sigma, sig));
            for (int m = tid; m < M; m += nth)
                if (!(sv[m] >= lo && sv[m] <= hi)) sv[m] = __int_as_float(0x7fc00000);
            __syncthreads();
        }
        int c = 0;
        for (int m = tid; m < M; m += nth) c += isnan(sv[m]) ? 0 : 1;
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
        if (lane == 0) redn[warp] = c;
        __syncthreads();
        if (tid == 0) {
            int n = 0;
            for (int w = 0; w < nw; ++w) n += redn[w];
            s_n = n;
            s_pref = 0u;
            s_rank = n > 0 ? (unsigned int)((n - 1) / 2) : 0u;
        }
        __syncthreads();
        if (s_n == 0) { if (tid == 0) out[blockIdx.x] = 0.0f; return; }
        for (int sh = 24; sh >= 0; sh -= 8) {
            for (int i = tid; i < 256; i += nth) hist[i] = 0u;
            __syncthreads();
            const unsigned int pref = s_pref;
            const unsigned int hmask = sh == 24 ? 0u : (0xffffffffu << (sh + 8));
            for (int m = tid; m < M; m += nth) {
                const float v = sv[m];
                if (isnan(v)) continue;
                const unsigned int k = l1_key(v);
                if ((k & hmask) == pref) atomicAdd(&hist[(k >> sh) & 0xFFu], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int r = s_rank, cum = 0u, bin = 0u;
                for (bin = 0u; bin < 256u; ++bin) {
                    if (cum + hist[bin] > r) break;
                    cum += hist[bin];
                }
                s_rank = r - cum;
                s_pref = pref | (bin << sh);
            }
            __syncthreads();
        }
        const float med = l1_unkey(s_pref);
        float acc = 0.0f;
        for (int m = tid; m < M; m += nth) {
            const float v = sv[m];
            if (!isnan(v)) { const float d = __fsub_rn(v, med); acc = __fadd_rn(acc, __fmul_rn(d, d)); }
        }
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nw; ++w) tot += red[w];
            s_med = med;
            s_sig = sqrtf(tot / (float)s_n);
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = isnan(s_sig) ? 0.0f : s_sig;
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

/* lane_dif32 with the five per-lane stage twiddles precomputed (dif_tw) */
__device__ __forceinline__ void dif_tw(float2 (&w)[5], const float2* tw, int N, int lane) {
    #pragma unroll
    for (int s = 0, h = 16; h >= 1; h >>= 1, ++s) w[s] = tw[((lane & (h - 1)) * (N / (2 * h))) % N];
}

template <int R1>
__device__ __forceinline__ void lane_dif32w(float2 (&A)[R1], const float2 (&w)[5], int lane) {
    #pragma unroll
    for (int s = 0, h = 16; h >= 1; h >>= 1, ++s) {
        const int top = (lane & h) == 0;
        #pragma unroll
        for (int i = 0; i < R1; ++i) {
            float2 p;
            p.x = __shfl_xor_sync(0xffffffffu, A[i].x, h);
            p.y = __shfl_xor_sync(0xffffffffu, A[i].y, h);
            A[i] = top ? make_float2(A[i].x + p.x, A[i].y + p.y)
                       : cmul(make_float2(p.x - A[i].x, p.y - A[i].y), w[s]);
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

/* K2+K3a fused (fft_mode "fused"): one block per (residue y2 = y mod R1, image b).
   Phase 1: for every FFT line, the K2 column IFFT restricted to the outputs
   y = y2 + R1 * y1 (y1 = 0..31) -- the same operations as k2_body for that y2
   (one R1-point partial DFT per lane, twiddle, one value through the 32-lane
   DIF) -- into shared memory Ys[line][y1] (stride 33: no bank conflicts); the
   fp32 intermediate never goes to global memory. A lane's inputs come straight
   from uv through the static table lidx[line][position] (uv index or -1).
   Phase 2: the K3a row C2R of those 32 output rows (k3a_body's operations); a
   lane reads its half-axis values straight from Ys / uv through wsrc[w] (line
   slot >= 0, constant line -(uv index + 2), or -1 = zero), and the image row is
   staged in shared memory for one coalesced 512-byte store (col fold).
   Dynamic shared memory: twiddles N float2 + Ys n_fc * 33 float2 (+ pad) + one
   N/2 float2 staging buffer per warp. */
__device__ __forceinline__ float2 k23_src(int s, const float2* Ys, const float2* uvb, int y1) {
    return s >= 0 ? Ys[s * 33 + y1] : (s <= -2 ? uvb[-(s + 2)] : make_float2(0.f, 0.f));
}

template <int LN, int RN>
__device__ __forceinline__ void k23_body(const float2* __restrict__ uv,
    const int* __restrict__ lidx, const int* __restrict__ lstart,
    const unsigned int* __restrict__ lbm, const int* __restrict__ wsrc,
    const float2* __restrict__ tw_g, const float* __restrict__ pix,
    const unsigned char* __restrict__ pcode, float pc, __half* __restrict__ ring,
    int B, int n_act, int n_fc, int n_t, int F, long long t0, int R_img, int transpose,
    int contig)
{
    const int R1 = 8, R2 = 4, N = 256, M = 128;
    extern __shared__ float2 sm[];
    /* twiddles: a shared copy for the broadcast / unit-stride reads; the DIF
       stages index with 64-128 B lane strides (8-16-way bank conflicts in
       shared memory) and read the global table through L1 instead */
    float2* tw = sm;                         /* N */
    float2* Ys = sm + N;                     /* n_fc * 33 (stride 33: no bank conflicts) */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float2* wb = sm + N + ((n_fc * 33 + 1) & ~1) + warp * (N / 2);   /* 16-byte aligned */
    const int y2 = blockIdx.x, b = blockIdx.y;
    if (b >= B) return;
    for (int i = threadIdx.x; i < N; i += blockDim.x) tw[i] = tw_g[i];
    __syncthreads();
    const float2* uvb = uv + (long long)b * n_act;
    /* phase 1: partial column IFFTs */
    float2 wy[R1];
    #pragma unroll
    for (int j = 0; j < R1; ++j) wy[j] = tw[(j * y2 * (N / R1)) % N];
    const float2 wl = tw[(lane * y2) % N];
    float2 wd[5];
    dif_tw(wd, tw_g, N, lane);
    const int y1o = __brev(lane) >> 27;
    /* LN lines per warp at once: independent values through the same DIF (ILP) */
    for (int c0 = warp * LN; c0 < n_fc; c0 += nw * LN) {
        float2 A[LN];
        #pragma unroll
        for (int l = 0; l < LN; ++l) {
            const int ci = min(c0 + l, n_fc - 1);
            float2 acc = make_float2(0.f, 0.f);
            if (contig) {
                /* the line's cells are uv[lstart .. ) in position order: rank by popcount */
                const unsigned int* bm = lbm + ci * R1;
                int pre = __ldg(lstart + ci);
                const unsigned int lt = (1u << lane) - 1u;
                #pragma unroll
                for (int j = 0; j < R1; ++j) {
                    const unsigned int m = __ldg(bm + j);
                    const float2 xj = ((m >> lane) & 1u) ? uvb[pre + __popc(m & lt)]
                                                         : make_float2(0.f, 0.f);
                    pre += __popc(m);
                    const float2 t = cmul(xj, wy[j]);
                    acc.x += t.x; acc.y += t.y;
                }
            } else {
                const int* li = lidx + ci * N + lane;
                #pragma unroll
                for (int j = 0; j < R1; ++j) {
                    const int e = __ldg(li + 32 * j);
                    const float2 xj = e >= 0 ? uvb[e] : make_float2(0.f, 0.f);
                    const float2 t = cmul(xj, wy[j]);
                    acc.x += t.x; acc.y += t.y;
                }
            }
            A[l] = cmul(acc, wl);
        }
        lane_dif32w<LN>(A, wd, lane);
        #pragma unroll
        for (int l = 0; l < LN; ++l)
            if (c0 + l < n_fc) Ys[(c0 + l) * 33 + y1o] = A[l];
    }
    __syncthreads();
    /* phase 2: row C2R of the 32 rows y = y2 + R1 * y1 */
    const int f = b / n_t, t = b % n_t;
    const long long slot = ((t0 + t) % R_img + R_img) % R_img;
    __half* dst = ring + (slot * F + f) * (long long)N * N;
    int ws[R2], wr[R2];
    float2 wk[R2], wo[R2], wm[R2 * R2];
    #pragma unroll
    for (int k2 = 0; k2 < R2; ++k2) {
        const int k = lane + 32 * k2;
        ws[k2] = __ldg(wsrc + k);
        wr[k2] = __ldg(wsrc + (k == 0 ? M : M - k));
        wk[k2] = tw[k];
        wo[k2] = tw[(2 * lane * k2) % N];
        #pragma unroll
        for (int n2 = 0; n2 < R2; ++n2) wm[k2 * R2 + n2] = tw[(k2 * n2 * (N / R2)) % N];
    }
    /* RN rows per warp at once (ILP through the DIF) */
    for (int r0 = warp * RN; r0 < 32; r0 += nw * RN) {
        float2 Bv[RN * R2];
        #pragma unroll
        for (int rr = 0; rr < RN; ++rr) {
            const int y1 = r0 + rr;
            float2 A[R2];
            #pragma unroll
            for (int k2 = 0; k2 < R2; ++k2) {
                const int k = lane + 32 * k2;
                float2 yk = k23_src(ws[k2], Ys, uvb, y1), yr;
                const float2 r = k23_src(wr[k2], Ys, uvb, y1);
                if (k == 0) { yk = make_float2(yk.x, 0.f); yr = make_float2(r.x, 0.f); }
                else { yr = make_float2(r.x, -r.y); }
                float2 fe = make_float2(yk.x + yr.x, yk.y + yr.y);
                float2 fo = cmul(make_float2(yk.x - yr.x, yk.y - yr.y), wk[k2]);
                A[k2] = make_float2(fe.x - fo.y, fe.y + fo.x);
            }
            #pragma unroll
            for (int n2 = 0; n2 < R2; ++n2) {
                float2 acc = make_float2(0.f, 0.f);
                #pragma unroll
                for (int k2 = 0; k2 < R2; ++k2) {
                    float2 tt = cmul(A[k2], wm[k2 * R2 + n2]);
                    acc.x += tt.x; acc.y += tt.y;
                }
                Bv[rr * R2 + n2] = cmul(acc, wo[n2]);
            }
        }
        lane_dif32w<RN * R2>(Bv, wd, lane);
      #pragma unroll
      for (int rr = 0; rr < RN; ++rr) {
        const int y1 = r0 + rr;
        const int y = y2 + R1 * y1;
        const int n1 = __brev(lane) >> 27;
        __half* hb = (__half*)wb;              /* N halves */
        #pragma unroll
        for (int n2 = 0; n2 < R2; ++n2) {
            const int nn = n2 + R2 * n1;
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int x = 2 * nn + q;
                const long long pidx = transpose ? (long long)x * N + y : (long long)y * N + x;
                const float raw = q ? Bv[rr * R2 + n2].y : Bv[rr * R2 + n2].x;
                float val;
                if (pcode) {           /* pix in {+pc, -pc, +0, -0}: exactly raw * pix */
                    const unsigned int cd = __ldg(pcode + pidx);
                    val = cd < 2u ? (cd ? -(raw * pc) : raw * pc) : raw * (cd == 2u ? 0.0f : -0.0f);
                } else {
                    val = raw * pix[pidx];
                }
                if (transpose) dst[pidx] = __float2half_rn(val);
                else hb[x] = __float2half_rn(val);
            }
        }
        if (!transpose) {
            __syncwarp();
            ((uint4*)(dst + (long long)y * N))[lane] = ((const uint4*)hb)[lane];
            __syncwarp();
        }
      }
    }
}

#define K23_ARGS const float2* __restrict__ uv, const int* __restrict__ lidx, \
    const int* __restrict__ lstart, const unsigned int* __restrict__ lbm, \
    const int* __restrict__ wsrc, const float2* __restrict__ tw_g, const float* __restrict__ pix, \
    const unsigned char* __restrict__ pcode, float pc, __half* __restrict__ ring, int B, \
    int n_act, int n_fc, int n_t, int F, long long t0, int R_img, int transpose, int contig
#define K23_PASS uv, lidx, lstart, lbm, wsrc, tw_g, pix, pcode, pc, ring, B, n_act, n_fc, n_t, \
    F, t0, R_img, transpose, contig
extern "C" __global__ void k23_fused_256(K23_ARGS) { k23_body<4, 2>(K23_PASS); }


/* ---- K2+K3a fused v2 (k23v2_256, fft_mode "fused"): the k23_fused_256 data
   flow (one block per (residue y2 = y mod 8, image); phase 1 the column IFFT
   at that residue into shared Ys, phase 2 the row C2R of its 32 rows) with
   fewer instructions and half the shared memory, so two 16-warp blocks fit
   an SM:
   * phase 1 is compiled per residue (template Y2): the in-lane 8-point DFT
     at Y2 uses the exact 8th roots (+-1, +-i, one sqrt(1/2) multiply), and a
     gap-free line (contiguous positions AND uv cells) is gathered by range,
     a gapped one through its 256-bit occupancy;
   * the DIF butterflies need no select (s = p + sign * a, then * w, w = 1 on
     the top half) and the last stage no multiply;
   * phase 2 reads every half-axis value from Ys through one per-lane row
     offset (constant lines are materialised as rows, zeros as a zero row),
     the 4-point stage uses the exact 4th roots, and each lane stores its
     eight consecutive pixels as ONE 16-byte write (col fold) -- no staging,
     no shared twiddle copy.
   Same arithmetic up to fp32 rounding (exact twiddles instead of the table's
   cos(pi/2) = 6e-17); checked against the twin like the other FFT paths. */
template <int E> __device__ __forceinline__ float2 k23_rot8(float2 v) {
    if (E == 0) return v;
    if (E == 2) return make_float2(-v.y, v.x);
    if (E == 4) return make_float2(-v.x, -v.y);
    if (E == 6) return make_float2(v.y, -v.x);
    if (E == 1) return make_float2(v.x - v.y, v.x + v.y);     /* / sqrt(2) later */
    if (E == 3) return make_float2(-v.x - v.y, v.x - v.y);
    if (E == 5) return make_float2(v.y - v.x, -v.x - v.y);
    return make_float2(v.x + v.y, v.y - v.x);                  /* E == 7 */
}

template <int Y2, int J> struct K23Dft8 {
    static __device__ __forceinline__ void acc(float2& ev, float2& od, const float2 (&x)[8]) {
        const float2 t = k23_rot8<(J * Y2) & 7>(x[J]);
        if (((J * Y2) & 7) & 1) { od.x += t.x; od.y += t.y; }
        else { ev.x += t.x; ev.y += t.y; }
        K23Dft8<Y2, J + 1>::acc(ev, od, x);
    }
};
template <int Y2> struct K23Dft8<Y2, 8> {
    static __device__ __forceinline__ void acc(float2&, float2&, const float2 (&)[8]) {}
};

struct K23Dif { float2 w[4]; float sg[5]; };

__device__ __forceinline__ void k23_dif_prep(K23Dif& d, const float2* __restrict__ tw_g, int lane) {
    #pragma unroll
    for (int s = 0, h = 16; h >= 1; h >>= 1, ++s) {
        const bool top = (lane & h) == 0;
        d.sg[s] = top ? 1.0f : -1.0f;
        if (s < 4) d.w[s] = top ? make_float2(1.0f, 0.0f)
                                : __ldg(tw_g + ((lane & (h - 1)) * (256 / (2 * h))) % 256);
    }
}

template <int R>
__device__ __forceinline__ void k23_dif32(float2 (&A)[R], const K23Dif& d) {
    #pragma unroll
    for (int s = 0, h = 16; h >= 1; h >>= 1, ++s) {
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            float2 p;
            p.x = __shfl_xor_sync(0xffffffffu, A[i].x, h);
            p.y = __shfl_xor_sync(0xffffffffu, A[i].y, h);
            const float2 t = make_float2(fmaf(d.sg[s], A[i].x, p.x), fmaf(d.sg[s], A[i].y, p.y));
            A[i] = (h == 1) ? t : cmul(t, d.w[s < 4 ? s : 0]);
        }
    }
}

template <int Y2, int LN>
__device__ __forceinline__ void k23v2_p1(const float2* __restrict__ uvb,
    const int* __restrict__ lstart, const uint4* __restrict__ lbm4, const int2* __restrict__ lspan,
    const float2* __restrict__ tw_g, float2* Ys, const K23Dif& d, int n_fc, int lane, int warp,
    int nw)
{
    const float2 wl = __ldg(tw_g + ((lane * Y2) & 255));
    const unsigned int lt = (1u << lane) - 1u;
    const int y1o = __brev(lane) >> 27;
    /* lines warp, warp + nw, ... ; LN of them through the DIF at once */
    for (int it = 0; warp + it * nw < n_fc; it += LN) {
        float2 A[LN];
        #pragma unroll
        for (int l = 0; l < LN; ++l) {
            const int ci = min(warp + (it + l) * nw, n_fc - 1);
            const int st = __ldg(lstart + ci);
            const int2 sp = __ldg(lspan + ci);          /* (p0, n) gap-free, else (0, -1) */
            float2 x[8];
            if (sp.y >= 0) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned int e = (unsigned int)(lane + 32 * j - sp.x);
                    x[j] = e < (unsigned int)sp.y ? uvb[st + (int)e] : make_float2(0.f, 0.f);
                }
            } else {
                const uint4 m0 = __ldg(lbm4 + 2 * ci), m1 = __ldg(lbm4 + 2 * ci + 1);
                const unsigned int m[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
                int pre = st;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x[j] = ((m[j] >> lane) & 1u) ? uvb[pre + __popc(m[j] & lt)]
                                                 : make_float2(0.f, 0.f);
                    pre += __popc(m[j]);
                }
            }
            float2 ev = make_float2(0.f, 0.f), od = make_float2(0.f, 0.f);
            K23Dft8<Y2, 0>::acc(ev, od, x);
            if (Y2 & 1) {
                const float r = 0.70710678118654752f;
                ev.x = fmaf(r, od.x, ev.x);
                ev.y = fmaf(r, od.y, ev.y);
            }
            A[l] = cmul(ev, wl);
        }
        k23_dif32<LN>(A, d);
        #pragma unroll
        for (int l = 0; l < LN; ++l) {
            const int ci = warp + (it + l) * nw;
            if (ci < n_fc) Ys[ci * 33 + y1o] = A[l];
        }
    }
}

template <int RN>
__device__ __forceinline__ void k23v2_p2(const float2* Ys, const int* __restrict__ wrow,
    const float2* __restrict__ tw_g, const float* __restrict__ pix,
    const unsigned char* __restrict__ pcode, float pc, __half* __restrict__ dst, const K23Dif& d,
    int y2, int transpose, int lane, int warp, int nw)
{
    const int N = 256, M = 128;
    int ok[4], orr[4];
    float2 wk[4], wo[4];
    #pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) {
        const int k = lane + 32 * k2;
        ok[k2] = __ldg(wrow + k) * 33;
        orr[k2] = __ldg(wrow + (k == 0 ? M : M - k)) * 33;
        wk[k2] = __ldg(tw_g + k);
        wo[k2] = __ldg(tw_g + ((2 * lane * k2) & 255));
    }
    const float m0 = lane == 0 ? 0.0f : 1.0f;   /* k = 0: DC / Nyquist real parts only */
    const int n1 = __brev(lane) >> 27;
    const unsigned int cb = __float_as_uint(pc);
    for (int r0 = warp * RN; r0 < 32; r0 += nw * RN) {
        float2 Bv[RN * 4];
        #pragma unroll
        for (int rr = 0; rr < RN; ++rr) {
            const int y1 = r0 + rr;
            float2 A[4];
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                float2 yk = Ys[ok[k2] + y1];
                const float2 r = Ys[orr[k2] + y1];
                float2 yr = make_float2(r.x, -r.y);
                if (k2 == 0) { yk.y *= m0; yr.y *= m0; }
                const float2 fe = make_float2(yk.x + yr.x, yk.y + yr.y);
                const float2 fo = cmul(make_float2(yk.x - yr.x, yk.y - yr.y), wk[k2]);
                A[k2] = make_float2(fe.x - fo.y, fe.y + fo.x);
            }
            const float2 s02 = make_float2(A[0].x + A[2].x, A[0].y + A[2].y);
            const float2 d02 = make_float2(A[0].x - A[2].x, A[0].y - A[2].y);
            const float2 s13 = make_float2(A[1].x + A[3].x, A[1].y + A[3].y);
            const float2 d13 = make_float2(A[1].x - A[3].x, A[1].y - A[3].y);
            Bv[rr * 4 + 0] = make_float2(s02.x + s13.x, s02.y + s13.y);
            Bv[rr * 4 + 1] = cmul(make_float2(d02.x - d13.y, d02.y + d13.x), wo[1]);
            Bv[rr * 4 + 2] = cmul(make_float2(s02.x - s13.x, s02.y - s13.y), wo[2]);
            Bv[rr * 4 + 3] = cmul(make_float2(d02.x + d13.y, d02.y - d13.x), wo[3]);
        }
        k23_dif32<RN * 4>(Bv, d);
        #pragma unroll
        for (int rr = 0; rr < RN; ++rr) {
            const int y = y2 + 8 * (r0 + rr);
            /* lane: pixels x = 8 n1 + i, i = 2 n2 + q */
            float v[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = (i & 1) ? Bv[rr * 4 + (i >> 1)].y : Bv[rr * 4 + (i >> 1)].x;
            if (!transpose) {
                const long long p0 = (long long)y * N + 8 * n1;
                if (pcode) {            /* pix in {+pc, -pc, +0, -0}: exactly raw * pix */
                    const uint2 c8 = __ldg((const uint2*)(pcode + p0));
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const unsigned int cd = ((i < 4 ? c8.x : c8.y) >> (8 * (i & 3))) & 3u;
                        v[i] *= __uint_as_float(((cd & 2u) ? 0u : cb) | ((cd & 1u) << 31));
                    }
                } else {
                    const float4 q0 = __ldg((const float4*)(pix + p0));
                    const float4 q1 = __ldg((const float4*)(pix + p0 + 4));
                    v[0] *= q0.x; v[1] *= q0.y; v[2] *= q0.z; v[3] *= q0.w;
                    v[4] *= q1.x; v[5] *= q1.y; v[6] *= q1.z; v[7] *= q1.w;
                }
                __half2 h[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
                *(uint4*)(dst + p0) = *(const uint4*)h;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const long long pidx = (long long)(8 * n1 + i) * N + y;
                    float f;
                    if (pcode) {
                        const unsigned int cd = __ldg(pcode + pidx);
                        f = __uint_as_float(((cd & 2u) ? 0u : cb) | ((cd & 1u) << 31));
                    } else {
                        f = pix[pidx];
                    }
                    dst[pidx] = __float2half_rn(v[i] * f);
                }
            }
        }
    }
}

template <int Y2, int LN>
__device__ __forceinline__ void k23v2_block(const float2* __restrict__ uv,
    const int* __restrict__ lstart, const uint4* __restrict__ lbm4, const int2* __restrict__ lspan,
    const int* __restrict__ wrow, const int* __restrict__ cidx, const float2* __restrict__ tw_g,
    const float* __restrict__ pix, const unsigned char* __restrict__ pcode, float pc,
    __half* __restrict__ ring, int B, int n_act, int n_fc, int n_cl, int n_t, int F, long long t0,
    int R_img, int transpose)
{
    extern __shared__ float2 sm[];
    float2* Ys = sm;                       /* (n_fc + n_cl + 1) rows x 33 */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int b = blockIdx.y;
    const float2* uvb = uv + (long long)b * n_act;
    K23Dif d;
    k23_dif_prep(d, tw_g, lane);
    /* constant lines (position 0 only: the same value at every y) and the zero row */
    for (int i = threadIdx.x; i < (n_cl + 1) * 32; i += blockDim.x) {
        const int c = i >> 5;
        Ys[(n_fc + c) * 33 + (i & 31)] = c < n_cl ? uvb[__ldg(cidx + c)] : make_float2(0.f, 0.f);
    }
    k23v2_p1<Y2, LN>(uvb, lstart, lbm4, lspan, tw_g, Ys, d, n_fc, lane, warp, nw);
    __syncthreads();
    const int f = b / n_t, t = b % n_t;
    const long long slot = ((t0 + t) % R_img + R_img) % R_img;
    k23v2_p2<2>(Ys, wrow, tw_g, pix, pcode, pc, ring + (slot * F + f) * 65536LL, d, Y2,
                transpose, lane, warp, nw);
}

#define K23V2_ARGS const float2* __restrict__ uv, const int* __restrict__ lstart, \
    const uint4* __restrict__ lbm4, const int2* __restrict__ lspan, const int* __restrict__ wrow, \
    const int* __restrict__ cidx, const float2* __restrict__ tw_g, const float* __restrict__ pix, \
    const unsigned char* __restrict__ pcode, float pc, __half* __restrict__ ring, int B, \
    int n_act, int n_fc, int n_cl, int n_t, int F, long long t0, int R_img, int transpose
#define K23V2_PASS uv, lstart, lbm4, lspan, wrow, cidx, tw_g, pix, pcode, pc, ring, B, n_act, \
    n_fc, n_cl, n_t, F, t0, R_img, transpose
#define K23V2_KERNEL(NAME, LN, THREADS, MINB) \
extern "C" __global__ void __launch_bounds__(THREADS, MINB) NAME(K23V2_ARGS) { \
    switch (blockIdx.x) { \
    case 0: k23v2_block<0, LN>(K23V2_PASS); break; \
    case 1: k23v2_block<1, LN>(K23V2_PASS); break; \
    case 2: k23v2_block<2, LN>(K23V2_PASS); break; \
    case 3: k23v2_block<3, LN>(K23V2_PASS); break; \
    case 4: k23v2_block<4, LN>(K23V2_PASS); break; \
    case 5: k23v2_block<5, LN>(K23V2_PASS); break; \
    case 6: k23v2_block<6, LN>(K23V2_PASS); break; \
    default: k23v2_block<7, LN>(K23V2_PASS); break; \
    } }
/* 8 warps, two lines per warp through the phase-1 DIF, two blocks per SM: the
   fastest of (LN 2/4/8) x (8/16 warps) on the 2x maps (n13, 2080 Ti) */
K23V2_KERNEL(k23v2_256, 2, 256, 2)

/* ---- k23v2p_256: residue PAIRS (y2, y2 + 4) per block: one gather and one
   in-lane DFT serve both (W_8^{j (y2+4)} = (-1)^j W_8^{j y2}: a = E + O,
   b = E - O over even / odd j), two DIFs; Ys holds both residues (2 x 30 KB,
   one 16-warp block per SM); phase 2 does the pair's 64 rows. */
template <int Y2, int J> struct K23Dft8EO {
    static __device__ __forceinline__ void acc(float2& e, float2& o, const float2 (&x)[8]) {
        const float2 t = k23_rot8<(J * Y2) & 7>(x[J]);
        if (J & 1) { o.x += t.x; o.y += t.y; }
        else { e.x += t.x; e.y += t.y; }
        K23Dft8EO<Y2, J + 1>::acc(e, o, x);
    }
};
template <int Y2> struct K23Dft8EO<Y2, 8> {
    static __device__ __forceinline__ void acc(float2&, float2&, const float2 (&)[8]) {}
};

template <int Y2, int LN>
__device__ __forceinline__ void k23p_p1(const float2* __restrict__ uvb,
    const int* __restrict__ lstart, const uint4* __restrict__ lbm4, const int2* __restrict__ lspan,
    const float2* __restrict__ tw_g, float2* Ys, int ys_half, const K23Dif& d, int n_fc, int lane,
    int warp, int nw)
{
    const float2 wla = __ldg(tw_g + ((lane * Y2) & 255));
    const float2 wlb = __ldg(tw_g + ((lane * (Y2 + 4)) & 255));
    const unsigned int lt = (1u << lane) - 1u;
    const int y1o = __brev(lane) >> 27;
    for (int it = 0; warp + it * nw < n_fc; it += LN) {
        float2 A[2 * LN];
        #pragma unroll
        for (int l = 0; l < LN; ++l) {
            const int ci = min(warp + (it + l) * nw, n_fc - 1);
            const int st = __ldg(lstart + ci);
            const int2 sp = __ldg(lspan + ci);
            float2 x[8];
            if (sp.y >= 0) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned int e = (unsigned int)(lane + 32 * j - sp.x);
                    x[j] = e < (unsigned int)sp.y ? uvb[st + (int)e] : make_float2(0.f, 0.f);
                }
            } else {
                const uint4 m0 = __ldg(lbm4 + 2 * ci), m1 = __ldg(lbm4 + 2 * ci + 1);
                const unsigned int m[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
                int pre = st;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    x[j] = ((m[j] >> lane) & 1u) ? uvb[pre + __popc(m[j] & lt)]
                                                 : make_float2(0.f, 0.f);
                    pre += __popc(m[j]);
                }
            }
            float2 e = make_float2(0.f, 0.f), o = make_float2(0.f, 0.f);
            K23Dft8EO<Y2, 0>::acc(e, o, x);
            float2 a, b;
            if (Y2 & 1) {                      /* odd j <-> odd exponent: the sqrt(1/2) */
                const float r = 0.70710678118654752f;
                a = make_float2(fmaf(r, o.x, e.x), fmaf(r, o.y, e.y));
                b = make_float2(fmaf(-r, o.x, e.x), fmaf(-r, o.y, e.y));
            } else {
                a = make_float2(e.x + o.x, e.y + o.y);
                b = make_float2(e.x - o.x, e.y - o.y);
            }
            A[l] = cmul(a, wla);
            A[LN + l] = cmul(b, wlb);
        }
        k23_dif32<2 * LN>(A, d);
        #pragma unroll
        for (int l = 0; l < LN; ++l) {
            const int ci = warp + (it + l) * nw;
            if (ci < n_fc) {
                Ys[ci * 33 + y1o] = A[l];
                Ys[ys_half + ci * 33 + y1o] = A[LN + l];
            }
        }
    }
}

template <int Y2, int LN>
__device__ __forceinline__ void k23p_block(K23V2_ARGS)
{
    extern __shared__ float2 sm[];
    const int rows = n_fc + n_cl + 1;
    const int ys_half = ((rows * 33 + 1) & ~1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int b = blockIdx.y;
    const float2* uvb = uv + (long long)b * n_act;
    K23Dif d;
    k23_dif_prep(d, tw_g, lane);
    for (int i = threadIdx.x; i < 2 * (n_cl + 1) * 32; i += blockDim.x) {
        const int h = i / ((n_cl + 1) * 32), r = i - h * (n_cl + 1) * 32;
        const int c = r >> 5;
        sm[h * ys_half + (n_fc + c) * 33 + (r & 31)] =
            c < n_cl ? uvb[__ldg(cidx + c)] : make_float2(0.f, 0.f);
    }
    k23p_p1<Y2, LN>(uvb, lstart, lbm4, lspan, tw_g, sm, ys_half, d, n_fc, lane, warp, nw);
    __syncthreads();
    const int f = b / n_t, t = b % n_t;
    const long long slot = ((t0 + t) % R_img + R_img) % R_img;
    __half* dst = ring + (slot * F + f) * 65536LL;
    /* the pair's 64 rows: warps [0, nw/2) residue Y2, [nw/2, nw) residue Y2 + 4 */
    const int hw = warp >= nw / 2;
    k23v2_p2<2>(sm + hw * ys_half, wrow, tw_g, pix, pcode, pc, dst, d, Y2 + 4 * hw, transpose, lane,
                warp - hw * (nw / 2), nw / 2);
}

/* LN 1: one line per warp through the (two-residue) phase-1 DIF -- 0.97 us/image
   vs 1.01 at LN 2, 1.02 at 32 warps (64 registers) and 1.39 for k23v2_256 (ns4
   map, n13 microbench) */
extern "C" __global__ void __launch_bounds__(512, 1) k23v2p_256(K23V2_ARGS) {
    switch (blockIdx.x) {
    case 0: k23p_block<0, 1>(K23V2_PASS); break;
    case 1: k23p_block<1, 1>(K23V2_PASS); break;
    case 2: k23p_block<2, 1>(K23V2_PASS); break;
    default: k23p_block<3, 1>(K23V2_PASS); break;
    }
}
"""

_MOD = None


def get_module():
    """Compile (NVRTC) on first use -- GPU nodes at a checkpoint only."""
    global _MOD
    if _MOD is None:
        import cupy as cp
        _MOD = cp.RawModule(code=CUDA_SOURCE, backend="nvrtc", options=("--std=c++14",))
    return _MOD


def k23_tables(lines: dict, n_fl: int, device, pix=None) -> dict:
    """Static tables of the fused FFT from the K2/K3a line bookkeeping.

    * ``wsrc [129]``: per half-axis index the FFT line slot, ``-(uv index +
      2)`` for a constant line, -1 = zero;
    * if every line's cells are a contiguous uv range in position order (the
      owner CSR's line-major order -- every DSA-110 2x map): ``lstart [n_fl]``
      + ``lbm [n_fl, 8]`` 256-bit occupancy (a lane ranks its positions by
      popcount: 36 B per line instead of a 1 KB index row); otherwise the
      dense ``lidx [n_fl, 256]`` (uv index or -1);
    * ``pix`` given and every value in ``{+c, -c, +0, -0}``: one byte per
      pixel + ``c`` instead of the fp32 table (exactly ``raw * pix``).
    """
    import torch

    lp = lines["line_ptr"].cpu().numpy()
    li = lines["line_idx"].cpu().numpy()
    lpos = lines["line_pos"].cpu().numpy()
    fl = lines["fl"].cpu().numpy()
    cci = lines["cc_idx"].cpu().numpy()
    ccl = lines["cc_line"].cpu().numpy()
    nl = max(n_fl, 1)
    lidx = np.full((nl, 256), -1, dtype=np.int32)
    lstart = np.zeros(nl, dtype=np.int32)
    lbm = np.zeros((nl, 8), dtype=np.uint32)
    contig = True
    for ci in range(n_fl):
        e0, e1 = int(lp[ci]), int(lp[ci + 1])
        idx, pos = li[e0:e1], lpos[e0:e1]
        lidx[ci, pos] = idx
        if e1 > e0:
            lstart[ci] = idx[0]
            contig &= bool(np.all(idx == idx[0] + np.arange(e1 - e0))
                           and np.all(np.diff(pos) > 0))
        for q in pos:
            lbm[ci, q >> 5] |= np.uint32(1) << np.uint32(q & 31)
    wsrc = np.full(129, -1, dtype=np.int32)
    wsrc[fl[:n_fl]] = np.arange(n_fl, dtype=np.int32)
    wsrc[ccl] = -(cci.astype(np.int32) + 2)
    # v2: one Ys row per half-axis index (FFT line slot, constant line n_fl + i,
    # zero row n_fl + n_cl); gap-free lines as (first position, count)
    n_cl = int(cci.size)
    wrow = np.full(129, n_fl + n_cl, dtype=np.int32)
    wrow[fl[:n_fl]] = np.arange(n_fl, dtype=np.int32)
    wrow[ccl] = n_fl + np.arange(n_cl, dtype=np.int32)
    lspan = np.zeros((nl, 2), dtype=np.int32)
    lspan[:, 1] = -1
    for ci in range(n_fl):
        e0, e1 = int(lp[ci]), int(lp[ci + 1])
        idx, pos = li[e0:e1], lpos[e0:e1]
        if e1 > e0 and np.all(np.diff(pos) == 1) and np.all(np.diff(idx) == 1):
            lspan[ci] = (pos[0], e1 - e0)
        elif e1 == e0:
            lspan[ci] = (0, 0)
    out = dict(n_fl=n_fl, contig=int(contig), wsrc=torch.as_tensor(wsrc, device=device),
               wrow=torch.as_tensor(wrow, device=device), n_cl=n_cl,
               cidx=torch.as_tensor(np.concatenate([cci.astype(np.int32), [0]]), device=device),
               lspan=torch.as_tensor(lspan, device=device),
               lidx=torch.as_tensor(lidx, device=device),
               lstart=torch.as_tensor(lstart, device=device),
               lbm=torch.as_tensor(lbm.view(np.int32), device=device), pcode=None, pc=0.0)
    if pix is not None:
        pv = np.asarray(pix.detach().cpu().numpy() if hasattr(pix, "detach") else pix,
                        dtype=np.float32).reshape(-1)
        nz = pv[pv != 0]
        c = float(np.abs(nz).max()) if nz.size else 0.0
        code = np.full(pv.shape, 255, dtype=np.uint8)
        code[pv == np.float32(c)] = 0
        code[pv == np.float32(-c)] = 1
        z = pv == 0
        code[z & ~np.signbit(pv)] = 2
        code[z & np.signbit(pv)] = 3
        if c > 0 and not (code == 255).any():
            out["pcode"] = torch.as_tensor(code, device=device)
            out["pc"] = c
    return out


def launch_l1(image, l1_t, l1_p, out, *, F: int, P: int, R_img: int, t0: int, M: int,
              stream, n_sigma: float = 3.0, n_iter: int = 3) -> None:
    """Layer-1 sigma of one DM-cycle in one launch (``l1_clipstd``): ``out [F]``
    float32; ``l1_t``/``l1_p`` int32 ``[F, M]`` (sample rows / pixels)."""
    import cupy as cp

    from ...image.tiled_combine_cuda import _ptr

    if M > 10240:
        raise ValueError(f"M={M} > 10240 Layer-1 samples per fine DM")
    with cp.cuda.ExternalStream(stream.cuda_stream):
        get_module().get_function("l1_clipstd")(
            (int(F),), (256,),
            (_ptr(image), _ptr(l1_t), _ptr(l1_p), _ptr(out), np.int32(F), np.int32(P),
             np.int32(R_img), np.int64(t0), np.int32(M), np.float32(n_sigma), np.int32(n_iter)))


def launch_clipstd_rows(x, out, *, stream, n_sigma: float = 3.0, n_iter: int = 3) -> None:
    """``clipstd_rows``: per-row clipped sigma of ``x [R, M]`` float32 (contiguous,
    MUTATED: inactive samples become NaN) into ``out [R]``."""
    import cupy as cp

    from ...image.tiled_combine_cuda import _ptr

    R, M = int(x.shape[0]), int(x.shape[1])
    with cp.cuda.ExternalStream(stream.cuda_stream):
        get_module().get_function("clipstd_rows")(
            (R,), (1024,), (_ptr(x), _ptr(out), np.int32(M), np.float32(n_sigma),
                            np.int32(n_iter)))


def launch_k0_tile(draw, dval, nv, nh, bits, codes, vout, nodata, *, S: int, n: int,
                   row_bytes: int, V: int, R: int, row0: int, stream, ndc=None) -> None:
    """Frame-v2 K0 in 2-D tiles (``k0_f2_tile2``: 32 rows x 128 values x stream)
    + the per-row validity pass (``k0_f2_valid``). Same outputs as
    ``k0_unpack_frame2``. ``ndc``: optional int32 scratch ``[S * n]``."""
    import cupy as cp
    import torch

    from ...image.tiled_combine_cuda import _ptr

    mod = get_module()
    if ndc is None or ndc.numel() < S * n:
        ndc = torch.zeros(S * n, dtype=torch.int32, device=draw.device)
    else:
        ndc = ndc[:S * n]
        ndc.zero_()
    with cp.cuda.ExternalStream(stream.cuda_stream):
        mod.get_function("k0_f2_tile2")(
            ((n + 31) // 32, (V + 127) // 128, S), (256,),
            (_ptr(draw), _ptr(dval), _ptr(nv), _ptr(nh), _ptr(bits), _ptr(codes), _ptr(ndc),
             np.int32(S), np.int32(n), np.int32(row_bytes), np.int32(V), np.int32(R),
             np.int64(row0)))
        mod.get_function("k0_f2_valid")(
            ((S * n + 255) // 256,), (256,),
            (_ptr(dval), _ptr(nv), _ptr(ndc), _ptr(vout), _ptr(nodata), np.int32(S),
             np.int32(n), np.int32(R), np.int64(row0)))


def launch_k23(uv, tables: dict, tw, pix, ring, *, B: int, n_act: int, n_t: int, F: int,
               t0: int, R_img: int, transpose: int, stream, warps: Optional[int] = None,
               kname: str = "k23v2p_256") -> None:
    """K2+K3a fused: ``uv`` float32 ``[B, n_act, 2]`` (``b = f * n_t + t``) ->
    fp16 image ring ``[R_img, F, N*N]`` rows ``t0 .. t0 + n_t - 1``; ``tables``
    = :func:`k23_tables`. ``kname``: ``k23v2p_256`` (default: residue pairs),
    ``k23v2_256`` (one residue per block) -- both fall back to v1 for a map
    whose line cells are not uv-contiguous -- or ``k23_fused_256`` (v1)."""
    import cupy as cp

    from ...image.tiled_combine_cuda import _ptr

    n_fl = int(tables["n_fl"])
    if kname.startswith("k23v2") and not tables["contig"]:
        kname = "k23_fused_256"          # v2 gathers line-contiguous uv cells only
    if kname.startswith("k23v2"):
        warps = warps or 8
        k = get_module().get_function(kname)
        n_cl = int(tables["n_cl"])
        smem = (((n_fl + n_cl + 1) * 33 + 1) & ~1) * 8
        if kname.startswith("k23v2p"):                 # residue pairs: two Ys, 16 warps
            smem, warps = 2 * smem, 16
        if smem > 48 * 1024 and getattr(k, "max_dynamic_shared_size_bytes", 0) < smem:
            if hasattr(k, "max_dynamic_shared_size_bytes"):      # cupy RawKernel
                k.max_dynamic_shared_size_bytes = smem
            else:
                cp.cuda.driver.funcSetAttribute(k.ptr, 8, smem)
        pcode = tables["pcode"]
        with cp.cuda.ExternalStream(stream.cuda_stream):
            k((4 if kname.startswith("k23v2p") else 8, int(B)),
              (32 * warps,),
              (_ptr(uv), _ptr(tables["lstart"]), _ptr(tables["lbm"]), _ptr(tables["lspan"]),
               _ptr(tables["wrow"]), _ptr(tables["cidx"]), _ptr(tw), _ptr(pix),
               np.uint64(0) if pcode is None else _ptr(pcode), np.float32(tables["pc"]),
               _ptr(ring), np.int32(B), np.int32(n_act), np.int32(n_fl), np.int32(n_cl),
               np.int32(n_t), np.int32(F), np.int64(t0), np.int32(R_img), np.int32(transpose)),
              shared_mem=smem)
        return
    warps = warps or 16
    k = get_module().get_function(kname)
    smem = (256 + ((n_fl * 33 + 1) & ~1) + warps * 128) * 8
    if smem > 48 * 1024 and getattr(k, "max_dynamic_shared_size_bytes", 0) < smem:
        if hasattr(k, "max_dynamic_shared_size_bytes"):          # cupy RawKernel
            k.max_dynamic_shared_size_bytes = smem
        else:                       # Function: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cp.cuda.driver.funcSetAttribute(k.ptr, 8, smem)
    pcode = tables["pcode"]
    with cp.cuda.ExternalStream(stream.cuda_stream):
        k((8, int(B)), (32 * warps,),
          (_ptr(uv), _ptr(tables["lidx"]), _ptr(tables["lstart"]), _ptr(tables["lbm"]),
           _ptr(tables["wsrc"]), _ptr(tw), _ptr(pix),
           np.uint64(0) if pcode is None else _ptr(pcode), np.float32(tables["pc"]), _ptr(ring),
           np.int32(B), np.int32(n_act), np.int32(n_fl), np.int32(n_t), np.int32(F),
           np.int64(t0), np.int32(R_img), np.int32(transpose), np.int32(tables["contig"])),
          shared_mem=smem)


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

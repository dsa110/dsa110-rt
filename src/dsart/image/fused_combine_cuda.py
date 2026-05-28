"""src/dsart/image/fused_combine_cuda.py — fused per-fdm combine CUDA
kernel for the M5 imager.

Memory pattern of the chunk-6c-follow-up bench combine step:

  for f in range(N_fdm):
      uv.zero_()                       # 1 write
      for g in range(N_chgroup):
          uv += streams[g, s_g:s_g+T_det]   # 16 × (2 reads + 1 write)

That's ``N_fdm × (1 + 16 × 3) × T_det × N²`` cfp16 of memory traffic =
49× the slab volume per fdm. At production geometry T=256 N_fdm=32
N_grid=256 cfp16, that's ~98 GiB / cube. Observed 193 ms → ~510 GB/s,
already near the 616 GB/s peak HBM bandwidth on a 2080 Ti.

This module ships a fused per-fdm kernel that reads each chgroup once
and writes the output once. Memory pattern reduces to
``N_fdm × (16 + 1) × T_det × N²`` cfp16 = 17× slab volume per cube ≈
34 GiB at the same geometry → ~67 ms theoretical (~3× speedup,
including kernel-overhead inefficiencies).

The kernel handles the boundary case ``s + t >= T_stream`` (a chgroup
that shifts off the end of the stream is treated as zero, matching
the bench's Python ``if s + T_det <= T_stream: uv.add_(...)`` guard
cell-wise).

JIT-compiled at first use via ``cupy.RawKernel``, which goes through
NVRTC (CUDA's runtime compiler) — no host gcc, no cpp_extension
ninja-build dance. The compiled cubin is cached by cupy under
``$CUPY_CACHE_DIR`` (default ``~/.cupy/kernel_cache``); re-runs are
near-instant after the first.

Used by ``bench/imager_only_gpu.py`` when ``--combine-impl
fused_cuda`` is passed (the default; pass ``--combine-impl
python_addloop`` for the chunk-6c-follow-up A/B baseline).

NOTE: this module imports ``cupy`` at top level. The dsa110-rt conda
env on h01 ships cupy 13.x; CI without cuda skips both this module
and the corresponding tests.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

_LOG = logging.getLogger(__name__)

# CAVEAT: cupy's NVRTC compile path calls ``setlocale(LC_ALL, "C")``
# inside its toolchain bootstrap, which clobbers Python's default
# ``locale.getpreferredencoding(False)`` from "UTF-8" to ASCII for
# the rest of the process. Python's ``setlocale(LC_ALL, "")``
# afterwards silently no-ops (the locale fastpath cache stays in
# the wrong state), so the only workable defence is for *every*
# file I/O in this codebase to pass ``encoding="utf-8"`` explicitly.
# tools/viz/* + tests/test_search*viz*.py have been audited; new
# code that touches text files in the same process as fused_combine
# must do the same.

# Lazy imports of cupy: this module is only useful on a cuda host.
# We defer the import to the first call so the rest of the package
# remains importable on cuda-less CI.
_cp: Optional[object] = None


def _get_cupy():
    global _cp
    if _cp is not None:
        return _cp
    import cupy as cp  # noqa: WPS433  # intentional lazy import
    _cp = cp
    return cp


# ---------------------------------------------------------------------------
# CUDA source (compiled via NVRTC; no host gcc involvement)
# ---------------------------------------------------------------------------

# Two kernels: one for cfp16 (production target) and one for cfp32
# (numerical-precision audit fallback). PyTorch's complex32 is laid
# out as ``__half2`` (interleaved real/imag fp16); complex64 is
# ``float2`` (interleaved fp32). We reinterpret the raw GPU pointer
# at kernel-call time.
_CUDA_SOURCE_CF16 = r"""
#include <cuda_fp16.h>

extern "C" __global__ void fused_combine_per_fdm_cf16(
    const __half2* __restrict__ streams,
    const int*     __restrict__ shifts,
    __half2*       __restrict__ output,
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    // 64-bit strides: n_chg*t_stream*n_grid_sq overflows int32 at the
    // production owned_dm=7 op-point (n_chg=16, t_stream~1548, N=256:
    // 16 * 1548 * 65536 = 3.24e9 > INT_MAX). The MMU faults observed
    // on n13/n09 (owned_dm 5..7) were the canonical symptom; lower
    // owned_dm cases worked because the product stayed under 2.1e9.
    // NOTE: keep the CUDA source strictly ASCII -- NVRTC + the CuPy
    // compile-cache write the source as bytes via the C-locale and
    // will UnicodeEncodeError on em-dash / mu / superscript chars.
    const long long chg_stride = (long long)t_stream * n_grid_sq;

    // Plan section 3.6.3 sign convention: at output cube-time t,
    // chgroup g contributes streams[g, t - shifts[g]]. Out-of-range
    // reads (negative or >= t_stream) yield a zero contribution,
    // matching combine_chgroups zero-fill on cube-time samples not
    // yet present in the stream. shifts[g] is non-negative
    // (plan section 3.6.3 invariant).
    __half2 acc = __floats2half2_rn(0.0f, 0.0f);
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * n_grid_sq
                                 + spatial_offset;
            const __half2 val = streams[base];
            acc = __hadd2(acc, val);
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = acc;
}
"""

_CUDA_SOURCE_CF32 = r"""
extern "C" __global__ void fused_combine_per_fdm_cf32(
    const float2* __restrict__ streams,
    const int*    __restrict__ shifts,
    float2*       __restrict__ output,
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long chg_stride = (long long)t_stream * n_grid_sq;

    // Plan section 3.6.3 sign convention:
    //   out[t] = sum_g streams[g, t - shifts[g]].
    float2 acc = make_float2(0.0f, 0.0f);
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * n_grid_sq
                                 + spatial_offset;
            const float2 val = streams[base];
            acc.x += val.x;
            acc.y += val.y;
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = acc;
}
"""


# Fused dequant + per-fdm combine: reads cint8 streams directly, no
# intermediate cfp16 buffer. Streams layout matches the M3 wire
# payload + bench staging: ``[N_chgroup, T_stream, 2, N_grid, N_grid]``
# int8, re plane at axis_2=0 + im plane at axis_2=1 (split, NOT
# interleaved). Per-block scale/offset are not yet baked in here —
# the bench currently uses scale=1, offset=0 (random-fill
# pessimistic case); production wiring will pass per-chgroup scale +
# offset float arrays for the cast.
#
# Memory traffic at production T=256 N_fdm=32 N_grid=256:
#   reads:  32 fdm × 16 chg × T × N² × 2 B = 17.2 GiB cint8 (half of
#           the cfp16 fused-combine kernel)
#   writes: 32 fdm × T × N² × 4 B = 2 GiB cfp16
#   total:  19.2 GiB / cube → ~38 ms theoretical at 510 GB/s
# vs the current scatter (35.6 ms, 4-pass dequant) + fused_combine
# (72.1 ms, cfp16 reads) = 107.7 ms. Projected ~3× win on the
# combined dequant+combine path; lands T_det=256 inside the 8 cubes/s
# plan §8 budget.
#
# Numerical: int32 accumulation across all 16 chgroups is exact (max
# ±16 × 127 = ±2032 fits comfortably). The single fp16 cast at the
# end carries ≤1 fp16 ULP (≈ 2 at magnitudes near 2k). The
# python_addloop reference path dequantises EACH chgroup to cfp16
# then accumulates as cfp16, taking ~16 ULP of fp16 reduction error;
# this fused kernel is therefore strictly more accurate.
_CUDA_SOURCE_CF16_DEQUANT = r"""
#include <cuda_fp16.h>

extern "C" __global__ void fused_dequant_combine_per_fdm_cint8_to_cf16(
    const signed char* __restrict__ streams,  // [N_chg, T, 2, N, N] int8
    const int*         __restrict__ shifts,   // [N_chg]
    __half2*           __restrict__ output,   // [T_det, N, N] cfp16
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    // 64-bit strides -- see CF16/CF32 kernels above for the rationale.
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    // Plan section 3.6.3 sign convention:
    //   out[t] = sum_g streams[g, t - shifts[g]].
    int acc_re = 0;
    int acc_im = 0;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            acc_re += (int)streams[base];                       // re plane
            acc_im += (int)streams[base + n_grid_sq];           // im plane
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = __floats2half2_rn(
        (float)acc_re, (float)acc_im
    );
}
"""

_CUDA_SOURCE_CF32_DEQUANT = r"""
extern "C" __global__ void fused_dequant_combine_per_fdm_cint8_to_cf32(
    const signed char* __restrict__ streams,
    const int*         __restrict__ shifts,
    float2*            __restrict__ output,
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    // Plan section 3.6.3 sign convention:
    //   out[t] = sum_g streams[g, t - shifts[g]].
    int acc_re = 0;
    int acc_im = 0;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            acc_re += (int)streams[base];
            acc_im += (int)streams[base + n_grid_sq];
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = make_float2(
        (float)acc_re, (float)acc_im
    );
}
"""


# Per-chgroup (scale, offset) calibration variants of the fused
# dequant+combine kernel (chunk-8(c)). Production M3 emits cint8
# streams alongside per-chgroup ``scale[g]`` (real fp32) +
# ``offset[g]`` (complex fp32) calibration metadata that compensates
# antenna-gain spread across the 16 chgroups; the int32-accumulation
# fast path above implicitly assumes a uniform unit scale + zero
# offset, which is "pessimistic" (slightly distorts the cross-chgroup
# magnitude balance the coherent dedispersion sum depends on).
#
# The dequant model is:
#   z[g, t, u, v] = scale[g] * cint8[g, t, u, v] + offset[g]
# where scale[g] is real-valued and offset[g] is a complex DC term
# (typically ~0 for visibilities post-static-sky-subtract; non-trivial
# for autocorrelation-with-DC chgroups). Both apply only to in-range
# stream cells (``0 <= t - shifts[g] < T_stream``); out-of-range
# samples zero-fill cell-wise per the §3.6.3 boundary, and crucially
# the offset is NOT added for out-of-range samples (only in-range
# chgroups contribute their DC).
#
# Accumulation switches to fp32 because per-chgroup multiplication
# kills the int-exact property; fp32 has 23-bit mantissa which is
# >> log2(16 chgroups × 127 max × scale) ≈ 11 bits of effective
# precision, so reduction error is sub-ULP on the final fp16 cast.
# The fma-per-cell cost is essentially free vs the int add (modern
# GPUs do mul-add in one cycle), so the calib variant runs at
# ~the same speed as the int variant.
_CUDA_SOURCE_CF16_DEQUANT_CALIB = r"""
#include <cuda_fp16.h>

extern "C" __global__ void fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf16(
    const signed char* __restrict__ streams,    // [N_chg, T, 2, N, N] int8
    const int*         __restrict__ shifts,     // [N_chg] int32
    const float*       __restrict__ scales,     // [N_chg] fp32
    const float*       __restrict__ offset_re,  // [N_chg] fp32
    const float*       __restrict__ offset_im,  // [N_chg] fp32
    __half2*           __restrict__ output,     // [T_det, N, N] cfp16
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    // Plan section 3.6.3 sign convention:
    //   out[t] = sum_g (scale[g] * cint8[g, t - shifts[g]] + offset[g])
    // restricted to in-range chgroups (out-of-range contribute 0).
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            const float c_re = (float)streams[base];
            const float c_im = (float)streams[base + n_grid_sq];
            const float sc   = scales[g];
            acc_re = fmaf(sc, c_re, acc_re) + offset_re[g];
            acc_im = fmaf(sc, c_im, acc_im) + offset_im[g];
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = __floats2half2_rn(acc_re, acc_im);
}
"""

_CUDA_SOURCE_CF32_DEQUANT_CALIB = r"""
extern "C" __global__ void fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf32(
    const signed char* __restrict__ streams,
    const int*         __restrict__ shifts,
    const float*       __restrict__ scales,
    const float*       __restrict__ offset_re,
    const float*       __restrict__ offset_im,
    float2*            __restrict__ output,
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    // Plan section 3.6.3 sign convention:
    //   out[t] = sum_g (scale[g] * cint8[g, t - shifts[g]] + offset[g])
    // restricted to in-range chgroups (out-of-range contribute 0).
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            const float c_re = (float)streams[base];
            const float c_im = (float)streams[base + n_grid_sq];
            const float sc   = scales[g];
            acc_re = fmaf(sc, c_re, acc_re) + offset_re[g];
            acc_im = fmaf(sc, c_im, acc_im) + offset_im[g];
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = make_float2(acc_re, acc_im);
}
"""


# M7.4 — per-(chgroup, t_src) scale + offset variant
# ---------------------------------------------------
# The cube assembler (rx_ring_assemble_dense_block) emits a per-(corr, t)
# scale/offset sidecar — see ProductionRxRingSource._assemble_cube. The
# wire side computes scale per (cube, dm, t_idx) over the FILLED cells
# only (tx.py::_compute_scale_offset), so the per-cube dynamic range
# varies by ~1-2 bits across the t_det window. Compressing all of that
# into a single per-chgroup scalar (the legacy ``[N_chg]`` calib path)
# would either truncate dynamic range or force a wasteful re-quantise
# on the search side. The per-t kernel below dequantises with the
# native per-(corr, t_src) scale at zero extra cost (the lookup just
# adds a multiply by ``t_src`` to the existing index math).
#
# Kernel signature differs ONLY in the indexing of scales/offsets:
#   per-chg: scales[g]
#   per-t  : scales[g * t_stream + t_src]
# Layout for the per-t arrays is row-major ``[N_chg, T_stream]`` f32,
# matching how the assembler writes them (see recv_ring.c::
# rx_ring_assemble_dense_block).
_CUDA_SOURCE_CF16_DEQUANT_CALIB_PER_T = r"""
#include <cuda_fp16.h>

extern "C" __global__ void fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf16(
    const signed char* __restrict__ streams,    // [N_chg, T, 2, N, N] int8
    const int*         __restrict__ shifts,     // [N_chg] int32
    const float*       __restrict__ scales,     // [N_chg, T] fp32 -- per-(g, t_src)
    const float*       __restrict__ offset_re,  // [N_chg, T] fp32
    const float*       __restrict__ offset_im,  // [N_chg, T] fp32
    __half2*           __restrict__ output,     // [T_det, N, N] cfp16
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    // out[t] = sum_g (scale[g, t_src] * cint8[g, t_src] + offset[g, t_src]).
    // scale==0 from the assembler signals "skip this (g, t_src) slot"
    // (zerofill, pattern-mismatch, or overrun); the multiply zeros the
    // contribution and we still add 0 to the accumulator.
    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            const float c_re = (float)streams[base];
            const float c_im = (float)streams[base + n_grid_sq];
            const int   gt   = g * t_stream + t_src;
            const float sc   = scales[gt];
            acc_re = fmaf(sc, c_re, acc_re) + offset_re[gt];
            acc_im = fmaf(sc, c_im, acc_im) + offset_im[gt];
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = __floats2half2_rn(acc_re, acc_im);
}
"""

_CUDA_SOURCE_CF32_DEQUANT_CALIB_PER_T = r"""
extern "C" __global__ void fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf32(
    const signed char* __restrict__ streams,
    const int*         __restrict__ shifts,
    const float*       __restrict__ scales,     // [N_chg, T]
    const float*       __restrict__ offset_re,  // [N_chg, T]
    const float*       __restrict__ offset_im,  // [N_chg, T]
    float2*            __restrict__ output,
    int n_chgroup, int t_stream, int t_det, int n_grid)
{
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (v >= n_grid || u >= n_grid || t >= t_det) return;

    const int n_grid_sq      = n_grid * n_grid;
    const int spatial_offset = u * n_grid + v;
    const long long t_stride   = (long long)2 * n_grid_sq;
    const long long chg_stride = (long long)t_stream * t_stride;

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = t - s;
        if (t_src >= 0 && t_src < t_stream) {
            const long long base = (long long)g * chg_stride
                                 + (long long)t_src * t_stride
                                 + spatial_offset;
            const float c_re = (float)streams[base];
            const float c_im = (float)streams[base + n_grid_sq];
            const int   gt   = g * t_stream + t_src;
            const float sc   = scales[gt];
            acc_re = fmaf(sc, c_re, acc_re) + offset_re[gt];
            acc_im = fmaf(sc, c_im, acc_im) + offset_im[gt];
        }
    }
    output[(long long)t * n_grid_sq + spatial_offset] = make_float2(acc_re, acc_im);
}
"""


# Cache the compiled cupy.RawKernel objects (one per dtype × variant).
# Each RawKernel call compiles the kernel source via NVRTC; the
# compiled cubin is then cached on disk by cupy.
_KERNEL_CF16: Optional[object] = None
_KERNEL_CF32: Optional[object] = None
_KERNEL_CF16_DEQUANT: Optional[object] = None
_KERNEL_CF32_DEQUANT: Optional[object] = None
_KERNEL_CF16_DEQUANT_CALIB: Optional[object] = None
_KERNEL_CF32_DEQUANT_CALIB: Optional[object] = None
_KERNEL_CF16_DEQUANT_CALIB_PER_T: Optional[object] = None
_KERNEL_CF32_DEQUANT_CALIB_PER_T: Optional[object] = None


def _get_kernel_cf16():
    global _KERNEL_CF16
    if _KERNEL_CF16 is not None:
        return _KERNEL_CF16
    cp = _get_cupy()
    _LOG.info("compiling fused_combine_per_fdm_cf16 via NVRTC...")
    # NVRTC doesn't accept -O3 (it's the default and the explicit
    # flag is rejected). --use_fast_math enables denormals-to-zero
    # and approximate transcendentals, both safe here (we only do
    # fp16 add).
    _KERNEL_CF16 = cp.RawKernel(
        code=_CUDA_SOURCE_CF16,
        name="fused_combine_per_fdm_cf16",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_combine_per_fdm_cf16 ready")
    return _KERNEL_CF16


def _get_kernel_cf32():
    global _KERNEL_CF32
    if _KERNEL_CF32 is not None:
        return _KERNEL_CF32
    cp = _get_cupy()
    _LOG.info("compiling fused_combine_per_fdm_cf32 via NVRTC...")
    _KERNEL_CF32 = cp.RawKernel(
        code=_CUDA_SOURCE_CF32,
        name="fused_combine_per_fdm_cf32",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_combine_per_fdm_cf32 ready")
    return _KERNEL_CF32


def _get_kernel_cf16_dequant():
    global _KERNEL_CF16_DEQUANT
    if _KERNEL_CF16_DEQUANT is not None:
        return _KERNEL_CF16_DEQUANT
    cp = _get_cupy()
    _LOG.info("compiling fused_dequant_combine_per_fdm_cint8_to_cf16 via NVRTC...")
    _KERNEL_CF16_DEQUANT = cp.RawKernel(
        code=_CUDA_SOURCE_CF16_DEQUANT,
        name="fused_dequant_combine_per_fdm_cint8_to_cf16",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_dequant_combine_per_fdm_cint8_to_cf16 ready")
    return _KERNEL_CF16_DEQUANT


def _get_kernel_cf32_dequant():
    global _KERNEL_CF32_DEQUANT
    if _KERNEL_CF32_DEQUANT is not None:
        return _KERNEL_CF32_DEQUANT
    cp = _get_cupy()
    _LOG.info("compiling fused_dequant_combine_per_fdm_cint8_to_cf32 via NVRTC...")
    _KERNEL_CF32_DEQUANT = cp.RawKernel(
        code=_CUDA_SOURCE_CF32_DEQUANT,
        name="fused_dequant_combine_per_fdm_cint8_to_cf32",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_dequant_combine_per_fdm_cint8_to_cf32 ready")
    return _KERNEL_CF32_DEQUANT


def _get_kernel_cf16_dequant_calib():
    global _KERNEL_CF16_DEQUANT_CALIB
    if _KERNEL_CF16_DEQUANT_CALIB is not None:
        return _KERNEL_CF16_DEQUANT_CALIB
    cp = _get_cupy()
    _LOG.info(
        "compiling fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf16 via NVRTC..."
    )
    _KERNEL_CF16_DEQUANT_CALIB = cp.RawKernel(
        code=_CUDA_SOURCE_CF16_DEQUANT_CALIB,
        name="fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf16",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf16 ready")
    return _KERNEL_CF16_DEQUANT_CALIB


def _get_kernel_cf32_dequant_calib():
    global _KERNEL_CF32_DEQUANT_CALIB
    if _KERNEL_CF32_DEQUANT_CALIB is not None:
        return _KERNEL_CF32_DEQUANT_CALIB
    cp = _get_cupy()
    _LOG.info(
        "compiling fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf32 via NVRTC..."
    )
    _KERNEL_CF32_DEQUANT_CALIB = cp.RawKernel(
        code=_CUDA_SOURCE_CF32_DEQUANT_CALIB,
        name="fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf32",
        options=("--use_fast_math",),
    )
    _LOG.info("fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf32 ready")
    return _KERNEL_CF32_DEQUANT_CALIB


def _get_kernel_cf16_dequant_calib_per_t():
    global _KERNEL_CF16_DEQUANT_CALIB_PER_T
    if _KERNEL_CF16_DEQUANT_CALIB_PER_T is not None:
        return _KERNEL_CF16_DEQUANT_CALIB_PER_T
    cp = _get_cupy()
    _LOG.info(
        "compiling fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf16 via NVRTC..."
    )
    _KERNEL_CF16_DEQUANT_CALIB_PER_T = cp.RawKernel(
        code=_CUDA_SOURCE_CF16_DEQUANT_CALIB_PER_T,
        name="fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf16",
        options=("--use_fast_math",),
    )
    _LOG.info(
        "fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf16 ready"
    )
    return _KERNEL_CF16_DEQUANT_CALIB_PER_T


def _get_kernel_cf32_dequant_calib_per_t():
    global _KERNEL_CF32_DEQUANT_CALIB_PER_T
    if _KERNEL_CF32_DEQUANT_CALIB_PER_T is not None:
        return _KERNEL_CF32_DEQUANT_CALIB_PER_T
    cp = _get_cupy()
    _LOG.info(
        "compiling fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf32 via NVRTC..."
    )
    _KERNEL_CF32_DEQUANT_CALIB_PER_T = cp.RawKernel(
        code=_CUDA_SOURCE_CF32_DEQUANT_CALIB_PER_T,
        name="fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf32",
        options=("--use_fast_math",),
    )
    _LOG.info(
        "fused_dequant_scale_offset_per_t_combine_per_fdm_cint8_to_cf32 ready"
    )
    return _KERNEL_CF32_DEQUANT_CALIB_PER_T


# ---------------------------------------------------------------------------
# Torch-to-cupy zero-copy view
# ---------------------------------------------------------------------------


def _torch_to_cupy_view(t: torch.Tensor) -> object:
    """Return a zero-copy cupy ndarray view of a cuda torch tensor.

    cupy 13.x's complex32 dtype is experimental and not always
    binary-compatible with torch.complex32 across cupy/torch
    versions; we sidestep by reinterpreting the underlying memory
    as ``uint32`` (cfp16 = 32 bits/element) or ``uint64`` (cfp32 =
    64 bits/element) — same bytes, just an opaque type to cupy. The
    kernel re-casts the pointer to ``__half2`` / ``float2`` inside.

    Args:
        t: cuda torch tensor, contiguous.

    Returns:
        cupy.ndarray viewing the same GPU memory.
    """
    cp = _get_cupy()
    if not t.is_cuda:
        raise ValueError(f"expected cuda tensor; got device={t.device}")
    if not t.is_contiguous():
        raise ValueError("tensor must be contiguous")

    n_bytes = t.numel() * t.element_size()
    # Reinterpret the dtype to something cupy is happy about.
    if t.dtype == torch.complex32 or t.dtype == torch.float16:
        cp_dtype = cp.uint32
        elem_size = 4 if t.dtype == torch.complex32 else 2
    elif t.dtype == torch.complex64 or t.dtype == torch.float32:
        cp_dtype = cp.uint64 if t.dtype == torch.complex64 else cp.uint32
        elem_size = 8 if t.dtype == torch.complex64 else 4
    elif t.dtype == torch.int32:
        cp_dtype = cp.int32
        elem_size = 4
    elif t.dtype == torch.int8:
        cp_dtype = cp.int8
        elem_size = 1
    else:
        raise ValueError(f"unsupported torch dtype for cupy view: {t.dtype}")

    n_elems = n_bytes // elem_size
    mem = cp.cuda.UnownedMemory(t.data_ptr(), n_bytes, owner=t, device_id=t.device.index)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape=(n_elems,), dtype=cp_dtype, memptr=memptr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_combine_per_fdm(
    streams: torch.Tensor,
    shifts: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run the fused per-fdm combine kernel in place into ``output``.

    Args:
        streams: ``[N_chgroup, T_stream, N_grid, N_grid]`` ``complex32``
            or ``complex64``, cuda, contiguous.
        shifts: ``[N_chgroup]`` ``int32``, cuda. Per-chgroup time shift
            for the (single) fdm trial this call produces.
        output: ``[T_det, N_grid, N_grid]`` same dtype as ``streams``,
            cuda, contiguous. Overwritten with the combined uv slab.

    §3.6.3 sign convention (matches ``fine_dm/combiner.py::combine_chgroups``):

        ``output[t] = sum_g streams[g, t - shifts[g]]``

    Equivalent Python add-loop reference:

        ``output.zero_()
          for g in range(N_chgroup):
              s = int(shifts[g])
              t_in_lo = max(0, s)
              t_in_hi = min(T_det, T_stream + s)
              if t_in_hi > t_in_lo:
                  output[t_in_lo:t_in_hi] += streams[g, t_in_lo - s : t_in_hi - s]``

    Boundary: chgroups whose ``t - shift`` index is negative or past
    ``T_stream`` zero-fill cell-wise (no exception). Combined with the
    §3.6.3 invariant ``shifts[g] >= 0``, this means chgroup g's
    contribution clips to ``output[shifts[g] : T_det]``; cube-time
    samples in ``[0, shifts[g])`` get a zero contribution from that
    chgroup. The chunk-2 cube-validity gate is responsible for
    rejecting cubes whose receive-ring slots haven't filled enough.
    """
    if not (streams.is_cuda and shifts.is_cuda and output.is_cuda):
        raise RuntimeError("fused_combine_per_fdm: all tensors must be cuda")
    if streams.dtype != output.dtype:
        raise RuntimeError(
            f"streams ({streams.dtype}) and output ({output.dtype}) "
            "dtype must match"
        )
    if shifts.dtype != torch.int32:
        raise RuntimeError(f"shifts must be int32; got {shifts.dtype}")
    if streams.dim() != 4 or output.dim() != 3:
        raise RuntimeError(
            f"expected streams.dim()=4, output.dim()=3; got "
            f"{streams.dim()=} {output.dim()=}"
        )

    n_chgroup, t_stream, n_grid, n_grid_y = streams.shape
    if n_grid_y != n_grid:
        raise RuntimeError("streams must be square in u/v")
    if shifts.shape != (n_chgroup,):
        raise RuntimeError(
            f"shifts shape mismatch: got {tuple(shifts.shape)}, "
            f"expected ({n_chgroup},)"
        )
    t_det, n_out_x, n_out_y = output.shape
    if n_out_x != n_grid or n_out_y != n_grid:
        raise RuntimeError("output spatial dims must match streams")

    cp = _get_cupy()
    streams_cp = _torch_to_cupy_view(streams)
    shifts_cp = _torch_to_cupy_view(shifts)
    output_cp = _torch_to_cupy_view(output)

    if streams.dtype == torch.complex32:
        kernel = _get_kernel_cf16()
    elif streams.dtype == torch.complex64:
        kernel = _get_kernel_cf32()
    else:
        raise RuntimeError(f"unsupported dtype: {streams.dtype}")

    # Block (32, 4, 8) = 1024 threads max per 2080 Ti SM (Turing).
    # The 32-thread x-axis maps one warp's load to a 128 B coalesced
    # transaction (32 × 4 B per cfp16 element).
    block: Tuple[int, int, int] = (32, 4, 8)
    grid: Tuple[int, int, int] = (
        (n_grid + block[0] - 1) // block[0],
        (n_grid + block[1] - 1) // block[1],
        (t_det  + block[2] - 1) // block[2],
    )

    # Use the active torch CUDA stream so the kernel synchronises
    # with the surrounding pytorch ops naturally.
    torch_stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(torch_stream.cuda_stream):
        kernel(
            grid, block,
            (streams_cp, shifts_cp, output_cp,
             int(n_chgroup), int(t_stream), int(t_det), int(n_grid)),
        )


def fused_dequant_combine_per_fdm(
    streams_cint8: torch.Tensor,
    shifts: torch.Tensor,
    output: torch.Tensor,
    *,
    scales: Optional[torch.Tensor] = None,
    offsets_re: Optional[torch.Tensor] = None,
    offsets_im: Optional[torch.Tensor] = None,
) -> None:
    """Run the fused cint8-input dequant+combine kernel into ``output``.

    Two dispatch modes:

    1. **Unit-scale fast path** (default; ``scales is None`` and both
       ``offsets_re``/``offsets_im`` are ``None``): single-pass NVRTC
       kernel reads cint8 streams directly, accumulates per-fdm across
       N_chgroup in int32 registers (exact for N_chg ≤ 16, max ±2032),
       and writes one output cfp16 / cfp32 cell. Used by every existing
       caller (bench / pre-chunk-8c CubePipeline) where the global
       ``quantise_streams_global_cint8`` scale lives downstream of the
       imager (Layer-1 σ-clip normalises it out cell-wise).

    2. **Per-chgroup calibrated path** (chunk-8(c); any of ``scales``
       / ``offsets_re`` / ``offsets_im`` provided): kernel applies
       ``scale[g] * cint8[g] + offset[g]`` per in-range chgroup with
       fp32 accumulation, matching the production M3 quantizer
       ``z[g] = scale[g] * cint8[g] + offset[g]`` so the imager output
       is in physical visibility units. Out-of-range chgroups (cube-
       time ``t < shifts[g]`` or past ``T_stream``) zero-fill including
       the offset (only in-range chgroups contribute their DC). The
       fp32 mul-add is essentially free vs the int add (modern GPUs
       fuse it into one cycle), so the calib path is bandwidth-bound
       at the same rate as the unit-scale path.

       When all three of ``scales``, ``offsets_re``, ``offsets_im`` are
       provided as scalars-on-host (length-N_chgroup numpy / Python
       sequences), the wrapper materialises them as cuda fp32 tensors
       and forwards. To enable persistent caching across cubes, pass
       pre-allocated cuda fp32 tensors directly.

    Args:
        streams_cint8: ``[N_chgroup, T_stream, 2, N_grid, N_grid]``
            ``int8``, cuda, contiguous. axis_2 is re/im split (re
            plane first, then im plane). Matches the M3 wire-payload
            staging in ``bench.imager_only_gpu._build_synthetic_streams``.
        shifts: ``[N_chgroup]`` ``int32``, cuda. Per-chgroup time shift
            for the (single) fdm trial this call produces.
        output: ``[T_det, N_grid, N_grid]`` ``complex32`` or
            ``complex64``, cuda, contiguous. Overwritten with the
            combined uv slab.
        scales: optional ``[N_chgroup]`` ``float32`` cuda tensor. Per-
            chgroup multiplicative scale. ``None`` → unit-scale fast
            path. ``M3 → M5`` will populate this from the per-chgroup
            quantiser metadata; bench fallbacks pass
            ``1 / quantise_global_scale`` broadcast over all chgroups
            to put the imager output back into physical units.
        offsets_re: optional ``[N_chgroup]`` ``float32`` cuda tensor.
            Per-chgroup real-part DC offset. ``None`` defaults to
            zeros. Typically ~0 for visibilities post-static-sky-
            subtract; non-trivial for autocorrelation chgroups.
        offsets_im: optional ``[N_chgroup]`` ``float32`` cuda tensor.
            Per-chgroup imag-part DC offset. ``None`` defaults to zeros.

    §3.6.3 sign convention (matches ``fine_dm/combiner.py::combine_chgroups``):

        unit-scale: ``output[t] = sum_g dequant(streams_cint8[g, t - shifts[g]])``
        calib    : ``output[t] = sum_g (scale[g] * dequant(streams_cint8[g,
                                  t - shifts[g]]) + offset[g])``

    Equivalent Python add-loop reference (calib):

        ``streams_cf = streams_cint8.astype(cf32)   # cint8 → cf32 (exact)
          output.zero_()
          for g in range(N_chgroup):
              s = int(shifts[g])
              t_in_lo = max(0, s)
              t_in_hi = min(T_det, T_stream + s)
              if t_in_hi > t_in_lo:
                  output[t_in_lo:t_in_hi] += (
                      scale[g] * streams_cf[g, t_in_lo - s : t_in_hi - s]
                      + complex(offsets_re[g], offsets_im[g]))``

    cint8-to-cf16/cf32 dequant is exact for the cint8 [-127, +127]
    range since both fp16 and fp32 mantissas have ≥ 7 bits. The
    int32 accumulation in the unit-scale path is also exact
    (max sum ±2032 fits in int16); the calib path's fp32 reduction
    has ≪1 fp32 ULP error across N_chg ≤ 16. Boundary check
    (``0 <= t - shift < T_stream``) is cell-wise inside the kernel:
    cube-time samples with no in-range stream sample for chgroup g
    get a zero contribution from that chgroup (and zero offset DC).
    """
    if not (streams_cint8.is_cuda and shifts.is_cuda and output.is_cuda):
        raise RuntimeError("fused_dequant_combine_per_fdm: all tensors must be cuda")
    if streams_cint8.dtype != torch.int8:
        raise RuntimeError(
            f"streams_cint8 must be int8; got {streams_cint8.dtype}"
        )
    if shifts.dtype != torch.int32:
        raise RuntimeError(f"shifts must be int32; got {shifts.dtype}")
    if streams_cint8.dim() != 5:
        raise RuntimeError(
            "streams_cint8 must be 5-D [N_chg, T_stream, 2, N, N]; "
            f"got dim={streams_cint8.dim()}"
        )
    if output.dim() != 3:
        raise RuntimeError("output must be 3-D [T_det, N, N]")

    n_chgroup, t_stream, two, n_grid, n_grid_y = streams_cint8.shape
    if two != 2:
        raise RuntimeError(
            f"streams_cint8 axis 2 must be 2 (re/im split); got {two}"
        )
    if n_grid_y != n_grid:
        raise RuntimeError("streams_cint8 must be square in u/v")
    if shifts.shape != (n_chgroup,):
        raise RuntimeError(
            f"shifts shape mismatch: got {tuple(shifts.shape)}, "
            f"expected ({n_chgroup},)"
        )
    t_det, n_out_x, n_out_y = output.shape
    if n_out_x != n_grid or n_out_y != n_grid:
        raise RuntimeError("output spatial dims must match streams")

    use_calib = (
        scales is not None or offsets_re is not None or offsets_im is not None
    )

    cp = _get_cupy()
    streams_cp = _torch_to_cupy_view(streams_cint8)
    shifts_cp = _torch_to_cupy_view(shifts)
    output_cp = _torch_to_cupy_view(output)

    block: Tuple[int, int, int] = (32, 4, 8)
    grid: Tuple[int, int, int] = (
        (n_grid + block[0] - 1) // block[0],
        (n_grid + block[1] - 1) // block[1],
        (t_det  + block[2] - 1) // block[2],
    )
    torch_stream = torch.cuda.current_stream()

    if not use_calib:
        if output.dtype == torch.complex32:
            kernel = _get_kernel_cf16_dequant()
        elif output.dtype == torch.complex64:
            kernel = _get_kernel_cf32_dequant()
        else:
            raise RuntimeError(
                f"unsupported output dtype: {output.dtype}; "
                "expected complex32 or complex64"
            )
        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            kernel(
                grid, block,
                (streams_cp, shifts_cp, output_cp,
                 int(n_chgroup), int(t_stream), int(t_det), int(n_grid)),
            )
        return

    # Calibrated path: validate / materialise scales + offsets.
    if scales is None:
        scales = torch.ones((n_chgroup,), dtype=torch.float32, device=output.device)
    if offsets_re is None:
        offsets_re = torch.zeros((n_chgroup,), dtype=torch.float32, device=output.device)
    if offsets_im is None:
        offsets_im = torch.zeros((n_chgroup,), dtype=torch.float32, device=output.device)

    # M7.4: detect per-(chgroup, t_src) calibration mode. We accept both
    # the legacy ``[N_chg]`` per-chgroup shape AND the new
    # ``[N_chg, T_stream]`` per-t shape from the same wrapper so callers
    # don't need to switch kernel entry points; the dispatch happens
    # here based on tensor rank. All three arrays must agree on shape.
    per_t_mode = all(t is not None and t.dim() == 2
                     for t in (scales, offsets_re, offsets_im))
    if any(t.dim() == 2 for t in (scales, offsets_re, offsets_im)) and not per_t_mode:
        raise RuntimeError(
            "fused_dequant_combine_per_fdm: scales/offsets_re/offsets_im "
            "must all be 1-D [N_chg] OR all be 2-D [N_chg, T_stream]; "
            f"got shapes {tuple(scales.shape)}, {tuple(offsets_re.shape)}, "
            f"{tuple(offsets_im.shape)}"
        )

    expected_shape = (n_chgroup, t_stream) if per_t_mode else (n_chgroup,)
    for name, t in (("scales", scales), ("offsets_re", offsets_re),
                    ("offsets_im", offsets_im)):
        if not t.is_cuda:
            raise RuntimeError(f"{name} must be cuda; got device={t.device}")
        if t.dtype != torch.float32:
            raise RuntimeError(f"{name} must be float32; got {t.dtype}")
        if t.shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: got {tuple(t.shape)}, "
                f"expected {expected_shape}"
            )
        if not t.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    if per_t_mode:
        if output.dtype == torch.complex32:
            kernel = _get_kernel_cf16_dequant_calib_per_t()
        elif output.dtype == torch.complex64:
            kernel = _get_kernel_cf32_dequant_calib_per_t()
        else:
            raise RuntimeError(
                f"unsupported output dtype: {output.dtype}; "
                "expected complex32 or complex64"
            )
    else:
        if output.dtype == torch.complex32:
            kernel = _get_kernel_cf16_dequant_calib()
        elif output.dtype == torch.complex64:
            kernel = _get_kernel_cf32_dequant_calib()
        else:
            raise RuntimeError(
                f"unsupported output dtype: {output.dtype}; "
                "expected complex32 or complex64"
            )

    scales_cp = _torch_to_cupy_view(scales)
    offsets_re_cp = _torch_to_cupy_view(offsets_re)
    offsets_im_cp = _torch_to_cupy_view(offsets_im)
    with cp.cuda.ExternalStream(torch_stream.cuda_stream):
        kernel(
            grid, block,
            (streams_cp, shifts_cp,
             scales_cp, offsets_re_cp, offsets_im_cp,
             output_cp,
             int(n_chgroup), int(t_stream), int(t_det), int(n_grid)),
        )


def get_module(verbose: bool = False) -> object:
    """Pre-warm the kernel compile (compatibility shim with the old
    cpp_extension API used by the bench).

    Triggers compilation of the cfp16 kernel (the original chunk-6c
    production target); the other 3 variants are compiled lazily on
    first use. The bench's ``--combine-impl`` selector calls the
    matching ``_get_kernel_*`` getter at startup as well.
    """
    return _get_kernel_cf16()

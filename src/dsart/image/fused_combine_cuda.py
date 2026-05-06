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
    const int chg_stride     = t_stream * n_grid_sq;

    __half2 acc = __floats2half2_rn(0.0f, 0.0f);
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = s + t;
        if (t_src < t_stream) {
            const __half2 val = streams[
                g * chg_stride + t_src * n_grid_sq + spatial_offset
            ];
            acc = __hadd2(acc, val);
        }
    }
    output[t * n_grid_sq + spatial_offset] = acc;
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
    const int chg_stride     = t_stream * n_grid_sq;

    float2 acc = make_float2(0.0f, 0.0f);
    for (int g = 0; g < n_chgroup; ++g) {
        const int s = shifts[g];
        const int t_src = s + t;
        if (t_src < t_stream) {
            const float2 val = streams[
                g * chg_stride + t_src * n_grid_sq + spatial_offset
            ];
            acc.x += val.x;
            acc.y += val.y;
        }
    }
    output[t * n_grid_sq + spatial_offset] = acc;
}
"""

# Cache the compiled cupy.RawKernel objects (one per dtype). Each
# RawKernel call compiles the kernel source via NVRTC; the compiled
# cubin is then cached on disk by cupy.
_KERNEL_CF16: Optional[object] = None
_KERNEL_CF32: Optional[object] = None


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

    Semantic equivalence to the Python add-loop:
        ``output.zero_()
          for g in range(N_chgroup):
              if shifts[g] + T_det <= T_stream:
                  output += streams[g, shifts[g] : shifts[g] + T_det]``

    Boundary: chgroups whose shift would read past T_stream are skipped
    cell-wise (so partial cube tails inherit zero-fill, matching the
    Python guard).
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


def get_module(verbose: bool = False) -> object:
    """Pre-warm the kernel compile (compatibility shim with the old
    cpp_extension API used by the bench).

    Triggers compilation of the cfp16 kernel (the production target);
    the cfp32 kernel is compiled lazily on first use.
    """
    return _get_kernel_cf16()

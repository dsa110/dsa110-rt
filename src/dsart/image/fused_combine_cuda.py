"""src/dsart/image/fused_combine_cuda.py — fused per-fdm combine CUDA
kernel for the M5 imager.

NOTE: requires GCC ≥ 9 (PyTorch 2.x ABI requirement). On h01 the
system gcc is 7.5; we install ``gcc_linux-64`` / ``gxx_linux-64`` via
conda-forge into the ``dsa110-rt`` env (gives gcc 15.2.0) and point
``CC`` / ``CXX`` at the conda compiler before calling load_inline.
``ninja`` is also a build-time dep (``pip install ninja`` in the
conda env). One-time conda env setup:

    conda activate dsa110-rt
    conda install -c conda-forge "gxx_linux-64>=9" "gcc_linux-64>=9"
    pip install ninja


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
34 GiB at the same geometry → ~67 ms theoretical (~3× speedup).

The kernel handles the boundary case ``s + t >= T_stream`` (a chgroup
that shifts off the end of the stream is treated as zero, matching
the bench's Python ``if s + T_det <= T_stream: uv.add_(...)`` guard).

Compiled at first import via ``torch.utils.cpp_extension.load_inline``
(takes ~30 s; cached under ``$TORCH_HOME/cuda``). Subsequent runs
load the compiled .so instantly.

Used by ``bench/imager_only_gpu.py`` when ``--combine-impl
fused_cuda`` is passed (the default; pass ``--combine-impl
python_addloop`` for the A/B-compare baseline).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

_LOG = logging.getLogger(__name__)

# C++ forward-declaration required by the load_inline auto-generated
# main.cpp (which contains the pybind11 binding and references the
# function defined in the .cu file). Without this, the main.cpp
# pybind11 macro can't resolve the symbol.
_CPP_DECL = r"""
#include <torch/extension.h>

void fused_combine_per_fdm(
    torch::Tensor streams,
    torch::Tensor shifts,
    torch::Tensor output);
"""

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------

# All-cfp16 (complex32) variant. PyTorch's complex32 has the same
# memory layout as ``__half2`` (interleaved real/imag fp16), so we
# reinterpret_cast on the data pointer.
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Per-fdm fused combine. One launch produces one [T_det, N, N] cfp16
// uv slab. shifts is a small host-staged [N_chgroup] int32 tensor on
// device; the kernel reads each chgroup's contiguous slice once and
// accumulates into the per-output element directly.
//
// Block layout: each thread produces one (t, u, v) output element.
// We tile (v_pix, u_pix, t) over the grid for coalesced reads/writes
// (the v_pix axis is the innermost, matches cfp16 stride 1).
__global__ void fused_combine_per_fdm_cf16_kernel(
    const __half2* __restrict__ streams,    // [N_chg, T_stream, N, N]
    const int*     __restrict__ shifts,     // [N_chg]
    __half2*       __restrict__ output,     // [T_det, N, N]
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
    // shifts is read by every thread in the warp; the L1 cache makes
    // this essentially free after the first warp loads it.
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

// All-cfp32 (complex64) variant. Mirrors the cf16 kernel structure;
// used only by the A/B sanity check in tests, never on the M5 hot
// path. complex64 is two doubles? — no: complex64 is two fp32. We
// keep the dispatch open in case the operator wants a fp32 fallback
// for numerical-precision audits.
__global__ void fused_combine_per_fdm_cf32_kernel(
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

void fused_combine_per_fdm(
    torch::Tensor streams,  // [N_chgroup, T_stream, N, N] cfp16 OR cfp32
    torch::Tensor shifts,   // [N_chgroup] int32 (cuda)
    torch::Tensor output)   // [T_det, N, N] same dtype as streams
{
    TORCH_CHECK(streams.is_cuda(), "streams must be cuda");
    TORCH_CHECK(shifts.is_cuda(),  "shifts must be cuda");
    TORCH_CHECK(output.is_cuda(),  "output must be cuda");
    TORCH_CHECK(streams.is_contiguous(), "streams must be contiguous");
    TORCH_CHECK(output.is_contiguous(),  "output must be contiguous");
    TORCH_CHECK(shifts.scalar_type() == torch::kInt32, "shifts must be int32");
    TORCH_CHECK(streams.scalar_type() == output.scalar_type(),
                "streams and output dtype must match");

    const int n_chgroup = streams.size(0);
    const int t_stream  = streams.size(1);
    const int n_grid    = streams.size(2);
    TORCH_CHECK(streams.size(3) == n_grid, "streams must be square in u/v");
    const int t_det     = output.size(0);
    TORCH_CHECK(output.size(1) == n_grid && output.size(2) == n_grid,
                "output spatial dims must match streams");
    TORCH_CHECK(shifts.size(0) == n_chgroup,
                "shifts must have N_chgroup entries");

    // Block: (32, 4, 8) = 1024 threads. v=inner so 32 threads per warp
    // map to a contiguous run of v_pix pairs (cfp16 = 4 B/element →
    // 32 × 4 = 128 B = one HBM transaction).
    dim3 block(32, 4, 8);
    dim3 grid(
        (n_grid + block.x - 1) / block.x,
        (n_grid + block.y - 1) / block.y,
        (t_det  + block.z - 1) / block.z
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (streams.scalar_type() == torch::kComplexHalf) {
        fused_combine_per_fdm_cf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half2*>(streams.data_ptr()),
            shifts.data_ptr<int>(),
            reinterpret_cast<__half2*>(output.data_ptr()),
            n_chgroup, t_stream, t_det, n_grid
        );
    } else if (streams.scalar_type() == torch::kComplexFloat) {
        fused_combine_per_fdm_cf32_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(streams.data_ptr()),
            shifts.data_ptr<int>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            n_chgroup, t_stream, t_det, n_grid
        );
    } else {
        TORCH_CHECK(false, "unsupported dtype: ", streams.scalar_type());
    }
}

"""

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_MODULE_CACHE: Optional[object] = None


def _ensure_conda_gcc_on_path() -> None:
    """Set CC/CXX env vars to the conda-forge gcc/g++ if available.

    PyTorch 2.x C++ extensions require GCC ≥ 9; h01's system gcc is
    7.5. The dsa110-rt conda env carries gcc_linux-64 / gxx_linux-64
    (15.x); we point the build at it via env vars. No-op on systems
    where the conda compilers aren't installed (caller falls back to
    whatever ``cc`` / ``c++`` resolve to and may fail with the 7.5
    error — that's recoverable: re-run ``conda install gxx_linux-64``).
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    bindir = Path(conda_prefix) / "bin"
    cc_path = bindir / "x86_64-conda-linux-gnu-gcc"
    cxx_path = bindir / "x86_64-conda-linux-gnu-g++"
    if cc_path.is_file() and cxx_path.is_file():
        os.environ["CC"] = str(cc_path)
        os.environ["CXX"] = str(cxx_path)


def get_module(verbose: bool = False) -> object:
    """Return the compiled CUDA extension. JIT-compile on first call.

    Cached under ``$TORCH_EXTENSIONS_DIR`` (default
    ``~/.cache/torch_extensions/<py_ver>_<cu_ver>``); recompiles only
    when the source changes.

    Raises:
        RuntimeError if cuda is not available, or compile fails.
    """
    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE
    if not torch.cuda.is_available():
        raise RuntimeError("fused_combine_cuda requires cuda")
    _ensure_conda_gcc_on_path()
    from torch.utils.cpp_extension import load_inline
    _LOG.info("compiling fused_combine_cuda extension (this takes ~30 s on first run)...")
    mod = load_inline(
        name="dsart_fused_combine_cuda",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SOURCE,
        functions=["fused_combine_per_fdm"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )
    _LOG.info("fused_combine_cuda extension compiled")
    _MODULE_CACHE = mod
    return mod


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
    mod = get_module()
    mod.fused_combine_per_fdm(streams, shifts, output)

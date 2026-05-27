"""src/dsart/transport/gpu_scatter.py — M7.4.1 GPU-side COO→dense
scatter kernel.

Replaces the dense-CPU scatter + dense-H2D path with a compact COO
H2D + GPU scatter. Memory math on the production geometry
(N_corr=16, T_det=192, n_filled_max=5000, N_grid=256):

    Dense plane bytes  = N_corr * T_stream * 2 * N²
                       = 16 * 270 * 2 * 65536
                       = 565 MiB
    Compact bytes      = N_corr * T_det * n_filled_max * 2
                       = 16 * 192 * 5000 * 2
                       = 29.3 MiB
    Reduction          = 19×

The compact buffer is what ``rx_ring_assemble_compact_block`` writes
on the CPU (one ``memcpy(slot, dest, n_filled_max*2)`` per slot, no
per-cell loop, no per-corr LUT touch). The kernel here scatters the
compact wire bytes into the dense plane on the GPU; the imager kernel
downstream still reads the dense plane (unchanged), preserving
byte-for-byte parity with the M7.4 CPU-scatter baseline.

Wire→dense mapping (mirrors ``rx_ring_assemble_dense_block``):

    For each (corr, t):
        For each cell k in [0, n_filled_per_corr[corr]):
            lin = lut[corr, k]                         # int32, in [0, N²)
            re  = cells_packed[corr, t, 2*k]
            im  = cells_packed[corr, t, 2*k + 1]
            dense[corr, t, 0, lin] = re
            dense[corr, t, 1, lin] = im

The kernel writes ONLY rows [0, t_det) of the dense plane (matching the
M7.4 CPU path) and leaves the lookahead rows [t_det, T_stream) at
whatever value they had before — the M7.4 CPU path's contract is that
the caller has already cleared those rows OR is using zero-init. This
module exposes a ``zero_dense_rows()`` helper so callers can zero the
[0, t_det) rows before launching the scatter (avoids carry-over from
the previous cube).

NVRTC compile cost: ~0.6 s first call, then cached under
``$CUPY_CACHE_DIR`` (default ``~/.cupy/kernel_cache``). Same pattern as
``fused_combine_cuda.py``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

_LOG = logging.getLogger(__name__)


# Lazy cupy import — keeps the module loadable on cuda-less CI.
_cp: Optional[object] = None


def _get_cupy():
    global _cp
    if _cp is not None:
        return _cp
    import cupy as cp  # noqa: WPS433  # intentional lazy import
    _cp = cp
    return cp


# ---------------------------------------------------------------------------
# CUDA source (compiled via NVRTC).
# ---------------------------------------------------------------------------

# Block layout:
#   blockIdx.z = corr (in [0, N_corr))
#   blockIdx.y = t    (in [0, T_det))
#   blockIdx.x * blockDim.x + threadIdx.x = k (cell index, in [0, n_filled_max))
#
# Per-thread guards:
#   - k < n_filled_per_corr[corr]    (skip unused tail of the slot)
#   - 0 <= lut[corr, k] < N_grid²    (defensive bound, mirrors recv_ring.c)
#
# Memory math per kernel:
#   reads:  cells_packed (compact)  ≤ 30 MiB
#           lut          (i32)      ~ 320 KiB
#           n_filled     (i32)      64 B
#   writes: dense_plane (sparse)    ≤ 12 MiB (2 bytes/cell × 6M cells)
# Total traffic ≪ memset of the 565 MiB dense plane (≥ 1.13 GB).
_CUDA_SOURCE = r"""
extern "C" __global__ void scatter_compact_to_dense_int8(
    const signed char* __restrict__ cells_packed,  /* [N_corr, T_det, n_filled_max*2] */
    const int*         __restrict__ lut,           /* [N_corr, lut_stride] */
    const int*         __restrict__ n_filled,      /* [N_corr] */
    signed char*       __restrict__ dense,         /* [N_corr, T_stream, 2, N_grid*N_grid] */
    int n_corr,
    int t_det,
    int t_stream,
    int n_grid,
    int n_filled_max,
    int lut_stride
) {
    const int corr = blockIdx.z;
    const int t    = blockIdx.y;
    const int k    = blockIdx.x * blockDim.x + threadIdx.x;
    if (corr >= n_corr || t >= t_det || k >= n_filled_max) return;

    const int nf = n_filled[corr];
    if (nf <= 0)        return;  /* silent corr (-1) OR no cells */
    if (k >= nf)        return;  /* tail: producer's wire-padding zeros */

    const int lin = lut[corr * lut_stride + k];
    const int n_grid_sq = n_grid * n_grid;
    if ((unsigned)lin >= (unsigned)n_grid_sq) return;  /* defensive */

    /* Compact source: row stride = n_filled_max * 2, interleaved (re, im). */
    const long long src_base = ((long long)corr * t_det + t)
                               * ((long long)n_filled_max * 2);
    const signed char re = cells_packed[src_base + (long long)k * 2 + 0];
    const signed char im = cells_packed[src_base + (long long)k * 2 + 1];

    /* Dense destination: corr_stride = t_stream * 2 * n_grid * n_grid */
    const long long corr_stride = (long long)t_stream * 2 * n_grid_sq;
    const long long t_stride    = (long long)2 * n_grid_sq;
    const long long base = (long long)corr * corr_stride
                         + (long long)t * t_stride;

    /* Plane 0 = re, plane 1 = im (matches recv_ring.c split-plane layout). */
    dense[base + 0LL * n_grid_sq + lin] = re;
    dense[base + 1LL * n_grid_sq + lin] = im;
}
"""

# Zero kernel for rows [0, t_det) of the dense plane. We could call
# ``cudaMemsetAsync`` instead, but using a kernel keeps the launch
# stream-attached and avoids a synchronous fallback in cupy.
# Cost: ~0.6 ms for 384 MiB at ~650 GB/s (RTX 2080 Ti).
_CUDA_SOURCE_ZERO = r"""
extern "C" __global__ void zero_dense_t_rows_int8(
    signed char* __restrict__ dense,  /* [N_corr, T_stream, 2, N_grid*N_grid] */
    int n_corr,
    int t_det,
    int t_stream,
    int n_grid
) {
    /* One thread per (corr, t, xy) cell. ~6.3M threads total for the
     * production geometry. Bandwidth-bound at ~600 GB/s on the 2080 Ti
     * memory subsystem -- ~0.6 ms for the 384 MiB we touch. */
    const int corr = blockIdx.z;
    const int t    = blockIdx.y;
    const int xy   = blockIdx.x * blockDim.x + threadIdx.x;

    const int n_grid_sq = n_grid * n_grid;
    if (corr >= n_corr || t >= t_det || xy >= n_grid_sq) return;

    const long long corr_stride = (long long)t_stream * 2 * n_grid_sq;
    const long long t_stride    = (long long)2 * n_grid_sq;
    const long long base = (long long)corr * corr_stride
                         + (long long)t * t_stride;
    dense[base + 0LL * n_grid_sq + xy] = 0;
    dense[base + 1LL * n_grid_sq + xy] = 0;
}
"""


# ---------------------------------------------------------------------------
# Compiled kernel cache.
# ---------------------------------------------------------------------------

_KERNEL_SCATTER = None
_KERNEL_ZERO    = None


def _get_kernel_scatter():
    global _KERNEL_SCATTER
    if _KERNEL_SCATTER is not None:
        return _KERNEL_SCATTER
    cp = _get_cupy()
    _LOG.info("compiling scatter_compact_to_dense_int8 via NVRTC...")
    _KERNEL_SCATTER = cp.RawKernel(
        code=_CUDA_SOURCE,
        name="scatter_compact_to_dense_int8",
        options=("--use_fast_math",),
    )
    _LOG.info("scatter_compact_to_dense_int8 ready")
    return _KERNEL_SCATTER


def _get_kernel_zero():
    global _KERNEL_ZERO
    if _KERNEL_ZERO is not None:
        return _KERNEL_ZERO
    cp = _get_cupy()
    _LOG.info("compiling zero_dense_t_rows_int8 via NVRTC...")
    _KERNEL_ZERO = cp.RawKernel(
        code=_CUDA_SOURCE_ZERO,
        name="zero_dense_t_rows_int8",
        options=("--use_fast_math",),
    )
    _LOG.info("zero_dense_t_rows_int8 ready")
    return _KERNEL_ZERO


# ---------------------------------------------------------------------------
# Public Python entry points.
# ---------------------------------------------------------------------------

# Block size for the scatter kernel along the k (cell) axis. Tuning
# notes: 128 is a safe default. The kernel is bandwidth-bound on the
# compact reads + sparse dense writes; the per-thread work is two byte
# loads + one indexed pair of byte stores. 256 also works but produces
# a few more launch-tail threads that get masked off.
_SCATTER_BLOCK = 128

# Block size for the zero kernel along the xy axis. Higher is OK since
# the kernel is 1 byte/thread bandwidth-bound and we want full 1024-thread
# warps per CTA.
_ZERO_BLOCK = 256


def _torch_int8_ptr(t: torch.Tensor) -> int:
    if t.dtype != torch.int8:
        raise TypeError(f"expected int8 tensor, got {t.dtype}")
    if not t.is_cuda:
        raise ValueError("expected CUDA tensor")
    if not t.is_contiguous():
        raise ValueError("expected C-contiguous tensor")
    return int(t.data_ptr())


def _torch_int32_ptr(t: torch.Tensor) -> int:
    if t.dtype != torch.int32:
        raise TypeError(f"expected int32 tensor, got {t.dtype}")
    if not t.is_cuda:
        raise ValueError("expected CUDA tensor")
    if not t.is_contiguous():
        raise ValueError("expected C-contiguous tensor")
    return int(t.data_ptr())


def zero_dense_rows(
    *,
    dense: torch.Tensor,    # int8 [N_corr, T_stream, 2, N_grid, N_grid]
    t_det: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Zero rows [0, t_det) of the dense per-(corr, t) split-plane buffer.

    Mirrors the per-cube ``memset(dense_corr, 0, t_det * t_stride)``
    that ``rx_ring_assemble_dense_block`` does on the CPU. Must be
    called BEFORE :func:`scatter_compact_to_dense` so invalid slots
    (those with scale==0) and unused tail cells (k >= n_filled[corr])
    end up at zero, matching the M7.4 CPU-scatter dense plane.

    Args:
        dense: int8 tensor of shape ``(N_corr, T_stream, 2, N_grid, N_grid)``.
            T_stream may be > t_det (lookahead rows); those are NOT touched.
        t_det: detector window length (rows to zero).
        stream: optional CUDA stream. Defaults to the current stream.
    """
    if dense.ndim != 5:
        raise ValueError(
            f"dense must be 5-D (N_corr, T_stream, 2, N_grid, N_grid); "
            f"got {dense.shape}"
        )
    n_corr, t_stream, two, n_grid_a, n_grid_b = dense.shape
    if two != 2 or n_grid_a != n_grid_b:
        raise ValueError(
            f"dense.shape={dense.shape}; expected (..., 2, N_grid, N_grid)"
        )
    if t_det > t_stream:
        raise ValueError(f"t_det={t_det} > t_stream={t_stream}")
    n_grid = int(n_grid_a)

    cp = _get_cupy()
    kernel = _get_kernel_zero()

    n_grid_sq = n_grid * n_grid
    grid = (
        (n_grid_sq + _ZERO_BLOCK - 1) // _ZERO_BLOCK,  # x
        int(t_det),                                    # y
        int(n_corr),                                   # z
    )
    block = (_ZERO_BLOCK, 1, 1)

    args = (
        dense.data_ptr(),
        int(n_corr),
        int(t_det),
        int(t_stream),
        int(n_grid),
    )
    # Launch on the current torch CUDA stream so it serialises with
    # all upstream / downstream work the imager pipeline issues.
    # cupy.cuda.ExternalStream wraps a non-cupy stream pointer; this is
    # the same pattern fused_combine_cuda.py uses.
    if stream is None:
        torch_stream = torch.cuda.current_stream()
    else:
        torch_stream = stream
    with cp.cuda.ExternalStream(torch_stream.cuda_stream):
        kernel(grid, block, args)


def scatter_compact_to_dense(
    *,
    cells_packed: torch.Tensor,   # int8 [N_corr, T_det, n_filled_max*2]
    lut: torch.Tensor,            # int32 [N_corr, lut_stride]
    n_filled_per_corr: torch.Tensor,  # int32 [N_corr]
    dense: torch.Tensor,          # int8 [N_corr, T_stream, 2, N_grid, N_grid]
    t_det: int,
    n_grid: int,
    n_filled_max: int,
    stream: Optional["torch.cuda.Stream"] = None,
) -> None:
    """Scatter compact COO cells into the dense per-(corr, t) plane (GPU).

    The dense plane must already have its [0, t_det) rows zeroed (see
    :func:`zero_dense_rows`). The kernel writes cells from
    ``cells_packed`` into ``dense`` at positions resolved via the
    per-corr LUT, mirroring the CPU path in
    ``recv_ring.c::rx_ring_assemble_dense_block`` byte-for-byte.

    Args:
        cells_packed: int8 ``(N_corr, T_det, n_filled_max * 2)``. Compact
            wire bytes (re, im interleaved). Zeros for invalid slots and
            for cells k >= n_filled[corr].
        lut: int32 ``(N_corr, lut_stride)``. Entry ``[c, k]`` =
            ``ix_row * N_grid + ix_col`` (flat dense-plane index).
        n_filled_per_corr: int32 ``(N_corr,)``. Number of valid cells
            per corr's sparsity pattern. ``-1`` marks a corr silent.
        dense: int8 ``(N_corr, T_stream, 2, N_grid, N_grid)``. Output
            buffer; only rows [0, t_det) are written.
        t_det: detector window length.
        n_grid: grid edge.
        n_filled_max: max cells per slot (wire-side dimension).
        stream: optional CUDA stream. Defaults to current stream.
    """
    if cells_packed.ndim != 3:
        raise ValueError(
            f"cells_packed must be 3-D (N_corr, T_det, n_filled_max*2); "
            f"got {cells_packed.shape}"
        )
    n_corr_c, t_det_c, two_nfm = cells_packed.shape
    if t_det_c != t_det:
        raise ValueError(
            f"cells_packed T_det={t_det_c} != t_det={t_det}"
        )
    if two_nfm != n_filled_max * 2:
        raise ValueError(
            f"cells_packed n_filled_max*2={two_nfm} != "
            f"{n_filled_max * 2}"
        )

    if lut.ndim != 2 or lut.shape[0] != n_corr_c:
        raise ValueError(
            f"lut.shape={lut.shape}; expected ({n_corr_c}, lut_stride)"
        )
    lut_stride = int(lut.shape[1])

    if n_filled_per_corr.shape != (n_corr_c,):
        raise ValueError(
            f"n_filled_per_corr.shape={n_filled_per_corr.shape}; "
            f"expected ({n_corr_c},)"
        )

    if dense.ndim != 5:
        raise ValueError(
            f"dense must be 5-D (N_corr, T_stream, 2, N_grid, N_grid); "
            f"got {dense.shape}"
        )
    n_corr_d, t_stream, two_d, n_grid_a, n_grid_b = dense.shape
    if n_corr_d != n_corr_c:
        raise ValueError(
            f"dense N_corr={n_corr_d} != cells_packed N_corr={n_corr_c}"
        )
    if two_d != 2 or n_grid_a != n_grid_b or n_grid_a != n_grid:
        raise ValueError(
            f"dense.shape={dense.shape}; expected (..., 2, {n_grid}, {n_grid})"
        )
    if t_det > t_stream:
        raise ValueError(f"t_det={t_det} > t_stream={t_stream}")

    cp = _get_cupy()
    kernel = _get_kernel_scatter()

    grid = (
        (int(n_filled_max) + _SCATTER_BLOCK - 1) // _SCATTER_BLOCK,  # k
        int(t_det),                                                   # t
        int(n_corr_c),                                                # corr
    )
    block = (_SCATTER_BLOCK, 1, 1)

    args = (
        _torch_int8_ptr(cells_packed),
        _torch_int32_ptr(lut),
        _torch_int32_ptr(n_filled_per_corr),
        _torch_int8_ptr(dense),
        int(n_corr_c),
        int(t_det),
        int(t_stream),
        int(n_grid),
        int(n_filled_max),
        int(lut_stride),
    )

    if stream is None:
        torch_stream = torch.cuda.current_stream()
    else:
        torch_stream = stream
    with cp.cuda.ExternalStream(torch_stream.cuda_stream):
        kernel(grid, block, args)


__all__ = [
    "scatter_compact_to_dense",
    "zero_dense_rows",
]

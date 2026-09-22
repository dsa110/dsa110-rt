"""src/dsart/image/tiled_combine_cuda.py — time-tiled fine-DM combine.

The problem, precisely
======================

``fused_combine_cuda`` is output-stationary over a dense plane and is
launched once per fine-DM trial. Its read volume at the production
op-point (n_sb=16, t_stream=322, N=256, t_out=192, n_fdm=34) is
**13.80 GB per cube** against a 0.68 GB stream — a 20.4x amplification
that is almost exactly ``n_fdm`` (34, scaled by 0.60 because each
fine-DM only touches 192 of 322 stream rows).

That amplification is *irreducible for an output-stationary kernel*.
Stream element ``(g, t_src, cell)`` is read by output ``(f, t_out)``
with ``t_out = t_src + shift[f][g] - t_lo`` — one reader per fine-DM
trial, each at a **different output time**. A thread that owns a fixed
``t_out`` therefore gets no reuse across ``f``.

Inverting to input-stationary (see :mod:`dsart.image.coo_combine_cuda`)
does capture the reuse, and is bit-identical, but pays 1.14 G atomics
and measured **191 ms vs 57.5 ms** on n09 — a net loss.

What this module does
=====================

Take the middle path: keep the accumulation output-stationary (no
atomics) but make the *reuse* happen in shared memory.

One block owns ``(output cell, tile of B output times)`` and every
fine-DM trial. It stages, for that cell and its conjugate mirror, the
stream window spanning all ``(f, t_out)`` in the tile — width
``W = B + spread`` where ``spread = max(shift) - min(shift)`` — for
every sub-band, then each thread sums over sub-bands out of shared
memory.

Global reads fall to ``cells * (t_out/B) * n_sb * W * 4 B`` =
**1.54 GB at B=24**, a 9.0x reduction, with no atomics.

Time-minor layout is mandatory
------------------------------

Staging is only cheap if a cell's time series is contiguous. In the
existing dense layout ``[n_sb, t_stream, 2, N, N]`` consecutive samples
of one cell are 131072 B apart, so each 1-byte read would pull a 32 B
sector and staging would cost ~32x more than it saves. This module
therefore uses a **time-minor** dense buffer
``[n_sb, 2, N*N, t_rows]`` and ships its own scatter
(:func:`scatter_compact_to_dense_tmin`) to fill it. The scatter's writes
become strided instead of its reads, which is the right trade: the
scatter moves 33 MB, the combine moves gigabytes.

Exactness
---------

Accumulation is in ``int`` exactly as the dense kernel does it, and the
0.5 conjugate fold plus the ``(-1)^(u+w)`` fftshift are applied
identically, so output is bit-identical to
``fused_dequant_combine_per_fdm_half``. Verified on n09 (RTX 2080 Ti,
2026-09-22) via ``bench/coo_combine_equiv.py``: ``torch.equal`` True
against the dense path at the production geometry.

Measured (n09, idle RTX 2080 Ti, n_fdm = 34 held at the production
value, N=256, t_stream=322, t_lo=64), ms/cube including the scatter or
transpose that feeds each:

    n_sb    dense    v1      v2      v3     v3 peak GiB
     16      57.5    45.4    17.6    19.5       0.91
     32     119.2   140.0    27.7    26.1       1.63
     40        -       -       -     31.9       2.31
     64     262.5   264.2    65.0    61.1       2.78

All three are bit-identical to ``fused_dequant_combine_per_fdm_half``
(``torch.equal``; tests/test_combine_equivalence.py, 17 passed).

v1 is kept only for the record: giving ~1000 threads two staging loads
each before a ``__syncthreads`` made it launch/sync-dominated, so it
barely beat the dense path at n_sb=16 and lost above it. **v3 is the one
to use.** It is ~4.3x faster than the dense path at n_sb=64 and needs a
0.21 GB buffer where the dense ping-pong needed 5.03 GiB — which is what
made n_sb=48 OOM on an 11 GiB card.

Consequence for sub-banding, with the fine grid held at 34 trials:
n_sub=4 (n_sb=64) costs 61.1 + 58.7 (fft+mask+detector) + 16.8 (H2D)
= ~137 ms of the 201.327 ms search budget, i.e. 68% on an idle GPU.
Graded sub-banding (n_sb=40) lands at ~101 ms = 50%.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

_cp: Optional[object] = None

#: Default output-time tile. Block is (n_fdm, TILE) threads, so this
#: must satisfy ``n_fdm * TILE <= 1024``.
DEFAULT_TILE: int = 24


def _get_cupy():
    global _cp
    if _cp is not None:
        return _cp
    import cupy as cp  # noqa: WPS433
    _cp = cp
    return cp


_CUDA_SOURCE = r"""
extern "C" __global__ void scatter_compact_to_dense_tmin(
    const signed char* __restrict__ cells_packed, /* [n_sb, t_rows, nfm*2] */
    const int*         __restrict__ lut,          /* [n_sb, lut_stride]    */
    const int*         __restrict__ n_filled,     /* [n_sb]                */
    signed char*       __restrict__ dense,        /* [n_sb, 2, N*N, t_rows]*/
    int n_sb, int t_rows, int n_grid, int n_filled_max, int lut_stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int g = blockIdx.z;
    if (g >= n_sb || t >= t_rows || k >= n_filled_max) return;
    const int nf = n_filled[g];
    if (nf <= 0 || k >= nf) return;

    const int n_grid_sq = n_grid * n_grid;
    const int lin = lut[g * lut_stride + k];
    if ((unsigned)lin >= (unsigned)n_grid_sq) return;

    const long long src = ((long long)g * t_rows + t)
                        * ((long long)n_filled_max * 2);
    const signed char re = cells_packed[src + (long long)k * 2 + 0];
    const signed char im = cells_packed[src + (long long)k * 2 + 1];

    const long long plane = (long long)n_grid_sq * t_rows;
    const long long base  = (long long)g * 2 * plane + (long long)lin * t_rows + t;
    dense[base]         = re;
    dense[base + plane] = im;
}

/* One block = (one half-plane output cell) x (tile of TILE output times).
   blockDim = (n_fdm, TILE). Shared memory stages, for this cell and its
   conjugate mirror, the stream window covering every (f, t_out) in the
   tile, for every sub-band. */
extern "C" __global__ void tiled_combine_per_fdm(
    const signed char* __restrict__ dense,   /* [n_sb, 2, N*N, t_rows] */
    const int*         __restrict__ shifts,  /* [n_fdm, n_sb]          */
    __half2*           __restrict__ out,     /* [n_fdm, t_out_len, N, n_half] */
    int n_sb, int t_rows, int n_grid, int n_half,
    int n_fdm, int t_out_len, int t_lo,
    int shift_max, int win_w, int tile, int fftshift)
{
    extern __shared__ signed char sh[];       /* 4 * n_sb * win_w bytes */

    const int cell_id = blockIdx.x;           /* half-plane cell index */
    const int tile_id = blockIdx.y;
    const int u = cell_id / n_half;
    const int w = cell_id - u * n_half;
    if (u >= n_grid) return;

    const int um = (n_grid - u) % n_grid;
    const int wm = (n_grid - w) % n_grid;
    const int lin  = u  * n_grid + w;
    const int linm = um * n_grid + wm;

    const int t_out0 = tile_id * tile;
    /* window start in stream coordinates */
    const int t_win0 = t_out0 + t_lo - shift_max;

    const long long plane = (long long)n_grid * n_grid * t_rows;
    signed char* s_re  = sh;
    signed char* s_im  = sh + (long long)n_sb * win_w;
    signed char* s_mre = sh + 2LL * n_sb * win_w;
    signed char* s_mim = sh + 3LL * n_sb * win_w;

    const int nthreads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid; i < n_sb * win_w; i += nthreads) {
        const int g  = i / win_w;
        const int jj = i - g * win_w;
        const int ts = t_win0 + jj;
        signed char re = 0, im = 0, mre = 0, mim = 0;
        if (ts >= 0 && ts < t_rows) {
            const long long gb = (long long)g * 2 * plane;
            re  = dense[gb + (long long)lin  * t_rows + ts];
            im  = dense[gb + plane + (long long)lin  * t_rows + ts];
            mre = dense[gb + (long long)linm * t_rows + ts];
            mim = dense[gb + plane + (long long)linm * t_rows + ts];
        }
        s_re[i] = re;  s_im[i] = im;  s_mre[i] = mre;  s_mim[i] = mim;
    }
    __syncthreads();

    const int f  = threadIdx.x;
    const int tl = threadIdx.y;
    if (f >= n_fdm) return;
    const int t_out = t_out0 + tl;
    if (t_out >= t_out_len) return;

    int acc_re = 0, acc_im = 0, accm_re = 0, accm_im = 0;
    for (int g = 0; g < n_sb; ++g) {
        const int ts = t_out + t_lo - shifts[f * n_sb + g];
        if (ts < 0 || ts >= t_rows) continue;
        const int jj = ts - t_win0;               /* always in [0, win_w) */
        const int idx = g * win_w + jj;
        acc_re  += (int)s_re[idx];
        acc_im  += (int)s_im[idx];
        accm_re += (int)s_mre[idx];
        accm_im += (int)s_mim[idx];
    }

    float o_re = 0.5f * (float)(acc_re + accm_re);
    float o_im = 0.5f * (float)(acc_im - accm_im);
    if (fftshift && ((u + w) & 1)) { o_re = -o_re; o_im = -o_im; }
    out[(long long)f * t_out_len * n_grid * n_half
        + (long long)t_out * n_grid * n_half
        + (long long)u * n_half + w] = __floats2half2_rn(o_re, o_im);
}
"""

_MOD = None


def _get_module():
    global _MOD
    if _MOD is None:
        cp = _get_cupy()
        _LOG.info("compiling tiled_combine via NVRTC...")
        _MOD = cp.RawModule(
            code="#include <cuda_fp16.h>\n" + _CUDA_SOURCE,
            backend="nvrtc", options=("--std=c++14",),
        )
        _LOG.info("tiled_combine ready")
    return _MOD


def _as_cupy(t: torch.Tensor):
    cp = _get_cupy()
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))


def scatter_compact_to_dense_tmin(
    cells_packed: torch.Tensor,
    lut: torch.Tensor,
    n_filled: torch.Tensor,
    dense: torch.Tensor,
    *,
    n_grid: int,
) -> None:
    """Scatter the compact COO block into a TIME-MINOR dense buffer.

    ``dense`` must be ``[n_sb, 2, n_grid*n_grid, t_rows]`` int8 and
    already zeroed (the pattern is static, so in steady state only the
    filled cells ever change and a one-time zero suffices — but any
    caller that reuses the buffer across cubes with a changed pattern
    must re-zero).
    """
    n_sb, t_rows, packed_w = cells_packed.shape
    n_filled_max = packed_w // 2
    kern = _get_module().get_function("scatter_compact_to_dense_tmin")
    threads = 128
    kern(
        ((n_filled_max + threads - 1) // threads, t_rows, n_sb),
        (threads, 1, 1),
        (
            _as_cupy(cells_packed), _as_cupy(lut.int()),
            _as_cupy(n_filled.int()), _as_cupy(dense),
            np.int32(n_sb), np.int32(t_rows), np.int32(n_grid),
            np.int32(n_filled_max), np.int32(lut.shape[1]),
        ),
    )


def combine_tiled(
    dense_tmin: torch.Tensor,
    shifts: torch.Tensor,
    *,
    n_grid: int,
    t_out_len: int,
    t_lo: int = 0,
    fftshift: bool = True,
    tile: int = DEFAULT_TILE,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """All fine-DM planes in one launch, reusing the stream in shared mem.

    Args:
        dense_tmin: ``[n_sb, 2, n_grid*n_grid, t_rows]`` int8.
        shifts: ``[n_fdm, n_sb]`` int32.
        n_grid: uv grid edge N.
        t_out_len: number of output rows (``t_det - t_lo``).
        t_lo: first output row.
        fftshift: apply the ``(-1)^(u+w)`` fold.
        tile: output-time tile; ``n_fdm * tile`` must be <= 1024.

    Returns:
        ``[n_fdm, t_out_len, n_grid, n_grid//2+1, 2]`` float16 — the
        same half2 plane layout the dense kernel writes.
    """
    n_sb, two, n_cells, t_rows = dense_tmin.shape
    if two != 2 or n_cells != n_grid * n_grid:
        raise ValueError(
            f"dense_tmin must be [n_sb, 2, n_grid^2, t_rows]; got "
            f"{tuple(dense_tmin.shape)} for n_grid={n_grid}"
        )
    n_fdm = int(shifts.shape[0])
    if shifts.shape[1] != n_sb:
        raise ValueError(
            f"shifts must be [n_fdm, n_sb={n_sb}]; got {tuple(shifts.shape)}"
        )
    if n_fdm * tile > 1024:
        raise ValueError(
            f"n_fdm({n_fdm}) * tile({tile}) = {n_fdm * tile} > 1024 threads; "
            f"lower `tile`"
        )
    n_half = n_grid // 2 + 1
    s_min = int(shifts.min().item())
    s_max = int(shifts.max().item())
    win_w = tile + (s_max - s_min)

    if out is None:
        out = torch.empty(
            (n_fdm, t_out_len, n_grid, n_half, 2),
            dtype=torch.float16, device=dense_tmin.device,
        )

    kern = _get_module().get_function("tiled_combine_per_fdm")
    n_tiles = (t_out_len + tile - 1) // tile
    shmem = 4 * n_sb * win_w
    kern(
        (n_grid * n_half, n_tiles),
        (n_fdm, tile, 1),
        (
            _as_cupy(dense_tmin), _as_cupy(shifts.int()), _as_cupy(out),
            np.int32(n_sb), np.int32(t_rows), np.int32(n_grid),
            np.int32(n_half), np.int32(n_fdm), np.int32(t_out_len),
            np.int32(t_lo), np.int32(s_max), np.int32(win_w),
            np.int32(tile), np.int32(1 if fftshift else 0),
        ),
        shared_mem=shmem,
    )
    return out


__all__ = [
    "DEFAULT_TILE",
    "combine_tiled",
    "scatter_compact_to_dense_tmin",
]


# ---------------------------------------------------------------------------
# v2: one block per uv cell, re/im interleaved, whole time axis staged
# ---------------------------------------------------------------------------
#
# v1 above gives ~1000 threads only two staging loads each before a
# __syncthreads, so it is launch/sync-dominated: measured 44.5 ms at
# n_sb=16 against a ~2.4 ms transaction bound. v2 fixes the amortisation:
#
#                      v1 (tile=30)      v2 (n_sb=16)
#   blocks               231168             33024
#   outputs per block      1020              6528
#   staged bytes          1.42 GB           0.55 GB
#
# and it stores re/im ADJACENT so a sample is one 2-byte load instead of
# two 1-byte loads from two different arrays. Layout is
# ``[n_sb, n_cells, t_pad, 2]`` int8 with ``t_pad`` rounded up to a
# multiple of 8 so each cell's series starts 16-byte aligned.
#
# The time tile is sized to the 48 KB shared-memory limit:
# ``W <= 49152 / (4 * n_sb)``. At n_sb <= 32 that covers the whole output
# axis in one tile; at n_sb=64 it falls to tile=126.

_CUDA_SOURCE_V2 = r"""
extern "C" __global__ void scatter_compact_to_dense_v2(
    const signed char* __restrict__ cells_packed, /* [n_sb, t_rows, nfm*2] */
    const int*         __restrict__ lut,          /* [n_sb, lut_stride]    */
    const int*         __restrict__ n_filled,     /* [n_sb]                */
    signed char*       __restrict__ dense,        /* [n_sb, n_cells, t_pad, 2] */
    int n_sb, int t_rows, int t_pad, int n_grid,
    int n_filled_max, int lut_stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int g = blockIdx.z;
    if (g >= n_sb || t >= t_rows || k >= n_filled_max) return;
    const int nf = n_filled[g];
    if (nf <= 0 || k >= nf) return;

    const int n_cells = n_grid * n_grid;
    const int lin = lut[g * lut_stride + k];
    if ((unsigned)lin >= (unsigned)n_cells) return;

    const long long src = ((long long)g * t_rows + t)
                        * ((long long)n_filled_max * 2);
    const long long dst = (((long long)g * n_cells + lin) * t_pad + t) * 2;
    dense[dst + 0] = cells_packed[src + (long long)k * 2 + 0];
    dense[dst + 1] = cells_packed[src + (long long)k * 2 + 1];
}

extern "C" __global__ void tiled_combine_per_fdm_v2(
    const signed char* __restrict__ dense,  /* [n_sb, n_cells, t_pad, 2] */
    const int*         __restrict__ shifts, /* [n_fdm, n_sb]             */
    __half2*           __restrict__ out,    /* [n_fdm, t_out_len, N, n_half] */
    int n_sb, int t_rows, int t_pad, int n_grid, int n_half,
    int n_fdm, int t_out_len, int t_lo,
    int shift_max, int win_w, int tile, int fftshift)
{
    extern __shared__ short sh[];       /* [2][n_sb][win_w] packed (re,im) */

    const int cell_id = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int u = cell_id / n_half;
    const int w = cell_id - u * n_half;
    if (u >= n_grid) return;

    const int um = (n_grid - u) % n_grid;
    const int wm = (n_grid - w) % n_grid;
    const int n_cells = n_grid * n_grid;
    const int lin  = u  * n_grid + w;
    const int linm = um * n_grid + wm;

    const int t_out0 = tile_id * tile;
    const int t_win0 = t_out0 + t_lo - shift_max;

    const short* src = (const short*)dense;   /* (re,im) as one 2-byte unit */
    const int nthreads = blockDim.x;
    const int tid = threadIdx.x;
    const int span = n_sb * win_w;

    for (int idx = tid; idx < 2 * span; idx += nthreads) {
        const int which = idx / span;
        const int rem   = idx - which * span;
        const int g     = rem / win_w;
        const int j     = rem - g * win_w;
        const int ts    = t_win0 + j;
        short v = 0;
        if (ts >= 0 && ts < t_rows) {
            const int c = which ? linm : lin;
            v = src[((long long)g * n_cells + c) * t_pad + ts];
        }
        sh[idx] = v;
    }
    __syncthreads();

    const int n_out = n_fdm * tile;
    for (int idx = tid; idx < n_out; idx += nthreads) {
        const int f  = idx / tile;
        const int tl = idx - f * tile;
        const int t_out = t_out0 + tl;
        if (t_out >= t_out_len) continue;

        int acc_re = 0, acc_im = 0, accm_re = 0, accm_im = 0;
        for (int g = 0; g < n_sb; ++g) {
            const int ts = t_out + t_lo - shifts[f * n_sb + g];
            if (ts < 0 || ts >= t_rows) continue;
            const int j = ts - t_win0;
            const short a = sh[g * win_w + j];
            const short b = sh[span + g * win_w + j];
            acc_re  += (int)(signed char)(a & 0xFF);
            acc_im  += (int)(signed char)((a >> 8) & 0xFF);
            accm_re += (int)(signed char)(b & 0xFF);
            accm_im += (int)(signed char)((b >> 8) & 0xFF);
        }
        float o_re = 0.5f * (float)(acc_re + accm_re);
        float o_im = 0.5f * (float)(acc_im - accm_im);
        if (fftshift && ((u + w) & 1)) { o_re = -o_re; o_im = -o_im; }
        out[(long long)f * t_out_len * n_grid * n_half
            + (long long)t_out * n_grid * n_half
            + (long long)u * n_half + w] = __floats2half2_rn(o_re, o_im);
    }
}
"""

_MOD2 = None
#: Shared memory a block may use, in bytes (Turing default cap).
SHMEM_CAP: int = 48 * 1024


def _get_module_v2():
    global _MOD2
    if _MOD2 is None:
        cp = _get_cupy()
        _LOG.info("compiling tiled_combine v2 via NVRTC...")
        _MOD2 = cp.RawModule(
            code="#include <cuda_fp16.h>\n" + _CUDA_SOURCE_V2,
            backend="nvrtc", options=("--std=c++14",),
        )
        _LOG.info("tiled_combine v2 ready")
    return _MOD2


def pad_t(t_rows: int) -> int:
    """Time stride rounded up so each cell's series is 16-byte aligned."""
    return ((int(t_rows) + 7) // 8) * 8


def max_tile_for(n_sb: int, spread: int, t_out_len: int) -> int:
    """Largest output-time tile whose staged window fits shared memory."""
    win_cap = SHMEM_CAP // (4 * int(n_sb))      # 2 cells * 2 bytes
    return max(1, min(int(t_out_len), win_cap - int(spread)))


def scatter_compact_to_dense_v2(
    cells_packed: torch.Tensor,
    lut: torch.Tensor,
    n_filled: torch.Tensor,
    dense: torch.Tensor,
    *,
    n_grid: int,
) -> None:
    """Scatter into the v2 ``[n_sb, n_cells, t_pad, 2]`` layout."""
    n_sb, t_rows, packed_w = cells_packed.shape
    n_filled_max = packed_w // 2
    t_pad = dense.shape[2]
    kern = _get_module_v2().get_function("scatter_compact_to_dense_v2")
    threads = 128
    kern(
        ((n_filled_max + threads - 1) // threads, t_rows, n_sb),
        (threads, 1, 1),
        (
            _as_cupy(cells_packed), _as_cupy(lut.int()),
            _as_cupy(n_filled.int()), _as_cupy(dense),
            np.int32(n_sb), np.int32(t_rows), np.int32(t_pad),
            np.int32(n_grid), np.int32(n_filled_max),
            np.int32(lut.shape[1]),
        ),
    )


def combine_tiled_v2(
    dense: torch.Tensor,
    shifts: torch.Tensor,
    *,
    n_grid: int,
    t_rows: int,
    t_out_len: int,
    t_lo: int = 0,
    fftshift: bool = True,
    threads: int = 512,
    tile: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One block per uv cell; stage the whole (tiled) time axis once."""
    n_sb, n_cells, t_pad, two = dense.shape
    if two != 2 or n_cells != n_grid * n_grid:
        raise ValueError(
            f"dense must be [n_sb, n_grid^2, t_pad, 2]; got "
            f"{tuple(dense.shape)} for n_grid={n_grid}"
        )
    n_fdm = int(shifts.shape[0])
    if shifts.shape[1] != n_sb:
        raise ValueError(
            f"shifts must be [n_fdm, n_sb={n_sb}]; got {tuple(shifts.shape)}"
        )
    n_half = n_grid // 2 + 1
    s_min = int(shifts.min().item())
    s_max = int(shifts.max().item())
    spread = s_max - s_min
    if tile is None:
        tile = max_tile_for(n_sb, spread, t_out_len)
    win_w = tile + spread
    shmem = 2 * n_sb * win_w * 2
    if shmem > SHMEM_CAP:
        raise ValueError(
            f"staged window needs {shmem} B > {SHMEM_CAP} B of shared "
            f"memory (n_sb={n_sb}, tile={tile}, spread={spread}); lower "
            f"`tile`"
        )
    if out is None:
        out = torch.empty(
            (n_fdm, t_out_len, n_grid, n_half, 2),
            dtype=torch.float16, device=dense.device,
        )
    kern = _get_module_v2().get_function("tiled_combine_per_fdm_v2")
    n_tiles = (t_out_len + tile - 1) // tile
    kern(
        (n_grid * n_half, n_tiles), (threads, 1, 1),
        (
            _as_cupy(dense), _as_cupy(shifts.int()), _as_cupy(out),
            np.int32(n_sb), np.int32(t_rows), np.int32(t_pad),
            np.int32(n_grid), np.int32(n_half), np.int32(n_fdm),
            np.int32(t_out_len), np.int32(t_lo), np.int32(s_max),
            np.int32(win_w), np.int32(tile), np.int32(1 if fftshift else 0),
        ),
        shared_mem=shmem,
    )
    return out


__all__ += [
    "SHMEM_CAP",
    "combine_tiled_v2",
    "max_tile_for",
    "pad_t",
    "scatter_compact_to_dense_v2",
]


# ---------------------------------------------------------------------------
# v3: v2, but the time-major buffer covers only the FILLED cells
# ---------------------------------------------------------------------------
#
# v2 allocates a full n_grid^2 plane per sub-band even though a sub-band
# fills only ~2134 of 65536 cells at n_sub=4 -- 97% of the buffer is
# zeros. v3 keeps the compact cell axis instead:
#
#     v2   [n_sb, n_grid^2,   t_pad, 2]   2.75 GB at n_sb=64
#     v3   [n_sb, n_filled_max, t_pad, 2] 0.21 GB at n_sb=64
#
# and resolves grid cell -> compact index through an inverse LUT
# ``inv_lut[g, lin]`` (-1 where the sub-band does not carry that cell).
# The LUT is static (the sparsity pattern is), costs n_sb * n_grid^2 * 4 B
# = 16.8 MB at n_sb=64, and is built once.
#
# This also turns the "scatter" into a pure (t, k) transpose of the
# compact buffer the RX ring already produces -- no expansion at all.
#
# Memory matters here as much as speed: the search-node card is at 85%
# occupancy at n_sb=16 today, and the dense ping-pong is what made
# n_sb=48 OOM.

_CUDA_SOURCE_V3 = r"""
extern "C" __global__ void transpose_compact_tmajor(
    const signed char* __restrict__ cells_packed, /* [n_sb, t_rows, nfm*2] */
    signed char*       __restrict__ dense,        /* [n_sb, nfm, t_pad, 2] */
    int n_sb, int t_rows, int t_pad, int n_filled_max)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    const int g = blockIdx.z;
    if (g >= n_sb || t >= t_rows || k >= n_filled_max) return;
    const long long src = ((long long)g * t_rows + t)
                        * ((long long)n_filled_max * 2);
    const long long dst = (((long long)g * n_filled_max + k) * t_pad + t) * 2;
    dense[dst + 0] = cells_packed[src + (long long)k * 2 + 0];
    dense[dst + 1] = cells_packed[src + (long long)k * 2 + 1];
}

extern "C" __global__ void tiled_combine_per_fdm_v3(
    const signed char* __restrict__ dense,   /* [n_sb, nfm, t_pad, 2] */
    const int*         __restrict__ inv_lut, /* [n_sb, n_grid^2]      */
    const int*         __restrict__ shifts,  /* [n_fdm, n_sb]         */
    __half2*           __restrict__ out,     /* [n_fdm, t_out_len, N, n_half] */
    int n_sb, int t_rows, int t_pad, int n_filled_max,
    int n_grid, int n_half, int n_fdm, int t_out_len, int t_lo,
    int shift_max, int win_w, int tile, int fftshift)
{
    extern __shared__ short sh[];       /* [2][n_sb][win_w] packed (re,im) */

    const int cell_id = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int u = cell_id / n_half;
    const int w = cell_id - u * n_half;
    if (u >= n_grid) return;

    const int um = (n_grid - u) % n_grid;
    const int wm = (n_grid - w) % n_grid;
    const int n_cells = n_grid * n_grid;
    const int lin  = u  * n_grid + w;
    const int linm = um * n_grid + wm;

    const int t_out0 = tile_id * tile;
    const int t_win0 = t_out0 + t_lo - shift_max;

    const short* src = (const short*)dense;
    const int nthreads = blockDim.x;
    const int tid = threadIdx.x;
    const int span = n_sb * win_w;

    for (int idx = tid; idx < 2 * span; idx += nthreads) {
        const int which = idx / span;
        const int rem   = idx - which * span;
        const int g     = rem / win_w;
        const int j     = rem - g * win_w;
        const int ts    = t_win0 + j;
        short v = 0;
        if (ts >= 0 && ts < t_rows) {
            const int c = which ? linm : lin;
            const int k = inv_lut[(long long)g * n_cells + c];
            if (k >= 0) {
                v = src[((long long)g * n_filled_max + k) * t_pad + ts];
            }
        }
        sh[idx] = v;
    }
    __syncthreads();

    const int n_out = n_fdm * tile;
    for (int idx = tid; idx < n_out; idx += nthreads) {
        const int f  = idx / tile;
        const int tl = idx - f * tile;
        const int t_out = t_out0 + tl;
        if (t_out >= t_out_len) continue;

        int acc_re = 0, acc_im = 0, accm_re = 0, accm_im = 0;
        for (int g = 0; g < n_sb; ++g) {
            const int ts = t_out + t_lo - shifts[f * n_sb + g];
            if (ts < 0 || ts >= t_rows) continue;
            const int j = ts - t_win0;
            const short a = sh[g * win_w + j];
            const short b = sh[span + g * win_w + j];
            acc_re  += (int)(signed char)(a & 0xFF);
            acc_im  += (int)(signed char)((a >> 8) & 0xFF);
            accm_re += (int)(signed char)(b & 0xFF);
            accm_im += (int)(signed char)((b >> 8) & 0xFF);
        }
        float o_re = 0.5f * (float)(acc_re + accm_re);
        float o_im = 0.5f * (float)(acc_im - accm_im);
        if (fftshift && ((u + w) & 1)) { o_re = -o_re; o_im = -o_im; }
        out[(long long)f * t_out_len * n_grid * n_half
            + (long long)t_out * n_grid * n_half
            + (long long)u * n_half + w] = __floats2half2_rn(o_re, o_im);
    }
}
"""

_MOD3 = None


def _get_module_v3():
    global _MOD3
    if _MOD3 is None:
        cp = _get_cupy()
        _LOG.info("compiling tiled_combine v3 via NVRTC...")
        _MOD3 = cp.RawModule(
            code="#include <cuda_fp16.h>\n" + _CUDA_SOURCE_V3,
            backend="nvrtc", options=("--std=c++14",),
        )
        _LOG.info("tiled_combine v3 ready")
    return _MOD3


def build_inverse_lut(
    lut: torch.Tensor, n_filled: torch.Tensor, *, n_grid: int,
) -> torch.Tensor:
    """grid cell -> compact index per sub-band, -1 where absent.

    Static for a given sparsity pattern; build once and keep.
    """
    n_sb = int(lut.shape[0])
    inv = torch.full(
        (n_sb, n_grid * n_grid), -1, dtype=torch.int32, device=lut.device,
    )
    for g in range(n_sb):
        nf = int(n_filled[g].item())
        if nf <= 0:
            continue
        cells = lut[g, :nf].long()
        valid = (cells >= 0) & (cells < n_grid * n_grid)
        idx = torch.arange(nf, dtype=torch.int32, device=lut.device)
        inv[g, cells[valid]] = idx[valid]
    return inv


def transpose_compact_tmajor(
    cells_packed: torch.Tensor, dense: torch.Tensor,
) -> None:
    """(t, k) -> (k, t) transpose of the compact COO buffer."""
    n_sb, t_rows, packed_w = cells_packed.shape
    n_filled_max = packed_w // 2
    t_pad = dense.shape[2]
    kern = _get_module_v3().get_function("transpose_compact_tmajor")
    threads = 128
    kern(
        ((n_filled_max + threads - 1) // threads, t_rows, n_sb),
        (threads, 1, 1),
        (
            _as_cupy(cells_packed), _as_cupy(dense),
            np.int32(n_sb), np.int32(t_rows), np.int32(t_pad),
            np.int32(n_filled_max),
        ),
    )


def combine_tiled_v3(
    dense: torch.Tensor,
    inv_lut: torch.Tensor,
    shifts: torch.Tensor,
    *,
    n_grid: int,
    t_rows: int,
    t_out_len: int,
    t_lo: int = 0,
    fftshift: bool = True,
    threads: int = 1024,
    tile: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """v2, indexed through the inverse LUT so no dense plane is needed."""
    n_sb, n_filled_max, t_pad, two = dense.shape
    if two != 2:
        raise ValueError(
            f"dense must be [n_sb, n_filled_max, t_pad, 2]; got "
            f"{tuple(dense.shape)}"
        )
    n_fdm = int(shifts.shape[0])
    if shifts.shape[1] != n_sb:
        raise ValueError(
            f"shifts must be [n_fdm, n_sb={n_sb}]; got {tuple(shifts.shape)}"
        )
    n_half = n_grid // 2 + 1
    s_min = int(shifts.min().item())
    s_max = int(shifts.max().item())
    spread = s_max - s_min
    if tile is None:
        tile = max_tile_for(n_sb, spread, t_out_len)
    win_w = tile + spread
    shmem = 2 * n_sb * win_w * 2
    if shmem > SHMEM_CAP:
        raise ValueError(
            f"staged window needs {shmem} B > {SHMEM_CAP} B of shared "
            f"memory (n_sb={n_sb}, tile={tile}, spread={spread})"
        )
    if out is None:
        out = torch.empty(
            (n_fdm, t_out_len, n_grid, n_half, 2),
            dtype=torch.float16, device=dense.device,
        )
    kern = _get_module_v3().get_function("tiled_combine_per_fdm_v3")
    n_tiles = (t_out_len + tile - 1) // tile
    kern(
        (n_grid * n_half, n_tiles), (threads, 1, 1),
        (
            _as_cupy(dense), _as_cupy(inv_lut.int()), _as_cupy(shifts.int()),
            _as_cupy(out),
            np.int32(n_sb), np.int32(t_rows), np.int32(t_pad),
            np.int32(n_filled_max), np.int32(n_grid), np.int32(n_half),
            np.int32(n_fdm), np.int32(t_out_len), np.int32(t_lo),
            np.int32(s_max), np.int32(win_w), np.int32(tile),
            np.int32(1 if fftshift else 0),
        ),
        shared_mem=shmem,
    )
    return out


__all__ += [
    "build_inverse_lut",
    "combine_tiled_v3",
    "transpose_compact_tmajor",
]

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

Measured — and where it does and does not help
----------------------------------------------

n09, idle RTX 2080 Ti, n_fdm=34, N=256, t_stream=322, t_lo=64,
n_filled=3247, tile=30 (ms/cube):

    n_sb     scatter   tiled combine   dense combine   dense scatter
     16        1.67        44.52           47.9            5.4
     32        2.41       137.74          103.2            9.8
     64        4.76       259.32          234.6           19.6

Peak GPU memory at n_sb=16: **3.09 GiB vs 5.17 GiB** for the dense
path (the ping-pong dense planes are what make n_sb=48 OOM on an
11 GiB card).

So this kernel is a win at the **current** op-point (n_sb=16): ~1.19x
faster end-to-end and 1.68x smaller. It LOSES above n_sb=16 and should
not be selected there.

It is not yet near its own bound. Staged reads are ~1.4 GB at n_sb=16,
which at ~500 GB/s is ~3 ms against the 44.5 ms measured — i.e. the
kernel is **latency-bound, not bandwidth-bound**. Each block issues 4
short strided bursts per sub-band (cell re/im and mirror re/im) across
231k blocks. The obvious next steps, untried:

  * pad the time stride to a multiple of 4 and stage with 4-byte
    loads (4x fewer load instructions);
  * give each block several cells to raise memory-level parallelism
    and amortise ``__syncthreads``;
  * stage re/im as one interleaved stream so the 4 bursts become 2.

Until then, do not expect this path to unlock larger ``n_sb``.
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

"""src/dsart/image/coo_combine_cuda.py — COO-direct fine-DM combine.

Why this exists
===============

``fused_combine_cuda.fused_dequant_combine_per_fdm_half`` is
*output-stationary over a dense plane*: one thread owns one output cell
``(t_out, u, w)`` and loops the chgroups, reading
``streams[g, t_out - shift[g], :, u, w]``. It is called once per fine-DM
trial. That costs, at the production op-point (n_sb=16, t_stream=322,
N=256, t_out=192, n_fdm=34):

    reads per fine-DM     405.8 MB
    x n_fdm = 34           13.80 GB per cube
    the stream itself       0.68 GB
    amplification           20.4x

**The 20.4x is the fine-DM re-read, not the sparsity.** It is almost
exactly ``n_fdm`` (34), scaled by 0.60 because each fine-DM only touches
192 of the 322 stream samples. Measured on n02 (2026-09-22) the combine
runs 47.9 ms/cube at n_sb=16 and 234.6 ms at n_sb=64, scaling as
``n_sb^1.14`` and DRAM-bandwidth-bound at 235-288 GB/s.

Sparsity is a **second, independent** factor: only ~3247 of the 65536
uv cells per (sub-band, sample) are non-zero (5.0%). The dense kernel
reads all of them.

This module takes both factors at once by inverting the loop:

  * **input-stationary** — one thread owns one compact COO entry
    ``(g, k, t_src)`` and loops the fine-DM trials, so each wire byte is
    read ONCE instead of ``n_fdm`` times;
  * **COO-direct** — it consumes ``cells_packed`` (the compact buffer
    the RX ring already assembles) rather than a scattered dense plane,
    so the 5% occupancy is not paid for.

Reads drop from 13.80 GB to **33.5 MB** per cube, and they are perfectly
coalesced: the compact row ``cells_packed[g, t, :]`` is contiguous in
``k``, so adjacent threads read adjacent bytes.

The price is atomics. Each (g, k, t_src) entry contributes to ``n_fdm``
output planes, and within a plane only chgroups collide (different
fine-DM trials write different planes), so contention is <= n_sb-way.

Two further consequences, both of which matter more than the speed:

  * **the dense planes disappear.** ``scatter_compact_to_dense`` and its
    ``[n_sb, t_stream, 2, N, N]`` int8 buffers (1.26 GiB ping-pong at
    n_sb=16, 5.03 GiB at n_sb=64) are not needed. That buffer is what
    made n_sb=48 OOM on an 11 GiB card.
  * the separate GPU-scatter kernel (0.306 ms per sub-band) goes away.

Exactness
---------

The dense kernel accumulates in ``int`` and converts once. Here the
accumulator is the output buffer itself. ``__half`` represents integers
exactly up to 2048, and the running sum is bounded by
``n_sb * 127``, so fp16 accumulation is **bit-exact for n_sb <= 16**
(2032 <= 2048) and NOT exact beyond it. ``combine_coo`` therefore
selects an fp32 accumulator automatically when ``n_sb > 16``; see
:func:`accumulator_dtype_for`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

_LOG = logging.getLogger(__name__)

_cp: Optional[object] = None

#: Largest ``n_sb`` for which an fp16 accumulator is bit-exact.
#: ``__half`` is exact on integers up to 2048 and the accumulated value
#: is bounded by ``n_sb * 127`` (int8 magnitude cap).
MAX_NSB_FP16_EXACT: int = 2048 // 127  # == 16


def _get_cupy():
    global _cp
    if _cp is not None:
        return _cp
    import cupy as cp  # noqa: WPS433  # lazy: cuda-less CI must still import
    _cp = cp
    return cp


def accumulator_dtype_for(n_sb: int) -> torch.dtype:
    """fp16 while it is exact, fp32 once it is not.

    See the module docstring: the sum of ``n_sb`` int8 values fits
    exactly in ``__half`` only while ``n_sb * 127 <= 2048``.
    """
    return torch.float16 if int(n_sb) <= MAX_NSB_FP16_EXACT else torch.float32


# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------
#
# Grid:  x -> k (compact cell index, contiguous => coalesced)
#        y -> t_src
#        z -> g (sub-band / corr)
# Each thread loops n_fdm and atomically accumulates into the output.
#
# Conjugate fold. The imager wants, for the half-spectrum w in [0,n_half):
#     out[u][w] = 0.5 * ( G[u][w] + conj(G[(N-u)%N][(N-w)%N]) )
# An input cell (a,b) carrying V therefore lands:
#     * as +V    at out[a][b]                       when b  < n_half
#     * as +conj(V) at out[(N-a)%N][(N-b)%N]        when bm < n_half
# Both fire when b == 0 or b == N/2 (there bm == b), which is exactly the
# self-conjugate column the dense kernel also counts twice. The 0.5 and
# the (-1)^(u+w) fftshift are applied in the finalise pass, not here, so
# the accumulator stays integral (and therefore exact in fp16).
_CUDA_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" __global__ void coo_combine_per_fdm_accum(
    const signed char* __restrict__ cells_packed, /* [n_sb, t_det, n_filled_max*2] */
    const int*         __restrict__ lut,          /* [n_sb, lut_stride] */
    const int*         __restrict__ n_filled,     /* [n_sb] */
    const int*         __restrict__ shifts,       /* [n_fdm, n_sb] */
    ACC_T*             __restrict__ out,          /* [n_fdm, t_out, N, n_half, 2] */
    int n_sb, int t_det, int t_out_len, int n_grid,
    int n_filled_max, int lut_stride, int n_fdm, int t_lo)
{
    const int k     = blockIdx.x * blockDim.x + threadIdx.x;
    const int t_src = blockIdx.y;
    const int g     = blockIdx.z;

    if (g >= n_sb || t_src >= t_det || k >= n_filled_max) return;
    const int nf = n_filled[g];
    if (nf <= 0 || k >= nf) return;          /* silent corr, or wire padding */

    const int n_half    = n_grid / 2 + 1;
    const int n_grid_sq = n_grid * n_grid;

    const int lin = lut[g * lut_stride + k];
    if ((unsigned)lin >= (unsigned)n_grid_sq) return;   /* defensive */

    const int a = lin / n_grid;
    const int b = lin - a * n_grid;
    const int am = (n_grid - a) % n_grid;
    const int bm = (n_grid - b) % n_grid;

    const long long src = ((long long)g * t_det + t_src)
                        * ((long long)n_filled_max * 2);
    const float re = (float)cells_packed[src + (long long)k * 2 + 0];
    const float im = (float)cells_packed[src + (long long)k * 2 + 1];
    if (re == 0.0f && im == 0.0f) return;    /* empty cell: nothing to add */

    const int  direct = (b  < n_half);
    const int  mirror = (bm < n_half);
    const long long plane = (long long)n_grid * n_half * 2;
    const long long o_direct = ((long long)a  * n_half + b ) * 2;
    const long long o_mirror = ((long long)am * n_half + bm) * 2;

    for (int f = 0; f < n_fdm; ++f) {
        const int t_o = t_src + shifts[f * n_sb + g] - t_lo;
        if (t_o < 0 || t_o >= t_out_len) continue;
        ACC_T* base = out + (long long)f * t_out_len * plane
                          + (long long)t_o * plane;
        if (direct) {
            atomicAdd(base + o_direct + 0, (ACC_T)re);
            atomicAdd(base + o_direct + 1, (ACC_T)im);
        }
        if (mirror) {
            /* conj(V): imaginary part negated */
            atomicAdd(base + o_mirror + 0, (ACC_T)re);
            atomicAdd(base + o_mirror + 1, (ACC_T)(-im));
        }
    }
}

/* Apply the 0.5 fold factor and the (-1)^(u+w) fftshift, and pack the
   interleaved (re,im) accumulator into the __half2 plane the imager's
   irfft2 consumes. One thread per output cell. */
extern "C" __global__ void coo_combine_finalise(
    const ACC_T* __restrict__ acc,   /* [n_fdm, t_out, N, n_half, 2] */
    __half2*     __restrict__ out,   /* [n_fdm, t_out, N, n_half]    */
    int n_fdm, int t_out_len, int n_grid, int fftshift)
{
    const int n_half = n_grid / 2 + 1;
    const long long total = (long long)n_fdm * t_out_len * n_grid * n_half;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const int w = (int)(i % n_half);
    const int u = (int)((i / n_half) % n_grid);

    float re = 0.5f * (float)acc[i * 2 + 0];
    float im = 0.5f * (float)acc[i * 2 + 1];
    if (fftshift && ((u + w) & 1)) { re = -re; im = -im; }
    out[i] = __floats2half2_rn(re, im);
}
"""


_KERNELS: dict[str, object] = {}


def _get_kernels(acc_dtype: torch.dtype):
    """Compile (once per accumulator dtype) and return both kernels."""
    key = "fp16" if acc_dtype == torch.float16 else "fp32"
    if key in _KERNELS:
        return _KERNELS[key]
    cp = _get_cupy()
    ctype = "__half" if key == "fp16" else "float"
    src = f"#define ACC_T {ctype}\n" + _CUDA_SOURCE
    _LOG.info("compiling coo_combine (ACC_T=%s) via NVRTC...", ctype)
    mod = cp.RawModule(code=src, backend="nvrtc", options=("--std=c++14",))
    pair = (
        mod.get_function("coo_combine_per_fdm_accum"),
        mod.get_function("coo_combine_finalise"),
    )
    _KERNELS[key] = pair
    _LOG.info("coo_combine (ACC_T=%s) ready", ctype)
    return pair


def _as_cupy(t: torch.Tensor):
    cp = _get_cupy()
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))


def combine_coo(
    cells_packed: torch.Tensor,
    lut: torch.Tensor,
    n_filled: torch.Tensor,
    shifts: torch.Tensor,
    *,
    n_grid: int,
    t_out_len: int,
    t_lo: int = 0,
    fftshift: bool = True,
    out: torch.Tensor | None = None,
    acc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine every fine-DM trial straight from the compact COO buffer.

    Args:
        cells_packed: ``[n_sb, t_rows, n_filled_max*2]`` int8, the buffer
            ``rx_ring_assemble_compact_block`` already produces. Note
            ``t_rows`` is the number of COMPACT rows — in the M7.7
            symmetric-padding path that is ``t_stream`` (= t_det +
            pad_left + pad_right, 322 at the production op-point), not
            ``t_det``; see ``cube_pipeline._stage_h2d``'s
            ``n_rows_compact``.
        lut: ``[n_sb, lut_stride]`` int32, compact index -> linear
            ``u * n_grid + w`` cell index (the M7.5 sparsity LUT).
        n_filled: ``[n_sb]`` int32.
        shifts: ``[n_fdm, n_sb]`` int32 per-fine-DM per-sub-band sample
            shift (what ``compute_time_shift_search`` returns).
        n_grid: uv grid size N.
        t_out_len: number of output time rows (``t_det - t_lo``).
        t_lo: first output row (M7.7.2 carry-over).
        fftshift: apply the ``(-1)^(u+w)`` fold.
        out: optional ``[n_fdm, t_out_len, N, N/2+1]`` complex-as-half2
            destination, supplied as a ``torch.float16`` tensor with a
            trailing size-2 axis.
        acc: optional accumulator scratch to re-use across cubes.

    Returns:
        The ``out`` tensor, shape ``[n_fdm, t_out_len, N, N/2+1, 2]``
        float16, matching the dense kernel's ``__half2`` plane layout.
    """
    cp = _get_cupy()
    if cells_packed.ndim != 3:
        raise ValueError(
            f"cells_packed must be [n_sb, t_det, n_filled_max*2], got "
            f"{tuple(cells_packed.shape)}"
        )
    n_sb, t_det, packed_w = cells_packed.shape
    if packed_w % 2:
        raise ValueError(
            f"cells_packed last axis must be even (re,im interleaved), "
            f"got {packed_w}"
        )
    n_filled_max = packed_w // 2
    if shifts.ndim != 2 or shifts.shape[1] != n_sb:
        raise ValueError(
            f"shifts must be [n_fdm, n_sb={n_sb}], got "
            f"{tuple(shifts.shape)}"
        )
    n_fdm = int(shifts.shape[0])
    n_half = n_grid // 2 + 1
    lut_stride = int(lut.shape[1])

    acc_dtype = accumulator_dtype_for(n_sb)
    acc_shape = (n_fdm, t_out_len, n_grid, n_half, 2)
    if acc is None:
        acc = torch.zeros(
            acc_shape, dtype=acc_dtype, device=cells_packed.device,
        )
    else:
        if tuple(acc.shape) != acc_shape or acc.dtype != acc_dtype:
            raise ValueError(
                f"acc must be {acc_shape} {acc_dtype}, got "
                f"{tuple(acc.shape)} {acc.dtype}"
            )
        acc.zero_()
    if out is None:
        out = torch.empty(
            acc_shape, dtype=torch.float16, device=cells_packed.device,
        )

    accum_k, final_k = _get_kernels(acc_dtype)

    threads = 128
    blocks = ((n_filled_max + threads - 1) // threads, t_det, n_sb)
    accum_k(
        blocks, (threads, 1, 1),
        (
            _as_cupy(cells_packed), _as_cupy(lut.int()),
            _as_cupy(n_filled.int()), _as_cupy(shifts.int()), _as_cupy(acc),
            np.int32(n_sb), np.int32(t_det), np.int32(t_out_len),
            np.int32(n_grid), np.int32(n_filled_max), np.int32(lut_stride),
            np.int32(n_fdm), np.int32(t_lo),
        ),
    )

    total = n_fdm * t_out_len * n_grid * n_half
    final_k(
        ((total + 255) // 256,), (256,),
        (
            _as_cupy(acc), _as_cupy(out),
            np.int32(n_fdm), np.int32(t_out_len), np.int32(n_grid),
            np.int32(1 if fftshift else 0),
        ),
    )
    return out


__all__ = [
    "MAX_NSB_FP16_EXACT",
    "accumulator_dtype_for",
    "combine_coo",
]

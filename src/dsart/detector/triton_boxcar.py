"""Triton kernels for boxcar evaluation on a precomputed padded cumsum.

This module hosts two kernels:

  * :func:`boxcar_from_padded_cumsum_triton` — one boxcar width per call::

        out[t, f, h, w] = cs[t + offset + width, f, h, w]
                          - cs[t + offset, f, h, w]

  * :func:`multi_boxcar_argmax_triton` — *all* K widths in one fused pass:

        For each (t, f, h, w) compute the SNR of K boxcars
        ``snr[k] = (cs[t + off[k] + w[k]] - cs[t + off[k]]) * sigma_inv[k]``
        and write the **maximum** SNR over k plus the **winner** k index.

Where ``offset[k] = (max_width // 2) - (widths[k] // 2)``.

The multi-boxcar argmax kernel is the M7.2 perf win: at production
geometry (T_search=192, N_fdm=34, N_grid=256, K=7) the per-kernel
boxcar+topk loop took ~7 ms × 7 kernels of cube-sized memory traffic.
The fused kernel reads the cumsum once per output cell and produces
the argmax SNR cube in a single pass, eliminating K-1 redundant passes
over the cube (~30 ms saved at production op-point).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

try:  # Optional dependency on CUDA hosts.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _boxcar_from_cs_kernel(
        cs_ptr,
        out_ptr,
        n_out,
        n_f,
        n_h,
        n_w,
        offset,
        width,
        t_base,
        cs_s_t,
        cs_s_f,
        cs_s_h,
        cs_s_w,
        out_s_t,
        out_s_f,
        out_s_h,
        out_s_w,
        BLOCK: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = n_out * n_f * n_h * n_w
        mask = offs < total

        hw = n_h * n_w
        fhw = n_f * hw

        t = offs // fhw
        rem = offs % fhw
        f = rem // hw
        rem2 = rem % hw
        h = rem2 // n_w
        w = rem2 % n_w

        t_abs = t + t_base
        base_low = (
            (t_abs + offset) * cs_s_t
            + f * cs_s_f
            + h * cs_s_h
            + w * cs_s_w
        )
        base_high = (
            (t_abs + offset + width) * cs_s_t
            + f * cs_s_f
            + h * cs_s_h
            + w * cs_s_w
        )

        low = tl.load(cs_ptr + base_low, mask=mask, other=0.0)
        high = tl.load(cs_ptr + base_high, mask=mask, other=0.0)
        val = high - low

        out_off = (
            t * out_s_t
            + f * out_s_f
            + h * out_s_h
            + w * out_s_w
        )
        if OUT_FP16:
            tl.store(out_ptr + out_off, val.to(tl.float16), mask=mask)
        else:
            tl.store(out_ptr + out_off, val, mask=mask)


def boxcar_from_padded_cumsum_triton(
    cs: torch.Tensor,
    *,
    axis: int,
    width: int,
    max_width: int,
    n_out: int,
    t_base: int = 0,
    out_dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Return triton result, or None if unsupported / unavailable."""
    if bool(int(os.environ.get("DSART_DISABLE_TRITON_BOXCAR", "0"))):
        return None
    if not _HAVE_TRITON:
        return None
    if cs.device.type != "cuda":
        return None
    if axis != 0:
        return None
    if cs.dim() != 4:
        return None
    if width < 1 or width > max_width:
        return None

    # axis=0 only path [Tpad, F, H, W]
    n_f, n_h, n_w = int(cs.shape[1]), int(cs.shape[2]), int(cs.shape[3])
    if n_out < 1:
        return None
    if t_base < 0:
        return None

    pad_left_full = max_width // 2
    pad_left_w = width // 2
    offset = pad_left_full - pad_left_w
    # Max index touched is (t_base + n_out - 1) + offset + width.
    if (t_base + n_out + offset + width) > int(cs.shape[0]):
        return None

    eff_dtype = out_dtype if out_dtype is not None else cs.dtype
    if eff_dtype not in (torch.float16, torch.float32):
        return None
    if cs.dtype not in (torch.float16, torch.float32):
        return None

    out = torch.empty((n_out, n_f, n_h, n_w), dtype=eff_dtype, device=cs.device)
    total = n_out * n_f * n_h * n_w
    block = 256
    grid = (triton.cdiv(total, block),)
    _boxcar_from_cs_kernel[grid](
        cs,
        out,
        n_out,
        n_f,
        n_h,
        n_w,
        int(offset),
        int(width),
        int(t_base),
        cs.stride(0),
        cs.stride(1),
        cs.stride(2),
        cs.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK=block,
        OUT_FP16=(eff_dtype == torch.float16),
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Multi-boxcar fused argmax kernel (M7.2 perf win)
# ---------------------------------------------------------------------------

if _HAVE_TRITON:

    @triton.jit
    def _multi_boxcar_argmax_kernel(
        cs_ptr,            # [Tpad, F, H, W] cube_dtype (fp16/fp32)
        sigma_inv_ptr,     # [K] fp32 per-kernel SNR scales (1/sigma_k_prev)
        widths_ptr,        # [K] int32 boxcar widths
        offsets_ptr,       # [K] int32 read offsets (pad_left_full - w[k]//2)
        out_max_ptr,       # [T_out, F, H, W] cube_dtype max-SNR cube
        out_win_ptr,       # [T_out, F, H, W] int16 winner kernel id
        n_out,
        n_f,
        n_h,
        n_w,
        t_base,
        cs_s_t,
        cs_s_f,
        cs_s_h,
        cs_s_w,
        out_max_s_t,
        out_max_s_f,
        out_max_s_h,
        out_max_s_w,
        out_win_s_t,
        out_win_s_f,
        out_win_s_h,
        out_win_s_w,
        K_STATIC: tl.constexpr,
        BLOCK: tl.constexpr,
        OUT_FP16: tl.constexpr,
    ):
        """For each output cell, compute K boxcar SNRs in registers and
        keep the maximum SNR + winner kernel index.

        Memory: ~2 cs reads × K + 1 max write + 1 winner write per cell.
        At K=7 cube_dtype=fp16, T*F*H*W=192*34*256*256 cells: 16 reads ×
        2 B + 4 B writes = ~36 B/cell × 428 M cells ≈ 15 GB of traffic.
        On a 2080 Ti with ~500 GB/s effective bandwidth this is ~30 ms,
        vs ~7 × per-kernel passes (~6 ms each) = ~42 ms in the
        unfused path. Crucially this also reduces the per-cube topk
        from K calls to 1.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = n_out * n_f * n_h * n_w
        mask = offs < total

        hw = n_h * n_w
        fhw = n_f * hw

        t = offs // fhw
        rem = offs % fhw
        f = rem // hw
        rem2 = rem % hw
        h = rem2 // n_w
        w = rem2 % n_w
        t_abs = t + t_base

        # Base offsets into the cs cube (no time offset yet).
        base = (
            f * cs_s_f + h * cs_s_h + w * cs_s_w
        )

        # Initialize running max + winner with k=0.
        off0 = tl.load(offsets_ptr + 0)
        w0 = tl.load(widths_ptr + 0)
        si0 = tl.load(sigma_inv_ptr + 0)
        low0 = tl.load(
            cs_ptr + base + (t_abs + off0) * cs_s_t,
            mask=mask, other=0.0,
        ).to(tl.float32)
        high0 = tl.load(
            cs_ptr + base + (t_abs + off0 + w0) * cs_s_t,
            mask=mask, other=0.0,
        ).to(tl.float32)
        max_snr = (high0 - low0) * si0
        winner = tl.zeros((BLOCK,), dtype=tl.int32)

        # Loop over remaining kernels — Triton unrolls static_range.
        for k in tl.static_range(1, K_STATIC):
            offk = tl.load(offsets_ptr + k)
            wk = tl.load(widths_ptr + k)
            sik = tl.load(sigma_inv_ptr + k)
            lowk = tl.load(
                cs_ptr + base + (t_abs + offk) * cs_s_t,
                mask=mask, other=0.0,
            ).to(tl.float32)
            highk = tl.load(
                cs_ptr + base + (t_abs + offk + wk) * cs_s_t,
                mask=mask, other=0.0,
            ).to(tl.float32)
            snrk = (highk - lowk) * sik
            better = snrk > max_snr
            max_snr = tl.where(better, snrk, max_snr)
            winner = tl.where(better, k, winner)

        out_max_off = (
            t * out_max_s_t
            + f * out_max_s_f
            + h * out_max_s_h
            + w * out_max_s_w
        )
        out_win_off = (
            t * out_win_s_t
            + f * out_win_s_f
            + h * out_win_s_h
            + w * out_win_s_w
        )
        if OUT_FP16:
            tl.store(out_max_ptr + out_max_off, max_snr.to(tl.float16), mask=mask)
        else:
            tl.store(out_max_ptr + out_max_off, max_snr, mask=mask)
        tl.store(out_win_ptr + out_win_off, winner.to(tl.int16), mask=mask)


def multi_boxcar_argmax_triton(
    cs: torch.Tensor,
    *,
    widths: torch.Tensor,
    offsets: torch.Tensor,
    sigma_inv: torch.Tensor,
    n_out: int,
    t_base: int = 0,
    out_max_dtype: Optional[torch.dtype] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Fused multi-boxcar argmax over K kernel widths.

    Args:
        cs: ``[Tpad, F, H, W]`` padded-cumsum tensor (cube_dtype),
            cuda. Produced by ``_get_or_build_amortise_cs`` /
            ``_fill_amortise_cs``.
        widths: ``[K] int32`` cuda tensor of boxcar widths.
        offsets: ``[K] int32`` cuda tensor of per-kernel read offsets
            ``(max_w // 2 - widths[k] // 2)``. Pre-computed on host.
        sigma_inv: ``[K] float32`` cuda tensor of per-kernel SNR scales
            ``1 / sigma_k_prev`` (used to compare boxcar SNRs).
        n_out: output time length.
        t_base: start index in cs of the first output time sample
            (per the existing ``boxcar_from_padded_cumsum`` convention).
        out_max_dtype: dtype for ``out_max``. Default ``cs.dtype``.

    Returns:
        ``(out_max, out_winner)`` tensors of shape
        ``[n_out, F, H, W]``. ``out_max`` is ``out_max_dtype``
        (default cube_dtype); ``out_winner`` is ``int16``. Returns
        ``None`` when the Triton path is unavailable.
    """
    if bool(int(os.environ.get("DSART_DISABLE_TRITON_BOXCAR", "0"))):
        return None
    if not _HAVE_TRITON:
        return None
    if cs.device.type != "cuda":
        return None
    if cs.dim() != 4:
        return None
    if widths.device != cs.device or offsets.device != cs.device:
        return None
    if widths.dtype != torch.int32 or offsets.dtype != torch.int32:
        return None
    if sigma_inv.dtype != torch.float32:
        return None
    K = int(widths.numel())
    if K < 1 or K > 32:
        # Triton static_range needs a compile-time bound; cap at 32.
        return None
    if int(offsets.numel()) != K or int(sigma_inv.numel()) != K:
        return None
    if cs.dtype not in (torch.float16, torch.float32):
        return None

    eff_dtype = out_max_dtype if out_max_dtype is not None else cs.dtype
    if eff_dtype not in (torch.float16, torch.float32):
        return None

    n_f, n_h, n_w = int(cs.shape[1]), int(cs.shape[2]), int(cs.shape[3])
    out_max = torch.empty(
        (n_out, n_f, n_h, n_w), dtype=eff_dtype, device=cs.device,
    )
    out_win = torch.empty(
        (n_out, n_f, n_h, n_w), dtype=torch.int16, device=cs.device,
    )
    total = n_out * n_f * n_h * n_w
    block = 256
    grid = (triton.cdiv(total, block),)
    _multi_boxcar_argmax_kernel[grid](
        cs,
        sigma_inv,
        widths,
        offsets,
        out_max,
        out_win,
        n_out,
        n_f,
        n_h,
        n_w,
        int(t_base),
        cs.stride(0),
        cs.stride(1),
        cs.stride(2),
        cs.stride(3),
        out_max.stride(0),
        out_max.stride(1),
        out_max.stride(2),
        out_max.stride(3),
        out_win.stride(0),
        out_win.stride(1),
        out_win.stride(2),
        out_win.stride(3),
        K_STATIC=K,
        BLOCK=block,
        OUT_FP16=(eff_dtype == torch.float16),
        num_warps=4,
    )
    return out_max, out_win


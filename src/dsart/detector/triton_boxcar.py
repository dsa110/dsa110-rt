"""Triton fast path for boxcar_from_padded_cumsum on axis=0.

This kernel computes one boxcar width from a precomputed padded cumsum:

    out[t, f, h, w] = cs[t + offset + width, f, h, w] - cs[t + offset, f, h, w]

where ``offset = (max_width // 2) - (width // 2)``.

It avoids Python-side tile loops and the intermediate per-tile tensors in
``boxcar_from_padded_cumsum``'s torch implementation.
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


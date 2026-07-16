"""Stokes-I dirty imager (plan §3.6.11 + §4.4 lines 1513-1515).

Per ``[fine_dm, t]`` slice, the imager:

  1. Takes the combiner's ``[head_block_samples, N_grid, N_grid] complex``
     uv-slab (single-side +uv only — the gridder per §3.6.5 G5 writes only
     ``+uv``; see §3.6.11 derivation for why ``Re(iFFT2)`` recovers the
     canonical Stokes-I dirty image up to a global factor of 2 that
     Layer-2 σ_k EMA absorbs).
  2. Computes ``Re(iFFT2(uv_slab))`` — single-side identity,
     §3.6.11 lines 1042-1064.
  3. Applies ``fftshift`` so DC lands at ``(N_grid//2, N_grid//2)``
     per §3.1.
  4. Applies the **edge mask** (§3.5 G11 + §4.4 line 1515): zeroes
     pixels in the outer ring covering FFT wraparound, gridding-kernel
     taper artifacts, and the image-plane envelope ``c(l, m)`` cut.
     Mask is multiplicative ``[N_grid, N_grid] float32``.
  5. Casts to fp16 to feed the detector cube ring (§3.6.11 dtype pin).

The production-grade GPU path lives in :mod:`dsart.image.imager_gpu`
(landed Chunk 8: D19/D20/D21 in ``M5_PLAN_FIXES.md``). That module
fuses the M3 cint8 dequant + per-fdm 16-chgroup combine into a single
NVRTC kernel, runs cuFFT-cfp16 ``ifft2`` + fftshift, and applies
:func:`compute_edge_mask` (defined here) at deployment geometry — 9.79
cubes/s at T_det=256 / N_fdm=32 / N_grid=256 on h01 GPU 1, comfortably
clearing the plan §8 8 cubes/s target. The live ``services/
search_compute.py`` service consumes :class:`dsart.image.imager_gpu.GpuImager`
directly; this Chunk-6a module retains the pure-numpy / pure-torch
helpers (:func:`compute_edge_mask`, :func:`dirty_image_from_uv_grid`,
:func:`apply_edge_mask`) that are needed for unit tests, the chunk-7
captured-mode loader's pre-quant cf64 reference path, and the cube-
injection bench's voltage-fixture cross-check.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

__all__ = [
    "compute_edge_mask",
    "dirty_image_from_uv_grid",
    "apply_edge_mask",
    "image_mask_npad",
]


# ---------------------------------------------------------------------------
# Edge mask
# ---------------------------------------------------------------------------


def image_mask_npad(
    *,
    n_grid: int,
    kernel_support: int = 5,
    sigma_l_pix: Optional[float] = None,
    envelope_threshold: float = 0.5,
) -> int:
    """Compute the per-side edge-mask pad in pixels (plan §3.5 G11).

        npad = max(
            ceil(kernel_support / 2) + 2,
            ceil(N_grid/2 - σ_l_pix · sqrt(2 · ln(1/envelope_threshold)))
        )

    Args:
        n_grid: grid side length.
        kernel_support: gridding-kernel support (odd, in cells).
        sigma_l_pix: image-plane envelope σ in pixels. The gridder's
            §3.6.5 truncated Gaussian with kernel-cell σ ``σ_kernel``
            produces an image-plane envelope ``c(l, m) ∝ exp(-(l² + m²)
            / (2·σ_l²))`` with ``σ_l_pix = N_grid / (2π · σ_kernel)``.
            Caller provides this directly because the kernel-cell σ
            depends on operating point (§3.6.5 G7) and the imager
            should not duplicate the math. If ``None``, the envelope
            term collapses to zero (only the kernel-support / FFT-
            wraparound term contributes).
        envelope_threshold: threshold below which the envelope is
            considered "too dim" to detect through; default 0.5
            (-3 dB) per §4.4 line 1515.

    Returns:
        Integer ``npad`` ≥ 0; the inner active region is
        ``[npad, N_grid - npad)`` along each spatial axis.
    """
    if n_grid <= 0 or n_grid & (n_grid - 1):
        raise ValueError(f"n_grid={n_grid}, expected positive power of two")
    if kernel_support < 1:
        raise ValueError(f"kernel_support={kernel_support}, expected ≥ 1")
    if not (0.0 < envelope_threshold <= 1.0):
        raise ValueError(
            f"envelope_threshold={envelope_threshold}, "
            f"expected ∈ (0, 1]"
        )

    pad_kernel = (kernel_support // 2) + 2
    if sigma_l_pix is not None and sigma_l_pix > 0:
        rad = float(sigma_l_pix) * math.sqrt(
            2.0 * math.log(1.0 / envelope_threshold)
        )
        pad_envelope = max(0, int(math.ceil((n_grid / 2.0) - rad)))
    else:
        pad_envelope = 0
    return max(pad_kernel, pad_envelope)


def compute_edge_mask(
    *,
    n_grid: int,
    kernel_support: int = 5,
    sigma_l_pix: Optional[float] = None,
    envelope_threshold: float = 0.5,
    drop_dc: bool = True,
    dtype: np.dtype = np.float32,
    apply_ifft2_dc_correction: bool = True,
) -> np.ndarray:
    """Return the multiplicative edge-mask, shape ``[N_grid, N_grid]``.

    Pixels in the active interior are 1.0; pixels in the outer
    ``image_mask_npad`` ring (and the DC cell when ``drop_dc=True``)
    are 0.0. Per plan §4.4 line 1515 the mask is applied before
    Layer-1 noise estimation and before the detector forward pass.

    The mask is constant per operating point; callers cache it once
    at startup and re-use across cubes.

    M7.4 ``apply_ifft2_dc_correction`` (2026-05-23): the gridder
    (:mod:`dsart.grid.sparsity_pattern`) writes ``V(u=0, v=0)`` at
    array index ``(N/2, N/2)`` — the natural radio-interferometry
    "DC at the center" convention. The imager pipeline then computes
    ``Re(fftshift(ifft2(uv_grid)))``, but ``ifft2`` assumes DC at
    index ``(0, 0)``. By the Fourier-shift theorem this introduces a
    per-pixel sign flip ``(-1)^(x+y)`` ("checkerboard") that
    ``fftshift`` does NOT undo (it's a translation, not a phase
    rotation; for even ``N=256`` the parity of ``(x+y)`` is
    preserved through ``fftshift``). The checkerboard manifests as
    the "strange positive/negative pattern" on dirty images and as
    a strong negative ``lag(0,1)`` / ``lag(1,0)`` spatial
    autocorrelation in the noise field (verified empirically:
    -0.64 at lag(0,1) on the 250924mptq replay, cube 8, t=60,
    fdm=33 — flips to +0.64 after correction). Multiplying the
    edge mask by ``(-1)^(i+j)`` undoes the checkerboard at zero
    added compute (the imager's edge-mask multiply already happens
    on every cube). Default ``True``; set ``False`` only to
    reproduce legacy pre-M7.4 imagery for regression studies.
    """
    npad = image_mask_npad(
        n_grid=n_grid,
        kernel_support=kernel_support,
        sigma_l_pix=sigma_l_pix,
        envelope_threshold=envelope_threshold,
    )
    mask = np.zeros((n_grid, n_grid), dtype=dtype)
    if 2 * npad < n_grid:
        mask[npad:n_grid - npad, npad:n_grid - npad] = 1.0
    if drop_dc:
        # Plan §4.4 line 1515: "The mask also excludes the single DC
        # cell at (N_grid/2, N_grid/2)" — an FFT residual that escapes
        # the static-sky IIR's per-uv-cell mean removal.
        dc = n_grid // 2
        if 0 <= dc < n_grid:
            mask[dc, dc] = 0.0
    if apply_ifft2_dc_correction:
        ii, jj = np.indices((n_grid, n_grid))
        # (-1)^(i+j) — equivalent to a per-pixel sign flip of every
        # odd-parity pixel. Encodes the missing ``ifftshift(uv)``
        # before ``ifft2`` (DC-at-center convention; see docstring).
        sign = (1.0 - 2.0 * ((ii + jj) & 1)).astype(dtype)
        mask = mask * sign
    return mask


# ---------------------------------------------------------------------------
# Dirty image
# ---------------------------------------------------------------------------


def _dirty_image_from_uv_grid_np(uv_grid: np.ndarray) -> np.ndarray:
    """Pure-numpy ``Re(iFFT2(uv_grid))`` + fftshift on the inner spatial
    axes; preserves any leading batch dims.
    """
    # iFFT operates on the last two axes regardless of leading shape.
    img = np.fft.ifft2(uv_grid, axes=(-2, -1))
    img = np.fft.fftshift(img, axes=(-2, -1))
    return np.real(img)


def _dirty_image_from_uv_grid_torch(uv_grid: torch.Tensor) -> torch.Tensor:
    img = torch.fft.ifft2(uv_grid, dim=(-2, -1))
    img = torch.fft.fftshift(img, dim=(-2, -1))
    return img.real


def dirty_image_from_uv_grid(
    uv_grid,
    *,
    out_dtype=None,
    prescale: float = 1.0,
):
    """Compute the canonical Stokes-I dirty image.

    Accepts numpy arrays (CPU; the chunk-6a unit-test path) or torch
    tensors (GPU; the chunk-6b services path that calls into a cached
    cuFFT plan via ``torch.fft.ifft2``). The single-side identity
    ``Re(iFFT2(V_pos)) = 0.5 · iFFT2(V_full)`` (§3.6.11) is what makes
    the +uv-only gridder lossless.

    Args:
        uv_grid: ``[..., N_grid, N_grid]`` complex array/tensor; the
            leading axes are pass-through (e.g.
            ``[head_block_samples, N_grid, N_grid] complex``).
        out_dtype: optional output dtype. Default keeps numpy as
            float32 and torch as the input's real-promote dtype
            (typically float32 if input is complex64, float16 if
            complex32 — but torch doesn't expose complex32 ifft, so
            we compute in complex64 and downcast).
        prescale: constant multiplied into ``uv_grid`` before the
            inverse FFT (the reference-path mirror of
            :attr:`dsart.image.imager_gpu.GpuImagerConfig.imager_uv_prescale`).
            The FFT is linear, so ``dirty_image_from_uv_grid(c·V) ==
            c·dirty_image_from_uv_grid(V)``; production uses ``c≈1/256``
            to keep the fp16 FFT butterflies in range for bright bursts.
            Default ``1.0`` skips the multiply (identity).

    Returns:
        Real ``[..., N_grid, N_grid]`` array/tensor of the same backend
        as ``uv_grid``.
    """
    if isinstance(uv_grid, torch.Tensor):
        # torch.fft.ifft2 requires complex64 or complex128. fp16 inputs
        # land here via the chunk-6b production path which casts to
        # complex64 once at the cuFFT plan boundary; we mirror that.
        if uv_grid.dtype == torch.complex32:
            uv_grid = uv_grid.to(torch.complex64)
        if prescale != 1.0:
            uv_grid = uv_grid * prescale
        out = _dirty_image_from_uv_grid_torch(uv_grid)
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out
    arr = np.asarray(uv_grid)
    if prescale != 1.0:
        arr = arr * prescale
    out = _dirty_image_from_uv_grid_np(arr)
    if out_dtype is not None:
        out = out.astype(out_dtype, copy=False)
    return out


def apply_edge_mask(
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
):
    """Apply the multiplicative edge mask to the (last-two-axes) image.

    Broadcasting rules: ``mask`` is ``[N_grid, N_grid]`` and broadcasts
    against any leading batch dims of ``image``.

    Returns the masked image with the same backend / dtype as ``image``
    (mask is upcast to match if necessary).
    """
    if isinstance(image, torch.Tensor):
        mask_t = (
            mask if isinstance(mask, torch.Tensor)
            else torch.from_numpy(np.asarray(mask))
        )
        return image * mask_t.to(dtype=image.dtype, device=image.device)
    arr = np.asarray(image)
    mask_arr = np.asarray(mask)
    return arr * mask_arr.astype(arr.dtype, copy=False)

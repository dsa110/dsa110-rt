"""1x-compat harness: the owner engine at today's parameters vs production.

Two pieces:

* :func:`production_cycle_cpu` -- today's search for ONE cube composed
  from the production functions that run without a GPU, in production
  order: the compact block -> ``build_inverse_lut`` /
  ``build_sparse_cell_csr`` (torch, CPU) -> a literal transcription of the
  ``sparse_combine_v4`` kernel (``combine.v4_emulate``) -> uv prescale ->
  irfft2 (float32; cuFFT-fp16 cannot run here -- its difference is the
  plan's FFT tolerance) -> fused edge mask x 1/sigma in fp16 ->
  ``_clamp_inf_to_finite`` -> 250-sigma clip -> fp16-output cumsum + the
  ``multi_boxcar_argmax`` arithmetic (fp32 difference x 1/s_k, strict ``>``,
  max stored fp16) -> ``decode_topk_argmax_lowmem`` ->
  ``merge_across_kernels_c1``. On a GPU node the checkpoint runner swaps in
  the real ``CubePipeline`` taps (``tests/x2`` ``x2_gpu`` tests).
* :func:`compare_candidates` -- the plan's candidate tolerances
  (t +- 1, fine DM +- 1, l/m +- 1, dS/N <= 0.1 for end-to-end; the K3b
  compat gate |dS/N| <= 0.02 is checked on matched maxima).

Also :func:`compact_block_to_rows` (today's compact block -> ring rows)
so the same recorded / synthetic 1x data feeds both chains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = [
    "ProductionCube",
    "production_cycle_cpu",
    "compact_block_to_rows",
    "compare_candidates",
]


@dataclass
class ProductionCube:
    half_spectrum: np.ndarray     # [F, T, N, nh, 2] fp16 (v4 output, pre-prescale)
    cube: np.ndarray              # [T, F, N, N] fp16, detector input (normalised, clipped)
    max_snr: np.ndarray           # [n_out, F, N, N] fp16
    winner: np.ndarray            # [n_out, F, N, N] int16
    candidates: list              # production Candidate list (post C1 merge)
    raw_candidates: list          # decoder output before the C1 merge


def compact_block_to_rows(cells: np.ndarray) -> np.ndarray:
    """``int8 [n_sb, t_rows, 2 * n_filled_max]`` -> ``uint8`` ring rows
    ``[n_sb, t_rows, 2 * n_filled_max]`` (the 1x ``cint8`` row format)."""
    return np.ascontiguousarray(cells).view(np.uint8)


def production_cycle_cpu(
    *,
    cells: np.ndarray,
    lut: np.ndarray,
    n_filled: np.ndarray,
    shifts: np.ndarray,
    pad_left: int,
    t_det: int,
    n_grid: int,
    sigma_inv: np.ndarray,
    s_k: np.ndarray,
    specnum_start: int,
    fine_dm: np.ndarray,
    widths=(1, 2, 4, 8, 16, 32, 64),
    prescale: float = 1.0 / 256.0,
    clip: float = 250.0,
    threshold: float = 9.55,
    n_top_base: int = 24,
    kernel_support: int = 1,
    c1_lm_max_cells: int = 5,
    c1_dm_max_trials: int = 2,
) -> ProductionCube:
    """One production cube on CPU (see module doc).

    Args:
        cells: compact block rows ``[specnum_start - pad_left, ...)``,
            ``int8 [n_sb, t_det + pad_left + pad_right, 2 * n_filled_max]``.
        shifts: the production table ``[F, n_sb]`` (NOT offset-adjusted).
        sigma_inv: ``[F]`` the 1/sigma the fused mask carries for this cube.
        s_k: ``[K]`` Layer-2 state used to decode this cube.
    """
    import torch

    from ...detector.decoder import decode_topk_argmax_lowmem
    from ...detector.merger import (DEFAULT_MERGE_RADIUS_FDM, DEFAULT_MERGE_RADIUS_LM,
                                    DEFAULT_MERGE_RADIUS_T, MergerConfig,
                                    merge_across_kernels_c1)
    from ...image.imager import compute_edge_mask
    from ...image.tiled_combine_cuda import build_inverse_lut, build_sparse_cell_csr, pad_t
    from .combine import v4_emulate

    n_sb, t_rows, pw = cells.shape
    nfm = pw // 2
    N = int(n_grid)
    nh = N // 2 + 1
    F = shifts.shape[0]
    # transpose_compact_tmajor
    dense = np.zeros((n_sb, nfm, pad_t(t_rows), 2), dtype=np.int8)
    dense[:, :, :t_rows] = cells.reshape(n_sb, t_rows, nfm, 2).transpose(0, 2, 1, 3)
    inv = build_inverse_lut(torch.from_numpy(np.ascontiguousarray(lut, dtype=np.int32)),
                            torch.from_numpy(np.ascontiguousarray(n_filled, dtype=np.int32)),
                            n_grid=N)
    csr = build_sparse_cell_csr(inv, n_grid=N)
    adj = (np.asarray(shifts, dtype=np.int32) - np.int32(pad_left)).astype(np.int32)
    half = v4_emulate(dense, csr, adj, t_rows=t_rows, t_out_len=t_det, t_lo=0, n_grid=N)
    uv = half.astype(np.float32)
    uv = (uv[..., 0] + 1j * uv[..., 1]) * np.float32(prescale)     # complex32 x 2^-8: exact
    uv16 = (uv.real.astype(np.float16).astype(np.float32)
            + 1j * uv.imag.astype(np.float16).astype(np.float32))
    img = np.fft.irfft2(uv16.astype(np.complex64), s=(N, N), axes=(-2, -1)).astype(np.float32)
    img16 = img.astype(np.float16)                                  # cuFFT-fp16 output dtype
    edge = compute_edge_mask(n_grid=N, kernel_support=kernel_support).astype(np.float16)
    mask_f = (edge[None] * np.asarray(sigma_inv, np.float32).astype(np.float16)[:, None, None])
    cube = (img16 * mask_f[:, None]).astype(np.float16)              # [F, T, N, N]
    cube = np.nan_to_num(cube, nan=0.0, posinf=60000.0, neginf=-60000.0)
    cube = np.clip(cube, np.float16(-clip), np.float16(clip)).astype(np.float16)
    cube = np.ascontiguousarray(np.moveaxis(cube, 0, 1))             # [T, F, N, N]
    # detector: fp16-output cumsum + multi_boxcar_argmax arithmetic
    maxw = int(max(widths))
    t_base = maxw // 2
    n_out = t_det - 2 * t_base
    cs = torch.zeros((t_det + maxw,) + cube.shape[1:], dtype=torch.float16)
    torch.cumsum(torch.from_numpy(cube), dim=0, dtype=torch.float16,
                 out=cs.narrow(0, 1 + maxw // 2, t_det))
    pr = maxw - 1 - maxw // 2
    if pr > 0:
        cs.narrow(0, 1 + maxw // 2 + t_det, pr).copy_(cs.narrow(0, maxw // 2 + t_det, 1))
    s_inv = (1.0 / np.maximum(np.asarray(s_k, np.float64), 1e-30)).astype(np.float32)
    best = None
    win = None
    for k, w in enumerate(widths):
        off = maxw // 2 - w // 2
        hi = cs.narrow(0, t_base + off + w, n_out).float()
        lo = cs.narrow(0, t_base + off, n_out).float()
        v = (hi - lo) * float(s_inv[k])
        if best is None:
            best = v
            win = torch.zeros(v.shape, dtype=torch.int16)
        else:
            b = v > best
            best = torch.where(b, v, best)
            win = torch.where(b, torch.full_like(win, k), win)
    max16 = best.to(torch.float16)
    import math
    n_top = max(n_top_base, int(math.ceil(n_top_base * t_det / n_out))) * len(widths)
    raw = decode_topk_argmax_lowmem(
        max16, win, threshold=threshold,
        kernel_ids=[f"unit:d1:b{w}" for w in widths], kernel_time_widths=list(widths),
        merge_radius_lm=DEFAULT_MERGE_RADIUS_LM, merge_radius_fdm=DEFAULT_MERGE_RADIUS_FDM,
        merge_radius_t=DEFAULT_MERGE_RADIUS_T, event_specnum=int(specnum_start) + t_base,
        fine_dm_pc_cm3=torch.as_tensor(np.asarray(fine_dm, np.float32)), n_top=n_top,
    )
    merged = merge_across_kernels_c1(raw, MergerConfig(lm_max_cells=c1_lm_max_cells,
                                                       dm_max_trials=c1_dm_max_trials))
    merged = [c for c in merged if float(c.snr) >= threshold]
    del F, nh
    return ProductionCube(half_spectrum=half, cube=cube, max_snr=max16.numpy(),
                          winner=win.numpy(), candidates=merged, raw_candidates=raw)


def compare_candidates(a: List, b: List, *, dt: int = 1, ddm: int = 1, dlm: int = 1,
                       dsnr: float = 0.1, dm_key=lambda c: c.dm_idx) -> dict:
    """Greedy one-to-one match of two candidate lists (objects with ``l, m,
    event_specnum, snr`` and ``dm_key``). Returns matched pairs, the
    unmatched of each side and the max |dS/N| over matches."""
    used = set()
    pairs = []
    for x in sorted(a, key=lambda c: -float(c.snr)):
        best: Optional[int] = None
        for j, y in enumerate(b):
            if j in used:
                continue
            if (abs(int(x.event_specnum) - int(y.event_specnum)) <= dt
                    and abs(int(dm_key(x)) - int(dm_key(y))) <= ddm
                    and abs(float(x.l) - float(y.l)) <= dlm
                    and abs(float(x.m) - float(y.m)) <= dlm
                    and abs(float(x.snr) - float(y.snr)) <= dsnr):
                best = j
                break
        if best is not None:
            used.add(best)
            pairs.append((x, b[best]))
    un_a = [x for x in a if all(x is not p[0] for p in pairs)]
    un_b = [y for j, y in enumerate(b) if j not in used]
    mx = max((abs(float(p[0].snr) - float(p[1].snr)) for p in pairs), default=0.0)
    return dict(pairs=pairs, unmatched_a=un_a, unmatched_b=un_b, max_dsnr=mx)

"""Fused HMMA-GEMM + post-GEMM Triton kernel for the M3 fast correlator.

Replaces *all* of ``corr_fast_kernel._compute_one_slab`` Stages 4-6
(2 stacked-K HMMA GEMMs + upper-tri gather + fp32 cast + Stokes-I
pol-sum + chan-sum + permute + complex64 combine) with one Triton
kernel that goes ``(A_re, A_im_b)`` fp16 → ``(vis_real, vis_imag)``
fp32 ``(n_fv, NBASE, NCHAN_eff)`` directly.

Key wins vs the Phase-8 PyTorch path:
  * **Skip the (B, 96, 96) fp16 cube write+read** — the V_real /
    V_imag intermediates are kept in registers/shared memory only.
    At the production op-point this saves ~864 MB write + ~864 MB
    read per slab × 16 slabs = ~28 GB / block of avoided memory
    traffic (~46 ms at the 2080Ti's 600 GB/s peak).
  * **Skip the 5 PyTorch ops** (gather → cast → sum → reshape-sum →
    permute → complex). They become 1 fp32 store per output cell
    inside the kernel.
  * **Each (fv, c_eff, baseline) output is written once, not
    atomic-summed** — we accumulate the (CSF * NVP = 16) batches
    in the kernel's fp32 register accumulator before storing.

Math (matches ``corr_fast_kernel`` Stage 4-6 fuse_stokes_i path):

    For each (fv, c_eff, bls) with bls(a, b) = a*(a+1)/2 + b, a >= b:
      vis[fv, bls, c_eff] = sum_{c_inner=0..CSF-1, pol=0..NVP-1}
                               V_x[fv, c_eff*CSF + c_inner, pol, b, a]
        where V_real[..., b, a] = (R^T R + I^T I)[..., b, a]
              V_imag[..., b, a] = (R^T I - I^T R)[..., b, a]

A single tl.dot tile is 16x16x16 fp16 → fp32 accumulate, which is
exactly one Turing HMMA tile. NANTS=96 maps to 6 M-tiles × 6 N-tiles
of which 21 are upper-tri (b ≤ a in V's m,n coords). The kernel
skips lower-tri tiles entirely.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_corr_post_kernel(
    A_re_ptr, A_im_b_ptr,                       # fp16 (B, K, NANTS)
    out_re_ptr, out_im_ptr,                     # fp32 (n_fv, NBASE, NCHAN_EFF)
    n_fv,
    NCHAN: tl.constexpr,                        # 384
    NCHAN_EFF: tl.constexpr,                    # 48
    NANTS: tl.constexpr,                        # 96
    NBASE: tl.constexpr,                        # 4656
    NVP: tl.constexpr,                          # 2 (nvolt_pol)
    CSF: tl.constexpr,                          # 8 (chan_sum_factor)
    K: tl.constexpr,                            # 16 (stacked-K)
    A_stride_b, A_stride_k, A_stride_n,
    OUT_stride_fv, OUT_stride_bls, OUT_stride_c,
    BLOCK_M: tl.constexpr,                      # = 16 (HMMA tile)
    BLOCK_N: tl.constexpr,                      # = 16 (HMMA tile)
):
    """One program → one (fv, c_eff) × (M-tile, N-tile) chunk of outputs.

    Grid: (n_fv, NCHAN_EFF, M_TILES * N_TILES) where M_TILES=N_TILES=NANTS/BLOCK.
    Programs with M-tile > N-tile (lower-tri in V's (m,n) coords) early-return.
    Diagonal tiles mask out the strict-lower-tri half (m > n).

    Per program:
      * Iterate over CSF * NVP batches in this c_eff group.
      * For each batch, compute one (BLOCK_M, BLOCK_N) HMMA tile of
        V_real and V_imag in fp32 register accumulators.
      * After the batch loop, store the accumulated (BLOCK_M, BLOCK_N)
        outputs to the correct upper-tri positions in
        ``out_re / out_im[fv, bls, c_eff]`` — one fp32 store per cell,
        no atomicAdds.
    """
    pid_fv = tl.program_id(0)
    pid_ce = tl.program_id(1)
    pid_mn = tl.program_id(2)

    M_TILES: tl.constexpr = NANTS // BLOCK_M
    N_TILES: tl.constexpr = NANTS // BLOCK_N
    pid_m = pid_mn // N_TILES
    pid_n = pid_mn %  N_TILES
    if pid_m > pid_n:
        return                                   # pure lower-tri → skip

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)            # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # (BLOCK_N,)
    offs_k = tl.arange(0, K)                                    # (K,)
    mask_m = offs_m < NANTS
    mask_n = offs_n < NANTS

    # fp32 accumulators (in registers)
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over (CSF * NVP) batches in this c_eff group
    for ci in tl.static_range(CSF):
        ch = pid_ce * CSF + ci
        for p in tl.static_range(NVP):
            batch = (pid_fv * NCHAN + ch) * NVP + p

            base_re = A_re_ptr  + batch * A_stride_b
            base_im = A_im_b_ptr + batch * A_stride_b

            # A_re[batch, :, M] as (BLOCK_M, K) for tl.dot left operand.
            ptrs_A_M = (base_re
                        + offs_m[:, None] * A_stride_n
                        + offs_k[None, :] * A_stride_k)
            # A_re[batch, :, N] as (K, BLOCK_N) for tl.dot right operand.
            ptrs_A_N = (base_re
                        + offs_k[:, None] * A_stride_k
                        + offs_n[None, :] * A_stride_n)
            # A_im_b[batch, :, N] as (K, BLOCK_N)
            ptrs_AIM_N = (base_im
                          + offs_k[:, None] * A_stride_k
                          + offs_n[None, :] * A_stride_n)

            A_M  = tl.load(ptrs_A_M,   mask=mask_m[:, None], other=0.0)   # (M, K)
            A_N  = tl.load(ptrs_A_N,   mask=mask_n[None, :], other=0.0)   # (K, N)
            AI_N = tl.load(ptrs_AIM_N, mask=mask_n[None, :], other=0.0)   # (K, N)

            # HMMA: A_re_M^T (M, K) @ A_re_N (K, N) → (M, N) fp32
            # Note: the data is (m, k) and (k, n) so a regular tl.dot
            # gives V[m, n] = sum_k A_re[m, k] * A_re[k, n] which is
            # what we want IF the K-dim of A_re is iterated. Recall
            # the upstream input is (batch, K, NANTS) with the K-dim
            # being the GEMM reduction. The formula
            #   V[m, n] = sum_k A_re[batch, k, m] * A_re[batch, k, n]
            # is satisfied by loading A_M = A_re[:, m] as (BLOCK_M, K)
            # (transposed at load via offs_m on the N-stride and
            # offs_k on the K-stride) and A_N = A_re[:, n] as (K, BLOCK_N).
            acc_re += tl.dot(A_M, A_N,  out_dtype=tl.float32)
            acc_im += tl.dot(A_M, AI_N, out_dtype=tl.float32)

    # ---- Scatter to upper-tri output positions ----
    # V[m, n] with m <= n maps to bls = n*(n+1)/2 + m
    # (legacy gather V_real_b[..., b_idx, a_idx] with b <= a means
    # the gather reads V[..., b, a] for bls(a, b) = a*(a+1)/2 + b;
    # in (m, n) coords this is m=b, n=a, so m <= n.)
    upper_mask = offs_m[:, None] <= offs_n[None, :]
    valid_mask = mask_m[:, None] & mask_n[None, :] & upper_mask

    n2d = offs_n[None, :]
    m2d = offs_m[:, None]
    bls_idx = n2d * (n2d + 1) // 2 + m2d                        # (BLOCK_M, BLOCK_N)

    out_offs = (pid_fv * OUT_stride_fv
                + bls_idx * OUT_stride_bls
                + pid_ce * OUT_stride_c)

    tl.store(out_re_ptr + out_offs, acc_re, mask=valid_mask)
    tl.store(out_im_ptr + out_offs, acc_im, mask=valid_mask)


def fused_corr_post_triton(
    A_re: torch.Tensor,                           # fp16 (B, K, NANTS)
    A_im_b: torch.Tensor,                         # fp16 (B, K, NANTS)
    *,
    n_fv: int,
    nchan: int,
    nchan_eff: int,
    nvp: int,
    csf: int,
    nbase: int,
    BLOCK_M: int = 16,
    BLOCK_N: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused HMMA + post-GEMM kernel.

    Returns (vis_real, vis_imag) of shape (n_fv, NBASE, NCHAN_EFF) fp32.
    Caller is responsible for the upstream stacked-K-pair construction
    (``A_re = cat([R, I])``, ``A_im_b = cat([I, -R])``) — same as
    Phase-8 production today.
    """
    assert A_re.is_contiguous() and A_im_b.is_contiguous()
    assert A_re.dtype == torch.float16 and A_im_b.dtype == torch.float16
    assert A_re.shape == A_im_b.shape
    nants = A_re.shape[2]
    K = A_re.shape[1]
    assert nants % BLOCK_M == 0 and nants % BLOCK_N == 0
    M_TILES = nants // BLOCK_M
    N_TILES = nants // BLOCK_N

    out_re = torch.empty(
        (n_fv, nbase, nchan_eff),
        dtype=torch.float32, device=A_re.device,
    )
    out_im = torch.empty_like(out_re)

    grid = (n_fv, nchan_eff, M_TILES * N_TILES)
    _fused_corr_post_kernel[grid](
        A_re, A_im_b,
        out_re, out_im,
        n_fv,
        nchan, nchan_eff, nants, nbase, nvp, csf, K,
        A_re.stride(0), A_re.stride(1), A_re.stride(2),
        out_re.stride(0), out_re.stride(1), out_re.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out_re, out_im

"""Microbench + correctness check for the fused HMMA + post-GEMM Triton kernel.

Compares against the Phase-8 PyTorch reference path (stacked-K pair
GEMM + upper-tri gather + Stokes-I + chan-sum) for ONE slab worth
of inputs at the production op-point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import NANTS, NBASE, NCHAN_PER_CHGROUP, NPOL  # noqa: E402
from dsart.services.slow_corr_kernel import upper_tri_indices  # noqa: E402
from dsart.services.triton_corr_fused import fused_corr_post_triton  # noqa: E402


def reference_pytorch(
    A_re: torch.Tensor,                       # fp16 (B, K, N)
    A_im_b: torch.Tensor,                     # fp16 (B, K, N)
    *,
    n_fv: int,
    nchan: int,
    nchan_eff: int,
    nvp: int,
    csf: int,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
):
    """Mirror of corr_fast_kernel._compute_one_slab Stages 4-6 fuse_stokes_i."""
    nants = A_re.shape[2]
    nbase = a_idx.shape[0]

    # Stage 4: 2 stacked-K HMMA GEMMs
    V_real = torch.matmul(A_re.transpose(-1, -2), A_re)            # (B, N, N) fp16
    V_imag = torch.matmul(A_re.transpose(-1, -2), A_im_b)

    # Stage 5: view as 5D (fv, ch, pol, N, N) — caller passes contiguous A
    # so this is a free view.
    V_real_5d = V_real.view(n_fv, nchan, nvp, nants, nants)
    V_imag_5d = V_imag.view(n_fv, nchan, nvp, nants, nants)

    # Stage 6: gather upper-tri (b ≤ a convention via _b_idx, _a_idx)
    vis_real_fp16 = V_real_5d[..., b_idx, a_idx]                    # (fv, ch, npol, NBASE) fp16
    vis_imag_fp16 = V_imag_5d[..., b_idx, a_idx]
    vis_real = vis_real_fp16.to(torch.float32).sum(dim=2)            # (fv, ch, NBASE) fp32
    vis_imag = vis_imag_fp16.to(torch.float32).sum(dim=2)
    vis_real = vis_real.reshape(n_fv, nchan_eff, csf, nbase).sum(dim=2)
    vis_imag = vis_imag.reshape(n_fv, nchan_eff, csf, nbase).sum(dim=2)
    vis_real = vis_real.permute(0, 2, 1).contiguous()                # (fv, NBASE, NCHAN_EFF)
    vis_imag = vis_imag.permute(0, 2, 1).contiguous()
    return vis_real, vis_imag


def main():
    device = torch.device("cuda")
    n_fv = 32
    nchan = NCHAN_PER_CHGROUP
    csf = 8
    nchan_eff = nchan // csf
    nants = NANTS
    nvp = NPOL                                  # nbada_pol == nvolt_pol == 2 in production
    K_combined = 8                              # NTIMES_PER_PACKET * packets_per_fv at t_int=8
    K = 2 * K_combined                          # 16, after stacked-K-pair
    nbase = nants * (nants + 1) // 2

    a_idx_np, b_idx_np = upper_tri_indices(nants)
    a_idx = torch.from_numpy(a_idx_np).to(device)
    b_idx = torch.from_numpy(b_idx_np).to(device)

    g = torch.Generator(device=device).manual_seed(0xBADD15)
    B = n_fv * nchan * nvp
    R = (torch.rand(B, K_combined, nants, generator=g, device=device,
                     dtype=torch.float16) * 2 - 1).contiguous()
    I = (torch.rand(B, K_combined, nants, generator=g, device=device,
                     dtype=torch.float16) * 2 - 1).contiguous()

    # Stacked-K-pair construction (matches Phase 8 production)
    A_re   = torch.cat([R,  I], dim=1).contiguous()                  # (B, K=16, N)
    A_im_b = torch.cat([I, -R], dim=1).contiguous()
    print(f"A_re shape: {tuple(A_re.shape)} fp16  = "
          f"{A_re.numel() * 2 / (1024**2):.1f} MB each")
    print(f"V cube (skipped by fused kernel): "
          f"{B * nants * nants * 2 / (1024**2):.1f} MB each (real+imag)")

    # ---------------- reference ----------------
    print("running reference PyTorch (Phase 8 path)...")
    ref_re, ref_im = reference_pytorch(
        A_re, A_im_b,
        n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
        nvp=nvp, csf=csf,
        a_idx=a_idx, b_idx=b_idx,
    )
    print(f"  ref shape: {tuple(ref_re.shape)}")

    # ---------------- triton ----------------
    print("running fused HMMA Triton kernel...")
    tri_re, tri_im = fused_corr_post_triton(
        A_re, A_im_b,
        n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
        nvp=nvp, csf=csf, nbase=nbase,
    )
    print(f"  tri shape: {tuple(tri_re.shape)}")

    # ---- correctness ----
    diff_r = (ref_re - tri_re).abs().max().item()
    diff_i = (ref_im - tri_im).abs().max().item()
    rel_r = diff_r / ref_re.abs().max().item()
    rel_i = diff_i / ref_im.abs().max().item()
    print(f"max |Δ| vs ref: real={diff_r:.3e}  imag={diff_i:.3e}  "
          f"(rel real={rel_r:.3e} imag={rel_i:.3e})")

    if max(rel_r, rel_i) > 1e-2:
        print("CORRECTNESS FAILURE — investigate before benchmarking.")
        return

    # ---------------- bench ----------------
    n_warm, n_iter = 5, 50
    def _ev(): return torch.cuda.Event(enable_timing=True)

    print("\nbench (per-slab, multiply by 16 slabs to get per-block):")

    # reference
    for _ in range(n_warm):
        reference_pytorch(A_re, A_im_b,
                          n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
                          nvp=nvp, csf=csf, a_idx=a_idx, b_idx=b_idx)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        e0, e1 = _ev(), _ev()
        e0.record()
        reference_pytorch(A_re, A_im_b,
                          n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
                          nvp=nvp, csf=csf, a_idx=a_idx, b_idx=b_idx)
        e1.record(); e1.synchronize()
        times.append(e0.elapsed_time(e1))
    p50_ref = float(np.median(times))
    print(f"  REFERENCE (Phase 8 PyTorch):  per-slab p50 = {p50_ref:.2f} ms  "
          f"(× 16 = {p50_ref*16:.0f} ms / block)")

    # triton — sweep BLOCK_M/N just to check
    for blk in [16]:
        for _ in range(n_warm):
            fused_corr_post_triton(A_re, A_im_b,
                                    n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
                                    nvp=nvp, csf=csf, nbase=nbase,
                                    BLOCK_M=blk, BLOCK_N=blk)
        torch.cuda.synchronize()
        times_t = []
        for _ in range(n_iter):
            e0, e1 = _ev(), _ev()
            e0.record()
            fused_corr_post_triton(A_re, A_im_b,
                                    n_fv=n_fv, nchan=nchan, nchan_eff=nchan_eff,
                                    nvp=nvp, csf=csf, nbase=nbase,
                                    BLOCK_M=blk, BLOCK_N=blk)
            e1.record(); e1.synchronize()
            times_t.append(e0.elapsed_time(e1))
        p50 = float(np.median(times_t))
        print(f"  TRITON FUSED BLOCK={blk:2d}: per-slab p50 = {p50:.2f} ms  "
              f"(× 16 = {p50*16:.0f} ms / block)  "
              f"speedup vs ref = {p50_ref/p50:.2f}x")


if __name__ == "__main__":
    main()

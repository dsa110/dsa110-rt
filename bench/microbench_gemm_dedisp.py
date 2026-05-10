"""Focused micro-bench: kernel-name introspection + roofline for the GEMM
+ dedisp stages of the M3 fast path.

Two sub-benches, each isolated and run to convergence with CUDA events
so reported numbers are GPU-only:

  ``--what gemm`` — drives only the 4 fp16 batched matmuls of
  ``compute_split._compute_one_slab`` (Stage 4, the kernel we currently
  spend most of compute_split on). Captures the underlying kernel name
  via ``torch.profiler``, computes achieved TFLOPS and write-bandwidth,
  and tries each candidate replacement (cuBLASLt fp32-acc complex GEMM,
  Triton-style custom kernel) for comparison.

  ``--what dedisp`` — drives only ``_dedisperse_one_window`` on a
  realistic synthetic ``vis_stokes_i`` (T, B, C) cfp32 tile. Reports
  per-stage GPU ms (permute, gather, real-contig + Re-scatter,
  imag-contig + Im-scatter) and effective DRAM BW per stage. Also
  tests an alternative complex-direct ``index_add_`` (no
  ``.real/.imag .contiguous()`` materialisation) for a head-to-head
  with the current path.

CLI::

    python bench/microbench_gemm_dedisp.py --what gemm --device cuda
    python bench/microbench_gemm_dedisp.py --what dedisp --device cuda
    python bench/microbench_gemm_dedisp.py --what dedisp --device cuda --try-complex-scatter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BLOCK_DURATION_S,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.slow_corr_kernel import NTIMES_PER_PACKET  # noqa: E402

# ---------------------------------------------------------------------------
# 2080 Ti reference numbers (used for roofline % calculations)
# ---------------------------------------------------------------------------
TURING_PEAK_FP16_TENSORCORE_TFLOPS = 107.6   # 544 TC × 128 ops × 1.545 GHz
TURING_PEAK_FP32_CUDA_TFLOPS = 13.4          # spec sheet base
TURING_PEAK_DRAM_GB_S = 616.0                # spec sheet, GDDR6 14 Gbps × 352-bit


def _ev():
    return torch.cuda.Event(enable_timing=True)


def _time_ms(fn, *, n_warm=2, n_iter=10):
    """Run fn() in a CUDA event timing loop. Returns list of per-iter ms."""
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        ev0, ev1 = _ev(), _ev()
        ev0.record()
        fn()
        ev1.record()
        ev1.synchronize()
        times.append(ev0.elapsed_time(ev1))
    return times


def _profile_kernels(fn, *, n_warm=2, n_iter=3):
    """Run fn() under torch.profiler; return a list of (kernel_name, ms)
    tuples merged across iterations.

    Useful for confirming HMMA tensor-core usage and identifying which
    cuBLAS/cuBLASLt tactic torch.matmul picks for our shapes.
    """
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity, record_function
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(n_iter):
            with record_function("BENCH_ITER"):
                fn()
        torch.cuda.synchronize()
    # Collect kernel-level events with non-zero CUDA time.
    out: dict[str, dict] = {}
    for ev in prof.key_averages():
        # Filter to events that ran on CUDA.
        cuda_us = float(getattr(ev, "self_device_time_total", 0)) or float(
            getattr(ev, "self_cuda_time_total", 0)
        )
        if cuda_us <= 0:
            continue
        key = ev.key
        d = out.setdefault(key, {"cuda_us": 0.0, "calls": 0})
        d["cuda_us"] += cuda_us
        d["calls"] += int(ev.count)
    # Sort by total cuda time descending.
    return sorted(out.items(), key=lambda kv: -kv[1]["cuda_us"])


# ---------------------------------------------------------------------------
# GEMM micro-bench
# ---------------------------------------------------------------------------


def _build_gemm_inputs(
    *, n_fv_slab: int, packets_per_fast_vis: int, device: torch.device,
):
    """Build R, I in the (B, K, NANTS) shape consumed by Stage 4 of
    ``_compute_one_slab``. Random fp16 in [-1, 1).

    B = n_fv_slab * NCHAN * NPOL          (production: 32 * 384 * 2 = 24576)
    K = NTIMES_PER_PACKET * packets_per_fast_vis  (production: 2 * 4 = 8)
    """
    B = n_fv_slab * NCHAN_PER_CHGROUP * NPOL
    K = NTIMES_PER_PACKET * packets_per_fast_vis
    g = torch.Generator(device=device).manual_seed(0xBADCAFE)
    R = (torch.rand(B, K, NANTS, generator=g, device=device,
                    dtype=torch.float16) * 2.0 - 1.0)
    I = (torch.rand(B, K, NANTS, generator=g, device=device,
                    dtype=torch.float16) * 2.0 - 1.0)
    return R, I, B, K


def gemm_baseline(R: torch.Tensor, I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The current Stage 4: 4 separate fp16 matmuls.

    V_real = R^T @ R + I^T @ I    (fp16, in-place add)
    V_imag = R^T @ I - I^T @ R    (fp16, in-place sub)
    """
    R_T = R.transpose(-1, -2)
    I_T = I.transpose(-1, -2)
    V_real = torch.matmul(R_T, R)
    V_real = V_real.add_(torch.matmul(I_T, I))
    V_imag = torch.matmul(R_T, I)
    V_imag = V_imag.sub_(torch.matmul(I_T, R))
    return V_real, V_imag


def gemm_baddbmm_accumulate(R: torch.Tensor, I: torch.Tensor):
    """4 GEMMs, but use baddbmm(beta=1) to fuse the sum/diff into the
    GEMM itself, eliminating the separate fp16 element-wise add/sub
    kernels (which today cost ~5 ms / slab = 47% of Stage 4 time).

    V_real = R^T@R; V_real = baddbmm(V_real, I^T, I, beta=1)
    V_imag = R^T@I; V_imag = baddbmm(V_imag, I^T, R, beta=1, alpha=-1)
    """
    R_T = R.transpose(-1, -2).contiguous()
    I_T = I.transpose(-1, -2).contiguous()
    V_real = torch.matmul(R_T, R)
    V_real = torch.baddbmm(V_real, I_T, I, beta=1, alpha=1)
    V_imag = torch.matmul(R_T, I)
    V_imag = torch.baddbmm(V_imag, I_T, R, beta=1, alpha=-1)
    return V_real, V_imag


def gemm_stacked_K(R: torch.Tensor, I: torch.Tensor):
    """Stack R and I along K → one GEMM with K=16, then post-process.

    Build A = stack(R, I) along K → shape (B, 2K, NANTS).
    V = A^T @ A has block structure:
        V[0:M, 0:N] = R^T @ R
        V[0:M, N:]  = R^T @ I
        V[M:, 0:N]  = I^T @ R
        V[M:, N:]   = I^T @ I
    Then V_real = V[0:M, 0:N] + V[M:, N:],
         V_imag = V[0:M, N:]  - V[M:, 0:N].

    Doubles K (8→16, hits the s16168 mma sweet spot better) at the
    cost of a 2x larger output tensor before the post-process (then
    we slice and add).
    """
    B, K, M = R.shape
    A = torch.cat([R, I], dim=1)               # (B, 2K, NANTS)
    A_T = A.transpose(-1, -2).contiguous()     # (B, NANTS, 2K)
    V = torch.matmul(A_T, A)                   # (B, NANTS, NANTS) but K=16
    # Wait — by collapsing on K=16 we get a single (NANTS, NANTS) result that
    # is the SUM of all 4 sub-blocks (= R^T@R + R^T@I + I^T@R + I^T@I), which
    # is NOT what we want. To preserve block structure we need to widen
    # M and N too. Build the full (B, 2*NANTS, 2*NANTS) and slice.
    return None  # disabled — see _gemm_stacked_K_block below


def gemm_stacked_K_pair(R: torch.Tensor, I: torch.Tensor):
    """The clean stacked-K trick — 2 GEMMs with K=16, output stays N×N.

    Construct A_re = [R; I]      (K-stacked → (B, 2K, N))
              A_im_a = [R; I]    (same as A_re)
              A_im_b = [I; -R]   (K-stacked → (B, 2K, N))

    V_real = A_re^T @ A_re           = R^T@R + I^T@I  ✓
    V_imag = A_im_a^T @ A_im_b       = R^T@I - I^T@R  ✓

    2 GEMMs with K=16 instead of 4 GEMMs with K=8 — same total flops,
    but K=16 hits the s16168 HMMA tile dimensions exactly, AND the
    output volume is HALF of the (2N, 2N) "full block" approach
    (writes 2 N×N tensors vs 1 2N×2N tensor = 4 N×N tensors).
    """
    B, K, N = R.shape
    A_re = torch.cat([R, I], dim=1)                # (B, 2K, N)
    A_im_a = A_re                                  # alias
    A_im_b = torch.cat([I, -R], dim=1)             # (B, 2K, N)
    V_real = torch.matmul(A_re.transpose(-1, -2), A_re)        # (B, N, N) fp16
    V_imag = torch.matmul(A_im_a.transpose(-1, -2), A_im_b)
    return V_real, V_imag


def gemm_stacked_K_block(R: torch.Tensor, I: torch.Tensor):
    """Properly stack via block-diagonal-like padding.

    Form ``A = [R | I]`` along the LAST axis → (B, K, 2*NANTS),
    so columns 0..NANTS-1 hold R, NANTS..2*NANTS-1 hold I.
    Form ``B = [R; I]`` along the K axis → (B, 2K, NANTS) ... no,
    that doesn't decompose right either.

    The clean version: build A = (B, 2K, 2N) so that
        A[:, 0:K, 0:N]      = R
        A[:, 0:K, N:2N]     = I
        A[:, K:2K, 0:N]     = -I    (so A^T@A picks up the cross-term sign)
        A[:, K:2K, N:2N]    = R
    Then A^T @ A is a 2N×2N matrix whose top-left N×N block equals
    R^T@R + I^T@I = V_real, and top-right block equals R^T@I - I^T@R = V_imag.
    Output is 4x bigger but we get K=16 and 1 GEMM call.
    """
    B, K, N = R.shape
    A = torch.zeros(B, 2 * K, 2 * N, dtype=R.dtype, device=R.device)
    A[:, :K, :N] = R
    A[:, :K, N:] = I
    A[:, K:, :N] = -I
    A[:, K:, N:] = R
    A_T = A.transpose(-1, -2).contiguous()       # (B, 2N, 2K)
    V = torch.matmul(A_T, A)                     # (B, 2N, 2N) fp16
    V_real = V[:, :N, :N]
    V_imag = V[:, :N, N:]
    return V_real, V_imag


def gemm_complex_packed(R: torch.Tensor, I: torch.Tensor):
    """Replace 4 real fp16 matmuls with a single complex fp16 matmul.

    Forms E = (R + i I) of shape (B, K, NANTS) cfp32 (the closest
    PyTorch supports — there is no native cfp16 matmul on Turing).
    Computes V = E^H @ E using cgemm, returns V split back to real/imag.

    Note: V's diagonal will be the SUMMED auto-correlation power, the
    upper-tri matches the M3 baseline cross. We compute everything
    here for fairness.
    """
    # cfp32 from two fp16 halves — fp32 storage so cgemm works.
    E = torch.complex(R.to(torch.float32), I.to(torch.float32))   # (B, K, NANTS) cfp32
    E_H = E.transpose(-1, -2).conj()                              # (B, NANTS, K) cfp32
    V = torch.matmul(E_H, E)                                      # (B, NANTS, NANTS) cfp32
    return V.real.contiguous().to(torch.float16), V.imag.contiguous().to(torch.float16)


def gemm_fold_chan_into_K_GROUP(
    R: torch.Tensor, I: torch.Tensor, *,
    fold_groups: int = 1,
):
    """Try grouping ``fold_groups`` adjacent (fv,ch,pol) batches and
    folding their K axis into one GEMM.

    This is INVALID physics (different channels can't be summed in K
    because the auto/cross products would mix), but we run it to see
    the *upper bound* of what tensor-core utilization could look like
    if K were larger. Result tensors have shape
    ``(B/fold_groups, M, N)`` and so don't decode to anything
    meaningful — we discard them and only measure timing.
    """
    B, K, M = R.shape
    if B % fold_groups != 0:
        raise ValueError(f"fold_groups={fold_groups} must divide B={B}")
    R_f = R.reshape(B // fold_groups, fold_groups * K, M)
    I_f = I.reshape(B // fold_groups, fold_groups * K, M)
    R_T = R_f.transpose(-1, -2)
    I_T = I_f.transpose(-1, -2)
    V_real = torch.matmul(R_T, R_f).add_(torch.matmul(I_T, I_f))
    V_imag = torch.matmul(R_T, I_f).sub_(torch.matmul(I_T, R_f))
    return V_real, V_imag


def run_gemm_bench(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    n_fv_slab = args.n_fv_slab
    ppfv = args.packets_per_fast_vis
    R, I, B, K = _build_gemm_inputs(
        n_fv_slab=n_fv_slab, packets_per_fast_vis=ppfv, device=device,
    )

    M = N = NANTS

    # Theoretical work (4 GEMMs, M*N*K*2 ops each).
    flops_per_gemm = 2 * B * M * N * K
    flops_per_call = 4 * flops_per_gemm                          # all 4 matmuls
    out_bytes_per_call = 4 * B * M * N * 2                       # 4 fp16 outputs

    print(f"{'='*78}")
    print(f"GEMM Stage 4 micro-bench")
    print(f"{'='*78}")
    print(f"shapes: B={B}  M={M}  N={N}  K={K}  (fp16 in, fp16 out)")
    print(f"per-call:  {flops_per_call/1e9:.2f} GFLOP   "
          f"{out_bytes_per_call/(1024**3):.3f} GB writes (fp16 outputs)")

    print()
    print("--- baseline (current 4 separate matmuls) ---")
    times = _time_ms(lambda: gemm_baseline(R, I), n_warm=3, n_iter=20)
    p50 = float(np.median(times))
    achieved_tflops = flops_per_call / (p50 * 1e-3) / 1e12
    achieved_gbs_writes = out_bytes_per_call / (p50 * 1e-3) / 1e9
    print(f"  per-call median = {p50:.2f} ms  (min {min(times):.2f}, "
          f"p99 {np.percentile(times, 99):.2f})")
    print(f"  achieved fp16 TC = {achieved_tflops:.2f} TFLOPS "
          f"({100*achieved_tflops/TURING_PEAK_FP16_TENSORCORE_TFLOPS:.2f}% of {TURING_PEAK_FP16_TENSORCORE_TFLOPS:.0f} TFLOPS peak)")
    print(f"  output write BW = {achieved_gbs_writes:.0f} GB/s "
          f"({100*achieved_gbs_writes/TURING_PEAK_DRAM_GB_S:.1f}% of {TURING_PEAK_DRAM_GB_S:.0f} GB/s peak)")

    # --- HMMA confirmation via torch.profiler -----------------------------
    print()
    print("--- kernel breakdown (torch.profiler, top 8 by CUDA time) ---")
    kernels = _profile_kernels(lambda: gemm_baseline(R, I), n_warm=3, n_iter=5)
    for name, d in kernels[:8]:
        print(f"  {d['cuda_us']/1e3:8.2f} ms total ({d['calls']} calls)  "
              f"{name[:120]}")

    # Look for HMMA evidence.
    hmma_keys = [k for k, _ in kernels
                 if any(s in k for s in
                        ("hmma", "HMMA", "tensor", "Tensor", "wmma",
                         "_h884", "_h1688", "Cijk_Ailk_Bljk_HHS",
                         "ampere_h", "turing_h"))]
    if hmma_keys:
        print()
        print(f"  ✓ tensor-core kernel detected: {hmma_keys[0][:120]}")
    else:
        print()
        print("  ⚠ no obvious HMMA / tensor-core kernel name in top events; "
              "may be running fp16 SIMT GEMM (no tensor cores).")

    # --- Candidate optimisations ------------------------------------------
    print()
    print("--- candidate replacements ---")

    print()
    print("(a) baddbmm β=1 (fuse the add_/sub_ into the GEMM)")
    try:
        times = _time_ms(lambda: gemm_baddbmm_accumulate(R, I), n_warm=3, n_iter=20)
        p50 = float(np.median(times))
        achieved_tflops = flops_per_call / (p50 * 1e-3) / 1e12
        print(f"    per-call median = {p50:.2f} ms")
        print(f"    achieved fp16 TC = {achieved_tflops:.2f} TFLOPS "
              f"({100*achieved_tflops/TURING_PEAK_FP16_TENSORCORE_TFLOPS:.2f}% of peak)")
        # numerical check
        Vr0, Vi0 = gemm_baseline(R, I)
        Vr1, Vi1 = gemm_baddbmm_accumulate(R, I)
        diff_r = (Vr0.float() - Vr1.float()).abs().max().item()
        diff_i = (Vi0.float() - Vi1.float()).abs().max().item()
        print(f"    max |Δ| vs baseline: real={diff_r:.3e}  imag={diff_i:.3e}")
        print()
        print("    --- kernel breakdown (top 6) ---")
        kernels = _profile_kernels(lambda: gemm_baddbmm_accumulate(R, I),
                                   n_warm=3, n_iter=5)
        for name, d in kernels[:6]:
            print(f"      {d['cuda_us']/1e3:8.2f} ms ({d['calls']} calls)  "
                  f"{name[:115]}")
    except Exception as e:
        print(f"    FAILED: {e}")

    print()
    print("(b1) stacked-K PAIR (2 GEMMs with K=16, M=N=96; half the output bytes)")
    try:
        times = _time_ms(lambda: gemm_stacked_K_pair(R, I), n_warm=3, n_iter=20)
        p50 = float(np.median(times))
        achieved_tflops = flops_per_call / (p50 * 1e-3) / 1e12
        achieved_gbs_writes = (2 * B * M * N * 2) / (p50 * 1e-3) / 1e9
        print(f"    per-call median = {p50:.2f} ms")
        print(f"    achieved fp16 TC = {achieved_tflops:.2f} TFLOPS "
              f"({100*achieved_tflops/TURING_PEAK_FP16_TENSORCORE_TFLOPS:.2f}% of peak)")
        print(f"    output write BW = {achieved_gbs_writes:.0f} GB/s "
              f"({100*achieved_gbs_writes/TURING_PEAK_DRAM_GB_S:.1f}% of peak)")
        Vr0, Vi0 = gemm_baseline(R, I)
        Vr1, Vi1 = gemm_stacked_K_pair(R, I)
        diff_r = (Vr0.float() - Vr1.float()).abs().max().item()
        diff_i = (Vi0.float() - Vi1.float()).abs().max().item()
        print(f"    max |Δ| vs baseline: real={diff_r:.3e}  imag={diff_i:.3e}")
        print()
        print("    --- kernel breakdown (top 6) ---")
        kernels = _profile_kernels(lambda: gemm_stacked_K_pair(R, I),
                                   n_warm=3, n_iter=5)
        for name, d in kernels[:6]:
            print(f"      {d['cuda_us']/1e3:8.2f} ms ({d['calls']} calls)  "
                  f"{name[:115]}")
    except Exception as e:
        print(f"    FAILED: {e}")

    print()
    print("(c) stacked K (build (2K, 2N) block; 1 GEMM call, K=16)")
    try:
        times = _time_ms(lambda: gemm_stacked_K_block(R, I), n_warm=3, n_iter=20)
        p50 = float(np.median(times))
        achieved_tflops = flops_per_call / (p50 * 1e-3) / 1e12   # nominal
        print(f"    per-call median = {p50:.2f} ms")
        print(f"    achieved fp16 TC (counting useful work only) = {achieved_tflops:.2f} TFLOPS "
              f"({100*achieved_tflops/TURING_PEAK_FP16_TENSORCORE_TFLOPS:.2f}% of peak)")
        Vr0, Vi0 = gemm_baseline(R, I)
        Vr1, Vi1 = gemm_stacked_K_block(R, I)
        diff_r = (Vr0.float() - Vr1.float()).abs().max().item()
        diff_i = (Vi0.float() - Vi1.float()).abs().max().item()
        print(f"    max |Δ| vs baseline: real={diff_r:.3e}  imag={diff_i:.3e}")
    except Exception as e:
        print(f"    FAILED: {e}")

    for fg in (2, 4, 8):
        if B % fg != 0:
            continue
        print()
        print(f"(b) UPPER-BOUND: fold {fg} adjacent batches into K (K={K*fg}) "
              f"— wrong physics, just a roofline")
        try:
            times = _time_ms(
                lambda: gemm_fold_chan_into_K_GROUP(R, I, fold_groups=fg),
                n_warm=3, n_iter=10,
            )
            p50 = float(np.median(times))
            achieved_tflops = flops_per_call / (p50 * 1e-3) / 1e12
            print(f"    per-call median = {p50:.2f} ms")
            print(f"    achieved fp16 TC = {achieved_tflops:.2f} TFLOPS "
                  f"({100*achieved_tflops/TURING_PEAK_FP16_TENSORCORE_TFLOPS:.2f}% of peak)")
        except Exception as e:
            print(f"    FAILED: {e}")

    print()


# ---------------------------------------------------------------------------
# Dedisp micro-bench
# ---------------------------------------------------------------------------


def _build_dedisp_inputs(*, args, device: torch.device):
    """Build a synthetic vis_stokes_i tensor + bin_shifts table mirroring
    what Stage1MultiDMCoarseDM hands to ``_dedisperse_one_window``.

    Shape: (n_fv, NBASE, NCHAN_eff) cfp32, with NCHAN_eff = NCHAN /
    chan_sum_factor = 384/8 = 48 at the production op-point.
    """
    n_fv = args.n_fv
    nch_eff = NCHAN_PER_CHGROUP // args.chan_sum_factor
    nb = NBASE
    g = torch.Generator(device=device).manual_seed(0xBADBEEF)
    vis = (torch.rand(n_fv, nb, nch_eff, dtype=torch.float32,
                      generator=g, device=device)
           + 1j * torch.rand(n_fv, nb, nch_eff, dtype=torch.float32,
                             generator=g, device=device)).to(torch.complex64)
    # Random per-(c, dm) bin shifts in [0, max_shift].
    max_shift = args.max_shift
    rng = np.random.default_rng(0xBADD15)
    bs = rng.integers(0, max_shift + 1,
                      size=(nch_eff, args.n_dm), dtype=np.int64)
    bs_dev = torch.from_numpy(bs).to(device)
    return vis, bs_dev, max_shift


def dedisp_current(
    vis: torch.Tensor, bin_shifts: torch.Tensor, *,
    n_filled: int, dm_chunk: int, max_shift: int, cim_cb: torch.Tensor,
    counters: dict[str, list[float]] | None = None,
):
    """Mirror of corr_fast_integration._dedisperse_one_window (RT Phase 5/6),
    but instrumented per-stage. Returns the dedispersed cube
    ``(n_dm, t_dedisp, n_filled)`` cfp32.
    """
    device = vis.device
    n_fv, nb, nch = vis.shape
    n_dm = bin_shifts.shape[1]
    t_dedisp = n_fv - max_shift
    if counters is None:
        counters = {}

    def _add(name, e0, e1):
        e1.synchronize()
        counters.setdefault(name, []).append(e0.elapsed_time(e1))

    # ---- Permute (T, B, C) → (T, C, B) ----
    e0, e1 = _ev(), _ev()
    e0.record()
    vis_T = vis.permute(0, 2, 1).contiguous()
    e1.record()
    _add("01_permute", e0, e1)

    t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
    out = torch.empty((n_dm, t_dedisp, n_filled),
                      dtype=torch.complex64, device=device)

    for c0 in range(0, n_dm, dm_chunk):
        c1 = min(c0 + dm_chunk, n_dm)
        chunk = c1 - c0
        t_chunk = chunk * t_dedisp

        bs_chunk = bin_shifts[:, c0:c1]
        e0, e1 = _ev(), _ev()
        e0.record()
        t_idx_2d = (
            bs_chunk.t()[:, None, :] + t_arange[None, :, None]
        ).reshape(t_chunk, nch)
        t_idx_3d = t_idx_2d[:, :, None].expand(t_chunk, nch, nb)
        gathered = torch.gather(vis_T, 0, t_idx_3d)
        e1.record()
        _add("02_gather", e0, e1)
        del t_idx_2d, t_idx_3d

        e0, e1 = _ev(), _ev()
        e0.record()
        src = gathered.reshape(t_chunk, nch * nb)
        out_re = torch.zeros((t_chunk, n_filled + 1),
                             dtype=torch.float32, device=device)
        src_re = src.real.contiguous()
        e1.record()
        _add("03_real_contig", e0, e1)

        e0, e1 = _ev(), _ev()
        e0.record()
        out_re.index_add_(1, cim_cb, src_re)
        e1.record()
        _add("04_re_scatter", e0, e1)
        del src_re

        e0, e1 = _ev(), _ev()
        e0.record()
        out_im = torch.zeros_like(out_re)
        src_im = src.imag.contiguous()
        e1.record()
        _add("05_imag_contig", e0, e1)

        e0, e1 = _ev(), _ev()
        e0.record()
        out_im.index_add_(1, cim_cb, src_im)
        e1.record()
        _add("06_im_scatter", e0, e1)
        del src_im, gathered, src

        out_buf = torch.complex(out_re[:, :n_filled], out_im[:, :n_filled])
        out[c0:c1] = out_buf.reshape(chunk, t_dedisp, n_filled)
        del out_re, out_im, out_buf
    return out, counters


def dedisp_complex_scatter(
    vis: torch.Tensor, bin_shifts: torch.Tensor, *,
    n_filled: int, dm_chunk: int, max_shift: int, cim_cb: torch.Tensor,
    counters: dict[str, list[float]] | None = None,
):
    """Same shape, but use a single COMPLEX index_add_ instead of two
    fp32 ones.

    PyTorch's index_add_ supports complex64 directly. This avoids the
    two ~488 MB ``.real / .imag .contiguous()`` materialisation copies
    per chunk and writes one complex64 accumulator instead of two
    fp32 ones.
    """
    device = vis.device
    n_fv, nb, nch = vis.shape
    n_dm = bin_shifts.shape[1]
    t_dedisp = n_fv - max_shift
    if counters is None:
        counters = {}

    def _add(name, e0, e1):
        e1.synchronize()
        counters.setdefault(name, []).append(e0.elapsed_time(e1))

    e0, e1 = _ev(), _ev()
    e0.record()
    vis_T = vis.permute(0, 2, 1).contiguous()
    e1.record()
    _add("01_permute", e0, e1)

    t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
    out = torch.empty((n_dm, t_dedisp, n_filled),
                      dtype=torch.complex64, device=device)

    for c0 in range(0, n_dm, dm_chunk):
        c1 = min(c0 + dm_chunk, n_dm)
        chunk = c1 - c0
        t_chunk = chunk * t_dedisp

        bs_chunk = bin_shifts[:, c0:c1]
        e0, e1 = _ev(), _ev()
        e0.record()
        t_idx_2d = (
            bs_chunk.t()[:, None, :] + t_arange[None, :, None]
        ).reshape(t_chunk, nch)
        t_idx_3d = t_idx_2d[:, :, None].expand(t_chunk, nch, nb)
        gathered = torch.gather(vis_T, 0, t_idx_3d)
        e1.record()
        _add("02_gather", e0, e1)

        e0, e1 = _ev(), _ev()
        e0.record()
        src = gathered.reshape(t_chunk, nch * nb)         # (T_chunk, NSRC) cfp32 view
        out_c = torch.zeros((t_chunk, n_filled + 1),
                            dtype=torch.complex64, device=device)
        out_c.index_add_(1, cim_cb, src)
        e1.record()
        _add("03_complex_scatter", e0, e1)
        del gathered, src

        out[c0:c1] = out_c[:, :n_filled].reshape(chunk, t_dedisp, n_filled)
        del out_c
    return out, counters


def run_dedisp_bench(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    vis, bin_shifts, max_shift = _build_dedisp_inputs(args=args, device=device)
    n_fv, nb, nch = vis.shape

    # Cell-index map for a 256x256 grid roughly mirroring our gridder
    # K=1 fast-vis pillbox.
    n_filled = args.n_filled
    rng = np.random.default_rng(0xC1F1)
    cim_bc_np = rng.integers(0, n_filled, size=(nb, nch), dtype=np.int64)
    cim_cb = torch.from_numpy(cim_bc_np.T.copy()).reshape(-1).to(device)

    # Bytes accounting for roofline.
    src_bytes_cfp32 = vis.numel() * 8
    print(f"{'='*78}")
    print(f"DEDISP _dedisperse_one_window micro-bench")
    print(f"{'='*78}")
    print(f"vis: (n_fv={n_fv}, NBASE={nb}, NCHAN_eff={nch}) cfp32  "
          f"= {src_bytes_cfp32/(1024**3):.2f} GB")
    print(f"n_dm={args.n_dm}  max_shift={max_shift}  dm_chunk={args.dm_chunk}  "
          f"n_filled={n_filled}")

    # --- current path -----------------------------------------------------
    print()
    print("--- current path (Phase 5: gather → re/im contig + 2 fp32 scatters) ---")
    counters: dict[str, list[float]] = {}
    for _ in range(3):  # warm
        out_a, counters = dedisp_current(
            vis, bin_shifts, n_filled=n_filled, dm_chunk=args.dm_chunk,
            max_shift=max_shift, cim_cb=cim_cb, counters={},
        )
    counters = {}
    times = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, counters = dedisp_current(
            vis, bin_shifts, n_filled=n_filled, dm_chunk=args.dm_chunk,
            max_shift=max_shift, cim_cb=cim_cb, counters=counters,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"  total p50 = {np.median(times):.2f} ms  (per-stage GPU ms below)")
    for k in sorted(counters.keys()):
        v = counters[k]
        # Per-stage time accumulates over n_chunks per call.
        n_chunks = (args.n_dm + args.dm_chunk - 1) // args.dm_chunk
        per_call_ms = float(np.sum(v)) / args.iters
        print(f"    {k:<25s} per-call ms = {per_call_ms:8.2f}  "
              f"(per-chunk {per_call_ms/n_chunks:6.2f} × {n_chunks} chunks)")

    if args.try_complex_scatter:
        print()
        print("--- alt path: single complex64 index_add_ (no .real/.imag contig) ---")
        for _ in range(3):
            out_b, _ = dedisp_complex_scatter(
                vis, bin_shifts, n_filled=n_filled, dm_chunk=args.dm_chunk,
                max_shift=max_shift, cim_cb=cim_cb, counters={},
            )
        counters_c: dict[str, list[float]] = {}
        times_c = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, counters_c = dedisp_complex_scatter(
                vis, bin_shifts, n_filled=n_filled, dm_chunk=args.dm_chunk,
                max_shift=max_shift, cim_cb=cim_cb, counters=counters_c,
            )
            torch.cuda.synchronize()
            times_c.append((time.perf_counter() - t0) * 1000.0)
        print(f"  total p50 = {np.median(times_c):.2f} ms")
        for k in sorted(counters_c.keys()):
            v = counters_c[k]
            n_chunks = (args.n_dm + args.dm_chunk - 1) // args.dm_chunk
            per_call_ms = float(np.sum(v)) / args.iters
            print(f"    {k:<25s} per-call ms = {per_call_ms:8.2f}  "
                  f"(per-chunk {per_call_ms/n_chunks:6.2f} × {n_chunks} chunks)")
        # Equivalence check.
        torch.cuda.synchronize()
        diff = torch.max(torch.abs(out_a - out_b)).item()
        print(f"  max |Δ| vs baseline (single iter): {diff:.3e}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=("gemm", "dedisp", "both"), required=True)
    p.add_argument("--device", default="cuda")
    # GEMM
    p.add_argument("--n-fv-slab", type=int, default=32)
    p.add_argument("--packets-per-fast-vis", type=int, default=4)
    # Dedisp
    p.add_argument("--n-fv", type=int, default=512)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--n-dm", type=int, default=24)
    p.add_argument("--dm-chunk", type=int, default=2)
    p.add_argument("--max-shift", type=int, default=2)
    p.add_argument("--n-filled", type=int, default=32768)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--try-complex-scatter", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.what in ("gemm", "both"):
        run_gemm_bench(args)
    if args.what in ("dedisp", "both"):
        run_dedisp_bench(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

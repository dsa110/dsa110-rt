"""compute_split internals breakdown — STANDALONE.

Constructs a FastCorrKernel + per-slab synthetic voltage inputs at the
production op-point and runs the per-slab body
(``_compute_one_slab``) once per measurement, timing each stage with
cuda.synchronize + perf_counter.

Skips the full process_block path — we only want to know where the
time inside compute_split lives, broken down by stage:

  A_slab_alloc   — voltage-slab .contiguous() before passing to slab
  B_stage2_perm  — Stage 2: permute(3,0,1,2,4,5).contiguous() (R + I)
  C_stage4_gemm  — Stage 4: 4 batched fp16 matmuls (V_real + V_imag)
  D_stage5_sum   — Stage 5: sum-over-t_sub + cast to fp32 (R + I)
  E_stage6_pack  — Stage 6: gather + permute + Stokes-I + chan-sum + complex

Production op-point: t_int_fast_native=8 (ppfv=4), n_fv_chunk=32 (auto)
gives 16 slabs/block. Per slab batch = 32 * 384 * 2 * 2 = 49152
batched (96,4) @ (96,4) → (96,96) fp16 GEMMs.

CLI::

    python bench/profile_compute_split_internals.py \\
        --report-dir /tmp/p7-corr-internals \\
        --warmup 2 --n-blocks 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dsart.common.constants import NANTS, NBASE, NCHAN_PER_CHGROUP
from dsart.services.corr_fast_kernel import (
    FastCorrKernel,
    _F31A_CHUNK_TARGET_BYTES,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)

LOG = logging.getLogger("profile_compute_split_internals")

NVOLT_POL = 2
NBADA_POL = 2


def _stamp_sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _profile_one_slab(
    kernel: FastCorrKernel,
    real_v: torch.Tensor,
    imag_v: torch.Tensor,
    n_fv_chunk: int,
    chan_sum_factor: int,
    counters: dict[str, list[float]],
) -> torch.Tensor:
    """Run compute_split with stage-level timers; return final cube."""

    def _add(name: str, ms: float) -> None:
        counters.setdefault(name, []).append(ms)

    n_fast_vis = real_v.shape[3] // (kernel.t_int_fast_native // NTIMES_PER_PACKET)
    packets_per_fast_vis = real_v.shape[3] // n_fast_vis
    n_fv_slab = n_fv_chunk
    nchan_eff = kernel.nchan // chan_sum_factor

    # Allocate output once (production fast path)
    out_full = torch.empty(
        (n_fast_vis, NBASE, nchan_eff),
        dtype=torch.complex64, device=kernel.device,
    )

    for fv0 in range(0, n_fast_vis, n_fv_chunk):
        fv1 = min(fv0 + n_fv_chunk, n_fast_vis)
        n_fv_slab_actual = fv1 - fv0
        p0 = fv0 * packets_per_fast_vis
        p1 = fv1 * packets_per_fast_vis

        t = _stamp_sync()

        # A. slab .contiguous() (slice + materialise)
        real_slab = real_v[:, :, :, p0:p1, :].contiguous()
        imag_slab = imag_v[:, :, :, p0:p1, :].contiguous()
        t1 = _stamp_sync(); _add("A_slab_contig", (t1 - t) * 1000); t = t1

        # ---- inline _compute_one_slab with stage timers ----

        # Stage 1: views (free)
        R5 = real_slab.view(
            kernel.nchan, NTIMES_PER_PACKET, kernel.nvolt_pol,
            n_fv_slab_actual, packets_per_fast_vis, kernel.nants,
        )
        I5 = imag_slab.view(
            kernel.nchan, NTIMES_PER_PACKET, kernel.nvolt_pol,
            n_fv_slab_actual, packets_per_fast_vis, kernel.nants,
        )

        # Stage 2: permute + contiguous
        R6 = R5.permute(3, 0, 1, 2, 4, 5).contiguous()
        I6 = I5.permute(3, 0, 1, 2, 4, 5).contiguous()
        del R5, I5
        t1 = _stamp_sync(); _add("B_stage2_perm", (t1 - t) * 1000); t = t1

        # Stage 3: reshape into batched
        new_batch = (
            n_fv_slab_actual * kernel.nchan
            * NTIMES_PER_PACKET * kernel.nvolt_pol
        )
        R = R6.reshape(new_batch, packets_per_fast_vis, kernel.nants)
        I = I6.reshape(new_batch, packets_per_fast_vis, kernel.nants)
        del R6, I6

        # Stage 4: 4 batched fp16 GEMMs
        R_T = R.transpose(-1, -2)
        I_T = I.transpose(-1, -2)
        V_real = torch.matmul(R_T, R)
        V_real = V_real.add_(torch.matmul(I_T, I))
        V_imag = torch.matmul(R_T, I)
        V_imag = V_imag.sub_(torch.matmul(I_T, R))
        del R, I, R_T, I_T
        t1 = _stamp_sync(); _add("C_stage4_gemm", (t1 - t) * 1000); t = t1

        # Stage 5: sum over t_sub + cast to fp32
        V_real_6d = V_real.view(
            n_fv_slab_actual, kernel.nchan, NTIMES_PER_PACKET,
            kernel.nvolt_pol, kernel.nants, kernel.nants,
        )
        V_imag_6d = V_imag.view(
            n_fv_slab_actual, kernel.nchan, NTIMES_PER_PACKET,
            kernel.nvolt_pol, kernel.nants, kernel.nants,
        )
        V_real = V_real_6d.sum(dim=2).to(kernel.accum_dtype)
        V_imag = V_imag_6d.sum(dim=2).to(kernel.accum_dtype)
        del V_real_6d, V_imag_6d
        V_real_b = V_real[:, :, :kernel.nbada_pol, :, :]
        V_imag_b = V_imag[:, :, :kernel.nbada_pol, :, :]
        t1 = _stamp_sync(); _add("D_stage5_sum", (t1 - t) * 1000); t = t1

        # Stage 6: gather + permute + Stokes-I + chan-sum + complex
        vis_real = V_real_b[..., kernel._b_idx, kernel._a_idx]
        vis_imag = V_imag_b[..., kernel._b_idx, kernel._a_idx]
        vis_real = vis_real.permute(0, 3, 1, 2).contiguous()
        vis_imag = vis_imag.permute(0, 3, 1, 2).contiguous()
        # Stokes-I sum over BADA pols
        vis_real = vis_real.sum(dim=-1)
        vis_imag = vis_imag.sum(dim=-1)
        # chan-sum
        if chan_sum_factor > 1:
            vis_real = vis_real.reshape(
                n_fv_slab_actual, NBASE, nchan_eff, chan_sum_factor,
            ).sum(dim=-1)
            vis_imag = vis_imag.reshape(
                n_fv_slab_actual, NBASE, nchan_eff, chan_sum_factor,
            ).sum(dim=-1)
        out_full[fv0:fv1] = torch.complex(vis_real, vis_imag)
        del vis_real, vis_imag, V_real, V_imag, V_real_b, V_imag_b
        t1 = _stamp_sync(); _add("E_stage6_pack", (t1 - t) * 1000); t = t1

        del real_slab, imag_slab

    return out_full


def _synth_voltage_block(
    kernel: FastCorrKernel,
    n_fast_vis: int,
    packets_per_fast_vis: int,
    seed: int = 20260508,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic int4-unpacked fp16 voltage block at production
    op-point shape (NCHAN, 2t, 2p, n_packets, NANTS).
    """
    n_packets = n_fast_vis * packets_per_fast_vis
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # int4 unpacked range is [-8, 7]. Use uniform over that range, fp16.
    real_v = torch.randint(
        -8, 8, (kernel.nchan, NTIMES_PER_PACKET, kernel.nvolt_pol,
                n_packets, kernel.nants), generator=g,
    ).to(torch.float16).to(kernel.device)
    imag_v = torch.randint(
        -8, 8, (kernel.nchan, NTIMES_PER_PACKET, kernel.nvolt_pol,
                n_packets, kernel.nants), generator=g,
    ).to(torch.float16).to(kernel.device)
    return real_v, imag_v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--t-int-fast-native", type=int, default=8)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--n-fv-chunk", type=int, default=None,
                   help="None = auto-pick (production behavior)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        LOG.error("CUDA required"); return 2
    device = torch.device("cuda")

    kernel = FastCorrKernel(
        device=device,
        t_int_fast_native=args.t_int_fast_native,
        nants=NANTS,
        nchan=NCHAN_PER_CHGROUP,
        nvolt_pol=NVOLT_POL,
        nbada_pol=NBADA_POL,
        accum_dtype=torch.float32,
    )

    packets_per_fast_vis = args.t_int_fast_native // NTIMES_PER_PACKET
    n_fast_vis = NPACKETS_PER_BLOCK // packets_per_fast_vis
    LOG.info("ready: t_int=%d ppfv=%d n_fv_per_block=%d",
             args.t_int_fast_native, packets_per_fast_vis, n_fast_vis)

    real_v, imag_v = _synth_voltage_block(
        kernel, n_fast_vis, packets_per_fast_vis,
    )
    voltage_gb = (real_v.numel() * 2 + imag_v.numel() * 2) / 1e9
    LOG.info("voltage block fp16: real+imag = %.2f GB", voltage_gb)

    n_fv_chunk = args.n_fv_chunk
    if n_fv_chunk is None:
        n_fv_chunk = kernel._auto_n_fv_chunk(n_fast_vis)
    LOG.info("n_fv_chunk=%d → %d slabs/block",
             n_fv_chunk, n_fast_vis // n_fv_chunk)

    counters_throwaway: dict = {}
    for _ in range(args.warmup):
        _profile_one_slab(
            kernel, real_v, imag_v, n_fv_chunk, args.chan_sum_factor,
            counters_throwaway,
        )

    counters: dict = {}
    LOG.info("warmup done; profiling %d blocks", args.n_blocks)
    for _ in range(args.n_blocks):
        _profile_one_slab(
            kernel, real_v, imag_v, n_fv_chunk, args.chan_sum_factor,
            counters,
        )

    n_blocks = args.n_blocks
    summary: dict = {}
    print(f"\ncompute_split internal breakdown ({n_blocks} blocks, "
          f"{n_fast_vis // n_fv_chunk} slabs/block)")
    print(f"{'phase':<24} {'mean ms / slab':>15} "
          f"{'slabs / block':>14} {'sum ms / block':>15}")
    print("-" * 72)
    for k in sorted(counters.keys()):
        v = np.asarray(counters[k])
        sum_per_block = v.sum() / n_blocks
        n_calls_per_block = len(v) / n_blocks
        mean_per_call = v.mean()
        summary[k] = {
            "n_calls_total": len(v),
            "n_calls_per_block": float(n_calls_per_block),
            "mean_ms_per_call": float(mean_per_call),
            "sum_ms_per_block": float(sum_per_block),
        }
        print(f"{k:<24} {mean_per_call:>15.3f} "
              f"{n_calls_per_block:>14.1f} {sum_per_block:>15.2f}")
    print("-" * 72)
    total = sum(s["sum_ms_per_block"] for s in summary.values())
    print(f"{'TOTAL':<24} {'':>15} {'':>14} {total:>15.2f}")

    out_path = args.report_dir / "compute_split_internals.json"
    out_path.write_text(json.dumps({
        "config": vars(args),
        "n_fv_chunk": int(n_fv_chunk),
        "summary": summary,
    }, indent=2, default=str))
    LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

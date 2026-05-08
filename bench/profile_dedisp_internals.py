"""RT Phase 5 dedisp internal-timing breakdown — STANDALONE.

Constructs a Stage1MultiDMCoarseDM at the production op-point + a
synthetic vis_stokes_i tensor and calls the dedisp inner method
directly, timing each phase with cuda.synchronize + perf_counter.

Skips the full process_block path (which requires fada raw bytes +
unpacking + GEMM) — we only want to know where the time inside
_dedisperse_one_window lives.

Phases (per call):

  A. permute_vis    — vis_stokes_i.permute(2, 0, 1).contiguous() [once]
  B. setup_indices  — bin_shifts to GPU + t_arange + cim_cb       [once]
  C_gather_chunk    — torch.gather (C, T_chunk, B) per dm_chunk   [N_chunks]
  D_scatter_chunk   — per-channel index_add                       [N_chunks]

CLI::

    python -m bench.profile_dedisp_internals \\
        --report-dir bench/reports/RT-phase5-dedisp-internals \\
        --warmup 3 --n-blocks 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dsart.common.constants import (
    NANTS,
    NATIVE_SAMPLE_US,
    NBASE,
    NCHAN_PER_CHGROUP,
)
from dsart.coarse_dm.dm_plan import DMPlan
from dsart.grid.kernel import FastVisGridder
from dsart.grid.sparsity_pattern import build_pattern
from dsart.services.corr_fast_integration import (
    Stage1MultiDMCoarseDM,
    _build_core_baseline_mask,
    compute_top_of_band_cell_lambda,
)
from bench.fast_path_throughput import (
    _build_synthetic_summed_plan,
    _synth_antpos,
)


LOG = logging.getLogger("profile_dedisp_internals")


def _stamp_sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def _profile_one_call(
    stage1: Stage1MultiDMCoarseDM,
    vis_joined: torch.Tensor,
    counters: dict,
) -> torch.Tensor:
    """Re-implements _dedisperse_one_window inline with per-phase timers."""
    n_fv = int(vis_joined.shape[0])
    t_dedisp = stage1.t_dedisp_for(n_fv)
    n_filled = int(stage1.gridder.pattern.n_filled)
    nb = int(vis_joined.shape[1])
    nch = int(vis_joined.shape[2])
    device = vis_joined.device

    def _add(name, ms):
        counters.setdefault(name, []).append(ms)

    t = _stamp_sync()

    # A. Permute vis to (C, T, B)
    vis_T = vis_joined.permute(2, 0, 1).contiguous()
    t1 = _stamp_sync(); _add("A_permute_vis", (t1 - t) * 1000); t = t1

    # B. Setup indices
    bin_shifts_full = stage1.plan.delay_bins_per_chgroup(stage1.chgroup)
    bin_shifts = bin_shifts_full[:nch, stage1._dm_idx_iter]
    bin_shifts_dev = torch.as_tensor(
        bin_shifts, dtype=torch.int64, device=device,
    )
    t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
    t1 = _stamp_sync(); _add("B_setup_indices", (t1 - t) * 1000); t = t1

    out = torch.empty(
        (stage1.n_dm, t_dedisp, n_filled),
        dtype=torch.complex64, device=device,
    )
    dm_chunk = max(1, int(getattr(stage1, "dm_chunk_size", 2)))
    dm_chunk = min(dm_chunk, stage1.n_dm)

    for c0 in range(0, stage1.n_dm, dm_chunk):
        c1 = min(c0 + dm_chunk, stage1.n_dm)
        chunk = c1 - c0

        # C. Gather chunk
        bs_chunk = bin_shifts_dev[:, c0:c1]
        t_idx = (
            bs_chunk[:, :, None] + t_arange[None, None, :]
        ).reshape(nch, chunk * t_dedisp)
        t_idx_b = t_idx[:, :, None].expand(nch, chunk * t_dedisp, nb)
        gathered = torch.gather(vis_T, 1, t_idx_b)
        t1 = _stamp_sync(); _add("C_gather_chunk", (t1 - t) * 1000); t = t1

        # D. Transpose gather output back to (T_chunk, B, C)
        buf = gathered.permute(1, 2, 0).contiguous()
        t1 = _stamp_sync(); _add("D_transpose_back", (t1 - t) * 1000); t = t1

        # E. Inline single-batch scatter (view_as_real, no .real/.imag copies)
        cell_index_map = stage1.gridder.cell_index_map
        t_chunk = chunk * t_dedisp
        src_re = torch.view_as_real(buf.reshape(t_chunk, nb * nch))
        out_re = torch.zeros(
            (t_chunk, n_filled + 1, 2),
            dtype=torch.float32, device=device,
        )
        out_re.index_add_(1, cell_index_map, src_re)
        out_buf = torch.view_as_complex(
            out_re[:, :n_filled, :].contiguous()
        )
        out[c0:c1] = out_buf.reshape(chunk, t_dedisp, n_filled)
        t1 = _stamp_sync(); _add("E_inline_scatter", (t1 - t) * 1000); t = t1

        del bs_chunk, t_idx, t_idx_b, gathered, buf, src_re, out_re, out_buf

    del vis_T, bin_shifts_dev, t_arange
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--t-int-fast-native", type=int, default=8)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-coarse-dm", type=int, default=24)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--dm-truth", type=float, default=1500.0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        LOG.error("CUDA required"); return 2
    device = torch.device("cuda")

    plan = _build_synthetic_summed_plan(
        n_coarse=args.n_coarse_dm,
        dm_max=2.0 * args.dm_truth,
        chan_sum_factor=args.chan_sum_factor,
        t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
    )

    antpos_e, antpos_n = _synth_antpos(seed=42)
    # bench/profile_fast_path_K1.py builds the production context WITHOUT
    # is_core_baseline_mask — the cell_lambda is sized to the longest
    # baseline including outriggers (96-ant set). To match its op-point
    # exactly (cell_lambda~148 → n_filled=460) we pass mask=None too.
    cell_lambda_common = compute_top_of_band_cell_lambda(
        antpos_e, antpos_n, n_grid=args.n_grid,
        is_core_baseline_mask=None,
    )
    LOG.info("common cell_lambda = %.4g (matches production op-point)",
             cell_lambda_common)
    pattern = build_pattern(
        antpos_e=antpos_e, antpos_n=antpos_n,
        chgroup=0, dec_deg=53.85, n_grid=args.n_grid,
        kernel_support=1, chan_sum_factor=args.chan_sum_factor,
        cell_lambda=cell_lambda_common,
        is_core_baseline_mask=None,
    )
    gridder = FastVisGridder.from_pattern(
        pattern, antpos_e, antpos_n,
        is_core_baseline_mask=None,
        device=device,
    )
    stage1 = Stage1MultiDMCoarseDM(
        plan=plan, gridder=gridder, chgroup=0,
        sliding_window=True,  # sliding window joins prev+curr
    )
    LOG.info("ready: n_filled=%d n_dm=%d n_grid=%d chan_sum=%d",
             pattern.n_filled, stage1.n_dm, args.n_grid,
             args.chan_sum_factor)

    # Build a synthetic joined-block vis tensor at the production op-
    # point shape: (2 * n_fv_per_block, NBASE, NCHAN_eff)
    from dsart.services.slow_corr_kernel import (
        NPACKETS_PER_BLOCK, NTIMES_PER_PACKET,
    )
    n_fv_per_block = (
        NPACKETS_PER_BLOCK * NTIMES_PER_PACKET // args.t_int_fast_native
    )
    n_fv_joined = 2 * n_fv_per_block
    nch_eff = NCHAN_PER_CHGROUP // args.chan_sum_factor
    LOG.info("joined vis shape: (%d, %d, %d) cfp32 = %.2f GB",
             n_fv_joined, NBASE, nch_eff,
             n_fv_joined * NBASE * nch_eff * 8 / 1e9)

    torch.manual_seed(20260508)
    vis_joined = torch.complex(
        torch.randn(n_fv_joined, NBASE, nch_eff),
        torch.randn(n_fv_joined, NBASE, nch_eff),
    ).to(device)

    # Warmup (no recording).
    counters_throwaway: dict = {}
    for _ in range(args.warmup):
        _profile_one_call(stage1, vis_joined, counters_throwaway)

    # Measurements.
    counters: dict = {}
    LOG.info("warmup done; profiling %d calls", args.n_blocks)
    for _ in range(args.n_blocks):
        _profile_one_call(stage1, vis_joined, counters)

    # Aggregate.
    n_blocks = args.n_blocks
    summary: dict = {}
    print(f"\nPhase 5 dedisp internal breakdown ({n_blocks} blocks)")
    print(f"{'phase':<24} {'mean ms / call':>15} {'calls / block':>14} "
          f"{'sum ms / block':>15}")
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
        print(f"{k:<24} {mean_per_call:>15.3f} {n_calls_per_block:>14.1f} "
              f"{sum_per_block:>15.2f}")
    total = sum(s["sum_ms_per_block"] for s in summary.values())
    print("-" * 72)
    print(f"{'TOTAL':<24} {' ':>15} {' ':>14} {total:>15.2f}")

    out = {
        "config": {
            "t_int_fast_native": args.t_int_fast_native,
            "n_grid": args.n_grid,
            "n_coarse_dm": args.n_coarse_dm,
            "chan_sum_factor": args.chan_sum_factor,
            "kernel_support": 1,
            "n_filled": int(pattern.n_filled),
            "n_fv_per_block": n_fv_per_block,
            "n_fv_joined": n_fv_joined,
        },
        "n_blocks": n_blocks,
        "phase_breakdown": summary,
        "total_ms_per_block": total,
    }
    (args.report_dir / "dedisp_internals.json").write_text(
        json.dumps(out, indent=2),
    )
    LOG.info("wrote %s", args.report_dir / "dedisp_internals.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

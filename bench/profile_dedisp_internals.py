"""RT Phase 5 dedisp internal-timing breakdown.

Drills into the new ``_dedisperse_one_window`` (m3/rt-phase5) and times
the three internal phases per chunk:

  A. permute_vis    — vis_stokes_i.permute(2, 0, 1).contiguous()  [once / call]
  B. gather_chunk   — torch.gather of (C, T_chunk, B) per dm_chunk  [once / chunk]
  C. scatter_chunk  — per-channel index_add_ × NCHAN_eff per chunk  [NCHAN_eff / chunk]
  D. host_overhead  — Python dispatch + tensor view ops            [residual]

Production op-point only: t_int_fast_native=8, n_grid=256,
n_coarse_dm=24, chan_sum_factor=8, sliding_window=True, K=1 pillbox.

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

from dsart.coarse_dm.dm_plan import DMPlan
from dsart.common.constants import (
    NANTS,
    NATIVE_SAMPLE_US,
    NBASE,
    NCHAN_PER_CHGROUP,
)
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    Stage1MultiDMCoarseDM,
    build_context,
    _build_core_baseline_mask,
    process_block,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


LOG = logging.getLogger("profile_dedisp_internals")


# ---------- duplicated synth helpers from profile_fast_path_K1.py ----------


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _build_synthetic_summed_plan(
    *, n_coarse: int, dm_max: float, chan_sum_factor: int,
    t_int_fast_us: float,
) -> DMPlan:
    from dsart.coarse_dm.dm_plan import (
        build_chgroup_freq_table_GHz,
        compute_delay_native_samples_table,
    )
    dm_pc_cc = np.linspace(0.0, dm_max, n_coarse, dtype=np.float64)
    chgroup_freqs_full = build_chgroup_freq_table_GHz()
    nchan_eff = NCHAN_PER_CHGROUP // chan_sum_factor
    chgroup_freqs = chgroup_freqs_full.reshape(
        chgroup_freqs_full.shape[0], nchan_eff, chan_sum_factor,
    ).mean(axis=2)
    table = compute_delay_native_samples_table(dm_pc_cc, chgroup_freqs)
    return DMPlan(
        dm_pc_cc=dm_pc_cc,
        n_fine_per_coarse=1,
        t_int_fast_us=t_int_fast_us,
        chgroup_freqs_GHz=chgroup_freqs,
        _delay_native_samples_table=table,
    )


# ---------- patched _dedisperse_one_window with internal timers ----------


def _make_instrumented_dedisp(orig_method, counters_dict):
    """Wrap ``Stage1MultiDMCoarseDM._dedisperse_one_window`` with per-phase
    cuda.Event timers. Re-implements the full method body inline so we
    can time each segment.
    """
    def instrumented(self, vis_stokes_i):
        n_fv = int(vis_stokes_i.shape[0])
        t_dedisp = self.t_dedisp_for(n_fv)
        n_filled = int(self.gridder.pattern.n_filled)
        nb = int(vis_stokes_i.shape[1])
        nch = int(vis_stokes_i.shape[2])
        device = vis_stokes_i.device

        def _new_event_pair():
            return (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        def _add(name, ms):
            counters_dict.setdefault(name, []).append(ms)

        # Phase A: permute
        ev0, ev1 = _new_event_pair(); ev0.record()
        vis_T = vis_stokes_i.permute(2, 0, 1).contiguous()
        ev1.record(); ev1.synchronize()
        _add("A_permute_vis", ev0.elapsed_time(ev1))

        # Pre-cache bin shifts + cell index map
        ev0, ev1 = _new_event_pair(); ev0.record()
        bin_shifts_full = self.plan.delay_bins_per_chgroup(self.chgroup)
        bin_shifts = bin_shifts_full[:nch, self._dm_idx_iter]
        bin_shifts_dev = torch.as_tensor(
            bin_shifts, dtype=torch.int64, device=device,
        )
        t_arange = torch.arange(t_dedisp, dtype=torch.int64, device=device)
        cim_2d = self.gridder.cell_index_map.reshape(nb, nch)
        cim_cb = cim_2d.t().contiguous()
        ev1.record(); ev1.synchronize()
        _add("B_setup_indices", ev0.elapsed_time(ev1))

        out = torch.empty(
            (self.n_dm, t_dedisp, n_filled),
            dtype=torch.complex64, device=device,
        )
        dm_chunk = max(1, int(getattr(self, "dm_chunk_size", 2)))
        dm_chunk = min(dm_chunk, self.n_dm)

        for c0 in range(0, self.n_dm, dm_chunk):
            c1 = min(c0 + dm_chunk, self.n_dm)
            chunk = c1 - c0

            # Phase C: gather
            ev0, ev1 = _new_event_pair(); ev0.record()
            bs_chunk = bin_shifts_dev[:, c0:c1]
            t_idx = (
                bs_chunk[:, :, None] + t_arange[None, None, :]
            ).reshape(nch, chunk * t_dedisp)
            t_idx_b = t_idx[:, :, None].expand(nch, chunk * t_dedisp, nb)
            gathered = torch.gather(vis_T, 1, t_idx_b)
            ev1.record(); ev1.synchronize()
            _add("C_gather_chunk", ev0.elapsed_time(ev1))

            # Phase D: per-channel scatter
            ev0, ev1 = _new_event_pair(); ev0.record()
            gathered_re = torch.view_as_real(gathered)
            out_re = torch.zeros(
                (chunk * t_dedisp, n_filled + 1, 2),
                dtype=torch.float32, device=device,
            )
            for c in range(nch):
                out_re.index_add_(1, cim_cb[c], gathered_re[c])
            out_buf = torch.view_as_complex(
                out_re[:, :n_filled, :].contiguous()
            )
            out[c0:c1] = out_buf.reshape(chunk, t_dedisp, n_filled)
            ev1.record(); ev1.synchronize()
            _add("D_scatter_chunk", ev0.elapsed_time(ev1))

            del bs_chunk, t_idx, t_idx_b, gathered, gathered_re, out_re, out_buf

        del vis_T, bin_shifts_dev, t_arange, cim_cb, cim_2d
        return out

    return instrumented


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOG.error("CUDA required for this profiler"); return 2
    LOG.info("device=%s", device)

    plan = _build_synthetic_summed_plan(
        n_coarse=args.n_coarse_dm,
        dm_max=2.0 * args.dm_truth,
        chan_sum_factor=args.chan_sum_factor,
        t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
    )
    antpos_e, antpos_n = _synth_antpos(seed=42)
    cfg = FastIntegrationConfig(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=args.n_grid,
        kernel_support=1,
        cell_lambda_mode="common",
        chan_sum_factor=args.chan_sum_factor,
        sliding_window=True,
        n_fv_chunk=None,
        t_int_fast_native=args.t_int_fast_native,
        rfi_enabled=False,
        static_sky_disabled=True,
    )
    ctx = build_context(
        cfg=cfg, device=device,
        antpos_e=antpos_e, antpos_n=antpos_n,
        is_core_baseline_mask=_build_core_baseline_mask(n_core=82),
        plan=plan,
    )

    rng = np.random.default_rng(seed=20260508)
    raw_blocks = [
        rng.integers(0, 256, size=ctx.kernel.fada_bytes_per_block,
                     dtype=np.uint8)
        for _ in range(args.warmup + args.n_blocks)
    ]

    LOG.info("ready: n_filled=%d n_dm=%d n_fv_per_block=%d",
             ctx.gridder.pattern.n_filled,
             ctx.coarse_dm.n_dm,
             ctx.kernel.n_fast_vis_per_full_block)

    # Patch in the instrumented dedisp.
    counters: dict[str, list[float]] = {}
    orig = Stage1MultiDMCoarseDM._dedisperse_one_window
    Stage1MultiDMCoarseDM._dedisperse_one_window = (
        _make_instrumented_dedisp(orig, counters)
    )

    try:
        # Warmup (no recording)
        for i in range(args.warmup):
            process_block(raw_blocks[i], ctx=ctx, block_n=i + 1)
        # Clear counters from warmup
        counters.clear()
        # Profile
        for i in range(args.warmup, args.warmup + args.n_blocks):
            process_block(raw_blocks[i], ctx=ctx, block_n=i + 1)
    finally:
        Stage1MultiDMCoarseDM._dedisperse_one_window = orig

    # Aggregate and report.
    n_blocks = args.n_blocks
    summary = {}
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
            "n_calls_per_block": n_calls_per_block,
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
            "n_filled": ctx.gridder.pattern.n_filled,
            "n_fv_per_block": ctx.kernel.n_fast_vis_per_full_block,
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

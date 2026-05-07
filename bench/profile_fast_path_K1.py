"""Phase-level perf profile of ``process_block`` at the M3 production
op-point, K=1. Companion to ``bench/fast_path_throughput.py``.

Goal
====
Break the per-block wall time into named phases so we can decide which
piece(s) of the fast-corr graph to optimise to close the 44× real-time
gap measured by ``fast_path_throughput.py`` at K=1 on a 2080 Ti
(p50 ≈ 5950 ms vs the 134 ms block period).

Phases reported (per block, after the warmup window):

A. voltage prep
   1. unpack_int4_split           — CPU→GPU + int4→fp32
   2. cal apply (apply_cal_split) — F21 DEC-phase fold
B. F31b streaming corr loop (one outer loop over fast-vis chunks)
   3. kernel.compute_split        — GEMM + reduce per slab
   4. stokes_i_pol_sum            — pol→Stokes I per slab
   5. F33 chan-sum                — per-slab chan-sum into vis_stokes_i
   (the script reports the SUM of these three across all chunks; the
   chrome trace can break by individual chunk.)
C. multi-DM stage-1
   6. F34 join                    — torch.cat prev + curr (sliding window)
   7. apply_stage1_shifts         — per-DM vis-domain shift (×N_DM)
   8. gridder.compute             — per-DM scatter (×N_DM)
   9. F34 emit slice              — slice the prev-block window out
D. tail
   10. stage2 fifo + transport tx — no-op stubs, expected ~0 ms

Method
======
* Build the exact same ctx + DMPlan as ``fast_path_throughput.py`` at
  the production op-point (t_int_fast_native=8, n_grid=256,
  n_coarse_dm=24, chan_sum_factor=8 (F33), sliding_window=True (F34),
  cell_lambda_mode=common (F28), kernel_support=1).
* Warm up ``--warmup`` blocks (default 3) so JIT, allocator caches,
  and the F34 ring buffer settle.
* Profile ``--n-blocks`` blocks (default 5) by replacing the
  inner functions (``unpack_int4_split``, ``apply_cal_split``,
  ``FastCorrKernel.compute_split``, ``stokes_i_pol_sum``,
  ``apply_stage1_shifts``, ``FastVisGridder.compute``) with thin
  ``torch.cuda.Event``-timing wrappers that aggregate per-block GPU
  ms into named phase counters.
* Optionally also run one block under ``torch.profiler.profile`` and
  emit a chrome trace at ``<report-dir>/trace.json``.

Output
======
* ``<report-dir>/phase_breakdown.json``  — per-phase ms (mean/p50/p99)
* ``<report-dir>/phase_breakdown.txt``   — pretty table
* ``<report-dir>/trace.json``            — torch.profiler chrome trace
                                              (single block, last warmup)

CLI::

    python -m bench.profile_fast_path_K1 \\
        --report-dir bench/reports/<UTC>/G7-perf-profile-K1 \\
        --warmup 3 --n-blocks 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from dsart.coarse_dm.dm_plan import DMPlan
from dsart.common.constants import (
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services import corr_fast_integration as cfi_mod
from dsart.services import corr_fast_kernel as kernel_mod
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)
from dsart.grid.kernel import FastVisGridder

# Reuse the same synthetic-plan helper as fast_path_throughput.py
from bench.fast_path_throughput import (
    _build_synthetic_summed_plan,
    _synth_antpos,
    _synth_voltage_block,
)


LOG = logging.getLogger("profile_fast_path_K1")


# ---------------------------------------------------------------------------
# Phase counters
# ---------------------------------------------------------------------------


@dataclass
class _PhaseCounters:
    """Per-block GPU ms accumulator. One key per phase.

    For phases that fire multiple times per block (e.g. compute_split
    fires once per F31b chunk; apply_stage1_shifts fires once per DM
    trial), we accumulate the SUM into the per-block bucket and emit
    that as the phase total. Per-call distributions can be read off
    the chrome trace.
    """

    per_block: dict[str, list[float]] = field(default_factory=dict)
    _curr_block: dict[str, float] = field(default_factory=dict)

    def reset_block(self) -> None:
        self._curr_block = {}

    def add(self, phase: str, ms: float) -> None:
        self._curr_block[phase] = self._curr_block.get(phase, 0.0) + float(ms)

    def commit_block(self) -> None:
        for k, v in self._curr_block.items():
            self.per_block.setdefault(k, []).append(v)
        self._curr_block = {}

    def stats(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for k, vs in self.per_block.items():
            arr = np.asarray(vs, dtype=np.float64)
            out[k] = {
                "n_blocks": int(arr.size),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p99_ms": float(np.percentile(arr, 99)) if arr.size > 1 else float(arr.mean()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "n_calls_per_block_estimate": (
                    # rough hint: phases with deterministic call counts
                    # are tagged in the printer; this just records the
                    # number of timed events per block as the number of
                    # individual ms entries summed (we lose that here by
                    # design — see chrome trace for per-call detail).
                    1
                ),
            }
        return out


def _time_cuda(
    fn: Callable[..., object],
    counters: _PhaseCounters,
    phase: str,
) -> Callable[..., object]:
    """Wrap ``fn`` so each call's GPU duration is added to ``counters[phase]``.

    Uses ``torch.cuda.Event`` so we time only the GPU work, not the
    Python launch overhead (which is small but noisy). For non-CUDA
    calls (e.g. CPU stage in cold-path) this falls back to wall time.
    """

    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            out = fn(*args, **kwargs)
            ev_end.record()
            ev_end.synchronize()
            ms = ev_start.elapsed_time(ev_end)
        else:
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
        counters.add(phase, ms)
        return out

    return wrapped


def _install_phase_wrappers(counters: _PhaseCounters) -> Callable[[], None]:
    """Monkey-patch the inner-loop primitives with timing wrappers.

    Returns an ``unwind`` callable that restores the originals.
    """
    # Unpack int4
    orig_unpack = cfi_mod.unpack_int4_split
    orig_apply_cal = cfi_mod.apply_cal_split
    orig_stokes_i = cfi_mod.stokes_i_pol_sum
    orig_apply_stage1 = cfi_mod.apply_stage1_shifts
    orig_compute_split = kernel_mod.FastCorrKernel.compute_split
    orig_grid_compute = FastVisGridder.compute

    cfi_mod.unpack_int4_split = _time_cuda(orig_unpack, counters, "unpack_int4_split")
    cfi_mod.apply_cal_split = _time_cuda(orig_apply_cal, counters, "apply_cal_split")
    cfi_mod.stokes_i_pol_sum = _time_cuda(orig_stokes_i, counters, "stokes_i_pol_sum")
    cfi_mod.apply_stage1_shifts = _time_cuda(
        orig_apply_stage1, counters, "apply_stage1_shifts__sum_over_DMs",
    )

    def patched_compute_split(self, *args, **kwargs):
        if torch.cuda.is_available():
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            out = orig_compute_split(self, *args, **kwargs)
            ev_end.record()
            ev_end.synchronize()
            ms = ev_start.elapsed_time(ev_end)
        else:
            t0 = time.perf_counter()
            out = orig_compute_split(self, *args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
        counters.add("compute_split__sum_over_chunks", ms)
        return out

    def patched_grid_compute(self, *args, **kwargs):
        if torch.cuda.is_available():
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            out = orig_grid_compute(self, *args, **kwargs)
            ev_end.record()
            ev_end.synchronize()
            ms = ev_start.elapsed_time(ev_end)
        else:
            t0 = time.perf_counter()
            out = orig_grid_compute(self, *args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000.0
        counters.add("gridder_compute__sum_over_DMs", ms)
        return out

    kernel_mod.FastCorrKernel.compute_split = patched_compute_split
    FastVisGridder.compute = patched_grid_compute

    def unwind() -> None:
        cfi_mod.unpack_int4_split = orig_unpack
        cfi_mod.apply_cal_split = orig_apply_cal
        cfi_mod.stokes_i_pol_sum = orig_stokes_i
        cfi_mod.apply_stage1_shifts = orig_apply_stage1
        kernel_mod.FastCorrKernel.compute_split = orig_compute_split
        FastVisGridder.compute = orig_grid_compute

    return unwind


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_breakdown(stats: dict[str, dict[str, float]], block_period_ms: float) -> str:
    rows = []
    rows.append(
        f"{'phase':<42}  {'mean (ms)':>12}  {'p50 (ms)':>10}  "
        f"{'p99 (ms)':>10}  {'% of p50 wall':>14}"
    )
    rows.append("-" * 96)
    # Try to surface the headline phases first; otherwise alphabetical.
    preferred = [
        "unpack_int4_split",
        "apply_cal_split",
        "compute_split__sum_over_chunks",
        "stokes_i_pol_sum",
        "apply_stage1_shifts__sum_over_DMs",
        "gridder_compute__sum_over_DMs",
    ]
    seen = set()
    total_p50 = sum(s["p50_ms"] for s in stats.values())
    for k in preferred:
        if k in stats:
            seen.add(k)
            s = stats[k]
            pct = (s["p50_ms"] / total_p50 * 100.0) if total_p50 > 0 else 0.0
            rows.append(
                f"{k:<42}  {s['mean_ms']:>12.2f}  {s['p50_ms']:>10.2f}  "
                f"{s['p99_ms']:>10.2f}  {pct:>13.1f}%"
            )
    for k in sorted(stats.keys()):
        if k in seen:
            continue
        s = stats[k]
        pct = (s["p50_ms"] / total_p50 * 100.0) if total_p50 > 0 else 0.0
        rows.append(
            f"{k:<42}  {s['mean_ms']:>12.2f}  {s['p50_ms']:>10.2f}  "
            f"{s['p99_ms']:>10.2f}  {pct:>13.1f}%"
        )
    rows.append("-" * 96)
    rows.append(
        f"{'sum of phase p50s':<42}  {'':>12}  {total_p50:>10.2f}  "
        f"{'':>10}  {'100.0%':>14}"
    )
    rows.append(
        f"realtime block period = {block_period_ms:.2f} ms; "
        f"phase-sum p50 / block_period = {total_p50 / block_period_ms:.2f}x"
    )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--n-blocks", type=int, default=5)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--t-int-fast-native", type=int, default=8)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--n-coarse-dm", type=int, default=24)
    p.add_argument("--chan-sum-factor", type=int, default=8)
    p.add_argument("--dm-truth", type=float, default=1500.0)
    p.add_argument("--no-trace", action="store_true",
                   help="skip the torch.profiler chrome trace.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    LOG.info("device=%s warmup=%d n_blocks=%d", device, args.warmup, args.n_blocks)

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
        dm_plan=plan,
    )
    LOG.info(
        "ready: n_filled=%d n_dm=%d n_fv_per_block=%d t_int=%.3f µs",
        ctx.gridder.pattern.n_filled,
        ctx.multi_dm_coarse_dm.n_dm,
        ctx.kernel.n_fast_vis_per_full_block,
        plan.t_int_fast_us,
    )

    counters = _PhaseCounters()
    unwind = _install_phase_wrappers(counters)

    rng = np.random.default_rng(seed=0)

    block_n = 0
    # Warmup (no counters; warmup also seeds the F34 ring buffer)
    for _ in range(args.warmup):
        block_n += 1
        raw = _synth_voltage_block(rng=rng)
        process_block(raw, ctx=ctx, block_n=block_n)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    LOG.info("warmup done; profiling %d blocks", args.n_blocks)
    per_block_wall_ms: list[float] = []
    for _ in range(args.n_blocks):
        block_n += 1
        raw = _synth_voltage_block(rng=rng)
        counters.reset_block()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        process_block(raw, ctx=ctx, block_n=block_n)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_block_wall_ms.append((time.perf_counter() - t0) * 1000.0)
        counters.commit_block()

    # Optionally one chrome trace block
    trace_path = None
    if not args.no_trace:
        try:
            from torch.profiler import profile, ProfilerActivity, schedule
            block_n += 1
            raw = _synth_voltage_block(rng=rng)
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            with profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
            ) as prof:
                process_block(raw, ctx=ctx, block_n=block_n)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            trace_path = args.report_dir / "trace.json"
            prof.export_chrome_trace(str(trace_path))
            LOG.info("wrote %s", trace_path)

            # Also dump the top-20 ops by self-CUDA time
            try:
                key = "self_cuda_time_total"
                top_str = prof.key_averages().table(
                    sort_by=key, row_limit=20,
                )
            except Exception:
                top_str = prof.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=20,
                )
            (args.report_dir / "top_ops.txt").write_text(top_str)
            LOG.info("wrote %s", args.report_dir / "top_ops.txt")
        except Exception as exc:                                  # noqa: BLE001
            LOG.warning("chrome trace failed: %s", exc)

    unwind()

    stats = counters.stats()
    block_period_ms = float(
        NPACKETS_PER_BLOCK * NTIMES_PER_PACKET * NATIVE_SAMPLE_US * 1e-3
    )

    # Report
    arr = np.asarray(per_block_wall_ms)
    summary = {
        "config": {
            "t_int_fast_native": args.t_int_fast_native,
            "n_grid": args.n_grid,
            "n_coarse_dm": args.n_coarse_dm,
            "chan_sum_factor": args.chan_sum_factor,
            "kernel_support": 1,
            "cell_lambda_mode": "common",
            "sliding_window": True,
            "device": str(device),
            "n_filled": int(ctx.gridder.pattern.n_filled),
            "n_fv_per_block": int(ctx.kernel.n_fast_vis_per_full_block),
        },
        "block_period_ms": block_period_ms,
        "wall_per_block_ms": {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)) if arr.size > 1 else float(arr.mean()),
            "n_blocks": int(arr.size),
        },
        "phase_breakdown": stats,
    }
    (args.report_dir / "phase_breakdown.json").write_text(json.dumps(summary, indent=2))
    LOG.info("wrote %s", args.report_dir / "phase_breakdown.json")

    text = _print_breakdown(stats, block_period_ms)
    text += (
        f"\n\nwall per block (perf_counter, full block):"
        f" mean={arr.mean():.2f} p50={np.percentile(arr, 50):.2f} "
        f"p99={(np.percentile(arr, 99) if arr.size > 1 else arr.mean()):.2f} ms"
        f"\nfull-block p50 / block_period_ms = "
        f"{np.percentile(arr, 50) / block_period_ms:.2f}x"
    )
    print(text)
    (args.report_dir / "phase_breakdown.txt").write_text(text)
    LOG.info("wrote %s", args.report_dir / "phase_breakdown.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

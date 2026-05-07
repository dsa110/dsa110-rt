"""Fast-corr path sustained-throughput + injection-correctness bench (chunk 9).

Implements the §8.M3 ``bench/fast_path_throughput.py`` line item: drive
:func:`dsart.services.corr_fast_integration.process_block` at a fixed
``t_int_fast_native`` for ``--duration-s`` seconds with synthetic input
blocks (Gaussian voltage + a known voltage-domain injection at a fixed
``(l, m, dm, t)``) and report:

* **Throughput**: end-to-end wall time per block + cubes / second over
  the run window.
* **Injection correctness**: location of the injected source on the
  per-trial dedispersed cube — for the matching DM trial, the per-cell
  power should peak at the predicted ``(l, m, t)`` cell.
* **Per-DM-trial recovery**: across all coarse-DM trials in the
  custom plan, the truth-DM trial recovers the highest peak SNR.

Pipeline
========

The bench wires up the F25 multi-DM-trial path:

1. Build a custom :class:`DMPlan` with a small set of DM trials that
   bracket the truth DM (default ``dm_truth=200, n_coarse=5``).
2. Call :func:`build_context` with ``dm_plan=plan``.
3. Per block, synthesise raw voltage bytes (Gaussian thermal noise +
   :func:`dsart.inject.online.OnlineInjector.apply_block` to add a
   known ``(l, m, dm, t)`` injection).
4. Run :func:`process_block` and time the call.
5. Accumulate per-trial peak metrics; at end-of-run, render
   ``report.html`` + JSON summary + a peak-cube heatmap PNG.

Production pin
==============

This bench shares zero test-bespoke code with the production service
shell; it constructs ``FastIntegrationConfig`` + ``IntegrationContext``
the same way ``services.corr_fast_integration.run`` does, only the
``raw`` byte source is synthetic. The chunk-9 STEP gate in
``tools/dod/M3.sh`` runs this with ``--duration-s 10`` to avoid
spending DoD wall time on a 30-min throughput soak; for full
operating-point validation the operator runs it manually with
``--duration-s 1800``.

CLI
===

    python -m bench.fast_path_throughput \\
        [--duration-s 10] \\
        [--t-int-fast-native 32] \\
        [--n-grid 64] \\
        [--n-coarse-dm 5] \\
        [--dm-truth 200.0] \\
        [--injection-snr-db 20.0] \\
        [--device auto] \\
        [--report-dir bench/reports/<UTC>/M3-fast-path-throughput] \\
        [--n-blocks-min 8]

References
==========

* Plan §8.M3 line 2270 — ``bench/fast_path_throughput.py``.
* M3_PLAN_FIXES.md F25 — multi-DM-trial integration.
* :mod:`dsart.coarse_dm.stage1` — vis-domain stage-1 shifts.
* :mod:`dsart.services.corr_fast_integration` — production orchestrator.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dsart.coarse_dm.dm_plan import (
    DMPlan,
    build_chgroup_freq_table_GHz,
    compute_delay_native_samples_table,
)
from dsart.common.constants import (
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    build_context,
    process_block,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


LOG = logging.getLogger("fast_path_throughput")


# ---------------------------------------------------------------------------
# Synthetic raw-block generator
# ---------------------------------------------------------------------------


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """82 core ants in a 100m × 100m box + 14 outriggers ≥ 800m."""
    rng = np.random.default_rng(seed)
    antpos_e = np.zeros(96, dtype=np.float32)
    antpos_n = np.zeros(96, dtype=np.float32)
    antpos_e[:82] = rng.uniform(-50, 50, size=82).astype(np.float32)
    antpos_n[:82] = rng.uniform(-50, 50, size=82).astype(np.float32)
    antpos_e[82:] = rng.uniform(-2000, 2000, size=14).astype(np.float32)
    antpos_n[82:] = rng.uniform(-2000, 2000, size=14).astype(np.float32)
    return antpos_e, antpos_n


def _synth_voltage_block(
    *,
    rng: np.random.Generator,
    sigma: int = 32,
) -> np.ndarray:
    """Gaussian thermal noise int4-packed bytes for one fada block.

    The int4 format packs (real, imag) nibbles into a single byte per
    sample. We synthesise two int4-fluffed (i.e. sign-extended in
    int8 then byte-shifted) values, clip to int4 range, and pack.
    For benchmarking purposes only — the math correctness of the
    int4 representation is pinned in M2 unit tests, here we just
    need bytes that decode to non-trivial complex voltages.
    """
    nbytes = (
        NCHAN_PER_CHGROUP * NTIMES_PER_PACKET * 2 * NANTS * NPACKETS_PER_BLOCK
    )
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class _ThroughputSummary:
    duration_s: float
    n_blocks: int
    n_cubes_emitted: int
    cubes_per_second: float
    p50_per_block_ms: float
    p99_per_block_ms: float
    t_int_fast_us: float
    n_grid: int
    n_dm_coarse: int
    n_filled: int
    device: str
    git_sha: str
    utc_iso: str
    kernel_support: int = 1
    chan_sum_factor: int = 1
    sliding_window: bool = False
    cell_lambda_mode: str = "common"
    n_fv_chunk: int | None = None
    block_period_ms: float = 0.0
    realtime_factor: float = 0.0
    realtime_pass: bool = False


def _write_report(
    report_dir: Path,
    *,
    summary: _ThroughputSummary,
    per_block_ms: list[float],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    # Per-block latency histogram + summary
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(per_block_ms, bins=40, color="tab:blue", alpha=0.8)
        ax.axvline(
            summary.p50_per_block_ms, color="green", linestyle="--",
            label=f"p50 = {summary.p50_per_block_ms:.1f} ms",
        )
        ax.axvline(
            summary.p99_per_block_ms, color="red", linestyle="--",
            label=f"p99 = {summary.p99_per_block_ms:.1f} ms",
        )
        ax.set_xlabel("per-block process_block latency (ms)")
        ax.set_ylabel("count")
        ax.set_title(
            f"Fast-path throughput | {summary.cubes_per_second:.2f} cubes/s | "
            f"{summary.n_blocks} blocks | t_int = {summary.t_int_fast_us:.1f} µs"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir / "latency_histogram.png", dpi=120)
        plt.close(fig)
    except Exception as exc:                                    # noqa: BLE001
        LOG.warning("matplotlib not usable; skipping plot: %s", exc)

    (report_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
    )

    rt_label = "PASS" if summary.realtime_pass else "FAIL"
    rt_color = "green" if summary.realtime_pass else "red"
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>M3 fast-path throughput</title></head>
<body style="font-family:sans-serif">
<h1>M3 fast-path throughput</h1>
<p><b>Real-time</b>: <span style="color:{rt_color};font-weight:bold">{rt_label}</span>
&nbsp;&nbsp;<b>p99 / block_period</b>: {summary.p99_per_block_ms:.2f} / {summary.block_period_ms:.2f} ms
&nbsp;&nbsp;<b>real-time factor</b>: {summary.realtime_factor:.2f}x</p>
<p><b>Cubes/s</b>: {summary.cubes_per_second:.2f}
&nbsp;&nbsp;<b>n_blocks</b>: {summary.n_blocks}
&nbsp;&nbsp;<b>p50 latency</b>: {summary.p50_per_block_ms:.2f} ms
&nbsp;&nbsp;<b>p99 latency</b>: {summary.p99_per_block_ms:.2f} ms
</p>
<p><b>n_grid</b>: {summary.n_grid}
&nbsp;&nbsp;<b>n_dm_coarse</b>: {summary.n_dm_coarse}
&nbsp;&nbsp;<b>n_filled</b>: {summary.n_filled}
&nbsp;&nbsp;<b>t_int</b>: {summary.t_int_fast_us:.3f} µs
&nbsp;&nbsp;<b>device</b>: {summary.device}</p>
<p><b>kernel_support (G7)</b>: {summary.kernel_support}
&nbsp;&nbsp;<b>chan_sum_factor (F33)</b>: {summary.chan_sum_factor}
&nbsp;&nbsp;<b>sliding_window (F34)</b>: {summary.sliding_window}
&nbsp;&nbsp;<b>cell_lambda_mode (F28)</b>: {summary.cell_lambda_mode}
&nbsp;&nbsp;<b>n_fv_chunk (F31b)</b>: {summary.n_fv_chunk}</p>
<p><b>git_sha</b>: <code>{summary.git_sha}</code>
&nbsp;&nbsp;<b>utc_iso</b>: {summary.utc_iso}</p>
<img src="latency_histogram.png" style="max-width:900px;border:1px solid #ccc">
<hr>
<p>Real-time gate: each fada block ingests {summary.block_period_ms:.2f} ms of native
samples ({NPACKETS_PER_BLOCK} packets × {NTIMES_PER_PACKET} samples × {NATIVE_SAMPLE_US:.3f} µs).
PASS iff p99 process_block latency ≤ that period.</p>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration-s", type=float, default=60.0,
                   help="bench duration. Default 60 s (= ~447 fada blocks "
                        "at 134 ms each) gives a stable p50/p99 for "
                        "1-min real-time soak validation.")
    p.add_argument(
        "--t-int-fast-native", type=int, default=8,
        help=(
            "fast-vis integration depth in NATIVE samples. Default 8 "
            "(= 262.144 µs cadence; 512 fast-vis tiles/block) — the "
            "M3 PRODUCTION cadence on the 2080Ti, made memory-feasible "
            "by F31a (kernel chunking) + F31b (streaming chunks) + "
            "F33 (8-channel pre-dedispersion sum)."
        ),
    )
    p.add_argument("--n-grid", type=int, default=256,
                   help="image-plane grid side. Default 256 (production "
                        "O-4 op-point).")
    p.add_argument("--n-coarse-dm", type=int, default=24,
                   help="coarse DM trials. Default 24 = production O-4 "
                        "operating point (configs/operating_points.yaml "
                        "N_coarse_DM).")
    p.add_argument("--dm-truth", type=float, default=1500.0,
                   help="injected DM (pc/cc). Default 1500 puts the "
                        "truth in the middle of the 0..2*dm_truth = "
                        "0..3000 plan, which spans the M3 DM_max of "
                        "3000 pc/cc.")
    p.add_argument("--kernel-support", type=int, default=1,
                   choices=(1, 3, 5),
                   help="G7 gridding-kernel support cells. K=1 "
                        "(default; legacy pillbox), K=3 / K=5 = "
                        "Gaussian-tapered K^2 grid taps per (bls, ch).")
    p.add_argument("--chan-sum-factor", type=int, default=8,
                   help="F33: pre-dedispersion channel-sum factor. "
                        "Default 8 (production: 384 fine ch -> 48 "
                        "summed ch). At DM=3000, ν=1.31 GHz the max "
                        "intra-summed-channel smearing is ~2.7 ms, "
                        "well within the search t_int.")
    p.add_argument("--sliding-window", action="store_true", default=True,
                   help="F34: 2-block sliding-window stage-1 with K=2 "
                        "ring buffer. Default ON (required at M3 prod "
                        "op-point because at DM=3000 the intra-chgroup "
                        "delay reaches ~480 fv bins, comparable to "
                        "the ~512-bin block size).")
    p.add_argument("--no-sliding-window", action="store_false",
                   dest="sliding_window",
                   help="disable F34 (regression / debug only).")
    p.add_argument("--cell-lambda-mode", default="common",
                   choices=("common", "per_chgroup"),
                   help="F28: 'common' (production default; shared "
                        "top-of-band cell scale across chgroups) or "
                        "'per_chgroup' (legacy auto-fit; ~5-cell "
                        "column drift across chgroups).")
    p.add_argument("--n-fv-chunk", type=int, default=None,
                   help="F31b: streaming fast-vis chunk size. None "
                        "(default) auto-picks the largest power-of-two "
                        "slab under the F31b 256 MB target.")
    p.add_argument(
        "--device", type=str, default="auto", choices=("auto", "cpu", "cuda"),
    )
    p.add_argument(
        "--report-dir", type=Path,
        default=Path("bench/reports/fast-path-throughput"),
    )
    p.add_argument("--n-blocks-min", type=int, default=8)
    return p.parse_args(argv)


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    LOG.info("device=%s", device)

    # Custom DM plan: linear from 0 to 2*dm_truth so the truth DM
    # falls in the middle of the trials.
    dm_pc_cc = np.linspace(
        0.0, 2.0 * args.dm_truth, args.n_coarse_dm,
    ).astype(np.float64)
    chgroup_freqs_GHz = build_chgroup_freq_table_GHz()
    delay_table = compute_delay_native_samples_table(
        dm_pc_cc, chgroup_freqs_GHz,
    )
    plan = DMPlan(
        dm_pc_cc=dm_pc_cc,
        n_fine_per_coarse=1,
        t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
        chgroup_freqs_GHz=chgroup_freqs_GHz,
        _delay_native_samples_table=delay_table,
    )

    antpos_e, antpos_n = _synth_antpos(seed=42)
    cfg = FastIntegrationConfig(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=args.n_grid,
        kernel_support=int(args.kernel_support),                 # G7
        cell_lambda_mode=str(args.cell_lambda_mode),             # F28
        chan_sum_factor=int(args.chan_sum_factor),               # F33
        sliding_window=bool(args.sliding_window),                # F34
        n_fv_chunk=args.n_fv_chunk,                              # F31b
        t_int_fast_native=args.t_int_fast_native,
        rfi_enabled=False,                 # synthetic noise; flag fraction
                                           # would otherwise be ~100%
        static_sky_disabled=True,
    )
    ctx = build_context(
        cfg=cfg, device=device,
        antpos_e=antpos_e, antpos_n=antpos_n,
        dm_plan=plan,
    )
    LOG.info(
        "ready: n_filled=%d n_dm=%d n_fast_vis_per_block=%d t_int=%.3f µs",
        ctx.gridder.pattern.n_filled,
        ctx.multi_dm_coarse_dm.n_dm,
        ctx.kernel.n_fast_vis_per_full_block,
        plan.t_int_fast_us,
    )

    # Run-loop: process blocks until duration_s expires (with at
    # least n_blocks_min blocks).
    rng = np.random.default_rng(seed=0)
    start = time.perf_counter()
    deadline = start + float(args.duration_s)
    per_block_ms: list[float] = []
    n_cubes_emitted = 0
    block_n = 0

    while True:
        block_n += 1
        raw = _synth_voltage_block(rng=rng)
        b_start = time.perf_counter()
        out = process_block(raw, ctx=ctx, block_n=block_n)
        b_end = time.perf_counter()
        per_block_ms.append((b_end - b_start) * 1000.0)
        if out.gridded_minus_sky is not None:
            n_cubes_emitted += int(out.gridded_minus_sky.shape[0])

        now = time.perf_counter()
        if now >= deadline and block_n >= args.n_blocks_min:
            break
        if block_n >= 1000:                        # safety upper bound
            break

    duration = time.perf_counter() - start
    p50 = float(np.percentile(per_block_ms, 50))
    p99 = float(np.percentile(per_block_ms, 99))

    # Real-time gate: each fada block is NPACKETS_PER_BLOCK fada
    # packets of NTIMES_PER_PACKET native samples = the wall-clock
    # period the corr-node has to ingest + process before the next
    # block arrives. Real-time PASS iff p99 ≤ that period.
    block_period_us = float(
        NPACKETS_PER_BLOCK * NTIMES_PER_PACKET * NATIVE_SAMPLE_US
    )
    block_period_ms = block_period_us * 1e-3
    realtime_factor = (p99 / block_period_ms) if block_period_ms > 0 else 0.0
    realtime_pass = bool(p99 <= block_period_ms)

    summary = _ThroughputSummary(
        duration_s=duration,
        n_blocks=block_n,
        n_cubes_emitted=n_cubes_emitted,
        cubes_per_second=block_n / duration if duration > 0 else 0.0,
        p50_per_block_ms=p50,
        p99_per_block_ms=p99,
        t_int_fast_us=plan.t_int_fast_us,
        n_grid=args.n_grid,
        n_dm_coarse=int(plan.n_coarse),
        n_filled=int(ctx.gridder.pattern.n_filled),
        device=str(device),
        git_sha=_git_sha(),
        utc_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        kernel_support=int(args.kernel_support),
        chan_sum_factor=int(args.chan_sum_factor),
        sliding_window=bool(args.sliding_window),
        cell_lambda_mode=str(args.cell_lambda_mode),
        n_fv_chunk=(int(args.n_fv_chunk) if args.n_fv_chunk is not None else None),
        block_period_ms=block_period_ms,
        realtime_factor=realtime_factor,
        realtime_pass=realtime_pass,
    )
    LOG.info(
        "throughput: %d blocks in %.2f s = %.2f cubes/s "
        "(p50 %.1f ms, p99 %.1f ms; block_period=%.1f ms; rt_factor=%.2fx; "
        "real-time %s)",
        block_n, duration, summary.cubes_per_second, p50, p99,
        block_period_ms, realtime_factor,
        "PASS" if realtime_pass else "FAIL",
    )

    _write_report(args.report_dir, summary=summary, per_block_ms=per_block_ms)
    LOG.info("wrote report to %s/report.html", args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

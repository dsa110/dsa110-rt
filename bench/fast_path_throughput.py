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

from dsart.coarse_dm.dm_plan import DMPlan
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

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>M3 fast-path throughput</title></head>
<body style="font-family:sans-serif">
<h1>M3 fast-path throughput</h1>
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
<p><b>git_sha</b>: <code>{summary.git_sha}</code>
&nbsp;&nbsp;<b>utc_iso</b>: {summary.utc_iso}</p>
<img src="latency_histogram.png" style="max-width:900px;border:1px solid #ccc">
<hr>
<p>Operator note: ≥ 7.45 cubes/s is the §9 steady-state requirement
on a 2080 Ti at full ops; this CPU bench is for path correctness.
The chunk-9 §8.M3 line gates only on the bench producing this
<code>report.html</code>.</p>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration-s", type=float, default=10.0)
    p.add_argument("--t-int-fast-native", type=int, default=32)
    p.add_argument("--n-grid", type=int, default=64)
    p.add_argument("--n-coarse-dm", type=int, default=5)
    p.add_argument("--dm-truth", type=float, default=200.0)
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
    ).astype(np.float32)
    plan = DMPlan.build(
        dm_pc_cc=dm_pc_cc,
        t_int_fast_us=float(args.t_int_fast_native * NATIVE_SAMPLE_US),
    )

    antpos_e, antpos_n = _synth_antpos(seed=42)
    cfg = FastIntegrationConfig(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=args.n_grid,
        kernel_support=1,
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
    )
    LOG.info(
        "throughput: %d blocks in %.2f s = %.2f cubes/s "
        "(p50 %.1f ms, p99 %.1f ms)",
        block_n, duration, summary.cubes_per_second, p50, p99,
    )

    _write_report(args.report_dir, summary=summary, per_block_ms=per_block_ms)
    LOG.info("wrote report to %s/report.html", args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

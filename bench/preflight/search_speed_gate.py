#!/usr/bin/env python3
"""bench/preflight/search_speed_gate.py — operator-driven search-side
speed gate. Run this BEFORE any restart of the fleet that touches the
search pipeline.

What it does
============

Drives ``bench/search_node_throughput.py`` at the EXACT production
op-point currently locked in ``configs/dsart_search_rt.yaml``::

  n_grid=256  n_fdm=34  t_det=192
  cube_dtype=fp16  device=cuda:0  image_backend=gpu
  detector-streaming + tile_size=256 + n_top=24 + boxcar_accum=fp16
  detector-layer2-max-samples=100000  layer1-max-samples=10000
  threshold-sigma=12.0  (matches c1.snr_min)
  bank-mask=k_img=unit;k_dm=d1;k_time=b1..b64
  --pipeline-overlap  --prequantise  --symmetric-shift-padding
  --dm-plan-path=<prod v2 plan>  --coarse-dm-owner-idx=<--owner-idx>
  --t-int-search-us=1048.576

It then:

  * Parses the per-cube ``stage_timings.ndjson`` from the bench run.
  * Drops the first 5 cubes (Layer-1 burn-in + GPU warmup; the median
    over the 6+ steady-state cubes is what production sees).
  * Prints a per-stage table (p50 / p90 / p99 in ms).
  * Asserts ``total_pipeline.p50 <= --budget-ms`` (default 134, the
    7.45 cubes/s production cadence).
  * Exits 0 if PASS, 1 if FAIL.

Why this script exists
======================

The search service has historically regressed silently between
restarts (M7.4.2 added a 1.4 ms / cube coverage-correction broadcast
that was misdiagnosed as 96 ms; M7.7 then shipped a fix to a problem
that was not the bottleneck). The fix-loop went:

  1. Code change pushed to the fleet.
  2. Operator restarts services.
  3. Cube cadence regresses.
  4. Run the fleet for 5 minutes, fire an injection, see if it
     detects, repeat.

The point of this gate is to break that loop: code change → run this
gate on n01 → PASS means production cadence will hold; FAIL means we
KNOW we need to fix the search-side perf BEFORE touching the fleet.
No live RFI, no clock drift, no cross-node coordination — just the
GPU pipeline at the production op-point.

CLI
===

::

  python -m bench.preflight.search_speed_gate \\
      [--n-cubes 60]                                    \\
      [--budget-ms 134]                                 \\
      [--owner-idx 0]                                   \\
      [--dm-plan-path /home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz] \\
      [--device cuda:0]                                 \\
      [--out /tmp/m77_speed_gate]

Typical run on a search node:

::

  CUDA_VISIBLE_DEVICES=0 python -m bench.preflight.search_speed_gate
  CUDA_VISIBLE_DEVICES=1 python -m bench.preflight.search_speed_gate \\
      --owner-idx 1                                                  \\
      --out /tmp/m77_speed_gate_half1

Run both halves to mimic the production two-half load on a single
node (sequentially is fine — the gate is a single-half test by
design; it's the per-half latency we care about, not the cross-half
contention which is a separate concern).

Exit code conventions
=====================

* 0 — total_pipeline p50 <= budget-ms (PASS). Safe to fleet-push.
* 1 — total_pipeline p50 > budget-ms (FAIL). Do NOT fleet-push;
      iterate on the bottleneck first. The script prints which
      sub-stage is over budget.
* 2 — bench run itself failed (subprocess crashed, file missing).

Required environment
====================

* GPU node with the dsa110-rt conda env active.
* ``CUDA_VISIBLE_DEVICES`` should pin a single GPU (the gate uses
  the half-0 device path).
* ``/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz``
  present (production DM plan); override with ``--dm-plan-path`` if
  testing a new plan.
* The production search service should be STOPPED on the test GPU
  (verify with ``nvidia-smi --query-compute-apps=pid --format=csv``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


# Production op-point (mirrors configs/dsart_search_rt.yaml as of M7.7
# 2026-06-04). Bump these when the yaml flips, NOT silently.
PROD_N_GRID = 256
PROD_N_FDM = 34
PROD_T_DET = 192
PROD_T_INT_SEARCH_US = 1048.576
PROD_THRESHOLD_SIGMA = 12.0  # c1.snr_min
PROD_BANK_MASK = "k_img=unit;k_dm=d1;k_time=b1,b2,b4,b8,b16,b32,b64"
PROD_BUDGET_MS = 134.0  # 7.45 cubes/s; cube cadence at t_int=1048.576 us, 128 samples
PROD_DM_PLAN_PATH = "/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz"
PROD_LAYER1_MAX_SAMPLES = 10000
PROD_LAYER2_MAX_SAMPLES = 100000
PROD_DET_TILE_SIZE = 256
PROD_DET_N_TOP = 24

# Number of cubes to discard from the front of the run before
# computing summary percentiles. Covers Layer-1 burn-in (5 cubes) +
# GPU warmup / NVRTC compile / mempool warm-up.
DEFAULT_WARMUP_CUBES = 5


def _percentiles(values_ms: Sequence[float]) -> Dict[str, float]:
    """Return {p50, p90, p99, mean, max} (ms) for a list of ms values."""
    if not values_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    sorted_vals = sorted(values_ms)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1) / 100.0))))
        return float(sorted_vals[idx])

    return {
        "p50": _pct(50.0),
        "p90": _pct(90.0),
        "p99": _pct(99.0),
        "mean": float(sum(sorted_vals) / n),
        "max": float(sorted_vals[-1]),
    }


def _load_stage_timings(ndjson_path: Path) -> List[Dict[str, float]]:
    """Load per-cube stage timings (ms) from the bench's ndjson output."""
    rows: List[Dict[str, float]] = []
    with ndjson_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "cube_id": int(rec["cube_id"]),
                "n_candidates": int(rec["n_candidates"]),
                "build_cube_ms": float(rec["build_cube_ns"]) / 1.0e6,
                "layer1_norm_ms": float(rec["layer1_norm_ns"]) / 1.0e6,
                "detector_forward_ms": float(rec["detector_forward_ns"]) / 1.0e6,
                "total_pipeline_ms": float(rec["total_pipeline_ns"]) / 1.0e6,
            })
    return rows


def _load_gpu_substage_ms(summary_path: Path) -> Optional[Dict[str, float]]:
    """Load the per-substage GPU-time breakdown from summary.json.

    Returns the dict (with keys imager_ms / validity_ms / layer1_ms /
    detector_ms / total_gpu_ms / n) when the bench produced it, or
    None when the bench was older / ran without GPU profiling.
    """
    if not summary_path.exists():
        return None
    try:
        with summary_path.open() as fh:
            doc = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    g = doc.get("gpu_substage_ms")
    if not isinstance(g, dict) or int(g.get("n", 0)) <= 0:
        return None
    return g


def _load_imager_substage_ms(summary_path: Path) -> Optional[Dict[str, float]]:
    """Load the per-cube imager-substage GPU breakdown (combine / fft /
    mask) from summary.json. Returns None when missing or n == 0."""
    if not summary_path.exists():
        return None
    try:
        with summary_path.open() as fh:
            doc = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    g = doc.get("imager_substage_ms")
    if not isinstance(g, dict) or int(g.get("n", 0)) <= 0:
        return None
    return g


def _run_bench(out_dir: Path, args: argparse.Namespace) -> int:
    """Invoke bench.search_node_throughput as a subprocess with the
    production op-point. Returns the subprocess exit code.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "bench.search_node_throughput",
        "--n-cubes", str(int(args.n_cubes)),
        "--t-det", str(PROD_T_DET),
        "--n-fdm", str(PROD_N_FDM),
        "--n-grid", str(PROD_N_GRID),
        "--device", str(args.device),
        "--cube-dtype", "fp16",
        "--image-backend", "gpu",
        "--threshold-sigma", str(PROD_THRESHOLD_SIGMA),
        "--bank-mask", PROD_BANK_MASK,
        "--detector-streaming",
        "--detector-streaming-tile-size", str(PROD_DET_TILE_SIZE),
        "--detector-streaming-decoder-n-top", str(PROD_DET_N_TOP),
        "--detector-boxcar-accum-dtype", "fp16",
        "--detector-layer2-max-samples", str(PROD_LAYER2_MAX_SAMPLES),
        "--layer1-max-samples", str(PROD_LAYER1_MAX_SAMPLES),
        "--pipeline-overlap",
        "--prequantise",
        "--t-int-search-us", str(PROD_T_INT_SEARCH_US),
        "--cube-cadence-s", "0.0",
        "--out", str(out_dir),
    ]
    if not bool(args.no_m77):
        cmd += [
            "--symmetric-shift-padding",
            "--dm-plan-path", str(args.dm_plan_path),
            "--coarse-dm-owner-idx", str(int(args.owner_idx)),
        ]
    else:
        # Legacy comparison path: still load the prod DM plan + owner
        # slice so the shifts geometry matches prod, but skip M7.7.
        # This is the "what would production look like without M7.7"
        # control we used to establish the M7.7 8.6 ms savings.
        cmd += [
            "--dm-plan-path", str(args.dm_plan_path),
            "--coarse-dm-owner-idx", str(int(args.owner_idx)),
        ]

    env = os.environ.copy()
    env.setdefault("DSART_ENABLE_GPU_BUF_REUSE", "1")
    # Match prod (commit ef6ffd5): pin one GPU per process. Operator can
    # override by exporting CUDA_VISIBLE_DEVICES BEFORE invoking this
    # script — we don't override an explicitly-set value.
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # PYTHONPATH=src so the bench can ``import dsart`` without a full
    # ``pip install -e .`` on the dev clone. (Production search nodes
    # have it installed; dev clones often don't.)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"

    print("[gate] running bench:")
    print("[gate]   " + " ".join(cmd))
    print(f"[gate] out  : {out_dir}")
    print(f"[gate] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    print()

    # Stream the bench's stdout/stderr to the operator so a hang is
    # visible immediately. We still keep the bench's own bench.log
    # for offline inspection.
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(proc.returncode)


def _print_imager_substage_table(
    imager: Dict[str, float],
    *,
    imager_total_gpu_ms: float,
) -> None:
    """Print the imager's per-substage breakdown (combine / fft / mask)
    and call out which sub-stage is dominant."""
    combine = float(imager.get("combine_ms", 0.0))
    fft = float(imager.get("fft_ms", 0.0))
    mask = float(imager.get("mask_ms", 0.0))
    total = float(imager.get("total_ms", combine + fft + mask))
    n = int(imager.get("n", 0))

    print(
        f"  Imager substage breakdown (cuda Events inside "
        f"GpuImager.process_cube, mean over {n} cubes):"
    )
    print(
        f"    combine_ms      {combine:8.2f}    "
        f"(fused dequant+combine NVRTC kernel — chunk-8 D21)"
    )
    print(
        f"    fft_ms          {fft:8.2f}    (cuFFT-cfp16 ifft2 + fftshift)"
    )
    print(
        f"    mask_ms         {mask:8.2f}    "
        f"(edge_mask multiply + permute + writeback)"
    )
    print(f"    total_ms        {total:8.2f}")
    if imager_total_gpu_ms > 0 and total > 0:
        coverage = 100.0 * total / imager_total_gpu_ms
        gap = imager_total_gpu_ms - total
        print(
            f"    coverage        {coverage:8.1f}%   "
            f"(of overlap-path imager_ms={imager_total_gpu_ms:.1f}; "
            f"gap={gap:+.1f} ms is launch / wait between per-batch "
            f"loop iterations)"
        )
    if total > 0:
        substage = [("combine", combine), ("fft", fft), ("mask", mask)]
        substage.sort(key=lambda kv: -kv[1])
        worst_name, worst_val = substage[0]
        worst_pct = 100.0 * worst_val / total
        print(
            f"    dominant        {worst_name:>8s}    "
            f"({worst_val:.1f} ms = {worst_pct:.0f}% of imager GPU work)"
        )
    print()


def _print_gpu_substage_table(
    gpu: Dict[str, float],
    *,
    wall_p50_ms: float,
) -> None:
    """Print the GPU-time per-substage table and the wall-vs-GPU gap.

    The GPU breakdown is a cube-mean (not a per-cube percentile) — the
    cuda Events are too cheap to also percentile per-cube, and the
    wall-clock stage timings already capture variability above.
    """
    imager = float(gpu.get("imager_ms", 0.0))
    validity = float(gpu.get("validity_ms", 0.0))
    layer1 = float(gpu.get("layer1_ms", 0.0))
    detector = float(gpu.get("detector_ms", 0.0))
    total_gpu = float(gpu.get("total_gpu_ms", imager + validity + layer1 + detector))
    n = int(gpu.get("n", 0))

    print(
        f"  GPU SM busy time per cube (cuda Events, mean over {n} cubes "
        f"in the overlap path):"
    )
    print(
        f"    imager_ms       {imager:8.2f}    (fused dequant+combine "
        f"+ cuFFT + edge mask + clamp)"
    )
    print(
        f"    validity_ms     {validity:8.2f}    (all-true cached tensor → "
        f"usually ~0 ms)"
    )
    print(
        f"    layer1_ms       {layer1:8.2f}    (sigma_clipped_std + "
        f"broadcast divide / fused-imager mask update)"
    )
    print(
        f"    detector_ms     {detector:8.2f}    (boxcar + threshold + "
        f"topk decode)"
    )
    print(f"    total_gpu_ms    {total_gpu:8.2f}")
    print()

    # Sustained-throughput interpretation. In ``--pipeline-overlap``
    # mode the per-iteration wall clock UNDER-reports the steady-state
    # cube cadence: iteration N's CPU work overlaps with iteration
    # N-1's residual GPU drain, so a 137 ms iteration is consistent
    # with 170 ms of GPU work per cube if the next iteration's CPU
    # syncs at L1 / detector wait for the previous cube's GPU to
    # finish. The true sustained throughput is therefore bounded by
    # ``max(wall_p50, total_gpu_ms)`` — that's what the cube cadence
    # will be in production where the CPU has no bench-overhead slack.
    if wall_p50_ms > 0:
        sustained_ms = max(wall_p50_ms, total_gpu)
        cps = 1000.0 / max(1e-9, sustained_ms)
        print(
            f"  per-iter wall p50  : {wall_p50_ms:.1f} ms (CPU iteration time; "
            f"underestimates steady-state because cube N+1's CPU work overlaps "
            f"cube N's GPU drain)"
        )
        print(
            f"  GPU main-stream    : {total_gpu:.1f} ms (genuine main-stream "
            f"serial GPU work per cube — this gates steady-state)"
        )
        print(
            f"  sustained estimate : {sustained_ms:.1f} ms / cube = "
            f"{cps:.2f} cubes/s    [≈ max(wall, GPU)]"
        )
        if total_gpu > wall_p50_ms + 5.0:
            print(
                f"  HINT: GPU work ({total_gpu:.1f} ms) exceeds the CPU "
                f"iter wall ({wall_p50_ms:.1f} ms). The bench's "
                f"pipeline-overlap masks this — real production cadence "
                f"will track total_gpu_ms above. Focus on the dominant "
                f"GPU substage:"
            )
            substage_pairs = [
                ("imager_ms", imager),
                ("layer1_ms", layer1),
                ("detector_ms", detector),
            ]
            substage_pairs.sort(key=lambda kv: -kv[1])
            for name, val in substage_pairs:
                print(f"             {name:14s} = {val:6.1f} ms")
    print()


def _print_summary(
    rows: List[Dict[str, float]],
    *,
    warmup_cubes: int,
    budget_ms: float,
    gpu_substage_ms: Optional[Dict[str, float]] = None,
    imager_substage_ms: Optional[Dict[str, float]] = None,
) -> int:
    """Print the per-stage summary table + PASS/FAIL banner. Returns
    the exit code (0=PASS, 1=FAIL).
    """
    if len(rows) <= warmup_cubes:
        print(
            f"[gate] FAIL: bench produced only {len(rows)} cubes; "
            f"need > {warmup_cubes} after warmup to compute meaningful "
            f"percentiles"
        )
        return 1

    steady = rows[warmup_cubes:]
    print(
        f"[gate] dropping first {warmup_cubes} cubes (warmup); "
        f"summarising over the remaining {len(steady)} cubes."
    )
    print()

    stages = (
        ("build_cube     ", [r["build_cube_ms"] for r in steady]),
        ("layer1_norm    ", [r["layer1_norm_ms"] for r in steady]),
        ("detector_forward", [r["detector_forward_ms"] for r in steady]),
        ("total_pipeline ", [r["total_pipeline_ms"] for r in steady]),
    )

    print(
        f"  {'stage':17s} {'p50 ms':>8s} {'p90 ms':>8s} {'p99 ms':>8s} "
        f"{'mean ms':>8s} {'max ms':>8s}"
    )
    print("  " + "-" * 67)
    summary: Dict[str, Dict[str, float]] = {}
    for name, vals in stages:
        p = _percentiles(vals)
        summary[name.strip()] = p
        print(
            f"  {name:17s} {p['p50']:8.2f} {p['p90']:8.2f} {p['p99']:8.2f} "
            f"{p['mean']:8.2f} {p['max']:8.2f}"
        )

    print()
    n_cands = sum(int(r["n_candidates"]) for r in steady)
    print(
        f"  steady-state cands_total = {n_cands} over {len(steady)} cubes "
        f"({n_cands / max(1, len(steady)):.1f} per cube)"
    )
    print()

    total = summary["total_pipeline"]
    # Sustained per-cube cost in production = max(CPU iter wall, GPU
    # main-stream work). The bench's --pipeline-overlap masks the GPU
    # cost (next iter's CPU work overlaps prev iter's GPU drain), so
    # the per-iter wall p50 alone under-reports steady-state. Use the
    # GPU substage breakdown when available; fall back to wall for
    # CPU runs / older benches that didn't emit gpu_substage_ms.
    gpu_total_ms = 0.0
    if gpu_substage_ms is not None:
        gpu_total_ms = float(gpu_substage_ms.get("total_gpu_ms", 0.0))
    sustained_ms = max(total["p50"], gpu_total_ms)
    target_cubes_per_s = 1000.0 / budget_ms
    actual_cubes_per_s = 1000.0 / max(1e-9, sustained_ms)
    margin_ms = budget_ms - sustained_ms
    print(
        f"  RT budget: {budget_ms:.1f} ms / cube = {target_cubes_per_s:.2f} cubes/s"
    )
    print(
        f"  sustained: {sustained_ms:.1f} ms / cube = {actual_cubes_per_s:.2f} cubes/s "
        f"(margin {margin_ms:+.1f} ms)    "
        f"[= max(wall p50 {total['p50']:.1f}, GPU {gpu_total_ms:.1f})]"
    )

    # Identify the dominant stage so the operator knows WHERE to
    # iterate when the gate fails.
    stage_medians = {k: v["p50"] for k, v in summary.items() if k != "total_pipeline"}
    if stage_medians:
        worst_stage = max(stage_medians, key=stage_medians.get)
        print(
            f"  dominant stage (p50): {worst_stage} = "
            f"{stage_medians[worst_stage]:.1f} ms"
        )

    print()
    # M7.7.1 Phase A.2: GPU SM busy time per cube. The wall-clock
    # stage table above shows what the SCHEDULER sees (perf_counter_ns
    # around CPU calls that mostly LAUNCH async GPU work, so the
    # wall-clock includes the previous kernel's tail). The GPU table
    # below shows what the GPU actually spent its time on. The
    # difference is CPU + sync + cross-iteration overhead — the lever
    # to pull when total_gpu_ms is well below budget.
    if gpu_substage_ms is not None:
        _print_gpu_substage_table(gpu_substage_ms, wall_p50_ms=total["p50"])

    if imager_substage_ms is not None:
        imager_total_gpu = 0.0
        if gpu_substage_ms is not None:
            imager_total_gpu = float(gpu_substage_ms.get("imager_ms", 0.0))
        _print_imager_substage_table(
            imager_substage_ms, imager_total_gpu_ms=imager_total_gpu,
        )

    if sustained_ms <= budget_ms:
        print(
            f"  ✓ PASS — sustained cube cadence {sustained_ms:.1f} ms ≤ "
            f"budget {budget_ms:.1f} ms. Fleet-push is safe (from a "
            f"search-side compute standpoint; this gate does not "
            f"cover transport / dump / C1 / C2)."
        )
        return 0

    print(
        f"  ✗ FAIL — sustained cube cadence {sustained_ms:.1f} ms > "
        f"budget {budget_ms:.1f} ms ({-margin_ms:.1f} ms over). DO NOT "
        f"fleet-push; iterate on the dominant GPU substage above first "
        f"(sustained = max(wall p50, GPU main-stream); the larger of "
        f"the two gates production cadence)."
    )
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search-side speed preflight gate (M7.7+; "
                    "run on a GPU node before any fleet-push)."
    )
    p.add_argument(
        "--n-cubes", type=int, default=60,
        help="Number of cubes to drive through the bench. 60 gives "
             "stable steady-state percentiles after the 5-cube warmup "
             "drop; bump for tighter percentiles. Default 60.",
    )
    p.add_argument(
        "--budget-ms", type=float, default=PROD_BUDGET_MS,
        help=f"Cube-cadence budget (ms). Default {PROD_BUDGET_MS:.1f} ms "
             "= 7.45 cubes/s (production op-point; t_int_search_us="
             "1048.576, cube_cadence_samples=128). PASS = p50 <= budget.",
    )
    p.add_argument(
        "--owner-idx", type=int, default=0,
        help="Coarse-DM owner index for the synthetic source — picks "
             "one of the 8 production DM owners (n01-h0 owns 0, n01-h1 "
             "owns 1, ..., n13-h1 owns 7). The shifts table only "
             "depends weakly on the choice (~±5 sample variation in "
             "max shift across owners). Default 0 (n01 half-0).",
    )
    p.add_argument(
        "--dm-plan-path", type=str, default=PROD_DM_PLAN_PATH,
        help=f"Production DM plan NPZ. Default {PROD_DM_PLAN_PATH}. "
             "Override when testing a new plan.",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="GPU device. Default cuda:0. Combine with "
             "CUDA_VISIBLE_DEVICES=<id> to pin a physical GPU.",
    )
    p.add_argument(
        "--out", type=str, default="/tmp/m77_speed_gate",
        help="Output directory for the bench's stage_timings.ndjson + "
             "summary.json + bench.log. Default /tmp/m77_speed_gate.",
    )
    p.add_argument(
        "--no-m77", action="store_true",
        help="DEBUG: run with --symmetric-shift-padding OFF (the "
             "pre-M7.7 path), still using the prod DM plan + owner "
             "slice. Use this to confirm the M7.7 8.6 ms savings. "
             "Operator A/B tool; do NOT use as the pre-deploy gate.",
    )
    p.add_argument(
        "--warmup-cubes", type=int, default=DEFAULT_WARMUP_CUBES,
        help=f"Number of front cubes to discard before computing "
             f"percentiles (covers Layer-1 burn-in + GPU/NVRTC warmup). "
             f"Default {DEFAULT_WARMUP_CUBES}.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not Path(args.dm_plan_path).exists():
        print(
            f"[gate] FATAL: --dm-plan-path {args.dm_plan_path} not "
            f"found. Production search nodes ship the v2 plan at "
            f"{PROD_DM_PLAN_PATH}; copy it onto this node before "
            f"running the gate.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out).resolve()
    print(
        f"[gate] dsa110-rt search speed preflight "
        f"(UTC {datetime.now(timezone.utc).isoformat(timespec='seconds')})"
    )
    print(
        f"[gate] op-point: n_grid={PROD_N_GRID} n_fdm={PROD_N_FDM} "
        f"t_det={PROD_T_DET} M7.7={'OFF' if args.no_m77 else 'ON'} "
        f"budget={args.budget_ms:.1f}ms"
    )
    print()

    rc = _run_bench(out_dir, args)
    if rc != 0:
        print(
            f"[gate] FATAL: bench subprocess exited {rc}; see "
            f"{out_dir}/bench.log",
            file=sys.stderr,
        )
        return 2

    ndjson = out_dir / "stage_timings.ndjson"
    if not ndjson.exists():
        print(
            f"[gate] FATAL: bench did not produce {ndjson}",
            file=sys.stderr,
        )
        return 2

    rows = _load_stage_timings(ndjson)
    summary_json = out_dir / "summary.json"
    gpu_substage_ms = _load_gpu_substage_ms(summary_json)
    imager_substage_ms = _load_imager_substage_ms(summary_json)
    print()
    print(f"[gate] loaded {len(rows)} per-cube records from {ndjson}")
    if gpu_substage_ms is not None:
        print(
            f"[gate] loaded GPU substage breakdown from {summary_json} "
            f"({int(gpu_substage_ms.get('n', 0))} cubes timed)"
        )
    else:
        print(
            f"[gate] no GPU substage breakdown in {summary_json} "
            f"(older bench, sync path, or CPU run)"
        )
    if imager_substage_ms is not None:
        print(
            f"[gate] loaded imager substage breakdown "
            f"({int(imager_substage_ms.get('n', 0))} cubes timed)"
        )
    print()
    return _print_summary(
        rows,
        warmup_cubes=int(args.warmup_cubes),
        budget_ms=float(args.budget_ms),
        gpu_substage_ms=gpu_substage_ms,
        imager_substage_ms=imager_substage_ms,
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""bench/search_node_throughput.py — M5 Chunk 6b-β throughput +
per-stage latency bench (plan §8 lines 2316-2317).

Drives the ``SearchComputeService`` against a ``SyntheticRxRingSource``
at the configured cube cadence, instruments per-stage wall-clock
timing (combiner+imager, Layer-1 normalisation, detector forward,
emitter fan-out), and writes a compact NDJSON / JSON result set the
operator inspects via ``tools/viz/search_detector_check.py --mode
throughput`` (the viz mode lands in chunk-7 hardening; chunk-6b-β
ships the producer + a CLI summary).

Bench rate budget (production, plan §8 line 2317):
    cube cadence at default ops ........ 134 ms (7.45 cubes/s)
    end-to-end per-cube budget ......... ~30 ms

The bench runs h01 alone and is the per-stage perf gate for the M5
detector pipeline. It does NOT depend on M3 / M4a — the synthetic
RX-ring fills the role of M4a's POSIX-shm ring.

CLI surface (see ``--help`` for the full grid):

  python -m bench.search_node_throughput \\
      [--n-cubes 100]                                \\
      [--cube-cadence-s 0.0]                         \\
      [--t-det 64] [--n-fdm 8] [--n-grid 32]         \\
      [--threshold-sigma 8.0]                        \\
      [--listener-port 11227]                        \\
      [--out bench/reports/<UTC>/throughput/M5/]     \\
      [--quick-smoke]

Outputs (under ``--out``):

  * ``stage_timings.ndjson`` — one record per cube
        ``{cube_id, build_cube_ns, layer1_norm_ns, detector_forward_ns,
           emitter_dispatch_ns, total_pipeline_ns, n_candidates}``
  * ``summary.json``         — config + percentile summary
        ``{config: {...}, n_cubes, percentiles: {p50, p90, p99} per
           stage}``
  * ``bench.log``            — human-readable progress log

Operator gate: per-stage p99 < 30 ms total at default ops geometry
(``T_det=512, N_fdm=32, N_grid=256`` on cuda). The chunk-6b-β default
geometry (T_det=64) is sized for h01-CPU smoke; the operator runs the
full geometry on h01-GPU during M5 hardening.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticRxRingSource,
)
from dsart.trigger.conditions import (  # noqa: E402
    PerCubePerKernelCap,
    PerCubeTotalCap,
    RateLimitTokenBucket,
    SnrThreshold,
)
from dsart.trigger.emitter import (  # noqa: E402
    ConnectionEndpoint,
    TriggerEmitter,
    TriggerEmitterConfig,
)
from dsart.trigger.holdoff import HoldoffStateMachine  # noqa: E402
from dsart.trigger.mock_listener import (  # noqa: E402
    MockListenerConfig,
    MockTriggerListener,
)


_LOG = logging.getLogger("bench.search_node_throughput")


# ---------------------------------------------------------------------------
# Bench config
# ---------------------------------------------------------------------------


# Default geometry: small enough for CPU h01 smoke; the GPU-heavy
# operator-facing run uses the plan-pinned T_det=512, N_fdm=32 (per-GPU),
# N_grid=256 via CLI overrides during M5 hardening.
DEFAULT_T_DET: int = 64
DEFAULT_N_FDM: int = 8
DEFAULT_N_GRID: int = 32
DEFAULT_N_CUBES: int = 50
DEFAULT_CUBE_CADENCE_S: float = 0.0
DEFAULT_THRESHOLD_SIGMA: float = 8.0
DEFAULT_LISTENER_PORT: int = 11227

# --quick-smoke: minimal pass (5 cubes) for the M5.sh DoD path; full
# perf characterisation lives in the operator-facing runs.
QUICK_SMOKE_N_CUBES: int = 5
QUICK_SMOKE_T_DET: int = 32
QUICK_SMOKE_N_FDM: int = 4
QUICK_SMOKE_N_GRID: int = 16


@dataclass(frozen=True, slots=True)
class StageTimingRecord:
    """One cube's stage-timing record (NDJSON record)."""
    cube_id: int
    n_candidates: int
    n_records: int
    build_cube_ns: int
    layer1_norm_ns: int
    detector_forward_ns: int
    emitter_dispatch_ns: int
    total_pipeline_ns: int

    def to_json(self) -> Dict[str, int]:
        return {
            "cube_id": self.cube_id,
            "n_candidates": self.n_candidates,
            "n_records": self.n_records,
            "build_cube_ns": self.build_cube_ns,
            "layer1_norm_ns": self.layer1_norm_ns,
            "detector_forward_ns": self.detector_forward_ns,
            "emitter_dispatch_ns": self.emitter_dispatch_ns,
            "total_pipeline_ns": self.total_pipeline_ns,
        }


# ---------------------------------------------------------------------------
# Helpers: percentile rollups
# ---------------------------------------------------------------------------


def percentiles(values_ns: Sequence[int]) -> Dict[str, float]:
    """Return {p50, p90, p99, mean, max} in milliseconds."""
    if not values_ns:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    arr = np.asarray(values_ns, dtype=np.int64)
    arr_ms = arr.astype(np.float64) / 1.0e6
    return {
        "p50": float(np.percentile(arr_ms, 50)),
        "p90": float(np.percentile(arr_ms, 90)),
        "p99": float(np.percentile(arr_ms, 99)),
        "mean": float(arr_ms.mean()),
        "max": float(arr_ms.max()),
    }


# ---------------------------------------------------------------------------
# DM grid (synthetic; the bench doesn't gate on dispersion correctness)
# ---------------------------------------------------------------------------


def _build_dm_grids(n_fdm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic coarse/fine DM grid for the synthetic source."""
    n_coarse = max(2, n_fdm // 2)
    n_fine_per_coarse = max(1, n_fdm // n_coarse)
    coarse = np.linspace(50.0, 200.0, n_coarse, dtype=np.float64)
    spacing = (
        (coarse[1] - coarse[0]) / n_fine_per_coarse if n_coarse > 1 else 1.0
    )
    fine = np.concatenate(
        [coarse[c] + np.arange(n_fine_per_coarse) * spacing for c in range(n_coarse)]
    )
    fine = fine[:n_fdm]
    fine_to_coarse = np.repeat(
        np.arange(n_coarse, dtype=np.int64), n_fine_per_coarse
    )[:n_fdm]
    return coarse, fine, fine_to_coarse


# ---------------------------------------------------------------------------
# Bench main
# ---------------------------------------------------------------------------


async def _bench_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_log_path = out_dir / "bench.log"
    bench_log_handler = logging.FileHandler(bench_log_path, mode="w")
    bench_log_handler.setLevel(logging.INFO)
    bench_log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(bench_log_handler)
    _LOG.addHandler(logging.StreamHandler(sys.stdout))

    if args.quick_smoke:
        n_cubes = QUICK_SMOKE_N_CUBES
        t_det = QUICK_SMOKE_T_DET
        n_fdm = QUICK_SMOKE_N_FDM
        n_grid = QUICK_SMOKE_N_GRID
    else:
        n_cubes = int(args.n_cubes)
        t_det = int(args.t_det)
        n_fdm = int(args.n_fdm)
        n_grid = int(args.n_grid)

    threshold_sigma = float(args.threshold_sigma)
    cube_cadence_s = float(args.cube_cadence_s)

    _LOG.info(
        "bench config: n_cubes=%d cadence=%.3fs T_det=%d N_fdm=%d N_grid=%d "
        "threshold=%.2fσ",
        n_cubes, cube_cadence_s, t_det, n_fdm, n_grid, threshold_sigma,
    )

    coarse_dm, fine_dm, fine_to_coarse = _build_dm_grids(n_fdm)
    src = SyntheticRxRingSource(
        n_cubes=n_cubes,
        t_det=t_det,
        n_fdm=n_fdm,
        n_grid=n_grid,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        rng=np.random.default_rng(int(args.rng_seed)),
        cube_cadence_s=cube_cadence_s,
    )
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cube_dtype = torch.float16 if (
        args.cube_dtype == "fp16" and device != "cpu"
    ) else torch.float32
    detector_dtype = torch.float16 if (
        args.cube_dtype == "fp16" and device != "cpu"
    ) else torch.float32
    _LOG.info("device=%s cube_dtype=%s", device, cube_dtype)
    detector = DeterministicDetector(
        threshold_sigma=threshold_sigma,
        detector_version="v1.M5",
        search_node_id=1,
        gpu_half=1,
        dtype=detector_dtype,
    )
    detector = detector.to(torch.device(device))
    pipeline = CubePipeline(
        config=CubePipelineConfig(
            n_grid=n_grid,
            edge_mask_kernel_support=5,
            cube_dtype=cube_dtype,
            device=device,
        ),
        detector=detector,
        layer1_state=Layer1State(n_fdm=n_fdm, n_burnin_cubes=5),
    )

    # ---- Listener + emitter setup ----
    listener_cfg = MockListenerConfig(
        accept_rate=1.0, accept_delay_ms=0.0, completed_delay_ms=0.5,
        send_completed=True,
    )
    listener = MockTriggerListener(
        host="127.0.0.1", port=int(args.listener_port), config=listener_cfg,
    )
    await listener.start()
    _LOG.info("MockTriggerListener up on %s:%d", listener.host, listener.port)
    endpoint = ConnectionEndpoint(host=listener.host, port=listener.port)
    emitter_cfg = TriggerEmitterConfig(
        search_node_id=1,
        gpu_half=1,
        endpoints=[endpoint],
        conditions=[
            SnrThreshold(min_snr=threshold_sigma),
            PerCubePerKernelCap(max_per_kernel=4),
            PerCubeTotalCap(max_total=16),
            RateLimitTokenBucket(rate_per_s=10.0, burst=10),
        ],
        holdoff=HoldoffStateMachine(holdoff_ms=50.0),
    )
    emitter = TriggerEmitter(emitter_cfg)
    await emitter.start()

    # ---- Drain the source, collecting stage timings ----
    records: List[StageTimingRecord] = []
    bench_start_ns = time.perf_counter_ns()
    try:
        async with src:
            async for slot in src:
                t_dispatch_start = time.perf_counter_ns()
                result = pipeline.process(slot)
                emit_t0 = time.perf_counter_ns()
                emitted = await emitter.process_candidates(
                    slot.cube_id, result.candidates,
                )
                emit_t1 = time.perf_counter_ns()
                rec = StageTimingRecord(
                    cube_id=slot.cube_id,
                    n_candidates=len(result.candidates),
                    n_records=len(emitted),
                    build_cube_ns=int(result.stage_timings_ns["build_cube"]),
                    layer1_norm_ns=int(result.stage_timings_ns["layer1_norm"]),
                    detector_forward_ns=int(
                        result.stage_timings_ns["detector_forward"]
                    ),
                    emitter_dispatch_ns=int(emit_t1 - emit_t0),
                    total_pipeline_ns=int(emit_t1 - t_dispatch_start),
                )
                records.append(rec)
                if (slot.cube_id + 1) % max(1, n_cubes // 10) == 0:
                    _LOG.info(
                        "cube=%d/%d total=%.2fms detector=%.2fms cands=%d",
                        slot.cube_id + 1, n_cubes,
                        rec.total_pipeline_ns / 1.0e6,
                        rec.detector_forward_ns / 1.0e6,
                        rec.n_candidates,
                    )
                await src.release(slot.cube_id)
    finally:
        await emitter.stop()
        await listener.stop()
    bench_wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9

    # ---- Write outputs ----
    ndjson_path = out_dir / "stage_timings.ndjson"
    with ndjson_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json()) + "\n")
    _LOG.info("wrote %s (%d records)", ndjson_path, len(records))

    summary = {
        "schema_version": 1,
        "bench": "search_node_throughput",
        "milestone": "M5",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cubes": n_cubes,
            "cube_cadence_s": cube_cadence_s,
            "t_det": t_det,
            "n_fdm": n_fdm,
            "n_grid": n_grid,
            "threshold_sigma": threshold_sigma,
            "rng_seed": int(args.rng_seed),
            "device": device,
            "cube_dtype": str(cube_dtype).rsplit(".", 1)[-1],
        },
        "wall_clock_s": bench_wall_s,
        "achieved_cubes_per_s": (
            len(records) / bench_wall_s if bench_wall_s > 0 else 0.0
        ),
        "n_cubes_processed": len(records),
        "n_candidates_total": int(sum(r.n_candidates for r in records)),
        "n_records_total": int(sum(r.n_records for r in records)),
        "percentiles_ms": {
            "build_cube": percentiles([r.build_cube_ns for r in records]),
            "layer1_norm": percentiles([r.layer1_norm_ns for r in records]),
            "detector_forward": percentiles(
                [r.detector_forward_ns for r in records]
            ),
            "emitter_dispatch": percentiles(
                [r.emitter_dispatch_ns for r in records]
            ),
            "total_pipeline": percentiles(
                [r.total_pipeline_ns for r in records]
            ),
        },
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    _LOG.info("wrote %s", summary_path)
    pct = summary["percentiles_ms"]["total_pipeline"]
    _LOG.info(
        "throughput summary: %.2f cubes/s · total p50=%.2fms p99=%.2fms",
        summary["achieved_cubes_per_s"], pct["p50"], pct["p99"],
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 chunk-6b-β throughput + per-stage latency bench"
    )
    parser.add_argument("--n-cubes", type=int, default=DEFAULT_N_CUBES)
    parser.add_argument(
        "--cube-cadence-s", type=float, default=DEFAULT_CUBE_CADENCE_S,
        help="Wall-clock cube cadence (default 0 = drain as fast as the "
             "consumer can; use ~0.134 to mimic production cadence).",
    )
    parser.add_argument("--t-det", type=int, default=DEFAULT_T_DET)
    parser.add_argument("--n-fdm", type=int, default=DEFAULT_N_FDM)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument(
        "--threshold-sigma", type=float, default=DEFAULT_THRESHOLD_SIGMA,
    )
    parser.add_argument(
        "--listener-port", type=int, default=DEFAULT_LISTENER_PORT,
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="torch device for the detector + cube ('cpu', 'cuda', "
             "'cuda:0', 'auto'). Combiner + imager always run on CPU "
             "in chunk-6b-α (the GPU sparse-scatter+cuFFT lands in the "
             "production hardening pass).",
    )
    parser.add_argument(
        "--cube-dtype", type=str, default="fp32", choices=("fp32", "fp16"),
        help="Cube + detector dtype. fp16 only valid with --device cuda* "
             "(plan §3.6.11 production pin).",
    )
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "throughput" / "M5"),
    )
    parser.add_argument("--quick-smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())

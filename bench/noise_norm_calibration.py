#!/usr/bin/env python3
"""bench/noise_norm_calibration.py — M5 Chunk 6b-γ noise-norm
calibration bench (plan §8 line 2316 + F13 + F3).

Drives the chunk-6b-α ``CubePipeline`` against a noise-only
``SyntheticRxRingSource`` (no injections; complex Gaussian per-cell
σ=1/√2 each component → unit-σ post-image after Layer-1 normalisation),
collects per-cube candidate distributions, and writes the empirical-
vs-analytic FAR table per kernel × θ ∈ {6, 7, 8, 9, 10}.

Plan §8 line 2316 specifies "FAR within [0.5×, 2.0×] of analytic at
θ=8 across 30 s of synthetic cubes". Per F13 the literal θ=8 check is
unobservable at bench-scale geometries (≪ 1 expected event in ≤ 30 s
of cubes), so the bench instead produces the empirical curve across
the broader θ-grid; the operator inspects shape vs the analytic
Gaussian tail and the {0.5×, 2.0×} bound is asserted at the lowest θ
where the analytic expected count is ≥ 10 (typically θ=6 at the bench
geometries).

Per-cube-per-kernel analytic count (F3):

    N_eff = (T_det · N_fdm · N_grid²) / (K_img · K_dm · K_time)
    expected_per_cube_per_kernel(θ) = N_eff · 0.5·erfc(θ/√2)

CLI surface:

  python -m bench.noise_norm_calibration \\
      [--n-cubes 200]                                      \\
      [--t-det 64] [--n-fdm 8] [--n-grid 32]               \\
      [--theta-grid 6 7 8 9 10]                            \\
      [--detector-threshold-sigma 6.0]                     \\
      [--burnin-cubes 5]                                   \\
      [--listener-port 11227]                              \\
      [--out bench/reports/<UTC>/noise_norm/M5/]           \\
      [--quick-smoke]

Outputs (under ``--out``):

  * ``noise_norm.ndjson``     — one record per (cube_id, kernel_id) of
        ``{cube_id, kernel_id, k_dm, k_time, snr_max, n_above_theta}``.
  * ``far_curve.json``        — per-kernel × θ table:
        ``{kernel_id: {theta: {expected, observed, ratio}}}``.
  * ``summary.json``          — config + bench-level rollup +
        gate-status stamp.
  * ``bench.log``             — human progress.

Operator gate: ``far_curve.json`` plus the curve viz (chunk-7 viz);
the bench's PASS/FAIL is informational at the lowest-θ-with-N≥10 cell
(ratio ∈ [0.5, 2.0]) but the operator can still sign off on the curve
shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

import torch  # noqa: E402

from dsart.detector.forward import DeterministicDetector  # noqa: E402
from dsart.detector.kernels import build_kernel_bank  # noqa: E402
from dsart.noise_norm.layer1 import Layer1State  # noqa: E402
from dsart.services.cube_pipeline import (  # noqa: E402
    CubePipeline,
    CubePipelineConfig,
)
from dsart.services.rx_ring import (  # noqa: E402
    SyntheticRxRingSource,
)


_LOG = logging.getLogger("bench.noise_norm_calibration")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_T_DET: int = 64
DEFAULT_N_FDM: int = 8
DEFAULT_N_GRID: int = 32
DEFAULT_N_CUBES: int = 200
DEFAULT_THETA_GRID: Tuple[float, ...] = (6.0, 7.0, 8.0, 9.0, 10.0)
DEFAULT_DETECTOR_THRESHOLD_SIGMA: float = 6.0
DEFAULT_BURNIN_CUBES: int = 5
DEFAULT_LISTENER_PORT: int = 11227

# --quick-smoke: minimal pass for the M5.sh DoD path; full FAR-curve
# characterisation lives in operator-facing runs.
QUICK_SMOKE_N_CUBES: int = 12
QUICK_SMOKE_T_DET: int = 32
QUICK_SMOKE_N_FDM: int = 4
QUICK_SMOKE_N_GRID: int = 16
QUICK_SMOKE_BURNIN_CUBES: int = 3


# ---------------------------------------------------------------------------
# Analytic helpers (F3)
# ---------------------------------------------------------------------------


def gaussian_tail(theta: float) -> float:
    """One-sided Gaussian tail probability ``0.5 · erfc(θ/√2)``.

    Same formula as ``tools.viz.search_helpers.gaussian_tail_far``;
    re-implemented here to keep the bench's analytic side without a
    cross-import (the bench is imported by the test path which doesn't
    need the full viz dependency).
    """
    return 0.5 * math.erfc(float(theta) / math.sqrt(2.0))


def n_eff_per_cube_per_kernel(
    *, t_det: int, n_fdm: int, n_grid: int,
    k_img_volume: int, k_dm_width: int, k_time_width: int,
) -> float:
    """Per-cube effective-number-of-cells per kernel triple (plan F3)."""
    if k_img_volume < 1 or k_dm_width < 1 or k_time_width < 1:
        raise ValueError(
            f"kernel widths must be ≥ 1; got img={k_img_volume}, "
            f"dm={k_dm_width}, time={k_time_width}"
        )
    n_cells = float(t_det) * float(n_fdm) * float(n_grid) * float(n_grid)
    return n_cells / (
        float(k_img_volume) * float(k_dm_width) * float(k_time_width)
    )


def lowest_theta_with_n_expected_geq(
    theta_grid: Sequence[float],
    n_expected_per_theta: Dict[float, float],
    *,
    n_min: float = 10.0,
) -> Optional[float]:
    """Find the lowest θ in the grid whose summed expected count
    across kernels is ≥ ``n_min``. Returns None if none qualifies.
    Used for the F13 {0.5×, 2.0×} ratio gate.
    """
    for theta in sorted(theta_grid):
        if n_expected_per_theta.get(theta, 0.0) >= n_min:
            return theta
    return None


# ---------------------------------------------------------------------------
# DM grid (deterministic; bench doesn't gate on dispersion correctness)
# ---------------------------------------------------------------------------


def _build_dm_grids(n_fdm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        burnin_cubes = QUICK_SMOKE_BURNIN_CUBES
    else:
        n_cubes = int(args.n_cubes)
        t_det = int(args.t_det)
        n_fdm = int(args.n_fdm)
        n_grid = int(args.n_grid)
        burnin_cubes = int(args.burnin_cubes)

    theta_grid = tuple(float(x) for x in args.theta_grid)
    detector_threshold_sigma = float(args.detector_threshold_sigma)

    if min(theta_grid) < detector_threshold_sigma:
        _LOG.warning(
            "theta_grid min=%.2f < detector_threshold_sigma=%.2f; lowest "
            "thresholds in the grid will see truncated counts. Consider "
            "lowering --detector-threshold-sigma.",
            min(theta_grid), detector_threshold_sigma,
        )

    _LOG.info(
        "bench config: n_cubes=%d burnin=%d T_det=%d N_fdm=%d N_grid=%d "
        "theta_grid=%s detector_θ=%.2f",
        n_cubes, burnin_cubes, t_det, n_fdm, n_grid,
        list(theta_grid), detector_threshold_sigma,
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
        cube_cadence_s=0.0,
    )

    pipeline = CubePipeline(
        config=CubePipelineConfig(
            n_grid=n_grid, edge_mask_kernel_support=5,
            cube_dtype=torch.float32, device="cpu",
        ),
        detector=DeterministicDetector(
            threshold_sigma=detector_threshold_sigma,
            detector_version="v1.M5",
            search_node_id=1,
            gpu_half=1,
            dtype=torch.float32,
        ),
        layer1_state=Layer1State(n_fdm=n_fdm, n_burnin_cubes=burnin_cubes),
    )

    # Build a kernel-id → (k_dm, k_time) lookup for the analytic table.
    bank = build_kernel_bank()
    kernel_lookup: Dict[str, Tuple[int, int]] = {
        k.kernel_id: (k.k_dm_width, k.k_time_width) for k in bank
    }

    # Per-(kernel_id, theta) observed counter; cumulative across cubes
    # (post-burnin). Burn-in cubes are recorded but excluded from the
    # gate calculation (Layer-2 σ_k EMA is still warming).
    observed_post_burnin: Dict[str, Dict[float, int]] = {
        kid: {t: 0 for t in theta_grid} for kid in kernel_lookup
    }
    observed_burnin: Dict[str, Dict[float, int]] = {
        kid: {t: 0 for t in theta_grid} for kid in kernel_lookup
    }

    # Per-cube NDJSON records; one record per (cube_id, kernel_id) with
    # the per-theta counts and the cube's max snr at that kernel.
    ndjson_path = out_dir / "noise_norm.ndjson"
    bench_start_ns = time.perf_counter_ns()
    with ndjson_path.open("w") as ndjson_fh:
        async with src:
            cube_idx = 0
            async for slot in src:
                result = pipeline.process(slot)
                in_burnin = cube_idx < burnin_cubes
                # Bin candidates per (kernel, theta).
                per_kernel_snrs: Dict[str, List[float]] = {}
                for cand in result.candidates:
                    per_kernel_snrs.setdefault(cand.kernel_id, []).append(
                        float(cand.snr)
                    )
                for kid, snrs in per_kernel_snrs.items():
                    snr_max = max(snrs)
                    record = {
                        "cube_id": int(slot.cube_id),
                        "kernel_id": kid,
                        "in_burnin": in_burnin,
                        "snr_max": snr_max,
                        "n_above_theta": {
                            f"{t:.1f}": sum(1 for s in snrs if s >= t)
                            for t in theta_grid
                        },
                    }
                    ndjson_fh.write(json.dumps(record) + "\n")
                    counter = (
                        observed_burnin if in_burnin
                        else observed_post_burnin
                    )
                    for t in theta_grid:
                        counter[kid][t] += sum(1 for s in snrs if s >= t)
                if (cube_idx + 1) % max(1, n_cubes // 10) == 0:
                    _LOG.info(
                        "cube=%d/%d (burnin=%s) total_cands=%d",
                        cube_idx + 1, n_cubes, in_burnin,
                        len(result.candidates),
                    )
                cube_idx += 1
                await src.release(slot.cube_id)
    bench_wall_s = (time.perf_counter_ns() - bench_start_ns) / 1.0e9

    # ---- Build the analytic table ----
    n_post_burnin = max(0, n_cubes - burnin_cubes)
    far_table: Dict[str, Dict[str, Dict[str, float]]] = {}
    summed_expected_per_theta: Dict[float, float] = {t: 0.0 for t in theta_grid}
    summed_observed_per_theta: Dict[float, int] = {t: 0 for t in theta_grid}
    for kid, (k_dm, k_time) in kernel_lookup.items():
        n_eff_per_cube = n_eff_per_cube_per_kernel(
            t_det=t_det, n_fdm=n_fdm, n_grid=n_grid,
            k_img_volume=1,  # v1 image kernels are 1×1 deltas (D10)
            k_dm_width=k_dm, k_time_width=k_time,
        )
        per_theta: Dict[str, Dict[str, float]] = {}
        for t in theta_grid:
            expected = n_eff_per_cube * gaussian_tail(t) * n_post_burnin
            observed = observed_post_burnin[kid][t]
            ratio = (observed / expected) if expected > 0 else float("inf")
            per_theta[f"{t:.1f}"] = {
                "expected": expected,
                "observed": int(observed),
                "ratio": float(ratio),
            }
            summed_expected_per_theta[t] += expected
            summed_observed_per_theta[t] += observed
        far_table[kid] = per_theta

    # F13 gate: lowest θ with summed expected ≥ 10; ratio ∈ [0.5, 2.0].
    gate_theta = lowest_theta_with_n_expected_geq(
        theta_grid, summed_expected_per_theta, n_min=10.0,
    )
    if gate_theta is None:
        gate_status = "no_observable_theta"
        gate_ratio = None
    else:
        gate_observed = summed_observed_per_theta[gate_theta]
        gate_expected = summed_expected_per_theta[gate_theta]
        gate_ratio = (
            gate_observed / gate_expected if gate_expected > 0 else float("inf")
        )
        if 0.5 <= gate_ratio <= 2.0:
            gate_status = "PASS"
        else:
            gate_status = "INFO_OUT_OF_BAND"

    far_curve_path = out_dir / "far_curve.json"
    with far_curve_path.open("w") as fh:
        json.dump(
            {
                "schema_version": 1,
                "kernels": far_table,
                "summed_expected_per_theta": {
                    f"{t:.1f}": v for t, v in summed_expected_per_theta.items()
                },
                "summed_observed_per_theta": {
                    f"{t:.1f}": v for t, v in summed_observed_per_theta.items()
                },
            },
            fh, indent=2, sort_keys=True,
        )

    summary_path = out_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "bench": "noise_norm_calibration",
        "milestone": "M5",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cubes": n_cubes,
            "n_post_burnin_cubes": n_post_burnin,
            "burnin_cubes": burnin_cubes,
            "t_det": t_det,
            "n_fdm": n_fdm,
            "n_grid": n_grid,
            "theta_grid": list(theta_grid),
            "detector_threshold_sigma": detector_threshold_sigma,
            "rng_seed": int(args.rng_seed),
            "device": "cpu",
            "cube_dtype": "float32",
        },
        "wall_clock_s": bench_wall_s,
        "n_kernels": len(kernel_lookup),
        "summed_observed_per_theta": {
            f"{t:.1f}": v for t, v in summed_observed_per_theta.items()
        },
        "summed_expected_per_theta": {
            f"{t:.1f}": v for t, v in summed_expected_per_theta.items()
        },
        "gate": {
            "theta": gate_theta,
            "ratio": gate_ratio,
            "status": gate_status,
            "ratio_band": [0.5, 2.0],
            "n_min": 10.0,
        },
    }
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    _LOG.info("wrote %s", ndjson_path)
    _LOG.info("wrote %s", far_curve_path)
    _LOG.info("wrote %s", summary_path)
    if gate_theta is not None:
        _LOG.info(
            "F13 gate: θ=%.1f, ratio=%.3f (band [0.5, 2.0]) → %s",
            gate_theta, gate_ratio if gate_ratio is not None else float("nan"),
            gate_status,
        )
    else:
        _LOG.info(
            "F13 gate: no θ in grid had ≥10 expected events; "
            "operator inspects the curve shape (chunk-7 viz)"
        )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 chunk-6b-γ noise-norm calibration bench"
    )
    parser.add_argument("--n-cubes", type=int, default=DEFAULT_N_CUBES)
    parser.add_argument("--t-det", type=int, default=DEFAULT_T_DET)
    parser.add_argument("--n-fdm", type=int, default=DEFAULT_N_FDM)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument(
        "--theta-grid", type=float, nargs="+",
        default=list(DEFAULT_THETA_GRID),
    )
    parser.add_argument(
        "--detector-threshold-sigma", type=float,
        default=DEFAULT_DETECTOR_THRESHOLD_SIGMA,
    )
    parser.add_argument(
        "--burnin-cubes", type=int, default=DEFAULT_BURNIN_CUBES,
    )
    parser.add_argument(
        "--listener-port", type=int, default=DEFAULT_LISTENER_PORT,
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "bench" / "reports" / "noise_norm" / "M5"),
    )
    parser.add_argument("--quick-smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_bench_main(args))


if __name__ == "__main__":
    sys.exit(main())

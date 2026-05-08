#!/usr/bin/env python3
"""bench/clusterer_throughput.py — M6 Chunk 6 D5 fallback gate.

Characterises the M6 per-cube clusterer
(``dsart.cluster.forward.cluster_candidates``) at production-like
candidate-stream loads, so the M6 D5 backend decision can be made:
HDBSCAN remains primary if p99 ≤ 50 ms at the production geometry, else
the runtime falls back to sklearn DBSCAN (M6_PLAN_FIXES.md D5).

Per-cube production geometry (D4/D5):
  * worst-case ~1000 candidates per cube at θ = 8 σ with the cross-
    kernel merger applied;
  * cube cadence ~134 ms (8 cubes/s) at the default ops point;
  * p99 latency budget = 50 ms (~38 % of the 134 ms cube period — any
    longer and the next cube's detector forward catches up to the
    clusterer worker on the ThreadPool).

This bench drives the public clusterer API ONLY (no service / pipeline
wiring); it consumes ``cluster_candidates(cands, geom, config=...)``
and times the wall clock per call.

CLI surface (see ``--help`` for the full grid):

  python -m bench.clusterer_throughput \\
      [--report-dir bench/reports/M6/clusterer_throughput] \\
      [--backend hdbscan]               \\
      [--feature-mode int]              \\
      [--n-cubes 200]                   \\
      [--n-cands-per-cube 1000]         \\
      [--rng-seed 42]                   \\
      [--device cpu]                    \\
      [--p99-budget-ms 50.0]

Outputs (under ``--report-dir``):

  * ``report.json`` — single-shot bench report with config, summary
    percentiles (p50/p90/p99/max/mean/n_cubes_run/n_cands_total/
    n_records_total/n_clusters_total/n_noise_total), and the D5
    fallback predicate (``passes`` = p99 ≤ budget).
  * ``per_cube.csv``  — one row per cube
    (``cube_id, n_cands, n_records, n_clusters, n_noise, wall_ms``).

Synthetic candidate-stream generator (per cube):

  * ``n_cands ~ Poisson(λ = n_cands_per_cube)`` so the load varies
    cube-to-cube. (Realistic — most cubes are quiet; occasional
    high-FAR cubes spike to the worst-case 1000+ load.)
  * One injected burst kernel of ``K ~ Uniform[2, 20]`` candidates
    co-located in (l_pix, m_pix, fine_dm_idx, t_in_cube) with a small
    jitter to exercise the clusterer's grouping logic.
  * Remaining candidates are noise — uniform over the cube's
    (l_pix, m_pix, fine_dm_idx, t_in_cube) grid with random
    ``width_samples`` ∈ {1, 2, 4, 8, 16, 32, 64, 128}.
  * SNR drawn ``8 + Exponential(scale=2)`` for noise,
    ``15 + Exponential(scale=10)`` for burst-kernel members.
  * Reproducibility: ``np.random.default_rng(rng_seed)`` per run; per-
    cube sub-RNG seeded ``rng_seed * 1_000_003 + cube_id``.

The bench is a producer of timing data; the M6.sh DoD step
``bench_clusterer_throughput`` invokes it with the production load
(``--n-cubes 200 --n-cands-per-cube 1000``) and the report.json is
inspected by the operator before flipping the runtime backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

from dsart.cluster.forward import (  # noqa: E402
    ClustererBackend,
    ClustererConfig,
    cluster_candidates,
)
from dsart.cluster.features import FeatureMode  # noqa: E402
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    CubeGeometry,
)


_LOG = logging.getLogger("bench.clusterer_throughput")


# ---------------------------------------------------------------------------
# Defaults — production-geometry numbers locked in M6_PLAN_FIXES.md D4/D5.
# ---------------------------------------------------------------------------

DEFAULT_REPORT_DIR: Path = (
    REPO_ROOT / "bench" / "reports" / "M6" / "clusterer_throughput"
)
DEFAULT_BACKEND: str = ClustererBackend.HDBSCAN
DEFAULT_FEATURE_MODE: str = FeatureMode.INT
DEFAULT_N_CUBES: int = 200
DEFAULT_N_CANDS_PER_CUBE: int = 1000
DEFAULT_RNG_SEED: int = 42
DEFAULT_DEVICE: str = "cpu"
DEFAULT_P99_BUDGET_MS: float = 50.0

# Cube geometry (production defaults — match the spec verbatim).
GEOM_SAMPLE_PERIOD_SPECNUM: int = 16
GEOM_T_DET: int = 256
GEOM_N_GRID: int = 256
GEOM_N_FDM_IN_CUBE: int = 32
GEOM_SAMPLE_PERIOD_US: float = 131.072
GEOM_CELL_L_RAD: float = 1.5e-4
GEOM_CELL_M_RAD: float = 1.5e-4
GEOM_L0_RAD: float = 0.0
GEOM_M0_RAD: float = 0.0
GEOM_FINE_DM_LO: float = 50.0
GEOM_FINE_DM_HI: float = 800.0
GEOM_MJD_START_BASE: float = 60942.0
GEOM_CUBE_PERIOD_S: float = 134e-3  # cube cadence at default ops

# Allowed boxcar widths (matches DETECTOR_K_TIME_WIDTHS in constants.py;
# kept inline here so the bench has no transitive dep on constants).
NOISE_WIDTH_CHOICES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

# Burst-kernel jitter radius (in detector-frame units). Tight enough
# that the clusterer reliably groups the burst members under default
# eps=10 cityblock; loose enough to exercise the inner-loop NMS.
BURST_LM_JITTER: int = 1
BURST_DM_JITTER: int = 1
BURST_T_JITTER: int = 2
BURST_K_MIN: int = 2
BURST_K_MAX: int = 20

# Noise + burst SNR distributions (M6 D5: matches what the M5 detector
# emits at θ=8σ). Noise candidates cluster around the threshold;
# burst-kernel members carry a heavier tail to mimic real bursts.
NOISE_SNR_OFFSET: float = 8.0
NOISE_SNR_SCALE: float = 2.0
BURST_SNR_OFFSET: float = 15.0
BURST_SNR_SCALE: float = 10.0

# Detector-frame defaults used to populate Candidate fields that the
# clusterer ignores but the dataclass __post_init__ still validates.
DEFAULT_KERNEL_ID: str = "unit:d1:b4"
DEFAULT_DETECTOR_VERSION: str = "v1.M6.bench"


# ---------------------------------------------------------------------------
# Per-cube wall-clock record (driver-internal; not persisted as-is).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CubeTimingRecord:
    """One cube's clusterer wall-clock + record-count snapshot."""

    cube_id: int
    n_cands: int
    n_records: int
    n_clusters: int
    n_noise: int
    wall_ms: float


# ---------------------------------------------------------------------------
# Synthetic candidate-stream generator
# ---------------------------------------------------------------------------


def _build_geometry(cube_id: int) -> CubeGeometry:
    """Build a production-geometry ``CubeGeometry`` for cube ``cube_id``.

    The sample0 specnum advances by ``t_det * sample_period_specnum`` per
    cube; ``mjd_start`` advances by 134 ms (the default-ops cube
    cadence). Both choices match the spec's reference geometry block.
    """
    specnum_start = int(cube_id * GEOM_T_DET * GEOM_SAMPLE_PERIOD_SPECNUM)
    fine_dm_pc_cc = np.linspace(
        GEOM_FINE_DM_LO, GEOM_FINE_DM_HI, GEOM_N_FDM_IN_CUBE, dtype=np.float64
    )
    mjd_start = GEOM_MJD_START_BASE + cube_id * GEOM_CUBE_PERIOD_S / 86400.0
    return CubeGeometry(
        cube_id=int(cube_id),
        specnum_start=specnum_start,
        sample_period_specnum=GEOM_SAMPLE_PERIOD_SPECNUM,
        t_det=GEOM_T_DET,
        n_grid=GEOM_N_GRID,
        n_fdm_in_cube=GEOM_N_FDM_IN_CUBE,
        sample_period_us=GEOM_SAMPLE_PERIOD_US,
        cell_l_rad=GEOM_CELL_L_RAD,
        cell_m_rad=GEOM_CELL_M_RAD,
        l0_rad=GEOM_L0_RAD,
        m0_rad=GEOM_M0_RAD,
        fine_dm_pc_cc=fine_dm_pc_cc,
        mjd_start=float(mjd_start),
    )


def _make_candidate(
    *,
    geom: CubeGeometry,
    l_pix: int,
    m_pix: int,
    fine_dm_idx: int,
    t_in_cube: int,
    width_samples: int,
    snr: float,
) -> Candidate:
    """Build one Candidate consistent with the cube geometry.

    The clusterer recovers ``fine_dm_idx`` from ``cand.dm_fine`` via
    searchsorted on ``geom.fine_dm_pc_cc``; we set ``dm_fine`` to the
    exact grid value so the round-trip is lossless. ``event_specnum``
    is computed from ``t_in_cube`` so the ``t_in_cube`` recovery in
    features.py matches the input we synthesised.
    """
    return Candidate(
        l=float(l_pix),
        m=float(m_pix),
        dm_fine=float(geom.fine_dm_pc_cc[fine_dm_idx]),
        dm_idx=int(fine_dm_idx),
        event_specnum=int(
            geom.specnum_start + t_in_cube * geom.sample_period_specnum
        ),
        width_samples=int(width_samples),
        kernel_id=DEFAULT_KERNEL_ID,
        snr=float(snr),
        detector_version=DEFAULT_DETECTOR_VERSION,
        flags=int(CandidateFlags.NONE),
        search_node_id=0,
        gpu_half=0,
    )


def _generate_cube_candidates(
    *,
    geom: CubeGeometry,
    n_cands_per_cube_lambda: int,
    rng: np.random.Generator,
) -> List[Candidate]:
    """Synthesise one cube's candidate list per the spec's recipe.

    Returns a list of ``Candidate`` records; length is Poisson-drawn
    around ``n_cands_per_cube_lambda``. The first ``K`` rows are the
    burst-kernel members (so a downstream consumer that wants to slice
    off "the burst" can take ``cands[:K]``); the rest are noise rows.
    """
    n_cands = int(rng.poisson(n_cands_per_cube_lambda))
    if n_cands <= 0:
        return []

    # ----- Burst kernel -----
    k_burst = int(rng.integers(BURST_K_MIN, BURST_K_MAX + 1))
    k_burst = min(k_burst, n_cands)
    centre_l = int(rng.integers(0, geom.n_grid))
    centre_m = int(rng.integers(0, geom.n_grid))
    centre_fdm = int(rng.integers(0, geom.n_fdm_in_cube))
    centre_t = int(rng.integers(0, geom.t_det))

    cands: List[Candidate] = []
    for _ in range(k_burst):
        l_pix = int(np.clip(
            centre_l + int(rng.integers(-BURST_LM_JITTER, BURST_LM_JITTER + 1)),
            0, geom.n_grid - 1,
        ))
        m_pix = int(np.clip(
            centre_m + int(rng.integers(-BURST_LM_JITTER, BURST_LM_JITTER + 1)),
            0, geom.n_grid - 1,
        ))
        fdm = int(np.clip(
            centre_fdm + int(rng.integers(-BURST_DM_JITTER, BURST_DM_JITTER + 1)),
            0, geom.n_fdm_in_cube - 1,
        ))
        t_in_cube = int(np.clip(
            centre_t + int(rng.integers(-BURST_T_JITTER, BURST_T_JITTER + 1)),
            0, geom.t_det - 1,
        ))
        # Burst members carry a single shared boxcar width (a real burst
        # at one DM trial is best matched by one (k_dm, k_time) triple).
        width = int(rng.choice(NOISE_WIDTH_CHOICES))
        snr = BURST_SNR_OFFSET + float(rng.exponential(BURST_SNR_SCALE))
        cands.append(_make_candidate(
            geom=geom,
            l_pix=l_pix, m_pix=m_pix,
            fine_dm_idx=fdm, t_in_cube=t_in_cube,
            width_samples=width, snr=snr,
        ))

    # ----- Noise rows -----
    n_noise = n_cands - k_burst
    if n_noise > 0:
        noise_l = rng.integers(0, geom.n_grid, size=n_noise)
        noise_m = rng.integers(0, geom.n_grid, size=n_noise)
        noise_fdm = rng.integers(0, geom.n_fdm_in_cube, size=n_noise)
        noise_t = rng.integers(0, geom.t_det, size=n_noise)
        noise_widths = rng.choice(NOISE_WIDTH_CHOICES, size=n_noise)
        noise_snrs = NOISE_SNR_OFFSET + rng.exponential(
            NOISE_SNR_SCALE, size=n_noise
        )
        for i in range(n_noise):
            cands.append(_make_candidate(
                geom=geom,
                l_pix=int(noise_l[i]),
                m_pix=int(noise_m[i]),
                fine_dm_idx=int(noise_fdm[i]),
                t_in_cube=int(noise_t[i]),
                width_samples=int(noise_widths[i]),
                snr=float(noise_snrs[i]),
            ))
    return cands


# ---------------------------------------------------------------------------
# Bench driver
# ---------------------------------------------------------------------------


def _percentiles_ms(values_ms: Sequence[float]) -> Dict[str, float]:
    """Compute p50/p90/p99/max/mean of a list of millisecond values.

    Empty input returns all-zero so the JSON report stays well-formed
    even on a degenerate (n_cubes=0) bench run; the predicate then
    trivially passes.
    """
    if not values_ms:
        return {
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
        }
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
    }


def _git_sha() -> str:
    """Return ``HEAD`` SHA, or ``"unknown"`` if not in a git tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _hostname() -> str:
    """Return the short hostname (matches the M6.sh ``host`` field convention)."""
    try:
        return socket.gethostname().split(".")[0]
    except OSError:
        return "unknown"


def _run_bench(
    *,
    backend: str,
    feature_mode: str,
    n_cubes: int,
    n_cands_per_cube: int,
    rng_seed: int,
) -> Tuple[List[CubeTimingRecord], List[float]]:
    """Drive the clusterer over ``n_cubes`` synthetic cubes.

    Returns:
        Tuple ``(per_cube_records, wall_ms_list)`` where the second
        element is a flat list of per-cube wall-times in milliseconds
        (used to compute summary percentiles).
    """
    cfg = ClustererConfig(backend=backend, feature_mode=feature_mode)
    master_rng = np.random.default_rng(rng_seed)
    # Per-cube sub-RNG is derived from a master-RNG-driven jump count so
    # results are reproducible AND independent across cubes.
    sub_seeds = master_rng.integers(0, 2**63 - 1, size=n_cubes, dtype=np.int64)

    per_cube_records: List[CubeTimingRecord] = []
    wall_ms_list: List[float] = []

    for cube_id in range(n_cubes):
        sub_rng = np.random.default_rng(int(sub_seeds[cube_id]))
        geom = _build_geometry(cube_id)
        cands = _generate_cube_candidates(
            geom=geom,
            n_cands_per_cube_lambda=n_cands_per_cube,
            rng=sub_rng,
        )

        t0 = time.perf_counter_ns()
        labels, records = cluster_candidates(cands, geom, config=cfg)
        t1 = time.perf_counter_ns()
        wall_ms = (t1 - t0) / 1e6

        n_clusters = int(np.sum(labels >= 0)) if labels.size else 0
        n_noise = int(np.sum(labels == -1)) if labels.size else 0
        record = CubeTimingRecord(
            cube_id=cube_id,
            n_cands=len(cands),
            n_records=len(records),
            n_clusters=n_clusters,
            n_noise=n_noise,
            wall_ms=wall_ms,
        )
        per_cube_records.append(record)
        wall_ms_list.append(wall_ms)

        if cube_id == 0 or (cube_id + 1) % 25 == 0 or cube_id == n_cubes - 1:
            _LOG.info(
                "cube %4d/%d : n_cands=%5d n_records=%5d wall=%.2f ms",
                cube_id + 1, n_cubes, record.n_cands,
                record.n_records, record.wall_ms,
            )

    return per_cube_records, wall_ms_list


def _write_per_cube_csv(
    path: Path, per_cube_records: Sequence[CubeTimingRecord]
) -> None:
    """Write the per-cube CSV with the spec's column schema.

    Columns: ``cube_id, n_cands, n_records, n_clusters, n_noise, wall_ms``.
    ``wall_ms`` is written with 6-decimal precision (≈1 ns at ms scale).
    """
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "cube_id", "n_cands", "n_records",
            "n_clusters", "n_noise", "wall_ms",
        ])
        for r in per_cube_records:
            writer.writerow([
                r.cube_id, r.n_cands, r.n_records,
                r.n_clusters, r.n_noise, f"{r.wall_ms:.6f}",
            ])


def _build_report(
    *,
    backend: str,
    feature_mode: str,
    n_cubes: int,
    n_cands_per_cube: int,
    rng_seed: int,
    p99_budget_ms: float,
    per_cube_records: Sequence[CubeTimingRecord],
    wall_ms_list: Sequence[float],
) -> Dict[str, Any]:
    """Assemble the JSON report dict per the spec schema."""
    pct = _percentiles_ms(wall_ms_list)
    n_cands_total = int(sum(r.n_cands for r in per_cube_records))
    n_records_total = int(sum(r.n_records for r in per_cube_records))
    n_clusters_total = int(sum(r.n_clusters for r in per_cube_records))
    n_noise_total = int(sum(r.n_noise for r in per_cube_records))

    summary = {
        "p50_ms": pct["p50_ms"],
        "p90_ms": pct["p90_ms"],
        "p99_ms": pct["p99_ms"],
        "max_ms": pct["max_ms"],
        "mean_ms": pct["mean_ms"],
        "n_cubes_run": int(len(per_cube_records)),
        "n_cands_total": n_cands_total,
        "n_records_total": n_records_total,
        "n_clusters_total": n_clusters_total,
        "n_noise_total": n_noise_total,
    }
    p99_observed = pct["p99_ms"]
    return {
        "git_sha": _git_sha(),
        "host": _hostname(),
        "config": {
            "backend": backend,
            "feature_mode": feature_mode,
            "n_cubes": int(n_cubes),
            "n_cands_per_cube": int(n_cands_per_cube),
            "rng_seed": int(rng_seed),
        },
        "summary": summary,
        "d5_fallback_predicate": {
            "p99_budget_ms": float(p99_budget_ms),
            "p99_observed_ms": float(p99_observed),
            "passes": bool(p99_observed <= p99_budget_ms),
        },
    }


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the bench's ``argparse`` parser.

    Exposed at module scope so the test suite can introspect the CLI
    surface without invoking ``main``.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--report-dir", type=str, default=str(DEFAULT_REPORT_DIR),
        help=f"Output directory (default: {DEFAULT_REPORT_DIR}).",
    )
    ap.add_argument(
        "--backend", type=str,
        choices=(ClustererBackend.HDBSCAN, ClustererBackend.DBSCAN),
        default=DEFAULT_BACKEND,
        help=f"Clusterer backend (default: {DEFAULT_BACKEND}).",
    )
    ap.add_argument(
        "--feature-mode", type=str,
        choices=(FeatureMode.INT, FeatureMode.REAL),
        default=DEFAULT_FEATURE_MODE,
        help=f"Clusterer feature mode (default: {DEFAULT_FEATURE_MODE}).",
    )
    ap.add_argument(
        "--n-cubes", type=int, default=DEFAULT_N_CUBES,
        help=f"Number of cubes to bench (default: {DEFAULT_N_CUBES}).",
    )
    ap.add_argument(
        "--n-cands-per-cube", type=int, default=DEFAULT_N_CANDS_PER_CUBE,
        help=f"Poisson lambda for per-cube candidate count "
             f"(default: {DEFAULT_N_CANDS_PER_CUBE}).",
    )
    ap.add_argument(
        "--rng-seed", type=int, default=DEFAULT_RNG_SEED,
        help=f"Master RNG seed (default: {DEFAULT_RNG_SEED}).",
    )
    ap.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help=f"Compute device label (default: {DEFAULT_DEVICE}). The "
             f"clusterer is CPU-only today; this is a label that lands "
             f"in report.json so future GPU backends can be diff'd.",
    )
    ap.add_argument(
        "--p99-budget-ms", type=float, default=DEFAULT_P99_BUDGET_MS,
        help=f"D5 fallback budget in ms (default: {DEFAULT_P99_BUDGET_MS}).",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """Bench entry point. Returns a Unix-style exit code (0 = success)."""
    argv = argv if argv is not None else sys.argv[1:]
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not _LOG.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        _LOG.addHandler(handler)
    _LOG.setLevel(logging.INFO)
    _LOG.info(
        "bench config: backend=%s feature_mode=%s n_cubes=%d "
        "n_cands_per_cube=%d rng_seed=%d device=%s p99_budget_ms=%.2f",
        args.backend, args.feature_mode, args.n_cubes,
        args.n_cands_per_cube, args.rng_seed,
        args.device, args.p99_budget_ms,
    )

    # The bench is deterministic via numpy's default_rng; we still seed
    # python's random + PYTHONHASHSEED-equivalent for any transitive
    # consumer (e.g. hdbscan's tie-breaks) that might reach for them.
    random.seed(args.rng_seed)

    per_cube_records, wall_ms_list = _run_bench(
        backend=args.backend,
        feature_mode=args.feature_mode,
        n_cubes=int(args.n_cubes),
        n_cands_per_cube=int(args.n_cands_per_cube),
        rng_seed=int(args.rng_seed),
    )

    csv_path = report_dir / "per_cube.csv"
    _write_per_cube_csv(csv_path, per_cube_records)

    report = _build_report(
        backend=args.backend,
        feature_mode=args.feature_mode,
        n_cubes=int(args.n_cubes),
        n_cands_per_cube=int(args.n_cands_per_cube),
        rng_seed=int(args.rng_seed),
        p99_budget_ms=float(args.p99_budget_ms),
        per_cube_records=per_cube_records,
        wall_ms_list=wall_ms_list,
    )
    report_path = report_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    _LOG.info(
        "p50=%.2f ms p90=%.2f ms p99=%.2f ms max=%.2f ms mean=%.2f ms "
        "(n_cubes=%d n_cands=%d n_records=%d n_clusters=%d n_noise=%d)",
        report["summary"]["p50_ms"],
        report["summary"]["p90_ms"],
        report["summary"]["p99_ms"],
        report["summary"]["max_ms"],
        report["summary"]["mean_ms"],
        report["summary"]["n_cubes_run"],
        report["summary"]["n_cands_total"],
        report["summary"]["n_records_total"],
        report["summary"]["n_clusters_total"],
        report["summary"]["n_noise_total"],
    )
    pred = report["d5_fallback_predicate"]
    _LOG.info(
        "D5 predicate: p99_observed=%.2f ms %s budget=%.2f ms (passes=%s)",
        pred["p99_observed_ms"],
        "<=" if pred["passes"] else ">",
        pred["p99_budget_ms"],
        pred["passes"],
    )
    _LOG.info("wrote report=%s per_cube=%s", report_path, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

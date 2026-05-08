#!/usr/bin/env python3
"""bench/clusterer_throughput.py — M6 chunk 6 clusterer throughput
gate (D5 fallback predicate).

Per M6 D5 the clusterer's primary backend is HDBSCAN (cityblock,
``min_cluster_size=2``, ``min_samples=1``,
``cluster_selection_epsilon=10.0``). The fallback is sklearn DBSCAN
(``eps=10``, ``min_samples=2``, cityblock). The decision between
backends is gated by **per-cube clustering wall-clock latency at
production candidate load**: HDBSCAN p99 latency at the production
candidate count (default 1000 candidates / cube) must be ≤ 50 ms.
If not, the operator flips ``ClustererConfig.backend`` to
``"dbscan"`` in ``configs/config_compute_search.yaml``.

This bench drives ``cluster.forward.cluster_candidates`` against
synthetic per-cube candidate streams and reports p50 / p95 / p99 /
mean / stddev latency plus throughput across a sweep of candidate
counts and backends. The D5 gate verdict is emitted in the JSON
report under ``d5_gate.pass``; the script exits 0 on PASS, 1 on FAIL.

The synthetic candidate generator plants ``cluster_fraction``
(default 0.7) of each cube's candidates into clusters of 5-20
candidates within a 5-pixel (l, m) radius; the remainder are
uniformly distributed noise. Generator output honours the M5
detector convention (``Candidate.l = float(l_pix)``,
``Candidate.m = float(m_pix)``,
``Candidate.dm_fine = fine_dm_pc_cc[fine_dm_idx]``).

CLI surface (see ``--help`` for the full grid):

  python -m bench.clusterer_throughput \\
      [--n-cubes 100]                                            \\
      [--candidate-counts 50 100 200 500 1000 2000]              \\
      [--backends hdbscan dbscan]                                \\
      [--cluster-fraction 0.7]                                   \\
      [--seed 42]                                                \\
      [--report-path bench/reports/M6/clusterer_throughput.json] \\
      [--p99-budget-ms 50.0]                                     \\
      [--production-candidate-count 1000]                        \\
      [--device cpu]

Output (single JSON file at ``--report-path``):

  {
    "schema_version": 1,
    "git_sha": "...",
    "host": "lxd110h01",
    "n_cubes": 100,
    "production_candidate_count": 1000,
    "p99_budget_ms": 50.0,
    "results": [
      {"backend": "hdbscan", "n_cands": 1000, "p50_ms": 12.3,
       "p95_ms": 24.1, "p99_ms": 41.8, "mean_ms": 14.2,
       "stddev_ms": 5.1, "throughput_cubes_s": 70.4,
       "n_clusters_mean": 14.0, "n_noise_mean": 280.0},
      ...
    ],
    "d5_gate": {
      "pass": true,
      "decision": "evaluated",
      "hdbscan_p99_at_production_ms": 41.8,
      "p99_budget_ms": 50.0,
      "fallback_required": false,
      "production_candidate_count": 1000
    }
  }

Skipped-gate policy: if the production candidate count is not present
in the HDBSCAN sweep (because the operator excluded ``hdbscan`` from
``--backends`` or excluded ``--production-candidate-count`` from
``--candidate-counts``), the gate is *not* failed — instead it emits
``decision="skipped"`` with ``pass=true`` and
``hdbscan_p99_at_production_ms=null``. The exit code stays 0. Rationale:
the gate's only failure mode that should block CI is "HDBSCAN ran at
the production point and was too slow"; missing-data cases are an
operator/CI configuration mistake rather than a real D5 violation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Enable contract __post_init__ checks for the synthetic Candidate /
# CubeGeometry constructions; matches the production search-compute
# default and keeps the bench honest about contract conformance.
os.environ.setdefault("DSART_TEST", "1")

from dsart.cluster.forward import (  # noqa: E402
    ClustererBackend,
    ClustererConfig,
    cluster_candidates,
)
from dsart.common.contracts import (  # noqa: E402
    Candidate,
    CandidateFlags,
    CubeGeometry,
)


_LOG = logging.getLogger("bench.clusterer_throughput")


# ---------------------------------------------------------------------------
# Defaults (D5 production lock; matches services.search_compute defaults)
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATE_COUNTS: Tuple[int, ...] = (50, 100, 200, 500, 1000, 2000)
DEFAULT_BACKENDS: Tuple[str, ...] = ("hdbscan", "dbscan")
DEFAULT_N_CUBES: int = 100
DEFAULT_CLUSTER_FRACTION: float = 0.7
DEFAULT_SEED: int = 42
DEFAULT_P99_BUDGET_MS: float = 50.0
DEFAULT_PRODUCTION_CANDIDATE_COUNT: int = 1000

DEFAULT_REPORT_PATH: Path = (
    REPO_ROOT / "bench" / "reports" / "M6" / "clusterer_throughput.json"
)

# Production-like cube geometry per M6 D1/D6 + plan §3.6.12 pins. The
# bench's clusterer is geometry-agnostic for timing — these values just
# need to be self-consistent and large enough that fine_dm / l_pix /
# m_pix have realistic dynamic range.
PROD_N_GRID: int = 256
PROD_T_DET: int = 512
PROD_N_FDM_IN_CUBE: int = 32
PROD_SAMPLE_PERIOD_US: float = 131.072
PROD_SAMPLE_PERIOD_SPECNUM: int = 16
PROD_CELL_L_RAD: float = 1.5e-4
PROD_CELL_M_RAD: float = 1.5e-4

# Detector kernel id used for synthetic Candidates. Must be a valid
# triple (image:dm:time) per Candidate._check_kernel_id; "unit:d1:b4"
# is the default M5 detector triple at narrow boxcar.
DETECTOR_KERNEL_ID: str = "unit:d1:b4"

# Synthetic-cluster geometry per spec.
SYNTH_CLUSTER_RADIUS_PIX: int = 5
SYNTH_CLUSTER_SIZE_MIN: int = 5
SYNTH_CLUSTER_SIZE_MAX: int = 20
SYNTH_WIDTHS: Tuple[int, ...] = (1, 2, 4, 8, 16)
SYNTH_SNR_MIN: float = 4.0
SYNTH_SNR_MAX: float = 50.0

# Search-node id reserved for synthetic candidates (must be 0..N_SEARCH-1
# per Candidate.__post_init__). 2 is the M5/M6 default detector node.
SYNTH_SEARCH_NODE_ID: int = 2
SYNTH_GPU_HALF: int = 1


# ---------------------------------------------------------------------------
# Geometry + Candidate factories
# ---------------------------------------------------------------------------


def _build_geometry(cube_id: int = 0) -> CubeGeometry:
    """Build a production-like ``CubeGeometry`` for a synthetic cube.

    Args:
        cube_id: monotonic cube counter for this cube. Mirrors
            ``CubeRingSlot.cube_id`` and propagates into
            ``ClusterRecord.cube_id``.

    Returns:
        CubeGeometry instance with the production-like geometry constants
        pinned at module top (n_grid=256, t_det=512, n_fdm_in_cube=32,
        sample_period_us=131.072, etc.).
    """
    fine_dm_pc_cc = np.linspace(
        50.0, 800.0, PROD_N_FDM_IN_CUBE, dtype=np.float64
    )
    return CubeGeometry(
        cube_id=int(cube_id),
        specnum_start=0,
        sample_period_specnum=PROD_SAMPLE_PERIOD_SPECNUM,
        t_det=PROD_T_DET,
        n_grid=PROD_N_GRID,
        n_fdm_in_cube=PROD_N_FDM_IN_CUBE,
        sample_period_us=PROD_SAMPLE_PERIOD_US,
        cell_l_rad=PROD_CELL_L_RAD,
        cell_m_rad=PROD_CELL_M_RAD,
        l0_rad=0.0,
        m0_rad=0.0,
        fine_dm_pc_cc=fine_dm_pc_cc,
        mjd_start=60942.0,
    )


def _make_candidate(
    *,
    l_pix: int,
    m_pix: int,
    fine_dm_idx: int,
    t_in_cube: int,
    width_samples: int,
    snr: float,
    geom: CubeGeometry,
    search_node_id: int = SYNTH_SEARCH_NODE_ID,
    gpu_half: int = SYNTH_GPU_HALF,
) -> Candidate:
    """Build one ``Candidate`` honouring the M5 detector convention.

    Sets ``l = float(l_pix)``, ``m = float(m_pix)``, and
    ``dm_fine = geom.fine_dm_pc_cc[fine_dm_idx]`` so that the clusterer's
    INT-mode feature recovery (in :mod:`dsart.cluster.features`) round-
    trips back to the planted indices.
    """
    return Candidate(
        l=float(l_pix),
        m=float(m_pix),
        dm_fine=float(geom.fine_dm_pc_cc[fine_dm_idx]),
        dm_idx=0,
        event_specnum=int(
            geom.specnum_start + t_in_cube * geom.sample_period_specnum
        ),
        width_samples=int(width_samples),
        kernel_id=DETECTOR_KERNEL_ID,
        snr=float(snr),
        detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=int(search_node_id),
        gpu_half=int(gpu_half),
    )


# ---------------------------------------------------------------------------
# Synthetic candidate generator
# ---------------------------------------------------------------------------


def synthesize_candidates(
    n_cands: int,
    cluster_fraction: float,
    geom: CubeGeometry,
    rng: np.random.Generator,
    *,
    search_node_id: int = SYNTH_SEARCH_NODE_ID,
    gpu_half: int = SYNTH_GPU_HALF,
) -> Tuple[List[Candidate], np.ndarray]:
    """Generate one cube's worth of synthetic ``Candidate`` records.

    A fraction ``cluster_fraction`` of the candidates are planted into
    clusters of size :data:`SYNTH_CLUSTER_SIZE_MIN`-
    :data:`SYNTH_CLUSTER_SIZE_MAX` within a
    :data:`SYNTH_CLUSTER_RADIUS_PIX`-pixel (l, m) radius, centred at a
    random integer (l_pix, m_pix, fine_dm_idx, t_in_cube). Members
    inherit the cluster centre's ``width_samples`` (so a cluster lands
    on the same boxcar response). The remaining
    ``round(n_cands * (1 - cluster_fraction))`` candidates are uniform
    noise across the (n_grid, n_grid, n_fdm_in_cube, t_det) grid.

    Cluster sizes are sampled iid uniformly until the running total
    would exceed the planted-cluster target. If the residual budget is
    smaller than :data:`SYNTH_CLUSTER_SIZE_MIN`, the residual rolls into
    the noise bucket (so ``actual_planted ≤ target``; the slack is at
    most ``SYNTH_CLUSTER_SIZE_MIN - 1`` candidates).

    Args:
        n_cands: total candidate count.
        cluster_fraction: fraction in [0, 1] of candidates to plant in
            clusters (the rest are noise).
        geom: cube geometry for unit conversion + index bounds.
        rng: numpy ``Generator`` (drives all sampling; for determinism,
            seed at the call site).
        search_node_id: ``Candidate.search_node_id``; default
            :data:`SYNTH_SEARCH_NODE_ID`.
        gpu_half: ``Candidate.gpu_half``; default
            :data:`SYNTH_GPU_HALF`.

    Returns:
        Tuple ``(cands, planted_cluster_ids)`` where:
          * ``cands`` is a list of length ``n_cands``.
          * ``planted_cluster_ids`` is an ``ndarray[int64, n_cands]``
            with ``-1`` for noise candidates and ``k ≥ 0`` for the
            ``k``-th planted cluster (per-cube, monotonic). The ordering
            is "all clustered candidates first, then noise" — within
            each cluster the candidates are contiguous in ``cands``.

    Raises:
        ValueError: if ``cluster_fraction`` is outside ``[0, 1]`` or
            ``n_cands < 0``.
    """
    if not 0.0 <= cluster_fraction <= 1.0:
        raise ValueError(
            f"cluster_fraction={cluster_fraction}, expected ∈ [0, 1]"
        )
    if n_cands < 0:
        raise ValueError(f"n_cands={n_cands}, expected ≥ 0")
    if n_cands == 0:
        return [], np.zeros(0, dtype=np.int64)

    n_target_cluster = int(round(n_cands * cluster_fraction))
    cluster_ids = np.full(n_cands, -1, dtype=np.int64)
    cands: List[Candidate] = []
    cluster_count = 0

    margin = SYNTH_CLUSTER_RADIUS_PIX + 1  # keep cluster centres off the edge

    while True:
        remaining = n_target_cluster - len(cands)
        if remaining < SYNTH_CLUSTER_SIZE_MIN:
            break
        size_max = min(SYNTH_CLUSTER_SIZE_MAX, remaining)
        size = int(rng.integers(SYNTH_CLUSTER_SIZE_MIN, size_max + 1))

        center_l = float(rng.uniform(margin, geom.n_grid - margin))
        center_m = float(rng.uniform(margin, geom.n_grid - margin))
        center_dm_idx = int(rng.integers(0, geom.n_fdm_in_cube))
        center_t = int(rng.integers(0, geom.t_det))
        cluster_width = int(rng.choice(np.asarray(SYNTH_WIDTHS)))

        for _ in range(size):
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            r = float(rng.uniform(0.0, float(SYNTH_CLUSTER_RADIUS_PIX)))
            l_pix = int(round(center_l + r * np.cos(theta)))
            m_pix = int(round(center_m + r * np.sin(theta)))
            l_pix = max(0, min(geom.n_grid - 1, l_pix))
            m_pix = max(0, min(geom.n_grid - 1, m_pix))
            # Slight in-cluster jitter on dm/time so cntb_dm > 1 is
            # plausible without breaking the eps=10 cityblock budget.
            dm_offset = int(rng.integers(-1, 2))   # {-1, 0, 1}
            t_offset = int(rng.integers(-2, 3))    # {-2,-1,0,1,2}
            dm_idx = max(
                0, min(geom.n_fdm_in_cube - 1, center_dm_idx + dm_offset)
            )
            t_in_cube = max(0, min(geom.t_det - 1, center_t + t_offset))
            snr = float(rng.uniform(SYNTH_SNR_MIN, SYNTH_SNR_MAX))
            cand = _make_candidate(
                l_pix=l_pix, m_pix=m_pix,
                fine_dm_idx=dm_idx, t_in_cube=t_in_cube,
                width_samples=cluster_width, snr=snr,
                geom=geom,
                search_node_id=search_node_id, gpu_half=gpu_half,
            )
            cands.append(cand)
            cluster_ids[len(cands) - 1] = cluster_count
        cluster_count += 1

    while len(cands) < n_cands:
        l_pix = int(rng.integers(0, geom.n_grid))
        m_pix = int(rng.integers(0, geom.n_grid))
        dm_idx = int(rng.integers(0, geom.n_fdm_in_cube))
        t_in_cube = int(rng.integers(0, geom.t_det))
        width = int(rng.choice(np.asarray(SYNTH_WIDTHS)))
        snr = float(rng.uniform(SYNTH_SNR_MIN, SYNTH_SNR_MAX))
        cand = _make_candidate(
            l_pix=l_pix, m_pix=m_pix,
            fine_dm_idx=dm_idx, t_in_cube=t_in_cube,
            width_samples=width, snr=snr,
            geom=geom,
            search_node_id=search_node_id, gpu_half=gpu_half,
        )
        cands.append(cand)

    return cands, cluster_ids


# ---------------------------------------------------------------------------
# Sweep + summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SweepResult:
    """Per-(backend, n_cands) sweep result; internal aggregation type."""
    backend: str
    n_cands: int
    timings_ms: np.ndarray
    n_clusters_per_cube: List[int]
    n_noise_per_cube: List[int]


def _percentile(values_ms: np.ndarray, q: float) -> float:
    """Compute percentile in milliseconds; safe on empty inputs."""
    return float(np.percentile(values_ms, q)) if values_ms.size else 0.0


def _run_sweep(
    backend: str,
    n_cands: int,
    n_cubes: int,
    cluster_fraction: float,
    rng: np.random.Generator,
) -> _SweepResult:
    """Run ``n_cubes`` per-cube clusterings of synthetic candidate
    streams; return the timing + record-count aggregate.

    Args:
        backend: ``"hdbscan"`` or ``"dbscan"``.
        n_cands: per-cube candidate count.
        n_cubes: number of cubes to time.
        cluster_fraction: fraction of synthetic candidates planted in
            clusters (passed to :func:`synthesize_candidates`).
        rng: pre-seeded RNG; the caller is responsible for determinism
            across re-runs.

    Returns:
        A :class:`_SweepResult` with per-cube wall-clock latencies and
        cluster / noise counts.
    """
    cfg = ClustererConfig(backend=backend)
    timings_ms = np.empty(n_cubes, dtype=np.float64)
    n_clusters: List[int] = []
    n_noise: List[int] = []
    for cube_idx in range(n_cubes):
        geom = _build_geometry(cube_id=cube_idx)
        cands, _ = synthesize_candidates(n_cands, cluster_fraction, geom, rng)
        t0 = time.perf_counter()
        labels, _records = cluster_candidates(cands, geom, config=cfg)
        t1 = time.perf_counter()
        timings_ms[cube_idx] = (t1 - t0) * 1000.0
        unique_labels = {int(lbl) for lbl in labels.tolist()}
        n_clusters.append(sum(1 for lbl in unique_labels if lbl >= 0))
        n_noise.append(int((labels == -1).sum()))
    return _SweepResult(
        backend=backend,
        n_cands=n_cands,
        timings_ms=timings_ms,
        n_clusters_per_cube=n_clusters,
        n_noise_per_cube=n_noise,
    )


def _summarise(result: _SweepResult) -> Dict[str, object]:
    """Reduce a sweep to a JSON-friendly summary row."""
    arr = result.timings_ms
    p50 = _percentile(arr, 50)
    p95 = _percentile(arr, 95)
    p99 = _percentile(arr, 99)
    mean_ms = float(arr.mean()) if arr.size else 0.0
    stddev_ms = float(arr.std(ddof=0)) if arr.size else 0.0
    throughput = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
    n_clusters = result.n_clusters_per_cube
    n_noise = result.n_noise_per_cube
    return {
        "backend": result.backend,
        "n_cands": int(result.n_cands),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean_ms,
        "stddev_ms": stddev_ms,
        "throughput_cubes_s": throughput,
        "n_clusters_mean": (
            float(np.mean(n_clusters)) if n_clusters else 0.0
        ),
        "n_noise_mean": float(np.mean(n_noise)) if n_noise else 0.0,
    }


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the HEAD git sha; ``"unknown"`` if git lookup fails."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _hostname() -> str:
    """Return the short hostname (= what ``hostname -s`` would emit)."""
    return socket.gethostname().split(".", 1)[0]


def _print_summary_table(rows: Sequence[Dict[str, object]]) -> None:
    """Pretty-print the sweep table to stdout."""
    header = (
        "backend", "n_cands", "p50_ms", "p95_ms", "p99_ms",
        "throughput_cubes_s",
    )
    fmt = "{:>10s} {:>8s} {:>10s} {:>10s} {:>10s} {:>20s}"
    print(fmt.format(*header))
    print(fmt.format(*("-" * len(h) for h in header)))
    row_fmt = (
        "{backend:>10s} {n_cands:>8d} {p50_ms:>10.3f} "
        "{p95_ms:>10.3f} {p99_ms:>10.3f} {throughput_cubes_s:>20.2f}"
    )
    for r in rows:
        print(row_fmt.format(
            backend=str(r["backend"]),
            n_cands=int(r["n_cands"]),
            p50_ms=float(r["p50_ms"]),
            p95_ms=float(r["p95_ms"]),
            p99_ms=float(r["p99_ms"]),
            throughput_cubes_s=float(r["throughput_cubes_s"]),
        ))


# ---------------------------------------------------------------------------
# D5 gate evaluation
# ---------------------------------------------------------------------------


def _evaluate_d5_gate(
    rows: Sequence[Dict[str, object]],
    *,
    p99_budget_ms: float,
    production_candidate_count: int,
) -> Dict[str, object]:
    """Apply the D5 fallback predicate to the sweep results.

    Per spec: the gate's ``pass`` field is ``true`` iff HDBSCAN p99
    latency at the production candidate count is ≤ ``p99_budget_ms``.
    If the HDBSCAN sweep doesn't include the production candidate count
    (because the operator excluded ``hdbscan`` from ``--backends`` or
    ``production_candidate_count`` from ``--candidate-counts``), the gate
    returns ``decision="skipped"`` with ``pass=true`` (does not block CI).
    """
    hdbscan_at_prod = next(
        (
            r for r in rows
            if r["backend"] == ClustererBackend.HDBSCAN
            and int(r["n_cands"]) == production_candidate_count
        ),
        None,
    )
    if hdbscan_at_prod is None:
        return {
            "pass": True,
            "decision": "skipped",
            "reason": (
                f"HDBSCAN result at n_cands={production_candidate_count}"
                f" not present in sweep"
            ),
            "hdbscan_p99_at_production_ms": None,
            "p99_budget_ms": float(p99_budget_ms),
            "fallback_required": False,
            "production_candidate_count": int(production_candidate_count),
        }
    p99 = float(hdbscan_at_prod["p99_ms"])
    passed = p99 <= float(p99_budget_ms)
    return {
        "pass": bool(passed),
        "decision": "evaluated",
        "hdbscan_p99_at_production_ms": p99,
        "p99_budget_ms": float(p99_budget_ms),
        "fallback_required": (not passed),
        "production_candidate_count": int(production_candidate_count),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M6 chunk 6 clusterer throughput bench — D5 fallback gate "
            "for HDBSCAN p99 ≤ 50 ms at production candidate load."
        )
    )
    parser.add_argument(
        "--n-cubes", type=int, default=DEFAULT_N_CUBES,
        help="Number of cubes to time per (backend, n_cands) sweep.",
    )
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+",
        default=list(DEFAULT_CANDIDATE_COUNTS),
        help="Sweep over per-cube candidate counts.",
    )
    parser.add_argument(
        "--backends", type=str, nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=(ClustererBackend.HDBSCAN, ClustererBackend.DBSCAN),
        help="Clusterer backends to sweep.",
    )
    parser.add_argument(
        "--cluster-fraction", type=float, default=DEFAULT_CLUSTER_FRACTION,
        help=(
            "Fraction of synthetic candidates planted in clusters of "
            f"{SYNTH_CLUSTER_SIZE_MIN}-{SYNTH_CLUSTER_SIZE_MAX} within a "
            f"{SYNTH_CLUSTER_RADIUS_PIX}-pixel (l, m) radius. The rest "
            "are uniform noise."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="np.random seed; controls synthetic candidate determinism.",
    )
    parser.add_argument(
        "--report-path", type=Path, default=DEFAULT_REPORT_PATH,
        help=f"JSON report destination. Default: {DEFAULT_REPORT_PATH}.",
    )
    parser.add_argument(
        "--p99-budget-ms", type=float, default=DEFAULT_P99_BUDGET_MS,
        help="D5 fallback predicate: HDBSCAN p99 budget in ms.",
    )
    parser.add_argument(
        "--production-candidate-count",
        type=int, default=DEFAULT_PRODUCTION_CANDIDATE_COUNT,
        help=(
            "Per-cube candidate count whose HDBSCAN p99 the D5 gate "
            "evaluates. Must be present in --candidate-counts for the "
            "gate to render an 'evaluated' verdict; otherwise the gate "
            "reports 'skipped' (pass=true, non-blocking)."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help=(
            "Informational only — clusterer is CPU-only. Recorded in "
            "the report for operator audit (matches the deployed device)."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress logging (only emit the summary table).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Bench entry point. Returns the process exit code (0 = D5 PASS,
    1 = D5 FAIL).

    Args:
        argv: optional argv override (for in-process test harnesses).
            Defaults to ``sys.argv[1:]``.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.quiet and not _LOG.handlers:
        _LOG.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        _LOG.addHandler(handler)

    n_cubes = int(args.n_cubes)
    backends = list(args.backends)
    candidate_counts = list(args.candidate_counts)
    cluster_fraction = float(args.cluster_fraction)
    seed = int(args.seed)
    p99_budget_ms = float(args.p99_budget_ms)
    production_candidate_count = int(args.production_candidate_count)
    report_path: Path = Path(args.report_path).resolve()

    if n_cubes <= 0:
        raise SystemExit(f"--n-cubes={n_cubes}, expected > 0")
    if not candidate_counts:
        raise SystemExit("--candidate-counts must be non-empty")
    if not backends:
        raise SystemExit("--backends must be non-empty")

    _LOG.info(
        "config: n_cubes=%d backends=%s candidate_counts=%s "
        "cluster_fraction=%.2f seed=%d p99_budget_ms=%.2f "
        "production_candidate_count=%d device=%s",
        n_cubes, backends, candidate_counts, cluster_fraction, seed,
        p99_budget_ms, production_candidate_count, args.device,
    )

    # Spawn deterministic per-(backend, n_cands) RNGs from a single
    # SeedSequence so that ordering of backends/candidate_counts in the
    # CLI doesn't change which sub-RNG runs each sweep. This is what
    # makes the bench bit-reproducible across re-runs with the same
    # ``--seed``.
    n_pairs = len(backends) * len(candidate_counts)
    child_seqs = np.random.SeedSequence(seed).spawn(n_pairs)

    rows: List[Dict[str, object]] = []
    pair_idx = 0
    for backend in backends:
        for n_cands in candidate_counts:
            rng = np.random.default_rng(child_seqs[pair_idx])
            pair_idx += 1
            _LOG.info(
                "sweep: backend=%s n_cands=%d (%d cubes)",
                backend, n_cands, n_cubes,
            )
            t0 = time.perf_counter()
            result = _run_sweep(
                backend, n_cands, n_cubes, cluster_fraction, rng,
            )
            wall_s = time.perf_counter() - t0
            row = _summarise(result)
            _LOG.info(
                "  -> p50=%.3fms p95=%.3fms p99=%.3fms "
                "mean=%.3fms throughput=%.1f cubes/s "
                "(sweep wall=%.1fs, n_clusters_mean=%.1f, "
                "n_noise_mean=%.1f)",
                row["p50_ms"], row["p95_ms"], row["p99_ms"],
                row["mean_ms"], row["throughput_cubes_s"], wall_s,
                row["n_clusters_mean"], row["n_noise_mean"],
            )
            rows.append(row)

    d5_gate = _evaluate_d5_gate(
        rows,
        p99_budget_ms=p99_budget_ms,
        production_candidate_count=production_candidate_count,
    )

    report = {
        "schema_version": 1,
        "bench": "clusterer_throughput",
        "milestone": "M6",
        "utc_iso": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "host": _hostname(),
        "n_cubes": n_cubes,
        "candidate_counts": candidate_counts,
        "backends": backends,
        "cluster_fraction": cluster_fraction,
        "seed": seed,
        "production_candidate_count": production_candidate_count,
        "p99_budget_ms": p99_budget_ms,
        "device": str(args.device),
        "results": rows,
        "d5_gate": d5_gate,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    _LOG.info("wrote %s", report_path)

    print()
    print("clusterer throughput summary")
    _print_summary_table(rows)
    print()
    print(
        f"D5 gate: pass={d5_gate['pass']} "
        f"decision={d5_gate['decision']!r}"
    )
    if d5_gate.get("hdbscan_p99_at_production_ms") is not None:
        print(
            f"  hdbscan p99 at n_cands={production_candidate_count}: "
            f"{d5_gate['hdbscan_p99_at_production_ms']:.3f} ms "
            f"(budget {p99_budget_ms:.3f} ms)"
        )
    elif d5_gate.get("decision") == "skipped":
        print(f"  skipped: {d5_gate.get('reason')}")
    if d5_gate.get("fallback_required"):
        print(
            "  fallback required: flip "
            "ClustererConfig.backend = 'dbscan' in "
            "configs/config_compute_search.yaml"
        )

    return 0 if d5_gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

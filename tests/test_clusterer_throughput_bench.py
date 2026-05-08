"""Tests for ``bench/clusterer_throughput.py`` (M6 chunk 6).

Coverage map (per the M6 chunk-6 spec):
  1. Synthetic candidate generator emits the right total count.
  2. ``cluster_fraction ≈ 0.7`` → ~70% of generator output is planted
     in clusters with cardinality ≥ 5 (smoke test for the synthesizer;
     ±20% slack).
  3. The bench module imports cleanly + exposes a callable ``main()``.
  4. Quick-sweep ``main()`` writes a JSON report with the documented
     schema (schema_version, results, d5_gate) and the sweep-row count
     matches ``len(backends) × len(candidate_counts)``.
  5. The D5 gate fails when ``--p99-budget-ms 0.001`` (impossibly low)
     and passes when ``--p99-budget-ms 100000.0``. Requires HDBSCAN.
  6. The bench is deterministic given a seed: two runs with
     ``--seed 42`` produce the same ``n_clusters_mean``.
  7. ``--production-candidate-count`` not in ``--candidate-counts`` →
     gate emits ``decision="skipped"`` with ``pass=true`` (non-blocking
     skip policy; documented in the bench module docstring).

Tests run cleanly under ``DSART_TEST=1`` (Candidate / CubeGeometry
contracts assert per-call). Tests that require HDBSCAN are gated by a
``pytest.mark.skipif`` so the suite still runs to completion in
environments where ``hdbscan`` is not installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bench.clusterer_throughput import (  # noqa: E402
    DEFAULT_CANDIDATE_COUNTS,
    DEFAULT_CLUSTER_FRACTION,
    SYNTH_CLUSTER_SIZE_MIN,
    _build_arg_parser,
    _build_geometry,
    _evaluate_d5_gate,
    main,
    synthesize_candidates,
)
from dsart.common.contracts import Candidate, CubeGeometry  # noqa: E402


_HDBSCAN_AVAILABLE = True
try:
    import hdbscan  # noqa: F401
except ImportError:
    _HDBSCAN_AVAILABLE = False


needs_hdbscan = pytest.mark.skipif(
    not _HDBSCAN_AVAILABLE, reason="hdbscan not installed"
)


def _read_report(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"report missing at {path}"
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Synthesizer total count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cands", [50, 100, 500, 1000])
def test_synthesizer_emits_exact_n_cands(n_cands: int) -> None:
    rng = np.random.default_rng(42)
    geom = _build_geometry()
    cands, planted = synthesize_candidates(
        n_cands=n_cands,
        cluster_fraction=DEFAULT_CLUSTER_FRACTION,
        geom=geom,
        rng=rng,
    )
    assert len(cands) == n_cands
    assert planted.shape == (n_cands,)
    assert planted.dtype == np.int64
    for c in cands:
        assert isinstance(c, Candidate)


def test_synthesizer_zero_returns_empty() -> None:
    rng = np.random.default_rng(0)
    geom = _build_geometry()
    cands, planted = synthesize_candidates(
        n_cands=0, cluster_fraction=0.7, geom=geom, rng=rng,
    )
    assert cands == []
    assert planted.shape == (0,)


# ---------------------------------------------------------------------------
# 2. cluster_fraction ≈ 0.7 → ~70% in clusters with cntc ≥ 5
# ---------------------------------------------------------------------------


def test_cluster_fraction_roughly_matches_target() -> None:
    """For ``cluster_fraction=0.7`` the synthesizer plants ~70% of the
    candidates into clusters of size ≥ 5 (the rest are uniform noise).
    Allow ±20% slack per the chunk-6 spec.
    """
    n_cands = 1000
    rng = np.random.default_rng(42)
    geom = _build_geometry()
    cands, planted = synthesize_candidates(
        n_cands=n_cands, cluster_fraction=0.7, geom=geom, rng=rng,
    )
    assert len(cands) == n_cands
    n_clustered = int((planted >= 0).sum())
    # Each planted cluster has size ≥ SYNTH_CLUSTER_SIZE_MIN by
    # construction, so n_clustered already counts only "cntc ≥ 5"
    # candidates. Verify by re-counting per cluster id.
    cluster_sizes = np.bincount(planted[planted >= 0]) if n_clustered else np.zeros(0)
    if cluster_sizes.size:
        assert cluster_sizes.min() >= SYNTH_CLUSTER_SIZE_MIN
    achieved_fraction = n_clustered / n_cands
    # ±20% slack relative to the 0.7 target → [0.56, 0.84].
    assert 0.56 <= achieved_fraction <= 0.84, (
        f"achieved cluster fraction {achieved_fraction:.3f} outside "
        f"slack window [0.56, 0.84]"
    )


def test_cluster_fraction_zero_yields_all_noise() -> None:
    rng = np.random.default_rng(42)
    geom = _build_geometry()
    cands, planted = synthesize_candidates(
        n_cands=200, cluster_fraction=0.0, geom=geom, rng=rng,
    )
    assert len(cands) == 200
    assert int((planted >= 0).sum()) == 0
    assert int((planted == -1).sum()) == 200


def test_cluster_fraction_one_yields_all_clustered() -> None:
    rng = np.random.default_rng(42)
    geom = _build_geometry()
    cands, planted = synthesize_candidates(
        n_cands=200, cluster_fraction=1.0, geom=geom, rng=rng,
    )
    n_clustered = int((planted >= 0).sum())
    # The slack from rounding cluster sizes to [5, 20] is at most
    # SYNTH_CLUSTER_SIZE_MIN - 1 = 4 candidates.
    assert n_clustered >= 200 - (SYNTH_CLUSTER_SIZE_MIN - 1)


# ---------------------------------------------------------------------------
# 3. Module imports + main() callable
# ---------------------------------------------------------------------------


def test_module_imports_and_main_callable() -> None:
    parser = _build_arg_parser()
    assert parser is not None
    assert callable(main)


# ---------------------------------------------------------------------------
# 4. Quick-sweep main() writes the documented JSON schema
# ---------------------------------------------------------------------------


def test_quick_sweep_emits_expected_schema(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "5",
        "--candidate-counts", "50",
        "--backends", "dbscan",
        "--report-path", str(report_path),
        "--quiet",
    ])
    assert rc == 0
    report = _read_report(report_path)
    for key in (
        "schema_version", "git_sha", "host", "n_cubes",
        "production_candidate_count", "p99_budget_ms",
        "results", "d5_gate",
    ):
        assert key in report, f"missing top-level key {key!r}"
    assert report["schema_version"] == 1
    assert report["n_cubes"] == 5
    assert isinstance(report["results"], list)
    # 1 backend × 1 candidate count = 1 row.
    assert len(report["results"]) == 1
    row = report["results"][0]
    for key in (
        "backend", "n_cands", "p50_ms", "p95_ms", "p99_ms",
        "mean_ms", "stddev_ms", "throughput_cubes_s",
        "n_clusters_mean", "n_noise_mean",
    ):
        assert key in row, f"missing result key {key!r}"
    assert row["backend"] == "dbscan"
    assert row["n_cands"] == 50
    assert row["p99_ms"] >= 0.0
    assert row["mean_ms"] >= 0.0
    gate = report["d5_gate"]
    for key in (
        "pass", "decision", "p99_budget_ms",
        "fallback_required", "production_candidate_count",
    ):
        assert key in gate, f"missing d5_gate key {key!r}"


def test_quick_sweep_emits_one_row_per_backend_x_cand(tmp_path: Path) -> None:
    """N rows = len(backends) × len(candidate_counts)."""
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "3",
        "--candidate-counts", "30", "60",
        "--backends", "dbscan",
        "--report-path", str(report_path),
        "--quiet",
    ])
    assert rc == 0
    report = _read_report(report_path)
    assert len(report["results"]) == 2
    pairs = {(r["backend"], r["n_cands"]) for r in report["results"]}
    assert pairs == {("dbscan", 30), ("dbscan", 60)}


# ---------------------------------------------------------------------------
# 5. D5 gate threshold sensitivity (HDBSCAN required)
# ---------------------------------------------------------------------------


@needs_hdbscan
def test_d5_gate_fails_at_impossibly_low_budget(tmp_path: Path) -> None:
    """Any real HDBSCAN call takes more than 0.001 ms; gate must FAIL."""
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "3",
        "--candidate-counts", "50",
        "--backends", "hdbscan",
        "--report-path", str(report_path),
        "--p99-budget-ms", "0.001",
        "--production-candidate-count", "50",
        "--quiet",
    ])
    assert rc == 1
    report = _read_report(report_path)
    gate = report["d5_gate"]
    assert gate["pass"] is False
    assert gate["decision"] == "evaluated"
    assert gate["fallback_required"] is True
    assert gate["hdbscan_p99_at_production_ms"] is not None


@needs_hdbscan
def test_d5_gate_passes_at_huge_budget(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "3",
        "--candidate-counts", "50",
        "--backends", "hdbscan",
        "--report-path", str(report_path),
        "--p99-budget-ms", "100000.0",
        "--production-candidate-count", "50",
        "--quiet",
    ])
    assert rc == 0
    report = _read_report(report_path)
    gate = report["d5_gate"]
    assert gate["pass"] is True
    assert gate["decision"] == "evaluated"
    assert gate["fallback_required"] is False


# ---------------------------------------------------------------------------
# 6. Determinism with --seed
# ---------------------------------------------------------------------------


def test_bench_deterministic_with_seed(tmp_path: Path) -> None:
    """Two runs with the same ``--seed`` produce identical
    ``n_clusters_mean`` (and full results array, since synthesis +
    clustering are both deterministic given fixed input).
    """
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    common = [
        "--n-cubes", "5",
        "--candidate-counts", "50", "100",
        "--backends", "dbscan",
        "--seed", "42",
        "--quiet",
    ]
    rc_a = main(common + ["--report-path", str(report_a)])
    rc_b = main(common + ["--report-path", str(report_b)])
    assert rc_a == 0 and rc_b == 0
    rep_a = _read_report(report_a)
    rep_b = _read_report(report_b)
    res_a = {(r["backend"], r["n_cands"]): r for r in rep_a["results"]}
    res_b = {(r["backend"], r["n_cands"]): r for r in rep_b["results"]}
    assert set(res_a.keys()) == set(res_b.keys())
    for key, row_a in res_a.items():
        row_b = res_b[key]
        assert row_a["n_clusters_mean"] == row_b["n_clusters_mean"], (
            f"non-deterministic n_clusters_mean for {key}: "
            f"{row_a['n_clusters_mean']} vs {row_b['n_clusters_mean']}"
        )
        assert row_a["n_noise_mean"] == row_b["n_noise_mean"]


def test_bench_different_seeds_yield_different_results(tmp_path: Path) -> None:
    """Sanity check on the determinism path: different seeds should
    *generally* produce different planted configurations. We assert at
    least one row's ``n_clusters_mean`` differs across runs.
    """
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    base = [
        "--n-cubes", "3",
        "--candidate-counts", "200",
        "--backends", "dbscan",
        "--quiet",
    ]
    main(base + ["--seed", "42", "--report-path", str(report_a)])
    main(base + ["--seed", "999", "--report-path", str(report_b)])
    rep_a = _read_report(report_a)
    rep_b = _read_report(report_b)
    # Either n_clusters_mean OR n_noise_mean should differ at least
    # once (random synthesis under different seeds → different cluster
    # counts in expectation).
    different = False
    for ra, rb in zip(rep_a["results"], rep_b["results"]):
        if (
            ra["n_clusters_mean"] != rb["n_clusters_mean"]
            or ra["n_noise_mean"] != rb["n_noise_mean"]
        ):
            different = True
            break
    assert different, "expected synthesis to differ across seeds"


# ---------------------------------------------------------------------------
# 7. production_candidate_count not in candidate_counts → 'skipped'
# ---------------------------------------------------------------------------


def test_d5_gate_skipped_when_production_count_absent(tmp_path: Path) -> None:
    """When ``--production-candidate-count`` is not in
    ``--candidate-counts``, the gate emits ``decision="skipped"`` and
    ``pass=true`` (non-blocking; rc=0). This is the documented policy
    for "missing-data" gate calls.
    """
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "3",
        "--candidate-counts", "50",
        "--backends", "dbscan",
        "--report-path", str(report_path),
        "--production-candidate-count", "12345",
        "--quiet",
    ])
    assert rc == 0
    report = _read_report(report_path)
    gate = report["d5_gate"]
    assert gate["decision"] == "skipped"
    assert gate["pass"] is True
    assert gate["fallback_required"] is False
    assert gate["hdbscan_p99_at_production_ms"] is None
    assert gate["production_candidate_count"] == 12345
    assert "reason" in gate


def test_d5_gate_skipped_when_hdbscan_not_in_backends(tmp_path: Path) -> None:
    """Even if --candidate-counts includes the production count, if
    HDBSCAN wasn't run, the gate is 'skipped' (the gate only evaluates
    HDBSCAN p99 — DBSCAN-only sweeps don't gate D5)."""
    report_path = tmp_path / "report.json"
    rc = main([
        "--n-cubes", "3",
        "--candidate-counts", "50", "1000",
        "--backends", "dbscan",
        "--report-path", str(report_path),
        "--production-candidate-count", "1000",
        "--quiet",
    ])
    assert rc == 0
    report = _read_report(report_path)
    gate = report["d5_gate"]
    assert gate["decision"] == "skipped"
    assert gate["pass"] is True


# ---------------------------------------------------------------------------
# Direct unit tests on the gate evaluator (no I/O; deterministic)
# ---------------------------------------------------------------------------


def test_evaluate_d5_gate_pass_branch() -> None:
    rows = [
        {"backend": "hdbscan", "n_cands": 1000, "p99_ms": 25.0,
         "p50_ms": 12.0, "p95_ms": 20.0, "mean_ms": 13.0,
         "stddev_ms": 4.0, "throughput_cubes_s": 76.9,
         "n_clusters_mean": 14.0, "n_noise_mean": 280.0},
    ]
    gate = _evaluate_d5_gate(
        rows, p99_budget_ms=50.0, production_candidate_count=1000,
    )
    assert gate["pass"] is True
    assert gate["decision"] == "evaluated"
    assert gate["fallback_required"] is False
    assert gate["hdbscan_p99_at_production_ms"] == pytest.approx(25.0)


def test_evaluate_d5_gate_fail_branch() -> None:
    rows = [
        {"backend": "hdbscan", "n_cands": 1000, "p99_ms": 75.0,
         "p50_ms": 30.0, "p95_ms": 60.0, "mean_ms": 35.0,
         "stddev_ms": 10.0, "throughput_cubes_s": 28.0,
         "n_clusters_mean": 14.0, "n_noise_mean": 280.0},
    ]
    gate = _evaluate_d5_gate(
        rows, p99_budget_ms=50.0, production_candidate_count=1000,
    )
    assert gate["pass"] is False
    assert gate["decision"] == "evaluated"
    assert gate["fallback_required"] is True


def test_evaluate_d5_gate_skipped_branch() -> None:
    rows = [
        {"backend": "dbscan", "n_cands": 1000, "p99_ms": 5.0,
         "p50_ms": 2.0, "p95_ms": 4.0, "mean_ms": 2.5,
         "stddev_ms": 1.0, "throughput_cubes_s": 400.0,
         "n_clusters_mean": 14.0, "n_noise_mean": 280.0},
    ]
    gate = _evaluate_d5_gate(
        rows, p99_budget_ms=50.0, production_candidate_count=1000,
    )
    assert gate["pass"] is True
    assert gate["decision"] == "skipped"
    assert gate["hdbscan_p99_at_production_ms"] is None
    assert gate["fallback_required"] is False


# ---------------------------------------------------------------------------
# Defaults sanity check (catches accidental override of the spec)
# ---------------------------------------------------------------------------


def test_default_candidate_counts_match_spec() -> None:
    assert tuple(DEFAULT_CANDIDATE_COUNTS) == (50, 100, 200, 500, 1000, 2000)


def test_default_geometry_production_like() -> None:
    geom = _build_geometry()
    assert isinstance(geom, CubeGeometry)
    assert geom.n_grid == 256
    assert geom.t_det == 512
    assert geom.n_fdm_in_cube == 32
    assert geom.sample_period_us == pytest.approx(131.072)
    assert geom.fine_dm_pc_cc.shape == (32,)
    assert geom.fine_dm_pc_cc.dtype.name == "float64"

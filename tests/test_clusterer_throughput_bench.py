"""Smoke tests for ``bench/clusterer_throughput.py`` (M6 Chunk 6).

The bench is the operator-facing throughput characterisation that gates
the M6 D5 fallback decision (HDBSCAN if p99 ≤ 50 ms at production load,
else fall back to DBSCAN). The unit tests here cover:

  1. CLI runs end-to-end with ``n_cubes=10``; ``report.json`` and
     ``per_cube.csv`` are produced and parsable.
  2. ``summary.n_cubes_run`` matches ``--n-cubes``.
  3. ``summary.n_records_total`` equals the sum of ``per_cube.csv``
     ``n_records``.
  4. ``d5_fallback_predicate.passes`` is computed correctly given the
     observed p99 (synthesise a tiny load so p99 is well below 50 ms).
  5. Backend dispatch: ``--backend dbscan`` exercises the DBSCAN code
     path and produces a different report (smoke check).
  6. Reproducibility: same ``--rng-seed`` produces the same
     ``n_records_total`` across two runs.
  7. ``--feature-mode real`` runs without crashing (smoke check).

The HDBSCAN-specific tests skip via ``pytest.importorskip("hdbscan")``
so the suite is portable to environments that pin the DBSCAN-only
fallback (M6 D5).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bench.clusterer_throughput import main  # noqa: E402


def _read_per_cube_csv(path: Path) -> list[dict]:
    """Load per_cube.csv into a list of dicts with int/float coercion."""
    rows: list[dict] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({
                "cube_id": int(r["cube_id"]),
                "n_cands": int(r["n_cands"]),
                "n_records": int(r["n_records"]),
                "n_clusters": int(r["n_clusters"]),
                "n_noise": int(r["n_noise"]),
                "wall_ms": float(r["wall_ms"]),
            })
    return rows


# ---------------------------------------------------------------------------
# HDBSCAN tests (skipped if hdbscan is not installed)
# ---------------------------------------------------------------------------


def test_cli_writes_report_and_csv(tmp_path: Path) -> None:
    """1. CLI runs end-to-end with ``n_cubes=10``; both files parse."""
    pytest.importorskip("hdbscan")
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "hdbscan",
        "--feature-mode", "int",
        "--n-cubes", "10",
        "--n-cands-per-cube", "20",
        "--rng-seed", "42",
    ])
    assert rc == 0

    report_path = tmp_path / "report.json"
    csv_path = tmp_path / "per_cube.csv"
    assert report_path.exists()
    assert csv_path.exists()

    report = json.loads(report_path.read_text())
    for k in ("git_sha", "host", "config", "summary", "d5_fallback_predicate"):
        assert k in report, f"missing top-level key {k}"
    for k in (
        "p50_ms", "p90_ms", "p99_ms", "max_ms", "mean_ms",
        "n_cubes_run", "n_cands_total", "n_records_total",
        "n_clusters_total", "n_noise_total",
    ):
        assert k in report["summary"], f"missing summary key {k}"
    rows = _read_per_cube_csv(csv_path)
    assert len(rows) == 10


def test_summary_n_cubes_run_matches_arg(tmp_path: Path) -> None:
    """2. ``summary.n_cubes_run`` field equals ``--n-cubes``."""
    pytest.importorskip("hdbscan")
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "hdbscan",
        "--n-cubes", "10",
        "--n-cands-per-cube", "20",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["summary"]["n_cubes_run"] == 10
    assert report["config"]["n_cubes"] == 10


def test_summary_n_records_matches_csv_sum(tmp_path: Path) -> None:
    """3. ``summary.n_records_total`` equals sum of CSV ``n_records``."""
    pytest.importorskip("hdbscan")
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "hdbscan",
        "--n-cubes", "10",
        "--n-cands-per-cube", "20",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    rows = _read_per_cube_csv(tmp_path / "per_cube.csv")
    csv_n_records_sum = sum(r["n_records"] for r in rows)
    csv_n_clusters_sum = sum(r["n_clusters"] for r in rows)
    csv_n_noise_sum = sum(r["n_noise"] for r in rows)
    csv_n_cands_sum = sum(r["n_cands"] for r in rows)
    assert report["summary"]["n_records_total"] == csv_n_records_sum
    assert report["summary"]["n_clusters_total"] == csv_n_clusters_sum
    assert report["summary"]["n_noise_total"] == csv_n_noise_sum
    assert report["summary"]["n_cands_total"] == csv_n_cands_sum


def test_d5_predicate_passes_under_tiny_load(tmp_path: Path) -> None:
    """4. ``d5_fallback_predicate.passes`` is True for a tiny load.

    With ``n_cands_per_cube=5`` the per-cube wall-clock is well under
    50 ms even on a slow CI box, so the predicate must report
    ``passes=True``.
    """
    pytest.importorskip("hdbscan")
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "hdbscan",
        "--n-cubes", "5",
        "--n-cands-per-cube", "5",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    pred = report["d5_fallback_predicate"]
    assert pred["p99_budget_ms"] == 50.0
    assert pred["p99_observed_ms"] >= 0.0
    assert pred["p99_observed_ms"] < 50.0
    assert pred["passes"] is True


def test_reproducible_n_records_under_same_seed(tmp_path: Path) -> None:
    """6. Same ``--rng-seed`` yields the same ``n_records_total`` twice."""
    pytest.importorskip("hdbscan")
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    args_common = [
        "--backend", "hdbscan",
        "--n-cubes", "5",
        "--n-cands-per-cube", "20",
        "--rng-seed", "1234",
    ]
    assert main(["--report-dir", str(out_a), *args_common]) == 0
    assert main(["--report-dir", str(out_b), *args_common]) == 0
    rep_a = json.loads((out_a / "report.json").read_text())
    rep_b = json.loads((out_b / "report.json").read_text())
    # Synthetic generator is deterministic via numpy default_rng → the
    # input candidate stream is bit-identical across runs, so the
    # clusterer's output record count must match.
    assert rep_a["summary"]["n_cands_total"] == rep_b["summary"]["n_cands_total"]
    assert rep_a["summary"]["n_records_total"] == rep_b["summary"]["n_records_total"]


def test_feature_mode_real_runs_without_crash(tmp_path: Path) -> None:
    """7. ``--feature-mode real`` smoke runs (n_cubes=5)."""
    pytest.importorskip("hdbscan")
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "hdbscan",
        "--feature-mode", "real",
        "--n-cubes", "5",
        "--n-cands-per-cube", "20",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["config"]["feature_mode"] == "real"
    assert report["summary"]["n_cubes_run"] == 5


# ---------------------------------------------------------------------------
# DBSCAN test (always runs — sklearn is a hard dep of the clusterer)
# ---------------------------------------------------------------------------


def test_dbscan_backend_dispatch(tmp_path: Path) -> None:
    """5. ``--backend dbscan`` exercises the DBSCAN code path.

    Smoke check: produces a parsable ``report.json`` with
    ``config.backend == "dbscan"``. No timing assertion (DBSCAN is
    always fast — the bench just confirms the dispatch + report path
    work for both backends so the M6 D5 fallback flip is no-op).
    """
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "dbscan",
        "--n-cubes", "10",
        "--n-cands-per-cube", "20",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["config"]["backend"] == "dbscan"
    assert report["summary"]["n_cubes_run"] == 10
    assert report["summary"]["n_cands_total"] > 0
    rows = _read_per_cube_csv(tmp_path / "per_cube.csv")
    assert len(rows) == 10


def test_dbscan_report_carries_backend_label(tmp_path: Path) -> None:
    """5b. Backend label round-trips into ``config.backend`` so the
    operator can distinguish the two backends' report.json files at
    a glance (used by the M6 D5 ledger).
    """
    rc = main([
        "--report-dir", str(tmp_path),
        "--backend", "dbscan",
        "--n-cubes", "5",
        "--n-cands-per-cube", "10",
        "--rng-seed", "42",
    ])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["config"]["backend"] == "dbscan"
    # Predicate fields are still present (even though DBSCAN is the
    # fallback path; the predicate just measures DBSCAN's own p99).
    assert report["d5_fallback_predicate"]["p99_budget_ms"] == 50.0
    assert report["d5_fallback_predicate"]["p99_observed_ms"] >= 0.0

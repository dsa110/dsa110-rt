"""Smoke tests for ``bench/search_node_throughput.py`` (M5 Chunk 6b-β).

Quick-smoke run end-to-end produces:
  * ``stage_timings.ndjson`` with N records, one per cube
  * ``summary.json`` with per-stage percentiles
  * ``bench.log`` with progress lines

The bench is gated by operator inspection of the percentiles
(p99 < 30 ms at production geometry on h01-GPU); the unit test only
asserts the file format and basic per-stage record sanity.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bench.search_node_throughput import (  # noqa: E402
    QUICK_SMOKE_N_CUBES,
    _build_arg_parser,
    main,
)


def test_quick_smoke_writes_outputs(tmp_path) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    assert rc == 0

    ndjson_path = tmp_path / "stage_timings.ndjson"
    summary_path = tmp_path / "summary.json"
    bench_log_path = tmp_path / "bench.log"
    assert ndjson_path.exists()
    assert summary_path.exists()
    assert bench_log_path.exists()

    records = []
    with ndjson_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    assert len(records) == QUICK_SMOKE_N_CUBES
    for r in records:
        for k in (
            "cube_id", "n_candidates", "n_records",
            "build_cube_ns", "layer1_norm_ns",
            "detector_forward_ns", "emitter_dispatch_ns",
            "total_pipeline_ns",
        ):
            assert k in r, f"missing key {k} in {r}"
        for k in (
            "build_cube_ns", "layer1_norm_ns",
            "detector_forward_ns", "total_pipeline_ns",
        ):
            assert r[k] >= 0, f"negative timing {k}={r[k]} in {r}"


def test_summary_has_expected_percentiles(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["bench"] == "search_node_throughput"
    assert summary["n_cubes_processed"] == QUICK_SMOKE_N_CUBES
    pct = summary["percentiles_ms"]
    for stage in (
        "build_cube", "layer1_norm",
        "detector_forward", "emitter_dispatch", "total_pipeline",
    ):
        assert stage in pct, f"missing stage {stage}"
        for p in ("p50", "p90", "p99", "mean", "max"):
            assert p in pct[stage]
            assert pct[stage][p] >= 0.0
    # Sanity: total p50 ≥ detector p50 (within numerical jitter).
    assert (
        pct["total_pipeline"]["p50"]
        >= 0.5 * pct["detector_forward"]["p50"]
    ), pct


def test_quick_smoke_with_bank_mask(tmp_path) -> None:
    """End-to-end smoke with the 1×1×8 = 8-kernel sub-bank (Chunk 6c-β).

    Asserts the resolved bank-mask is round-tripped into summary.json
    so the Chunk 6c-γ Pareto plot tool can label each run point.
    """
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
        "--bank-mask", "k_img=unit;k_dm=d1",
    ])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["config"]["bank_mask"] == "k_img=unit;k_dm=d1"
    resolved = summary["config"]["bank_mask_resolved"]
    assert resolved["k_img"] == ["unit"]
    assert resolved["k_dm"] == ["d1"]
    assert len(resolved["k_time"]) == 8
    assert resolved["n_kernels"] == 1 * 1 * 8

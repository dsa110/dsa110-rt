"""Smoke tests for ``tools/viz/search_perf_characterization.py``
(M5 Chunk 6c-γ).

The Pareto plot tool consumes one or more cube_injection summary.json
+ one or more search_node_throughput summary.json files and renders:

  * pareto.png + recovery_grid.png + throughput_stages.png
  * report.html (no PASS/FAIL banner)
  * pareto.json (machine-readable joined Pareto table)

These tests fabricate minimal-but-realistic summary.json fixtures
(matching the schema produced by Chunks 6c-α + 6c-β) and exercise:

  * The Pareto join (matching recovery to throughput by bank_mask)
  * The headline-finding logic across N_grid filtering
  * The HTML/PNG renderers (lazy-imported matplotlib)
  * The pareto.json machine-readable export
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.viz import search_perf_characterization as viz  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_recovery_summary(
    *,
    bank_mask: str | None,
    n_kernels: int,
    cells: list[dict] | None = None,
    t_det: int = 64,
    n_fdm: int = 8,
    n_grid: int = 64,
) -> dict:
    """Build a cube_injection summary.json dict (Chunk 6c-α schema)."""
    if cells is None:
        # Default: 5 SNRs × 7 widths grid with monotone recovery.
        cells = []
        for snr in (6.0, 8.0, 10.0, 12.0, 15.0):
            for width in (2, 4, 8, 16, 32, 64, 128):
                # Recovery goes up with snr; goes up with width up to ~32.
                base = max(0.0, min(1.0, (snr - 5.0) / 8.0))
                width_boost = min(width, 32) / 32.0
                rec_frac = min(1.0, base * (0.5 + 0.5 * width_boost))
                cells.append({
                    "injected": {
                        "snr": snr,
                        "width_samples": width,
                        "l_pix": n_grid // 2,
                        "m_pix": n_grid // 2,
                        "fine_dm_idx": n_fdm // 2,
                        "t_in_cube": t_det // 2,
                        "profile": "boxcar",
                    },
                    "n_trials": 3,
                    "n_recovered": int(round(rec_frac * 3)),
                    "recovery_fraction": rec_frac,
                    "snr_ratio_mean": (
                        0.92 if rec_frac > 0 else None
                    ),
                    "matched_kernel_id": (
                        "unit:d1:b32" if rec_frac > 0 else None
                    ),
                })
    return {
        "tool": "bench/cube_injection_detector.py",
        "version": "v1.M5",
        "config": {
            "T_det": t_det,
            "N_fdm": n_fdm,
            "N_grid": n_grid,
            "detector_threshold_sigma": 8.0,
            "bank_mask": bank_mask,
            "bank_mask_resolved": {
                "k_img": ["unit"],
                "k_dm": ["d1"],
                "k_time": [f"b{2**i}" for i in range(8)],
                "n_kernels": n_kernels,
            },
        },
        "cells": cells,
        "far": [
            {"theta": 8.0, "empirical_per_cube_per_kernel": 0.0,
             "n_cubes": 30, "n_kernels": n_kernels},
        ],
    }


def _make_throughput_summary(
    *,
    bank_mask: str | None,
    n_kernels: int,
    n_grid: int,
    cubes_per_s: float,
    detector_p50_ms: float,
    build_cube_p50_ms: float = 5.0,
    layer1_norm_p50_ms: float = 1.0,
    emitter_dispatch_p50_ms: float = 0.5,
    device: str = "cpu",
    cube_dtype: str = "float32",
) -> dict:
    total_p50 = (
        detector_p50_ms + build_cube_p50_ms
        + layer1_norm_p50_ms + emitter_dispatch_p50_ms
    )

    def _pcts(v: float) -> dict:
        return {
            "p50": v, "p90": 1.1 * v, "p99": 1.3 * v,
            "mean": v, "max": 1.5 * v,
        }

    return {
        "schema_version": 1,
        "bench": "search_node_throughput",
        "milestone": "M5",
        "config": {
            "n_cubes": 50,
            "cube_cadence_s": 0.0,
            "t_det": 64,
            "n_fdm": 8,
            "n_grid": n_grid,
            "threshold_sigma": 8.0,
            "device": device,
            "cube_dtype": cube_dtype,
            "bank_mask": bank_mask,
            "bank_mask_resolved": {
                "k_img": ["unit"],
                "k_dm": ["d1"],
                "k_time": [f"b{2**i}" for i in range(8)],
                "n_kernels": n_kernels,
            },
        },
        "wall_clock_s": 50.0 / cubes_per_s,
        "achieved_cubes_per_s": cubes_per_s,
        "n_cubes_processed": 50,
        "n_candidates_total": 0,
        "n_records_total": 0,
        "percentiles_ms": {
            "build_cube": _pcts(build_cube_p50_ms),
            "layer1_norm": _pcts(layer1_norm_p50_ms),
            "detector_forward": _pcts(detector_p50_ms),
            "emitter_dispatch": _pcts(emitter_dispatch_p50_ms),
            "total_pipeline": _pcts(total_p50),
        },
    }


def _write_summary(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(data, indent=2))
    return p


# ---------------------------------------------------------------------------
# Loader / parser tests
# ---------------------------------------------------------------------------


def test_label_from_bank_mask_handles_glob_and_none() -> None:
    assert viz._label_from_bank_mask(None) == "full"
    assert viz._label_from_bank_mask("") == "full"
    assert viz._label_from_bank_mask("  ") == "full"
    assert viz._label_from_bank_mask("*") == "full"
    assert viz._label_from_bank_mask("k_img=unit") == "k_img=unit"
    assert (
        viz._label_from_bank_mask("k_img=unit;k_dm=d1")
        == "k_img=unit;k_dm=d1"
    )


def test_load_recovery_summary_round_trips_bank_mask(tmp_path: Path) -> None:
    data = _make_recovery_summary(bank_mask="k_img=unit;k_dm=d1", n_kernels=8)
    p = _write_summary(tmp_path, "rec", data)
    rec = viz._load_recovery_summary(p)
    assert rec.label == "k_img=unit;k_dm=d1"
    assert rec.n_kernels == 8
    assert rec.t_det == 64
    assert len(rec.cells) == 5 * 7


def test_recovery_summary_cell_lookup() -> None:
    data = _make_recovery_summary(bank_mask=None, n_kernels=128)
    cells = tuple(data["cells"])
    rec = viz.RecoverySummary(
        label="full", bank_mask=None, n_kernels=128,
        cells=cells, t_det=64, n_fdm=8, n_grid=64,
        detector_threshold_sigma=8.0, far_samples=(),
        source_path="/tmp/rec",
    )
    assert rec.cell(8.0, 32) is not None
    assert rec.cell(99.0, 32) is None  # off-grid


# ---------------------------------------------------------------------------
# Pareto join tests
# ---------------------------------------------------------------------------


def test_join_pareto_matches_by_bank_mask(tmp_path: Path) -> None:
    """Recovery + throughput entries with the same bank_mask join into
    a single ParetoPoint that has both axes populated."""
    rec_full = _write_summary(
        tmp_path, "rec_full",
        _make_recovery_summary(bank_mask=None, n_kernels=128),
    )
    rec_8 = _write_summary(
        tmp_path, "rec_8",
        _make_recovery_summary(
            bank_mask="k_img=unit;k_dm=d1", n_kernels=8,
        ),
    )
    th_full = _write_summary(
        tmp_path, "th_full",
        _make_throughput_summary(
            bank_mask=None, n_kernels=128, n_grid=256,
            cubes_per_s=0.5, detector_p50_ms=1800.0,
        ),
    )
    th_8 = _write_summary(
        tmp_path, "th_8",
        _make_throughput_summary(
            bank_mask="k_img=unit;k_dm=d1", n_kernels=8, n_grid=256,
            cubes_per_s=4.0, detector_p50_ms=120.0,
        ),
    )
    recoveries = [
        viz._load_recovery_summary(rec_full),
        viz._load_recovery_summary(rec_8),
    ]
    throughputs = [
        viz._load_throughput_summary(th_full),
        viz._load_throughput_summary(th_8),
    ]
    points = viz.join_pareto(
        recoveries, throughputs, mid_snr=8.0, mid_width=32,
    )
    assert len(points) == 2
    by_label = {p.label: p for p in points}
    assert by_label["full"].cubes_per_s == pytest.approx(0.5)
    assert by_label["full"].recovery_at_mid is not None
    assert by_label["k_img=unit;k_dm=d1"].cubes_per_s == pytest.approx(4.0)
    assert by_label["k_img=unit;k_dm=d1"].n_kernels == 8


def test_join_pareto_handles_recovery_only_and_throughput_only(
    tmp_path: Path,
) -> None:
    """If only one axis is provided for a config, the other stays None."""
    rec_path = _write_summary(
        tmp_path, "rec_only",
        _make_recovery_summary(bank_mask="k_dm=d1", n_kernels=32),
    )
    th_path = _write_summary(
        tmp_path, "th_only",
        _make_throughput_summary(
            bank_mask="k_img=unit", n_kernels=32, n_grid=128,
            cubes_per_s=2.0, detector_p50_ms=400.0,
        ),
    )
    points = viz.join_pareto(
        [viz._load_recovery_summary(rec_path)],
        [viz._load_throughput_summary(th_path)],
        mid_snr=8.0, mid_width=32,
    )
    assert len(points) == 2
    by_label = {p.label: p for p in points}
    rec_only = by_label["k_dm=d1"]
    th_only = by_label["k_img=unit"]
    assert rec_only.recovery_at_mid is not None
    assert rec_only.cubes_per_s is None
    assert th_only.cubes_per_s == pytest.approx(2.0)
    assert th_only.recovery_at_mid is None


# ---------------------------------------------------------------------------
# Headline tests
# ---------------------------------------------------------------------------


def test_cheapest_viable_at_target_picks_smallest_bank() -> None:
    """Among points hitting the target, pick the one with fewest
    kernels (cheapest)."""
    points = [
        viz.ParetoPoint(
            label="full", bank_mask=None, n_kernels=128, n_grid=256,
            cubes_per_s=10.0, recovery_at_mid=0.95,
        ),
        viz.ParetoPoint(
            label="k_dm=d1", bank_mask="k_dm=d1", n_kernels=32,
            n_grid=256, cubes_per_s=12.0, recovery_at_mid=0.85,
        ),
        viz.ParetoPoint(
            label="k_img=unit;k_dm=d1",
            bank_mask="k_img=unit;k_dm=d1", n_kernels=8, n_grid=256,
            cubes_per_s=15.0, recovery_at_mid=0.80,
        ),
    ]
    cheapest = viz.cheapest_viable_at_target(
        points, target_cubes_per_s=8.0, n_grid_filter=256,
    )
    assert cheapest is not None
    assert cheapest.label == "k_img=unit;k_dm=d1"
    assert cheapest.n_kernels == 8


def test_cheapest_viable_filters_low_recovery() -> None:
    points = [
        viz.ParetoPoint(
            label="aggressive", bank_mask="k_img=unit;k_dm=d1",
            n_kernels=8, n_grid=256,
            cubes_per_s=20.0, recovery_at_mid=0.10,  # below 0.5 floor
        ),
        viz.ParetoPoint(
            label="balanced", bank_mask="k_dm=d1",
            n_kernels=32, n_grid=256,
            cubes_per_s=10.0, recovery_at_mid=0.85,
        ),
    ]
    cheapest = viz.cheapest_viable_at_target(
        points, target_cubes_per_s=8.0, n_grid_filter=256,
        min_recovery=0.5,
    )
    assert cheapest is not None
    assert cheapest.label == "balanced"


def test_render_headline_falls_back_to_other_n_grid() -> None:
    """If no N_grid=256 point hits the target but N_grid=128 does,
    the headline names the 128 point + flags the GPU port."""
    points = [
        viz.ParetoPoint(
            label="full", bank_mask=None, n_kernels=128, n_grid=256,
            cubes_per_s=2.0, recovery_at_mid=0.95,
        ),
        viz.ParetoPoint(
            label="full@128", bank_mask=None, n_kernels=128, n_grid=128,
            cubes_per_s=10.0, recovery_at_mid=0.90,
        ),
    ]
    headline = viz.render_headline(
        points, target_cubes_per_s=8.0, mid_snr=8.0, mid_width=32,
    )
    assert "N_grid=128" in headline
    assert "GPU port" in headline


def test_render_headline_no_viable_recommends_gpu() -> None:
    points = [
        viz.ParetoPoint(
            label="full", bank_mask=None, n_kernels=128, n_grid=256,
            cubes_per_s=0.5, recovery_at_mid=0.95,
        ),
    ]
    headline = viz.render_headline(
        points, target_cubes_per_s=8.0, mid_snr=8.0, mid_width=32,
    )
    assert "GPU port" in headline


# ---------------------------------------------------------------------------
# End-to-end report generation
# ---------------------------------------------------------------------------


def test_main_writes_full_report(tmp_path: Path) -> None:
    """Run the full CLI end-to-end and assert all expected files
    materialise + the HTML body contains the headline."""
    rec_full = _write_summary(
        tmp_path, "rec_full",
        _make_recovery_summary(bank_mask=None, n_kernels=128),
    )
    rec_8 = _write_summary(
        tmp_path, "rec_8",
        _make_recovery_summary(
            bank_mask="k_img=unit;k_dm=d1", n_kernels=8,
        ),
    )
    th_full = _write_summary(
        tmp_path, "th_full",
        _make_throughput_summary(
            bank_mask=None, n_kernels=128, n_grid=256,
            cubes_per_s=0.5, detector_p50_ms=1800.0,
        ),
    )
    th_8 = _write_summary(
        tmp_path, "th_8",
        _make_throughput_summary(
            bank_mask="k_img=unit;k_dm=d1", n_kernels=8, n_grid=256,
            cubes_per_s=4.0, detector_p50_ms=120.0,
        ),
    )
    out = tmp_path / "report"
    rc = viz.main([
        "--recovery-summary", str(rec_full),
        "--recovery-summary", str(rec_8),
        "--throughput-summary", str(th_full),
        "--throughput-summary", str(th_8),
        "--target-cubes-per-s", "8.0",
        "--mid-snr", "8.0",
        "--mid-width", "32",
        "--out", str(out),
    ])
    assert rc == 0
    for fn in (
        "pareto.png", "recovery_grid.png", "throughput_stages.png",
        "report.html", "pareto.json",
    ):
        assert (out / fn).is_file(), f"missing {fn}"

    html = (out / "report.html").read_text()
    assert "M5 Chunk 6c" in html
    # Must NOT carry an automated PASS/FAIL banner per plan §4.7.
    assert "PASSED" not in html
    assert "FAILED" not in html
    # The headline mentions either the bank mask or the GPU port.
    assert "k_img=unit;k_dm=d1" in html or "GPU port" in html

    pareto = json.loads((out / "pareto.json").read_text())
    assert isinstance(pareto, list)
    assert len(pareto) == 2
    labels = {row["label"] for row in pareto}
    assert labels == {"full", "k_img=unit;k_dm=d1"}


def test_main_requires_at_least_one_summary(tmp_path: Path) -> None:
    """argparse-level error: no summaries supplied."""
    out = tmp_path / "report"
    with pytest.raises(SystemExit):
        viz.main(["--out", str(out)])

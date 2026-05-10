"""Smoke tests for ``bench/noise_norm_calibration.py`` (M5 Chunk 6b-γ).

Quick-smoke run end-to-end produces:
  * ``noise_norm.ndjson`` with one record per (cube, kernel) that
    fires at least one candidate.
  * ``far_curve.json`` with per-kernel × θ table of
    ``{expected, observed, ratio}``.
  * ``summary.json`` with config + the F13 gate status (PASS or
    informational).
  * ``bench.log`` with progress.

Operator gate is curve-shape inspection (chunk-7 viz); the smoke
test only validates the bench's output file format and the F13
analytic table sanity (expected counts non-negative, summed expected
strictly decreases with θ, kernel-id parses cleanly).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bench"))

from noise_norm_calibration import (  # noqa: E402
    DEFAULT_THETA_GRID,
    QUICK_SMOKE_N_CUBES,
    gaussian_tail,
    main,
    n_eff_per_cube_per_kernel,
)


# ---------------------------------------------------------------------------
# Pure helpers (no IO)
# ---------------------------------------------------------------------------


def test_gaussian_tail_matches_known_values() -> None:
    # 0.5·erfc(0/√2) = 0.5
    assert abs(gaussian_tail(0.0) - 0.5) < 1e-12
    # ~standard tail values
    assert 0.158 < gaussian_tail(1.0) < 0.160
    assert 0.022 < gaussian_tail(2.0) < 0.024
    assert 1e-3 < gaussian_tail(3.0) < 2e-3


def test_n_eff_per_cube_per_kernel_basic() -> None:
    # 64 × 8 × 32² = 524288; / (1 · 1 · 1) = 524288
    assert n_eff_per_cube_per_kernel(
        t_det=64, n_fdm=8, n_grid=32,
        k_img_volume=1, k_dm_width=1, k_time_width=1,
    ) == 64 * 8 * 32 * 32
    # Doubling K_time halves N_eff.
    assert n_eff_per_cube_per_kernel(
        t_det=64, n_fdm=8, n_grid=32,
        k_img_volume=1, k_dm_width=1, k_time_width=2,
    ) == (64 * 8 * 32 * 32) // 2


def test_n_eff_rejects_zero_widths() -> None:
    with pytest.raises(ValueError):
        n_eff_per_cube_per_kernel(
            t_det=64, n_fdm=8, n_grid=32,
            k_img_volume=0, k_dm_width=1, k_time_width=1,
        )


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_quick_smoke_writes_outputs(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    assert rc == 0
    assert (tmp_path / "noise_norm.ndjson").exists()
    assert (tmp_path / "far_curve.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "bench.log").exists()


def test_summary_has_expected_structure(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["bench"] == "noise_norm_calibration"
    assert summary["config"]["n_cubes"] == QUICK_SMOKE_N_CUBES
    assert "summed_expected_per_theta" in summary
    assert "summed_observed_per_theta" in summary
    assert "gate" in summary
    assert "status" in summary["gate"]
    # Expected counts strictly decrease with θ (Gaussian tail is
    # monotone-decreasing).
    expected = summary["summed_expected_per_theta"]
    thetas = sorted(float(t) for t in expected.keys())
    expected_vals = [expected[f"{t:.1f}"] for t in thetas]
    for a, b in zip(expected_vals, expected_vals[1:]):
        assert a > b, f"summed_expected non-monotone: {expected_vals}"


def test_far_curve_contains_all_kernels(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--listener-port", "0",
    ])
    assert rc == 0
    far_curve = json.loads((tmp_path / "far_curve.json").read_text())
    kernels = far_curve["kernels"]
    # Per D2: K_img × K_dm × K_time = 4 × 4 × 8 = 128 kernel triples.
    assert len(kernels) == 128
    # Each kernel has all theta-grid entries.
    for kid, table in kernels.items():
        for theta in DEFAULT_THETA_GRID:
            assert f"{theta:.1f}" in table, (
                f"kernel {kid} missing theta {theta:.1f}"
            )
            row = table[f"{theta:.1f}"]
            assert "expected" in row
            assert "observed" in row
            assert "ratio" in row
            assert row["expected"] >= 0.0
            assert row["observed"] >= 0

"""Smoke tests for ``tools/viz/search_helpers.py`` and
``tools/viz/search_detector_check.py`` (M5 chunk 5).

These tests are pure-Python (no torch / no detector) so they run in
<2 s on any matplotlib-equipped env. They exercise:

  * Each rendering helper produces a non-empty PNG / HTML output.
  * ``stitch_search_html_report`` writes a self-contained ``report.html``
    with no PASS/FAIL banner.
  * The CLI in cube_injection mode parses bench-shaped NDJSON / summary
    JSON and writes a complete report directory.
  * ``--mode burst`` raises ``NotImplementedError`` (M5 chunk 7).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("DSART_TEST", "1")

from tools.viz import search_detector_check  # noqa: E402
from tools.viz.search_helpers import (  # noqa: E402
    CandidateRow,
    FarSample,
    FigureEntry,
    KernelScoreEntry,
    RecoveryCell,
    gaussian_tail_far,
    n_eff_per_cube_per_kernel,
    render_candidates_table_html,
    render_far_curve_png,
    render_recovery_heatmap_png,
    render_score_per_kernel_png,
    stitch_search_html_report,
)


def test_gaussian_tail_far_known_values() -> None:
    """Sanity-check the Gaussian-tail FAR helper against known limits."""
    assert gaussian_tail_far(0.0) == pytest.approx(0.5, abs=1e-6)
    # Standard normal one-sided tail at θ=1: ~0.1587
    assert gaussian_tail_far(1.0) == pytest.approx(0.1587, abs=1e-3)
    # θ=5: ~2.87e-7
    assert gaussian_tail_far(5.0) == pytest.approx(2.87e-7, rel=1e-2)


def test_n_eff_per_cube_per_kernel_formula() -> None:
    n_eff = n_eff_per_cube_per_kernel(
        t_det=512, n_fdm=8, n_grid=64,
        k_img_volume=1, k_dm_width=3, k_time_width=4,
    )
    # 512 * 8 * 64 * 64 / (1 * 3 * 4) = 16777216 / 12 ≈ 1.398e6
    assert n_eff == pytest.approx(16777216 / 12, rel=1e-9)


def test_n_eff_rejects_zero_widths() -> None:
    with pytest.raises(ValueError):
        n_eff_per_cube_per_kernel(
            t_det=64, n_fdm=4, n_grid=8,
            k_img_volume=0, k_dm_width=1, k_time_width=1,
        )


def test_render_recovery_heatmap_png(tmp_path: Path) -> None:
    cells = [
        RecoveryCell(injected_snr=8.0, width_samples=4,
                     n_injected=3, n_recovered=2, snr_ratio_mean=0.95),
        RecoveryCell(injected_snr=8.0, width_samples=8,
                     n_injected=3, n_recovered=3, snr_ratio_mean=0.97),
        RecoveryCell(injected_snr=12.0, width_samples=4,
                     n_injected=3, n_recovered=3, snr_ratio_mean=0.99),
        RecoveryCell(injected_snr=12.0, width_samples=8,
                     n_injected=3, n_recovered=3, snr_ratio_mean=1.00),
    ]
    out = tmp_path / "recovery_heatmap.png"
    render_recovery_heatmap_png(cells, out_path=out, title="recovery test")
    assert out.is_file()
    assert out.stat().st_size > 0
    # PNG magic bytes
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_score_per_kernel_png(tmp_path: Path) -> None:
    entries = []
    for i_img, img in enumerate(("unit", "psf")):
        for i_dm, dm_w in enumerate((1, 3, 5, 7)):
            for i_t, t_w in enumerate((1, 2, 4, 8)):
                entries.append(KernelScoreEntry(
                    image_token=img,
                    dm_token=f"d{dm_w}",
                    time_token=f"b{t_w}",
                    k_dm_width=dm_w,
                    k_time_width=t_w,
                    snr=float(8.0 + i_img + i_dm + i_t),
                ))
    out = tmp_path / "score_per_kernel.png"
    render_score_per_kernel_png(entries, out_path=out, title="score test")
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_far_curve_png(tmp_path: Path) -> None:
    samples = [
        FarSample(theta=t, empirical_per_cube_per_kernel=10**-t,
                  analytic_per_cube_per_kernel=2 * 10**-t,
                  n_cubes=30, n_kernels=128)
        for t in (5.0, 6.0, 7.0, 8.0, 9.0)
    ]
    out = tmp_path / "far.png"
    render_far_curve_png(samples, out_path=out)
    assert out.is_file()


def test_render_candidates_table_html(tmp_path: Path) -> None:
    rows = [
        CandidateRow(
            rank=1,
            observed={
                "kernel_id": "unit:d3:b16", "snr": 9.5,
                "l": 32, "m": 32, "dm_idx": 4,
                "event_specnum": 256, "width_samples": 16,
            },
            injected={
                "l_pix": 32, "m_pix": 32,
                "fine_dm_idx": 4, "t_in_cube": 256,
            },
        ),
        CandidateRow(
            rank=2,
            observed={
                "kernel_id": "?", "snr": 8.2,
                "l": 0, "m": 0, "dm_idx": 0,
                "event_specnum": 17, "width_samples": 0,
            },
            injected=None,
        ),
    ]
    out = tmp_path / "candidates.html"
    render_candidates_table_html(rows, out_path=out)
    text = out.read_text()
    assert "<!doctype html>" in text
    assert "unit:d3:b16" in text
    assert "(noise-only" in text


def test_stitch_search_html_report(tmp_path: Path) -> None:
    figures = [
        FigureEntry(
            png_filename="x.png",
            caption="x figure",
            observed="42",
            expected="≥40",
        ),
        FigureEntry(
            png_filename="y.png",
            caption="y figure (no metrics)",
        ),
    ]
    report_path = stitch_search_html_report(
        out_dir=tmp_path,
        title="Test report",
        header_meta={"Run": "test", "Mode": "cube_injection"},
        figures=figures,
        candidates_html_filename="candidates.html",
        extra_links=[("README", "README.txt")],
    )
    assert report_path == tmp_path / "report.html"
    text = report_path.read_text()
    # No PASS/FAIL banner anywhere.
    assert "PASS" not in text and "FAIL" not in text
    # Per plan: explicit "No PASS/FAIL" note.
    assert "No PASS/FAIL banner" in text
    # All figures are present.
    assert "x.png" in text
    assert "y.png" in text
    assert "candidates.html" in text
    assert "README" in text


def test_cli_burst_mode_not_implemented() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        search_detector_check.main([
            "--mode", "burst",
            "--out", "/tmp/should_not_exist",
        ])
    assert "chunk 7" in str(excinfo.value)


def test_cli_cube_injection_end_to_end(tmp_path: Path) -> None:
    """Feed a synthetic injection_log + summary through the CLI."""
    injection_log = tmp_path / "injection_log.ndjson"
    noise_only_log = tmp_path / "noise_only_log.ndjson"
    summary_json = tmp_path / "summary.json"
    out_dir = tmp_path / "report"

    # One supra-threshold injection cell with a sane recovery + a
    # sub-threshold cell with no recovery.
    inj_records = [
        {
            "kind": "injection",
            "injected": {
                "snr": 12.0, "width_samples": 16,
                "l_pix": 32, "m_pix": 32,
                "fine_dm_idx": 4, "t_in_cube": 256,
                "profile": "boxcar",
            },
            "n_trials": 3,
            "n_recovered": 3,
            "recovered_snrs": [11.7, 11.9, 12.1],
            "matched_kernel_id": "unit:d3:b16",
            "score_per_kernel_at_match": {
                "unit:d3:b16": 12.0,
                "unit:d3:b32": 9.5,
                "unit:d1:b16": 7.5,
            },
            "cube_geometry": {"T_det": 512, "N_fdm": 8, "N_grid": 64},
        },
        {
            "kind": "injection",
            "injected": {
                "snr": 6.0, "width_samples": 4,
                "l_pix": 32, "m_pix": 32,
                "fine_dm_idx": 4, "t_in_cube": 256,
                "profile": "boxcar",
            },
            "n_trials": 3,
            "n_recovered": 0,
            "recovered_snrs": [],
            "matched_kernel_id": None,
            "score_per_kernel_at_match": {},
            "cube_geometry": {"T_det": 512, "N_fdm": 8, "N_grid": 64},
        },
    ]
    injection_log.write_text(
        "\n".join(json.dumps(r) for r in inj_records) + "\n"
    )

    noise_records = [
        {
            "kind": "noise_only",
            "cube_id": i + 1,
            "candidate_snrs": [5.6, 6.2, 7.1],
            "cube_geometry": {"T_det": 512, "N_fdm": 8, "N_grid": 64},
        }
        for i in range(3)
    ]
    noise_only_log.write_text(
        "\n".join(json.dumps(r) for r in noise_records) + "\n"
    )

    summary = {
        "tool": "bench/cube_injection_detector.py",
        "config": {
            "T_det": 512, "N_fdm": 8, "N_grid": 64,
            "seed": 12345,
        },
        "far": [
            {"theta": 6.0, "empirical_per_cube_per_kernel": 1e-4,
             "n_cubes": 3, "n_kernels": 128},
            {"theta": 8.0, "empirical_per_cube_per_kernel": 1e-7,
             "n_cubes": 3, "n_kernels": 128},
        ],
    }
    summary_json.write_text(json.dumps(summary))

    rc = search_detector_check.main([
        "--mode", "cube_injection",
        "--injection-log", str(injection_log),
        "--noise-only-log", str(noise_only_log),
        "--summary", str(summary_json),
        "--out", str(out_dir),
    ])
    assert rc == 0

    assert (out_dir / "report.html").is_file()
    assert (out_dir / "report.txt").is_file()
    assert (out_dir / "recovery_heatmap.png").is_file()
    assert (out_dir / "noise_only_far.png").is_file()
    assert (out_dir / "candidates.html").is_file()
    # Per-cell score map written for the supra-threshold cell only.
    pngs = list(out_dir.glob("score_per_kernel_*.png"))
    assert len(pngs) == 1

    # Report.html header must include the operator-pinned tool name.
    report_text = (out_dir / "report.html").read_text()
    assert "search_detector_check" in report_text
    assert "No PASS/FAIL banner" in report_text

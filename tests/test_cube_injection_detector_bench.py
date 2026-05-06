"""Smoke tests for ``bench/cube_injection_detector.py`` (M5 chunk 5).

These are deliberately lightweight: drive the bench in --quick-sweep
mode against an embedded MockTriggerListener, then assert:

  * The bench exits 0 and writes the four expected output files.
  * ``injection_log.ndjson`` contains one record per (snr, width) cell
    with all required fields and a sane shape.
  * ``noise_only_log.ndjson`` contains one record per noise-only cube.
  * ``summary.json`` is well-formed, pins the bench config, and tracks
    a sane FAR table.
  * For supra-threshold injections (snr ≥ 12) the bench recovers the
    injection in at least 1 / n_trials trials at any width.

The Chunk-6 search_node_throughput bench will exercise the production-
sized cube + the full 30 s noise-only run; this test stays small (8×8
fdm × 64×64 grid × 1 trial × 4 noise cubes) so it finishes in <30 s on
h01 GPU 1.
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

from bench import cube_injection_detector as bench_mod  # noqa: E402


@pytest.fixture(scope="module")
def quick_sweep_outdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run --quick-sweep once per test module; cache outputs for
    multiple-assertion parsing without re-running the bench."""
    out = tmp_path_factory.mktemp("cube_injection_quick_sweep")
    rc = bench_mod.main([
        "--quick-sweep",
        "--listener-port", "0",
        "--seed", "42",
        "--out", str(out),
    ])
    assert rc == 0, f"bench exited {rc}"
    return out


def test_bench_writes_expected_files(quick_sweep_outdir: Path) -> None:
    """Bench must produce the four canonical output files."""
    assert (quick_sweep_outdir / "injection_log.ndjson").is_file()
    assert (quick_sweep_outdir / "noise_only_log.ndjson").is_file()
    assert (quick_sweep_outdir / "summary.json").is_file()
    assert (quick_sweep_outdir / "bench.log").is_file()


def test_injection_log_well_formed(quick_sweep_outdir: Path) -> None:
    """Each NDJSON line is a valid JSON object with required fields."""
    path = quick_sweep_outdir / "injection_log.ndjson"
    text = path.read_text().strip()
    assert text, "injection_log.ndjson is empty"
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    # Quick sweep is 2 SNRs × 2 widths = 4 cells.
    assert len(records) == len(bench_mod.QUICK_SWEEP_SNRS) * len(bench_mod.QUICK_SWEEP_WIDTHS)
    required_fields = {
        "kind", "injected", "n_trials", "n_recovered", "recovered_snrs",
        "matched_kernel_id", "score_per_kernel_at_match", "cube_geometry",
    }
    for rec in records:
        assert rec["kind"] == "injection"
        assert required_fields.issubset(rec.keys()), (
            f"missing fields in record: {required_fields - rec.keys()}"
        )
        injected = rec["injected"]
        assert {"snr", "width_samples", "l_pix", "m_pix",
                "fine_dm_idx", "t_in_cube", "profile"}.issubset(
            injected.keys()
        )
        assert injected["profile"] == "boxcar"
        assert isinstance(rec["recovered_snrs"], list)


def test_noise_only_log_well_formed(quick_sweep_outdir: Path) -> None:
    """Each line in noise_only_log.ndjson must parse and carry
    ``candidate_snrs``."""
    path = quick_sweep_outdir / "noise_only_log.ndjson"
    text = path.read_text().strip()
    if not text:
        pytest.skip("quick-sweep had 0 noise-only cubes")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(records) >= 1
    for rec in records:
        assert rec["kind"] == "noise_only"
        assert "candidate_snrs" in rec
        assert isinstance(rec["candidate_snrs"], list)


def test_summary_json_structure(quick_sweep_outdir: Path) -> None:
    summary = json.loads((quick_sweep_outdir / "summary.json").read_text())
    assert summary["tool"] == "bench/cube_injection_detector.py"
    assert summary["config"]["quick_sweep"] is True
    assert summary["config"]["seed"] == 42
    assert summary["config"]["recovery_snr_fraction"] == bench_mod.RECOVERY_SNR_FRACTION
    assert isinstance(summary["cells"], list)
    assert len(summary["cells"]) >= 1
    for cell in summary["cells"]:
        assert {"injected", "n_trials", "n_recovered",
                "recovery_fraction", "matched_kernel_id"}.issubset(cell.keys())
    assert isinstance(summary["far"], list)
    # Quick sweep has 4 noise cubes, so 5 θ samples are emitted.
    if summary["far"]:
        assert {"theta", "empirical_per_cube_per_kernel",
                "n_cubes", "n_kernels"}.issubset(summary["far"][0].keys())


def test_supra_threshold_injection_recovered(quick_sweep_outdir: Path) -> None:
    """For the strongest supra-threshold cell (snr=12, width=4) we
    expect at least one trial to recover with a sane SNR ratio.

    This is a sanity gate, not a full recovery sweep — the operator
    inspects the recovery_heatmap.png for the full-sweep numbers.
    """
    summary = json.loads((quick_sweep_outdir / "summary.json").read_text())
    strong_cells = [
        c for c in summary["cells"]
        if c["injected"]["snr"] >= 12.0
    ]
    assert strong_cells, "expected at least one snr ≥ 12 cell"
    any_recovered = any(c["n_recovered"] >= 1 for c in strong_cells)
    assert any_recovered, (
        f"no snr≥12 cell recovered the injection in any trial; "
        f"cells={strong_cells}"
    )


def test_bench_match_radius_helpers() -> None:
    """Sanity-check the match-radius derivation used by the bench's
    recovery scorer."""
    # k_dm = 1, k_t = 1 → minimal radii.
    fdm, t = bench_mod._kernel_match_radius("unit:d1:b1")
    assert fdm == bench_mod.RECOVERY_FDM_TOL
    assert t == bench_mod.RECOVERY_T_TOL
    # k_dm = 7, k_t = 128 → radii grow with kernel width.
    fdm, t = bench_mod._kernel_match_radius("unit:d7:b128")
    assert fdm == max(bench_mod.RECOVERY_FDM_TOL, 7 // 2 + 1)
    assert t == max(bench_mod.RECOVERY_T_TOL, 128 // 2 + 1)
    # Malformed kernel_id falls back to defaults.
    fdm, t = bench_mod._kernel_match_radius("garbage")
    assert fdm == bench_mod.RECOVERY_FDM_TOL
    assert t == bench_mod.RECOVERY_T_TOL

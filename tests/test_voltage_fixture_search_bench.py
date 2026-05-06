"""Smoke tests for ``bench/voltage_fixture_search.py`` (M5 Chunk 7).

Mode-synthetic produces a ``run.json`` with the expected fields and
the gate fires PASS for a strong injection at known (l_pix, m_pix,
fine_dm_idx, t_in_cube). Mode-captured (post F6 resolution
2026-05-06) loads the M3-published captured-NPZ set via
``dsart.transport.captured_npz`` and emits an INSPECTION_ONLY run
record (cube shape + T2 truth) — the operator-facing detector sweep
against the captured stack is the next chunk-7 hardening item.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("DSART_TEST", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bench"))

from voltage_fixture_search import (  # noqa: E402
    DEFAULT_FIXTURE,
    _load_captured_npz_set,
    main,
)


def test_synthetic_mode_writes_run_json(tmp_path) -> None:
    rc = main([
        "--mode", "synthetic",
        "--out", str(tmp_path),
        "--listener-port", "0",
        "--t-det", "32", "--n-fdm", "4", "--n-grid", "16",
        "--synthetic-amplitude", "300.0",
        "--threshold-sigma", "6.0",
    ])
    assert rc == 0
    run_path = tmp_path / "run.json"
    assert run_path.exists()
    record = json.loads(run_path.read_text())
    assert record["bench"] == "voltage_fixture_search"
    assert record["mode"] == "synthetic"
    assert record["fixture"]["name"] == DEFAULT_FIXTURE.name
    assert record["fixture"]["dm_pc_cm3"] == DEFAULT_FIXTURE.dm_pc_cm3
    assert "synthetic_burst" in record
    assert record["gate_status"] in ("PASS", "FAIL")


def test_synthetic_mode_recovers_strong_injection(tmp_path) -> None:
    rc = main([
        "--mode", "synthetic",
        "--out", str(tmp_path),
        "--listener-port", "0",
        "--t-det", "32", "--n-fdm", "4", "--n-grid", "16",
        "--synthetic-amplitude", "500.0",  # very strong
        "--threshold-sigma", "6.0",
    ])
    assert rc == 0
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["gate_status"] == "PASS", (
        f"strong injection should recover; got {record}"
    )
    assert record["recovered"] is not None
    burst = record["synthetic_burst"]
    rec = record["recovered"]
    assert abs(rec["l_pix"] - burst["l_pix"]) <= 2
    assert abs(rec["m_pix"] - burst["m_pix"]) <= 2


def test_captured_mode_raises_on_missing_manifest(tmp_path) -> None:
    """Captured-mode loader (F6 resolved) raises ``FileNotFoundError``
    if ``manifest.json`` is missing from the captured dir."""
    empty_dir = tmp_path / "fake_captures"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        _load_captured_npz_set(empty_dir)


def test_captured_mode_cli_requires_captured_dir(tmp_path) -> None:
    """``--mode captured`` without ``--captured-dir`` should fail
    with a clear error. The CLI exits via ``SystemExit`` before the
    loader is invoked.
    """
    with pytest.raises(SystemExit):
        main([
            "--mode", "captured",
            "--out", str(tmp_path),
            "--listener-port", "0",
        ])


def test_captured_mode_inspection_only_run_record(tmp_path) -> None:
    """``--mode captured --captured-dir <synthetic-fixture>`` writes
    an INSPECTION_ONLY ``run.json`` capturing the M3 → M5 cube shape
    + T2 truth. The detector sweep itself is a follow-up chunk-7
    hardening item; this test only verifies the loader-bench glue.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_captured_npz import _write_synthetic_fixture  # noqa: E402

    captured_dir = tmp_path / "captured"
    _write_synthetic_fixture(
        captured_dir,
        n_chgroups_present=4,
        n_fv_total=3,
        n_grid=8,
        run_id="testrun",
        is_burst=True,
    )
    out = tmp_path / "out"
    rc = main([
        "--mode", "captured",
        "--captured-dir", str(captured_dir),
        "--out", str(out),
        "--listener-port", "0",
    ])
    assert rc == 0
    run_path = out / "run.json"
    assert run_path.exists()
    rec = json.loads(run_path.read_text())
    assert rec["mode"] == "captured"
    assert rec["gate_status"] == "INSPECTION_ONLY"
    assert rec["manifest"]["run_id"] == "testrun"
    assert rec["manifest"]["src_truth"]["is_burst"] is True
    shp = rec["streams_shape"]
    assert shp["n_chgroup_total"] == 16
    assert shp["n_chgroup_present"] == 4
    assert shp["n_fv_total"] == 3
    assert shp["n_grid"] == 8
    assert sum(shp["valid_mask"]) == 4

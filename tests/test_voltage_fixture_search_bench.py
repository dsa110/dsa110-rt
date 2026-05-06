"""Smoke tests for ``bench/voltage_fixture_search.py`` (M5 Chunk 7).

Mode-synthetic produces a ``run.json`` with the expected fields and
the gate fires PASS for a strong injection at known (l_pix, m_pix,
fine_dm_idx, t_in_cube). Mode-captured raises ``NotImplementedError``
pending M3's captured-NPZ schema (F6); we test that it raises
cleanly with an actionable error message.
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


def test_captured_mode_raises_until_m3_schema_locked(tmp_path) -> None:
    """F6: M3 owns the captured-npz schema. Until the schema is
    published, the captured-mode loader raises NotImplementedError
    with an actionable message pointing at the chunk-7 hardening.
    """
    bogus_dir = tmp_path / "fake_captures"
    bogus_dir.mkdir()
    with pytest.raises(NotImplementedError, match="F6|captured-npz schema|M3"):
        _load_captured_npz_set(bogus_dir)


def test_captured_mode_cli_requires_captured_dir(tmp_path) -> None:
    """``--mode captured`` without ``--captured-dir`` should fail
    with a clear error. The CLI exits via SystemExit before the
    NotImplementedError fires.
    """
    with pytest.raises(SystemExit):
        main([
            "--mode", "captured",
            "--out", str(tmp_path),
            "--listener-port", "0",
        ])

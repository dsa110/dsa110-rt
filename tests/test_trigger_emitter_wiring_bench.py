"""Smoke tests for ``bench/trigger_emitter_wiring.py`` (M5 chunk 4(b)).

The bench drives ``TriggerEmitter`` against a 16-port mock listener fan
and validates four operationally observable properties (plan §8 line
2328):

  Gate A: accepted-ack p99 within ``--ack-p99-budget-ms``.
  Gate B: rate-limit / per-cube-cap drop counter > 0 in the blast phase.
  Gate C: listener flips to RECONNECTING/DOWN on kill, back to UP on
          restart.
  Gate D: steady-phase fan-out parity ≥ 0.99 across surviving listeners.

These tests cover:

  * the bench-side gate-evaluation helper (no IO),
  * the candidate-builder helpers (deterministic (l, m, kernel) and
    schema-valid Candidate objects),
  * an end-to-end quick-smoke run that produces the expected output
    files and stamps the gate.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from trigger_emitter_wiring import (  # noqa: E402
    DEFAULT_ACK_P99_BUDGET_MS,
    KERNEL_IDS_FOR_BENCH,
    QUICK_SMOKE_N_LISTENERS,
    _build_blast_cube,
    _build_steady_cube,
    _evaluate_gate,
    _make_candidate,
    main,
)

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.trigger.emitter import ConnState  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helpers (no IO)
# ---------------------------------------------------------------------------


def test_make_candidate_returns_valid_candidate() -> None:
    cand = _make_candidate(
        cube_id=3, cand_idx=2, snr=12.5,
        kernel_id="psf:d3:b16",
    )
    assert isinstance(cand, Candidate)
    assert cand.snr == 12.5
    assert cand.kernel_id == "psf:d3:b16"
    # Schema validation runs at __post_init__; the Candidate's
    # ``flags`` and ``detector_version`` defaults must satisfy it.
    assert cand.detector_version == "v1.M5"
    assert int(cand.flags) == int(CandidateFlags.NONE)


def test_make_candidate_distinct_lm_per_cand_idx() -> None:
    """The (l, m) deltas across cand_idx within one cube must be
    distinct so the bench's holdoff-defeat invariant holds."""
    cand_lm = {
        (
            _make_candidate(
                cube_id=0, cand_idx=i, snr=10.0,
                kernel_id="psf:d3:b16",
            ).l,
            _make_candidate(
                cube_id=0, cand_idx=i, snr=10.0,
                kernel_id="psf:d3:b16",
            ).m,
        )
        for i in range(8)
    }
    assert len(cand_lm) == 8, (
        f"(l, m) collisions across cand_idx: {cand_lm}"
    )


def test_build_steady_cube_size() -> None:
    cands = _build_steady_cube(cube_id=5, n_cands=4)
    assert len(cands) == 4
    assert all(isinstance(c, Candidate) for c in cands)
    # Kernel ids rotate through KERNEL_IDS_FOR_BENCH (mod len).
    assert cands[0].kernel_id == KERNEL_IDS_FOR_BENCH[0]
    assert cands[1].kernel_id == KERNEL_IDS_FOR_BENCH[1]


def test_build_blast_cube_uses_higher_snr_than_steady() -> None:
    steady = _build_steady_cube(cube_id=0, n_cands=1)[0]
    blast = _build_blast_cube(cube_id=0, n_cands=1)[0]
    # Both clear the SnrThreshold(min_snr=8.0) gate; blast is louder so
    # it doesn't get throttled by SNR thresholding when the chain is
    # exercised.
    assert steady.snr >= 8.0
    assert blast.snr >= steady.snr


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def _summary_template() -> dict:
    return {
        "phases": {
            "steady": {
                "accepted_ack_p99_ms": 5.0,
                "completed_ack_p99_ms": 12.0,
                "fanout_parity_min": 1.0,
                "fanout_parity_max": 1.0,
            },
            "listener_fail": {
                "state_after_kill": ConnState.RECONNECTING,
                "state_after_restart": ConnState.UP,
            },
            "rate_limit_blast": {
                "delta_dropped_by_condition": {
                    "ratelimit": 100,
                },
            },
        },
    }


def test_gate_passes_on_clean_summary() -> None:
    summary = _summary_template()
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["overall_pass"] is True
    assert gate["gate_a_accepted_ack_p99_within_budget"]["pass"] is True
    assert gate["gate_b_rate_limit_or_per_cube_cap_fired"]["pass"] is True
    assert gate["gate_c_listener_recovery"]["pass"] is True
    assert gate["gate_d_steady_fanout_parity_ge_0_99"]["pass"] is True


def test_gate_a_fails_when_ack_p99_exceeds_budget() -> None:
    summary = _summary_template()
    summary["phases"]["steady"]["accepted_ack_p99_ms"] = 25.0
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["overall_pass"] is False
    assert gate["gate_a_accepted_ack_p99_within_budget"]["pass"] is False


def test_gate_a_uses_accepted_ack_not_completed() -> None:
    """Plan §8 line 2328's 20 ms budget targets the wire-level
    accepted-ack stage; completed-ack (which includes voltage-dump
    latency) is informational only."""
    summary = _summary_template()
    summary["phases"]["steady"]["accepted_ack_p99_ms"] = 5.0
    summary["phases"]["steady"]["completed_ack_p99_ms"] = 999.0
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_a_accepted_ack_p99_within_budget"]["pass"] is True


def test_gate_b_fails_when_rate_limit_chain_silent() -> None:
    summary = _summary_template()
    summary["phases"]["rate_limit_blast"]["delta_dropped_by_condition"] = {}
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_b_rate_limit_or_per_cube_cap_fired"]["pass"] is False


def test_gate_b_per_cube_total_cap_drops_count() -> None:
    """PerCubeTotalCap drops are also acceptable evidence of the
    chain firing under blast (the 100 cands/cube blast trips the
    PerCubeTotalCap=16 threshold before RateLimit does on the very
    first cube)."""
    summary = _summary_template()
    summary["phases"]["rate_limit_blast"]["delta_dropped_by_condition"] = {
        "per_cube_total_cap": 84,
    }
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_b_rate_limit_or_per_cube_cap_fired"]["pass"] is True


def test_gate_c_fails_when_listener_never_recovers() -> None:
    summary = _summary_template()
    summary["phases"]["listener_fail"]["state_after_restart"] = (
        ConnState.RECONNECTING
    )
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_c_listener_recovery"]["pass"] is False


def test_gate_c_fails_when_listener_never_disconnects() -> None:
    """If the kill is invisible to the emitter (state stays UP), the
    bench cannot have exercised the reconnect path."""
    summary = _summary_template()
    summary["phases"]["listener_fail"]["state_after_kill"] = ConnState.UP
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_c_listener_recovery"]["pass"] is False


def test_gate_d_fails_when_fanout_skewed() -> None:
    summary = _summary_template()
    summary["phases"]["steady"]["fanout_parity_min"] = 0.50
    gate = _evaluate_gate(summary, ack_p99_budget_ms=20.0)
    assert gate["gate_d_steady_fanout_parity_ge_0_99"]["pass"] is False


# ---------------------------------------------------------------------------
# End-to-end quick-smoke
# ---------------------------------------------------------------------------


def test_quick_smoke_writes_outputs(tmp_path) -> None:
    """Quick-smoke runs all 3 phases against 5 mock listeners and
    writes summary.json + ndjson + bench.log."""
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
        "--ack-p99-budget-ms", str(DEFAULT_ACK_P99_BUDGET_MS),
    ])
    # Smoke run is small enough that the gate may FAIL on a noisy
    # event loop (single-cube p99 outliers). We only assert that the
    # bench completed the IO. Exit code is 0 (PASS) or 1 (FAIL).
    assert rc in (0, 1), f"unexpected exit code {rc}"
    assert (tmp_path / "trigger_records.ndjson").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "bench.log").exists()


def test_quick_smoke_summary_has_expected_structure(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
    ])
    assert rc in (0, 1)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == "M5.bench.trigger_emitter_wiring.v1"
    assert summary["config"]["n_listeners"] == QUICK_SMOKE_N_LISTENERS
    assert "phases" in summary
    assert set(summary["phases"].keys()) == {
        "steady", "listener_fail", "rate_limit_blast",
    }
    assert "per_listener_final" in summary
    assert "gate" in summary
    g = summary["gate"]
    for key in (
        "gate_a_accepted_ack_p99_within_budget",
        "gate_b_rate_limit_or_per_cube_cap_fired",
        "gate_c_listener_recovery",
        "gate_d_steady_fanout_parity_ge_0_99",
        "overall_pass",
    ):
        assert key in g, f"missing gate key {key}"


def test_quick_smoke_listener_recovery_gate_passes(tmp_path) -> None:
    """The reconnect path is deterministic on localhost — Gate C
    should consistently PASS even on a noisy CI loop."""
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
    ])
    assert rc in (0, 1)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["gate"]["gate_c_listener_recovery"]["pass"] is True


def test_quick_smoke_rate_limit_gate_passes(tmp_path) -> None:
    """The rate-limit blast deliberately exceeds the chain's burst
    capacity; Gate B should consistently PASS (drops > 0)."""
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
    ])
    assert rc in (0, 1)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["gate"]["gate_b_rate_limit_or_per_cube_cap_fired"]["pass"] is True


def test_quick_smoke_records_per_phase(tmp_path) -> None:
    rc = main([
        "--quick-smoke",
        "--out", str(tmp_path),
    ])
    assert rc in (0, 1)
    records_path = tmp_path / "trigger_records.ndjson"
    phases = set()
    with records_path.open() as f:
        for line in f:
            rec = json.loads(line)
            phases.add(rec["phase"])
    assert phases == {
        "steady", "listener_fail_pre_restart",
        "listener_fail_post_restart", "rate_limit_blast",
    }

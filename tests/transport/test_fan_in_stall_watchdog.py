"""Fan-in stall watchdog — pure-decision unit tests (2026-07-21 RCA).

The 2026-07-21 fleet-wide freeze was NOT a dump deadlock: a shared
correlator group (corr 15) dropped out on every search node, and with
``fan_in_min_corrs=15`` (16 corrs, zero margin) the ``ProductionRxRingSource``
consumer parked in its fan-in gate ``await asyncio.sleep`` and emitted no
cubes — SILENTLY. All 8 halves went quiet with no log line, and the
outage was initially mis-attributed to the C2 ``too_late`` cube-dump path.

``summarize_fan_in_shortfall`` is the pure decision the gate-fail branch
uses to log a loud, correctly-attributed WARNING (and bump the
``n_fan_in_stall`` status counter) the instant the pipeline starves. It
is pure (no ring / no C extension / no CUDA) so it is testable off the
search nodes.
"""

from __future__ import annotations

import pytest

try:
    from dsart.transport.production_rx_ring import summarize_fan_in_shortfall

    _IMPORT_OK = True
except Exception:  # pragma: no cover - import-time skip guard
    _IMPORT_OK = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK, reason="dsart.transport.production_rx_ring import failed"
)


def test_all_corrs_present_no_shortfall() -> None:
    wseqs = [100] * 16
    n_at_target, shortfall, behind = summarize_fan_in_shortfall(wseqs, 50, 15)
    assert n_at_target == 16
    assert shortfall == 0
    assert behind == []


def test_one_dead_corr_still_meets_min() -> None:
    # corr 15 below target — exactly the 1-corr margin fan_in_min=15 allows.
    wseqs = [100] * 15 + [5]
    n_at_target, shortfall, behind = summarize_fan_in_shortfall(wseqs, 50, 15)
    assert n_at_target == 15
    assert shortfall == 0  # gate would still PASS — no stall
    assert behind == [15]


def test_two_dead_corrs_is_unsatisfiable_stall() -> None:
    # The incident: corr 15 dead AND a second corr frozen below target.
    # Only 14 at target < 15 required -> permanent stall until a corr
    # returns. The watchdog names exactly these two corrs.
    wseqs = [100] * 14 + [5, 5]
    n_at_target, shortfall, behind = summarize_fan_in_shortfall(wseqs, 50, 15)
    assert n_at_target == 14
    assert shortfall == 1
    assert behind == [14, 15]


def test_behind_corrs_are_reported_in_index_order() -> None:
    wseqs = [100, 5, 100, 5, 100, 5]
    _, shortfall, behind = summarize_fan_in_shortfall(wseqs, 50, 6)
    assert behind == [1, 3, 5]
    assert shortfall == 3  # need 6, only 3 present


def test_boundary_equal_target_counts_as_present() -> None:
    # A corr exactly AT target is present (gate uses >=).
    wseqs = [50, 50, 49]
    n_at_target, shortfall, behind = summarize_fan_in_shortfall(wseqs, 50, 3)
    assert n_at_target == 2
    assert behind == [2]
    assert shortfall == 1

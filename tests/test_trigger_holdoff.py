"""Tests for ``dsart.trigger.holdoff`` (M5 chunk 4).

Plan §4.4 line 1718. Coverage:

  * make_cell_key rounds (l, m) by k_lm and includes kernel_id.
  * is_suppressed returns False for never-seen cells.
  * check_and_register suppresses repeated emits within holdoff_ms;
    allows after the holdoff window expires.
  * Different (l, m) cells are independent.
  * Different kernel_ids are independent (same (l, m) cell can fire
    for kernel A and kernel B within the holdoff window).
  * prune drops stale entries.
  * reset clears state.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate, CandidateFlags  # noqa: E402
from dsart.trigger.holdoff import (  # noqa: E402
    DEFAULT_HOLDOFF_MS,
    HoldoffStateMachine,
    make_cell_key,
)


def _cand(*, l: float = 4.0, m: float = 4.0,
          kernel_id: str = "psf:d3:b16",
          snr: float = 9.0) -> Candidate:
    return Candidate(
        l=l, m=m, dm_fine=10.0, dm_idx=10,
        event_specnum=256, width_samples=4,
        kernel_id=kernel_id, snr=snr, detector_version="v1.M5",
        flags=int(CandidateFlags.NONE),
        search_node_id=0, gpu_half=0,
    )


def test_default_holdoff_ms() -> None:
    assert DEFAULT_HOLDOFF_MS == 50.0


def test_make_cell_key_rounds_lm() -> None:
    key = make_cell_key(_cand(l=4.4, m=5.6), k_lm=1.0)
    assert key == (4, 6, "psf:d3:b16")
    key2 = make_cell_key(_cand(l=4.4, m=5.6), k_lm=2.0)
    # round(4.4/2) = 2, round(5.6/2) = 3
    assert key2 == (2, 3, "psf:d3:b16")


def test_make_cell_key_uses_kernel_id() -> None:
    a = make_cell_key(_cand(kernel_id="psf:d3:b16"))
    b = make_cell_key(_cand(kernel_id="unit:d1:b1"))
    assert a != b


def test_is_suppressed_false_on_first_seen() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.is_suppressed(_cand(), now_utc_ns=1_000_000_000)


def test_check_and_register_first_call_passes() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.check_and_register(_cand(), now_utc_ns=1_000_000_000)


def test_check_and_register_repeat_within_window_suppressed() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.check_and_register(_cand(), now_utc_ns=1_000_000_000)
    # 25 ms later — well within the 50 ms window.
    assert s.check_and_register(_cand(), now_utc_ns=1_025_000_000)


def test_check_and_register_after_window_passes() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.check_and_register(_cand(), now_utc_ns=1_000_000_000)
    # 51 ms later → window expired.
    assert not s.check_and_register(_cand(), now_utc_ns=1_051_000_000)


def test_different_lm_cells_independent() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.check_and_register(_cand(l=4.0, m=4.0), now_utc_ns=1_000_000_000)
    # 25 ms later, but at a different (l, m).
    assert not s.check_and_register(
        _cand(l=10.0, m=10.0), now_utc_ns=1_025_000_000
    )


def test_different_kernel_ids_independent() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    assert not s.check_and_register(
        _cand(kernel_id="psf:d3:b16"), now_utc_ns=1_000_000_000,
    )
    # Same (l, m), different kernel_id, within window → not suppressed.
    assert not s.check_and_register(
        _cand(kernel_id="unit:d1:b1"), now_utc_ns=1_025_000_000,
    )


def test_prune_drops_stale_cells() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    s.check_and_register(_cand(l=4.0, m=4.0), now_utc_ns=1_000_000_000)
    s.check_and_register(_cand(l=10.0, m=10.0), now_utc_ns=1_010_000_000)
    assert s.active_cells == 2
    # Prune at t = 1.080 s — both cells are >50 ms old.
    n_dropped = s.prune(now_utc_ns=1_080_000_000)
    assert n_dropped == 2
    assert s.active_cells == 0


def test_prune_keeps_recent_cells() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    s.check_and_register(_cand(l=4.0, m=4.0), now_utc_ns=1_000_000_000)
    # Prune just 10 ms later — the cell is well within the window.
    n_dropped = s.prune(now_utc_ns=1_010_000_000)
    assert n_dropped == 0
    assert s.active_cells == 1


def test_reset_clears_state() -> None:
    s = HoldoffStateMachine(holdoff_ms=50.0)
    s.check_and_register(_cand(), now_utc_ns=1_000_000_000)
    s.check_and_register(_cand(l=10.0, m=10.0), now_utc_ns=1_010_000_000)
    assert s.active_cells == 2
    s.reset()
    assert s.active_cells == 0


def test_zero_holdoff_never_suppresses() -> None:
    """holdoff_ms=0 → every emission passes through (useful for tests
    that want to disable holdoff)."""
    s = HoldoffStateMachine(holdoff_ms=0.0)
    assert not s.check_and_register(_cand(), now_utc_ns=1_000_000_000)
    assert not s.check_and_register(_cand(), now_utc_ns=1_000_000_001)


def test_negative_holdoff_rejected() -> None:
    with pytest.raises(ValueError, match="holdoff_ms"):
        HoldoffStateMachine(holdoff_ms=-1.0)


def test_zero_klm_rejected() -> None:
    with pytest.raises(ValueError, match="k_lm"):
        HoldoffStateMachine(k_lm=0.0)

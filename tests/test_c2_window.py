"""Tests for :mod:`dsart.coinc.window` (rolling time-window buffer)."""

from __future__ import annotations

from typing import List

import pytest

from dsart.coinc import wire
from dsart.coinc.window import TimeWindow, WindowEntry


def _hdr(
    n: int,
    *,
    mjd_start: float = 60781.0,
    event_specnum_start: int = 100_000,
    search_node_id: int = 1,
    gpu_half: int = 0,
    cube_id: int = 0,
    sample_period_us: float = 1048.576,
    sample_period_specnum: int = 1,
) -> wire.C1BatchHeader:
    return wire.build_header(
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        n_candidates=n,
    )


def _row(
    event_specnum: int,
    *,
    snr: float = 10.0,
    width_samples: int = 4,
    kernel_id: str = "unit:d1:b4",
) -> wire.C1CandidateRow:
    return wire.C1CandidateRow(
        snr=snr,
        l_rad=0.0,
        m_rad=0.0,
        l_pix=0,
        m_pix=0,
        dm_pc_cc=100.0,
        dm_idx_global=10,
        fine_dm_idx=0,
        event_specnum=event_specnum,
        width_samples=width_samples,
        kernel_id=kernel_id,
        flags=0,
    )


def test_window_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        TimeWindow(window_s=0.0)
    with pytest.raises(ValueError):
        TimeWindow(window_s=-1.0)


def test_window_basic_add_returns_inserted_entries() -> None:
    win = TimeWindow(window_s=5.0)
    header = _hdr(2)
    rows = [_row(100_000), _row(100_016)]
    inserted = win.add(header, rows)
    assert len(inserted) == 2
    assert win.snapshot() == inserted
    # No age-out on first batch — window has only just opened.
    assert win.aged_out() == []
    assert pytest.approx(win.latest_mjd, abs=1e-12) == inserted[-1].mjd


def test_window_ages_out_old_entries_on_new_batch() -> None:
    win = TimeWindow(window_s=1.0)  # tight 1-second window
    # First batch at t=0.0 s (relative to mjd_start). Use a small
    # sample_period so we can step many samples and exceed 1s easily.
    header_old = _hdr(
        1, mjd_start=60781.0, event_specnum_start=0,
        sample_period_us=1e6,  # 1 second per sample
    )
    win.add(header_old, [_row(event_specnum=0)])
    # Now insert something 2 seconds later — should evict the first.
    header_new = _hdr(
        1, mjd_start=60781.0,
        event_specnum_start=0,  # share the same epoch
        sample_period_us=1e6,
    )
    new = win.add(header_new, [_row(event_specnum=2)])  # 2 s later
    aged = win.aged_out()
    assert len(aged) == 1
    assert aged[0].event_specnum == 0
    snap = win.snapshot()
    assert len(snap) == 1
    assert snap[0] is new[0]


def test_window_sorts_late_inserts() -> None:
    win = TimeWindow(window_s=10.0)
    header = _hdr(2, sample_period_us=1e6)  # 1 sec per sample
    win.add(header, [_row(event_specnum=2)])
    win.add(header, [_row(event_specnum=1)])  # earlier in time
    snap = win.snapshot()
    assert snap[0].event_specnum == 1
    assert snap[1].event_specnum == 2


def test_window_keeps_entries_inside_window() -> None:
    # Use a generous window so we can verify "keep" vs "evict" without
    # tripping over float-precision drift at the boundary. window=5.5s,
    # spacing=1s.
    win = TimeWindow(window_s=5.5)
    for spn in (1, 2, 3, 4, 5):
        win.add(_hdr(1, sample_period_us=1e6), [_row(event_specnum=spn)])
    assert len(win) == 5
    # spn=6 lands 5 s after spn=1 → cutoff = 6 - 5.5 = 0.5 → keep all.
    win.add(_hdr(1, sample_period_us=1e6), [_row(event_specnum=6)])
    assert win.aged_out() == []
    assert len(win) == 6
    # spn=10 → cutoff = 10 - 5.5 = 4.5 → evict spn 1..4.
    win.add(_hdr(1, sample_period_us=1e6), [_row(event_specnum=10)])
    aged = win.aged_out()
    assert sorted(a.event_specnum for a in aged) == [1, 2, 3, 4]
    # Survivors: spn 5, 6, 10.
    assert sorted(e.event_specnum for e in win.snapshot()) == [5, 6, 10]


def test_window_consumes_aged_out_only_once() -> None:
    win = TimeWindow(window_s=1.0)
    h = _hdr(1, sample_period_us=1e6)
    win.add(h, [_row(event_specnum=0)])
    win.add(h, [_row(event_specnum=3)])
    aged1 = win.aged_out()
    assert len(aged1) == 1
    aged2 = win.aged_out()
    assert aged2 == []


def test_window_entry_from_row_preserves_provenance() -> None:
    header = _hdr(1, search_node_id=9, gpu_half=1, cube_id=42)
    row = _row(100_000, snr=12.3, width_samples=8, kernel_id="unit:d1:b8")
    win = TimeWindow(window_s=5.0)
    inserted = win.add(header, [row])
    e = inserted[0]
    assert e.search_node_id == 9
    assert e.gpu_half == 1
    assert e.cube_id == 42
    assert e.snr == 12.3
    assert e.width_samples == 8
    assert e.kernel_id == "unit:d1:b8"
    assert e.event_specnum == 100_000

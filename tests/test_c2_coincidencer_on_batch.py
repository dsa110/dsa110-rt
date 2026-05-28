"""Regression tests for :class:`CoincidencerService._on_batch` —
specifically the window/graph contract that emerged from the M7.4 burst
test: late-arriving candidates (rows with ``mjd`` already below the
window cutoff) must be inserted+aged-out by the window *and* skipped
from the components graph, otherwise the graph leaks them forever.

These tests drive the service's batch handler directly (no socket, no
broadcaster) so they're cheap to run and exercise the exact code path
that produced the ``graph_size = 4.4 × window_size`` observation in
``/mon/c2/h23`` on 2026-05-27.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from dsart.coinc import wire
from dsart.services.coincidencer import (
    CoincidencerConfig,
    CoincidencerService,
)


def asyncio_test(func):
    """Custom asyncio test decorator (no pytest-asyncio dependency).

    Mirrors the pattern in ``tests/test_search_compute_service.py`` so
    these tests run under the same plain pytest invocation used by the
    rest of the suite.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _NoopBroadcaster:
    def broadcast(self, **kwargs) -> Dict[int, bool]:
        return {}

    def close(self) -> None:
        pass


class _StubStore:
    def __init__(self) -> None:
        self.last_put: Dict[str, Any] = {}

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        return None

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        self.last_put = dict(value)


def _criteria_file(tmp_path: Path) -> Path:
    p = tmp_path / "c.yaml"
    p.write_text("""
trigger_classes:
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    return p


def _make_service(tmp_path: Path, window_s: float = 5.0) -> CoincidencerService:
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1",
        bind_port=0,
        window_s=window_s,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=_criteria_file(tmp_path),
        name_allocator_offline=True,
        gal_dm_poll_interval_s=60.0,
    )
    return CoincidencerService(
        config=cfg,
        mon_store=_StubStore(),
        broadcaster=_NoopBroadcaster(),
    )


def _batch(
    mjd_start: float,
    n: int,
    *,
    event_specnum_start: int = 0,
    # 0.1 s per sample → batches of n entries span (n-1)*0.1 s in MJD,
    # comfortably inside a multi-second window for the tests below.
    sample_period_us: float = 100_000.0,
    sample_period_specnum: int = 1,
    cube_id: int = 0,
    snr: float = 9.0,
) -> wire.C1Batch:
    header = wire.build_header(
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=n,
    )
    rows = tuple(
        wire.C1CandidateRow(
            snr=snr + 0.1 * i,
            l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
            dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
            event_specnum=event_specnum_start + i,
            width_samples=4, kernel_id="unit:d1:b4", flags=0,
        )
        for i in range(n)
    )
    return wire.C1Batch(header=header, candidates=rows)


# Convenience: build a batch whose rows are exactly the requested
# (mjd_offset_seconds) tuple. mjd_offset_seconds is added to the header's
# mjd_start so callers don't have to think about sample_period scaling.
def _batch_at_offsets(
    mjd_start: float,
    offsets_s: tuple,
    *,
    event_specnum_start: int = 0,
) -> wire.C1Batch:
    # Use sample_period_us=1e6 so 1 specnum == 1 second of mjd offset.
    sample_period_us = 1_000_000.0
    sample_period_specnum = 1
    header = wire.build_header(
        cube_id=0,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=len(offsets_s),
    )
    rows = tuple(
        wire.C1CandidateRow(
            snr=9.0, l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
            dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
            event_specnum=event_specnum_start + int(off),
            width_samples=4, kernel_id="unit:d1:b4", flags=0,
        )
        for off in offsets_s
    )
    return wire.C1Batch(header=header, candidates=rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@asyncio_test
async def test_normal_batch_advances_window_and_graph_in_lockstep(
    tmp_path: Path,
) -> None:
    """Baseline: in-window batches grow the graph by exactly the number
    of new entries, and old batches age out of both window and graph.
    """
    svc = _make_service(tmp_path, window_s=2.0)
    # Drive 5 batches at successive 0.5 s anchor offsets, each carrying
    # 3 rows within a tight ±0.05 s envelope (sample_period_us=1e5).
    # After the 4th batch, the first batch's rows are 2 s old and start
    # to age out of the 2 s window.
    base_mjd = 60781.0
    for k in (0, 1, 2, 3, 4):
        offset_mjd = base_mjd + (k * 0.5) / 86400.0
        await svc._on_batch(
            _batch(mjd_start=offset_mjd, n=3, event_specnum_start=0),
            peer_repr="x",
        )
    # Window snapshot should match graph membership.
    win = svc._window.snapshot()
    assert len(win) == len(svc._graph)
    # No silent leak.
    assert svc._counters.get("rows_late_drop", 0) == 0


@asyncio_test
async def test_late_arrivals_are_dropped_from_graph_not_just_window(
    tmp_path: Path,
) -> None:
    """Headline regression: if a batch's mjds are already older than the
    current window cutoff, every row is inserted+aged-out by the window
    and must NOT be added to the components graph.

    Reproduces the M7.4 mode where a search-side restart with
    mjd_at_specnum_0=0 ships a fresh batch whose mjd values are already
    below the C2 window's anchor; pre-fix this leaked into the graph
    forever and produced ``graph_size ≫ window_size`` in /mon/c2/h23.
    """
    svc = _make_service(tmp_path, window_s=2.0)

    # Establish the window anchor at +10 s; 5 rows at offsets 0..0.4 s
    # (sample_period_us=1e5) — all inside the 2 s window.
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=5, event_specnum_start=0),
        peer_repr="x",
    )
    win_after_fresh = len(svc._window)
    graph_after_fresh = len(svc._graph)
    assert win_after_fresh == graph_after_fresh == 5

    # Now ship a "restart" batch anchored at +1 s (≈9 s in the past,
    # well below the cutoff at 10 - 2 = 8 s). 8 rows, all late.
    late_mjd = 60781.0 + 1.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=late_mjd, n=8, event_specnum_start=0),
        peer_repr="y",
    )

    # The late batch should not have grown the graph.
    assert len(svc._graph) == graph_after_fresh, (
        f"graph leaked late arrivals: expected {graph_after_fresh}, "
        f"got {len(svc._graph)}"
    )
    # The window should also have dropped the late rows.
    assert len(svc._window) == win_after_fresh
    # And counters should record the drop for observability.
    assert svc._counters["rows_late_drop"] == 8
    # rows_in tracks ALL received rows (in-window + late) for diagnostics.
    assert svc._counters["rows_in"] == 5 + 8


@asyncio_test
async def test_mixed_batch_keeps_in_window_rows(tmp_path: Path) -> None:
    """A batch with both in-window and late rows must keep the in-window
    rows in the graph and drop only the late ones.

    Uses offsets that sit well away from the window cutoff so float-
    precision drift in ``cutoff = latest_mjd - window_s/86400`` can't
    flip a borderline entry; the regression we care about is "graph
    grows by late_count", not the exact boundary handling.
    """
    svc = _make_service(tmp_path, window_s=2.0)

    # Anchor at t=20 s with a single row → latest_mjd = 60781.0 + 20 s.
    # Window cutoff after this batch: latest - 2 = 60781.0 + 18 s.
    await svc._on_batch(
        _batch_at_offsets(mjd_start=60781.0, offsets_s=(20.0,)),
        peer_repr="x",
    )
    assert len(svc._graph) == 1
    assert len(svc._window) == 1

    # Second batch mixes offsets clearly inside/outside the new window:
    #   late (offset < 21 after latest advances to 23):  5, 6, 7      → 3 late
    #   survivors (offset > 21 after latest advances):  22, 23        → 2 in-window
    # The original anchor at offset=20 (< 21) gets aged out too — that
    # counts as a *real* graph removal, not a late drop.
    await svc._on_batch(
        _batch_at_offsets(
            mjd_start=60781.0,
            offsets_s=(5.0, 6.0, 7.0, 22.0, 23.0),
        ),
        peer_repr="y",
    )
    assert svc._counters["rows_late_drop"] == 3
    # Graph now: only the 2 in-window survivors (anchor was aged out).
    assert len(svc._graph) == 2
    # Window has the same 2.
    assert len(svc._window) == 2


@asyncio_test
async def test_no_leak_under_many_restart_cycles(tmp_path: Path) -> None:
    """Stress: simulate 50 search-side restart cycles each shipping a
    pre-window batch. Pre-fix, this drove graph_size up to ~50× window
    size; post-fix, graph_size never exceeds the in-window count.
    """
    svc = _make_service(tmp_path, window_s=2.0)
    # Anchor far ahead at +100 s with 10 in-window rows (sample_period=1e5
    # → rows span 0.9 s, well inside the 2 s window).
    anchor_mjd = 60781.0 + 100.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=10, event_specnum_start=0),
        peer_repr="x",
    )
    g0 = len(svc._graph)
    assert g0 == 10

    # Each restart cycle ships 10 rows anchored at +1 s (well below the
    # window cutoff at 100 − 2 = 98 s). Pre-fix the graph would leak
    # 50 × 10 = 500 entries; post-fix it stays at 10.
    late_mjd = 60781.0 + 1.0 / 86400.0
    for cycle in range(50):
        await svc._on_batch(
            _batch(mjd_start=late_mjd, n=10, event_specnum_start=cycle),
            peer_repr=f"r{cycle}",
        )

    assert len(svc._graph) == g0, (
        f"graph leaked: expected {g0}, got {len(svc._graph)} after 50 "
        f"restart cycles (each shipping 10 late rows)"
    )
    assert svc._counters["rows_late_drop"] == 50 * 10

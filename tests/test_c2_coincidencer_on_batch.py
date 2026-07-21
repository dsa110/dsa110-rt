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
import logging
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


def _make_service(
    tmp_path: Path, window_s: float = 5.0, startup_grace_s: float = 0.0,
    *, rescue_late_priority: bool = True, priority_snr: float = 30.0,
) -> CoincidencerService:
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
        startup_grace_s=startup_grace_s,
        rescue_late_priority=rescue_late_priority,
        priority_snr=priority_snr,
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


# ---------------------------------------------------------------------------
# Late-priority rescue (2026-07-21 last silent-loss path)
#
# Incident: an le_w64_r3 111.75 sigma injection at 21:05 UT was matched by
# the C2 inject matcher but never clustered — its C1 batch landed >5 s
# (window_s) behind the anchor during a transient C2 event-loop stall, so
# it was inserted+aged-out as a "late arrival" and never graph-added. A
# bright (>= priority_snr) late arrival must instead SURVIVE (graph-added,
# evaluated) and be WARNING-logged; a dim one is dropped-and-counted as
# before; and graph/window membership must stay in lock-step throughout.
# ---------------------------------------------------------------------------


@asyncio_test
async def test_late_bright_row_rescued_survives_and_warned(
    tmp_path: Path, caplog,
) -> None:
    """A bright late arrival (snr >= priority_snr) is re-admitted with its
    mjd clamped to the cutoff, graph-added, counted as rescued (not
    dropped), and WARNING-logged. Graph and window stay in lock-step.
    """
    svc = _make_service(tmp_path, window_s=2.0)  # priority_snr=30, rescue on
    # Anchor the window at +10 s with 3 dim in-window rows (offsets
    # 0/0.1/0.2 s via sample_period_us=1e5 → all inside the 2 s window).
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=3, snr=9.0), peer_repr="x",
    )
    g0 = len(svc._graph)
    assert len(svc._window) == g0 == 3

    # Bright row anchored at +1 s (≈9 s old; cutoff = 10.2 − 2 = 8.2 s),
    # i.e. ~7 s beyond the window tail — exactly the incident geometry.
    late_mjd = 60781.0 + 1.0 / 86400.0
    with caplog.at_level(logging.WARNING):
        await svc._on_batch(
            _batch(mjd_start=late_mjd, n=1, snr=111.75), peer_repr="y",
        )

    # Rescued, NOT dropped.
    assert svc._counters["rows_late_rescued"] == 1
    assert svc._counters["rows_late_drop"] == 0
    # rows_in still tracks every received row.
    assert svc._counters["rows_in"] == 3 + 1
    # Graph grew by exactly the rescued row; window mirrors graph.
    assert len(svc._graph) == g0 + 1
    assert len(svc._window) == len(svc._graph)
    # Never silent: WARNING carries snr for the operator.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("RESCUED" in m and "111.75" in m for m in msgs), msgs


@asyncio_test
async def test_rescued_row_ages_out_of_both_graph_and_window(
    tmp_path: Path,
) -> None:
    """Membership contract: a rescued (clamped) row must age out of BOTH
    the window and the graph once the anchor advances past it — no leak.
    """
    svc = _make_service(tmp_path, window_s=2.0)
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=3, snr=9.0), peer_repr="x",
    )
    late_mjd = 60781.0 + 1.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=late_mjd, n=1, snr=111.75), peer_repr="y",
    )
    assert svc._counters["rows_late_rescued"] == 1
    assert len(svc._window) == len(svc._graph) == 4

    # Advance the anchor well past the clamped row's cutoff-mjd: a fresh
    # in-window batch at +14 s → cutoff = 14 − 2 = 12 s ages out the
    # clamped row (at 8.2 s) AND the three +10 s rows.
    future_mjd = 60781.0 + 14.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=future_mjd, n=1, snr=9.0), peer_repr="z",
    )
    # Only the newest row survives; graph and window agree (no leak of the
    # rescued row into the graph).
    assert len(svc._window) == len(svc._graph) == 1


@asyncio_test
async def test_late_dim_row_dropped_and_counted_as_before(
    tmp_path: Path,
) -> None:
    """A dim late arrival (snr < priority_snr) is dropped-and-counted
    exactly as pre-2026-07-21 — rescue only ever protects bright rows.
    """
    svc = _make_service(tmp_path, window_s=2.0)
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=3, snr=9.0), peer_repr="x",
    )
    g0 = len(svc._graph)
    late_mjd = 60781.0 + 1.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=late_mjd, n=4, snr=12.0), peer_repr="y",
    )
    assert svc._counters["rows_late_rescued"] == 0
    assert svc._counters["rows_late_drop"] == 4
    assert len(svc._graph) == g0            # no graph growth
    assert len(svc._window) == len(svc._graph)


@asyncio_test
async def test_late_bright_row_dropped_when_rescue_disabled(
    tmp_path: Path, caplog,
) -> None:
    """With rescue_late_priority=False the pre-2026-07-21 behaviour holds:
    the bright late row is dropped-and-counted — but STILL WARNING-logged
    (requirement (a): a bright loss is never silent, even without rescue).
    """
    svc = _make_service(tmp_path, window_s=2.0, rescue_late_priority=False)
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch(mjd_start=anchor_mjd, n=3, snr=9.0), peer_repr="x",
    )
    g0 = len(svc._graph)
    late_mjd = 60781.0 + 1.0 / 86400.0
    with caplog.at_level(logging.WARNING):
        await svc._on_batch(
            _batch(mjd_start=late_mjd, n=1, snr=111.75), peer_repr="y",
        )
    assert svc._counters["rows_late_rescued"] == 0
    assert svc._counters["rows_late_drop"] == 1
    assert len(svc._graph) == g0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("DROPPED a bright late" in m and "111.75" in m
               for m in msgs), msgs


# ---------------------------------------------------------------------------
# Noise-warmup filter (M7.4 Phase 8, 2026-05-28)
# ---------------------------------------------------------------------------


def _batch_with_flags(
    mjd_start: float,
    flag_pattern: tuple[int, ...],
    *,
    event_specnum_start: int = 0,
    cube_id: int = 0,
) -> wire.C1Batch:
    """Build a batch where each row carries the flag value at the
    matching index in ``flag_pattern``. Used to drive the C2 warmup
    filter with a known mix of warmup-flagged + clean rows.
    """
    n = len(flag_pattern)
    header = wire.build_header(
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=1,
        sample_period_us=100_000.0,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=n,
    )
    rows = tuple(
        wire.C1CandidateRow(
            snr=9.0 + 0.1 * i,
            l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
            dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
            event_specnum=event_specnum_start + i,
            width_samples=4, kernel_id="unit:d1:b4",
            flags=int(flag_pattern[i]),
        )
        for i in range(n)
    )
    return wire.C1Batch(header=header, candidates=rows)


@asyncio_test
async def test_warmup_flagged_rows_skip_graph_and_window(tmp_path: Path) -> None:
    """The C2 batch handler must drop rows whose flags include
    ``CandidateFlags.NOISE_WARMUP`` (bit 3 = 8) before they hit the
    window or graph.

    Regression for the 2026-05-28 false-burst incident: during the
    Layer-2 σ_k EMA burn-in (30 cubes at production cadence) every
    emitted Candidate carries this flag, but C2 ingested them all,
    producing 407 spurious rows_in across 50 s with peak 107 rows/s
    and 25 false log_only triggers/s -- all post-burnin readings
    showed 0 rows/s with the same sky/RFI environment.
    """
    svc = _make_service(tmp_path, window_s=5.0)
    # Mix: 5 warmup-flagged + 3 clean rows in one batch.
    NOISE_WARMUP = 1 << 3  # CandidateFlags.NOISE_WARMUP
    pattern = (NOISE_WARMUP,) * 5 + (0, 0, 0)
    await svc._on_batch(
        _batch_with_flags(mjd_start=60781.0, flag_pattern=pattern),
        peer_repr="warmup-test",
    )
    # Only the 3 clean rows should be in the window + graph.
    assert len(svc._window) == 3
    assert len(svc._graph) == 3
    # Counters: rows_in counts only the survivors (warmup-dropped never
    # enter the window/graph accounting); rows_warmup_drop tracks the
    # filtered total separately for the monitor surface.
    assert svc._counters["rows_in"] == 3
    assert svc._counters["rows_warmup_drop"] == 5


@asyncio_test
async def test_all_warmup_batch_is_full_skip(tmp_path: Path) -> None:
    """A batch where every row is warmup-flagged must produce a full
    skip: no graph add, no window mutation, no late_drop accounting
    fired."""
    svc = _make_service(tmp_path, window_s=5.0)
    NOISE_WARMUP = 1 << 3
    # First push a real (clean) batch so the window/graph aren't empty.
    await svc._on_batch(
        _batch_with_flags(mjd_start=60781.0, flag_pattern=(0,) * 4),
        peer_repr="warm-up-pre",
    )
    w0, g0 = len(svc._window), len(svc._graph)
    assert w0 == g0 == 4
    # Then push 7 all-warmup rows.
    await svc._on_batch(
        _batch_with_flags(
            mjd_start=60781.0 + 0.1 / 86400.0,
            flag_pattern=(NOISE_WARMUP,) * 7,
            event_specnum_start=100,
        ),
        peer_repr="all-warmup",
    )
    # Nothing should have changed in the graph / window.
    assert len(svc._window) == w0
    assert len(svc._graph) == g0
    assert svc._counters["rows_warmup_drop"] == 7
    assert svc._counters.get("rows_late_drop", 0) == 0


@asyncio_test
async def test_warmup_filter_preserves_late_drop_semantics(tmp_path: Path) -> None:
    """The warmup filter runs BEFORE the window age-out, so warmup-
    flagged rows are not counted as late drops. Mixed batches with
    both old + warmup-flagged rows should attribute the drop to the
    right counter."""
    svc = _make_service(tmp_path, window_s=2.0)
    NOISE_WARMUP = 1 << 3
    # Anchor the window at +10 s with one fresh row so window_s=2 has
    # a clear cutoff at 8 s.
    anchor_mjd = 60781.0 + 10.0 / 86400.0
    await svc._on_batch(
        _batch_with_flags(mjd_start=anchor_mjd, flag_pattern=(0,)),
        peer_repr="anchor",
    )
    assert len(svc._graph) == 1
    # Now send 3 warmup rows at the same anchor (would have been in-
    # window if not warmup-flagged) plus 2 clean late rows (mjd 9 s
    # below cutoff). Expectation:
    #   rows_warmup_drop += 3
    #   rows_late_drop += 2
    #   graph_size unchanged
    late_mjd = 60781.0 + 1.0 / 86400.0
    await svc._on_batch(
        _batch_with_flags(
            mjd_start=late_mjd,
            flag_pattern=(NOISE_WARMUP, NOISE_WARMUP, NOISE_WARMUP, 0, 0),
            event_specnum_start=50,
        ),
        peer_repr="mixed",
    )
    assert svc._counters["rows_warmup_drop"] == 3
    assert svc._counters["rows_late_drop"] == 2
    assert len(svc._graph) == 1


# ---------------------------------------------------------------------------
# Startup grace window (Phase 8c) — suppress the corr-RFI-warmup false burst
# ---------------------------------------------------------------------------


@asyncio_test
async def test_startup_grace_suppresses_triggers(tmp_path: Path) -> None:
    """Within the startup grace window, a coincidence that would fire a
    trigger is suppressed: the action does not run, triggers_log_only
    stays 0, and triggers_startup_grace counts the would-be trigger.

    This is the fix for the ~3-minute post-restart false-trigger burst
    caused by the corr-side RFI bandpass warmup (~150 s) leaking RFI into
    cubes -- those candidates reach C2 UNFLAGGED so the NOISE_WARMUP
    filter can't catch them.
    """
    svc = _make_service(tmp_path, window_s=5.0, startup_grace_s=600.0)
    # n_events_min=1 in the criteria → a single clean row fires log_only.
    await svc._on_batch(
        _batch(mjd_start=60781.0, n=1, event_specnum_start=0),
        peer_repr="grace",
    )
    assert svc._counters["triggers_startup_grace"] >= 1
    assert svc._counters["triggers_log_only"] == 0


@asyncio_test
async def test_after_grace_triggers_fire_normally(tmp_path: Path) -> None:
    """Once the grace window has elapsed, triggers fire as usual."""
    svc = _make_service(tmp_path, window_s=5.0, startup_grace_s=600.0)
    # Pre-set the first-batch anchor far in the past so we're past grace
    # (the handler only sets it when None, so this sticks).
    import time as _t
    svc._first_batch_mono = _t.monotonic() - 100_000.0
    await svc._on_batch(
        _batch(mjd_start=60781.0, n=1, event_specnum_start=0),
        peer_repr="post-grace",
    )
    assert svc._counters["triggers_log_only"] == 1
    assert svc._counters["triggers_startup_grace"] == 0


@asyncio_test
async def test_startup_grace_disabled_when_zero(tmp_path: Path) -> None:
    """startup_grace_s=0 disables the window: triggers fire immediately
    from the very first batch."""
    svc = _make_service(tmp_path, window_s=5.0, startup_grace_s=0.0)
    await svc._on_batch(
        _batch(mjd_start=60781.0, n=1, event_specnum_start=0),
        peer_repr="no-grace",
    )
    assert svc._counters["triggers_log_only"] == 1
    assert svc._counters["triggers_startup_grace"] == 0

"""Unit tests for the SEFD view module (``dsa_monitor.sefd_view``).

The SEFD scanner (``sefd_dashboard.service``, casa38 conda env)
writes a JSON state file and a tree of per-day PNG plots; the
``dsa_monitor`` app (``dsart_h23`` conda env) reads those outputs
read-only via :class:`SefdView` and renders the SEFD tab natively.

These tests exercise the read side end-to-end with a tmp_path
sandbox so we never touch the real h23 scanner state.  In
particular we verify:

* the mtime-keyed cache only re-parses when ``state.json``
  changes (cheap hot path),
* malformed / missing / non-JSON state files degrade gracefully
  without raising,
* the per-day x per-source ``summary`` grid only includes entries
  in the lookback window and with a parseable date,
* ``source_entries`` newest-firsts and drops the
  ``pending`` status (which clutters the per-source view),
* ``day_entries`` filters to one date,
* ``list_day_plots`` includes the ``sefd/`` subdir plots,
* ``resolve_plot_path`` enforces the path-traversal guard.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
DSA_MONITOR_DIR = os.path.normpath(os.path.join(
    HERE, "..", "tools", "dashboard", "dsa_monitor",
))
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import sefd_view  # noqa: E402
from sefd_view import (  # noqa: E402
    DEFAULT_SOURCES,
    SCANNER_STALE_S,
    SefdEntry,
    SefdView,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _iso_date_n_days_ago(n: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(days=n)
    ).date().isoformat()


_LAST_MTIME = [0.0]


def _write_state(state_file, payload):
    """Write the state file with a strictly-monotonic mtime.

    tmp_path on tmpfs can have sub-second mtime resolution, but two
    test-helper writes in quick succession may still produce the same
    mtime due to the simple ``getmtime()+1`` bump we used before.
    Track the last mtime we issued at module scope and always bump
    strictly past it.
    """
    state_file.write_text(json.dumps(payload))
    natural = os.path.getmtime(state_file)
    bumped = max(natural, _LAST_MTIME[0]) + 5.0
    _LAST_MTIME[0] = bumped
    os.utime(state_file, (bumped, bumped))


@pytest.fixture()
def sandbox(tmp_path):
    """A tmp_path-based scanner sandbox with one complete entry, one
    light-done entry, one error entry, and one pending entry.

    Yields ``(view, state_file_path, results_dir_path)``.
    """
    state_file = tmp_path / "state.json"
    results = tmp_path / "results"
    results.mkdir()

    today = _iso_date_n_days_ago(0)
    yest  = _iso_date_n_days_ago(1)
    week  = _iso_date_n_days_ago(7)
    old   = _iso_date_n_days_ago(45)

    state = {
        f"{today}_0318+164": {
            "status": "complete",
            "updated": datetime.now(timezone.utc).isoformat(),
            "date": today, "source": "0318+164",
            "path": "/dataz/dsa110/operations/calibration/x.ms",
            "metrics": {
                "median_amplitude": 0.04,
                "median_noise": 0.011,
                "median_coherence": 0.92,
                "median_hi_peak_db": 1.6,
                "median_autocorr_xx": 38.4,
                "median_autocorr_yy": 38.8,
            },
            "full_metrics": {
                "median_sefd": 6000.0,
                "mean_sefd": 6200.0,
                "n_baselines": 4163,
                "sefd_0-200m": 6400.0,
            },
        },
        f"{yest}_2253+161": {
            "status": "light_done",
            "updated": datetime.now(timezone.utc).isoformat(),
            "date": yest, "source": "2253+161",
            "metrics": {"median_amplitude": 0.08, "median_coherence": 0.78},
        },
        f"{week}_0521+166": {
            "status": "light_error",
            "updated": datetime.now(timezone.utc).isoformat(),
            "date": week, "source": "0521+166",
            "error": "casa diagonalise blew up: SVD failed in subspace",
        },
        f"{old}_0318+164": {
            "status": "complete",
            "updated": "2026-01-01T00:00:00",
            "date": old, "source": "0318+164",
            "metrics": {"median_amplitude": 0.05},
            "full_metrics": {"median_sefd": 5000.0},
        },
        f"{today}_2253+161": {"status": "pending"},
        # Garbage entries: malformed date, malformed key, non-dict.
        "not_a_real_key": {"status": "complete"},
        "9999-99-99_0318+164": {
            "status": "complete", "date": "9999-99-99",
            "source": "0318+164",
        },
        f"{today}_garbage": "this isn't even a dict",
    }
    _write_state(state_file, state)

    # Drop a few PNGs in the per-day results tree.  We only need a
    # couple to exercise the directory walk + the sefd/ subdir.
    (results / "0318+164" / today).mkdir(parents=True)
    for fname in [
        "amp_vs_baseline.png",
        "noise_vs_baseline.png",
        "coherence_vs_freq.png",
        "hi_spectrum.png",
        "not_a_png.txt",
    ]:
        (results / "0318+164" / today / fname).write_bytes(b"png")
    (results / "0318+164" / today / "sefd").mkdir()
    for fname in [
        "sefd_results.png",
        "sefd_vs_time.png",
    ]:
        (results / "0318+164" / today / "sefd" / fname).write_bytes(b"png")

    view = SefdView(
        state_file=str(state_file), results_dir=str(results),
    )
    return view, state_file, results, today, yest, week, old


def _sandbox_dates(sb):
    """Convenience accessor: returns (today, yest, week, old)."""
    _, _, _, today, yest, week, old = sb
    return today, yest, week, old


# ---------------------------------------------------------------------------
# _read_state cache
# ---------------------------------------------------------------------------


class TestStateFileCache:
    def test_first_load_reads_file(self, sandbox):
        view, state_file, *_ = sandbox
        assert view._cache_mtime is None
        s = view._read_state()
        assert isinstance(s, dict)
        assert view._cache_mtime == os.path.getmtime(state_file)
        assert view.state_error() is None

    def test_second_load_uses_cache(self, sandbox, monkeypatch):
        view, state_file, *_ = sandbox
        view._read_state()  # prime
        n_opens = [0]
        real_open = open

        def counting_open(path, *a, **kw):
            if str(path) == str(state_file):
                n_opens[0] += 1
            return real_open(path, *a, **kw)
        import builtins
        monkeypatch.setattr(builtins, "open", counting_open)

        for _ in range(5):
            view._read_state()
        assert n_opens[0] == 0, (
            "cached load must not re-open state.json when mtime unchanged"
        )

    def test_mtime_change_invalidates_cache(self, sandbox):
        view, state_file, *_ = sandbox
        view._read_state()
        prev_mtime = view._cache_mtime
        # Rewrite with new content.
        _write_state(state_file, {"foo": {"status": "complete"}})
        s = view._read_state()
        assert view._cache_mtime != prev_mtime
        assert "foo" in s

    def test_missing_state_file(self, tmp_path):
        view = SefdView(
            state_file=str(tmp_path / "nope.json"),
            results_dir=str(tmp_path / "results"),
        )
        s = view._read_state()
        assert s == {}
        assert view.state_error() is not None
        assert "missing" in view.state_error()

    def test_malformed_state_file_keeps_last_good(self, sandbox):
        view, state_file, *_ = sandbox
        view._read_state()  # prime
        good_n = len(view._cache_state)
        prev_mtime = view._cache_mtime
        # Overwrite with garbage.  Force a fresh-looking mtime that's
        # strictly newer than the cached value (tmp_path may have
        # low-res mtime).
        state_file.write_text("} this isn't json {")
        bump = max(time.time(), prev_mtime + 5.0)
        os.utime(state_file, (bump, bump))
        s = view._read_state()
        assert view.state_error() is not None, (
            f"_cache_mtime={view._cache_mtime!r} fs_mtime="
            f"{os.path.getmtime(state_file)!r} prev={prev_mtime!r}"
        )
        assert "parse" in view.state_error()
        # Last good cache is still served.
        assert len(s) == good_n

    def test_non_dict_root_is_rejected(self, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(["not", "a", "dict"]))
        view = SefdView(
            state_file=str(state_file), results_dir=str(tmp_path),
        )
        assert view._read_state() == {}
        assert "not a dict" in view.state_error()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_grid_only_includes_lookback_window(self, sandbox):
        view, _, _, today, yest, week, old = sandbox
        summ = view.summary(lookback_days=7)
        assert today in summ.dates
        assert yest in summ.dates
        assert week in summ.dates
        assert old not in summ.dates  # 45 d > 7 d

    def test_lookback_clamping(self, sandbox):
        view, *_ = sandbox
        summ = view.summary(lookback_days=0)
        assert summ.lookback_days == 1
        summ = view.summary(lookback_days=10_000)
        assert summ.lookback_days == 365

    def test_grid_row_has_one_cell_per_source(self, sandbox):
        view, _, _, today, *_ = sandbox
        summ = view.summary(lookback_days=7)
        row = summ.grid[today]
        assert set(row.keys()) == set(DEFAULT_SOURCES.keys())
        assert isinstance(row["0318+164"], SefdEntry)

    def test_missing_source_cell_is_none(self, sandbox):
        view, _, _, today, yest, *_ = sandbox
        summ = view.summary(lookback_days=7)
        # 0521+166 has no entry on either today or yest.
        assert summ.grid[today]["0521+166"] is None
        assert summ.grid[yest]["0521+166"] is None

    def test_currently_processing_picked_up_from_state(self, tmp_path):
        state_file = tmp_path / "state.json"
        today = _iso_date_n_days_ago(0)
        _write_state(state_file, {
            f"{today}_0318+164": {
                "status": "full_processing",
                "date": today, "source": "0318+164",
                "updated": datetime.now(timezone.utc).isoformat(),
            },
        })
        view = SefdView(
            state_file=str(state_file), results_dir=str(tmp_path),
        )
        summ = view.summary(lookback_days=7)
        assert summ.currently_processing == f"{today}_0318+164"

    def test_scanner_alive_when_fresh_mtime(self, sandbox):
        view, *_ = sandbox
        summ = view.summary(lookback_days=7)
        assert summ.scanner_alive is True
        assert summ.scanner_age_s is not None
        assert summ.scanner_age_s < SCANNER_STALE_S

    def test_scanner_stale_when_old_mtime(self, sandbox):
        view, state_file, *_ = sandbox
        # Mark state.json as 2 hours old.
        old_mtime = time.time() - 2 * SCANNER_STALE_S
        os.utime(state_file, (old_mtime, old_mtime))
        # Force cache invalidation by clearing.
        view._cache_mtime = None
        summ = view.summary(lookback_days=7)
        assert summ.scanner_alive is False
        assert summ.scanner_age_s > SCANNER_STALE_S

    def test_heartbeat_file_used_for_liveness(self, sandbox, tmp_path):
        """If the scanner's heartbeat file exists and is fresher than
        state.json, the dashboard should use it for the liveness
        pill -- the whole point of the heartbeat is that a quiet
        night (no state.json mutation) does not show as a dead
        scanner."""
        view, state_file, *_ = sandbox
        # Age state.json by 2x the staleness window.
        old_mtime = time.time() - 2 * SCANNER_STALE_S
        os.utime(state_file, (old_mtime, old_mtime))
        view._cache_mtime = None
        # But the scanner has been touching its heartbeat file 5s ago.
        hb = tmp_path / "scanner_heartbeat"
        hb.write_text("")
        recent = time.time() - 5
        os.utime(hb, (recent, recent))
        view.heartbeat_file = str(hb)
        summ = view.summary(lookback_days=7)
        assert summ.scanner_alive is True
        assert summ.scanner_age_s < 30, (
            f"expected ~5s scanner_age, got {summ.scanner_age_s}"
        )

    def test_heartbeat_missing_falls_back_to_state_mtime(self, sandbox):
        """If the heartbeat file does not exist, fall back to
        state.json mtime (back-compat with pre-heartbeat scanner)."""
        view, state_file, *_ = sandbox
        view.heartbeat_file = "/dev/null/no-such-heartbeat"
        old_mtime = time.time() - 2 * SCANNER_STALE_S
        os.utime(state_file, (old_mtime, old_mtime))
        view._cache_mtime = None
        summ = view.summary(lookback_days=7)
        # state.json is 2 hours old, no heartbeat available, so
        # liveness should report stale.
        assert summ.scanner_alive is False

    def test_garbage_entries_dropped_silently(self, sandbox):
        view, *_ = sandbox
        # Should not raise even though state.json contains malformed
        # keys / dates / values.
        summ = view.summary(lookback_days=7)
        for row in summ.grid.values():
            assert "9999-99-99" not in row.keys()


# ---------------------------------------------------------------------------
# source_entries / day_entries
# ---------------------------------------------------------------------------


class TestSourceEntries:
    def test_drops_pending(self, sandbox):
        view, _, _, today, *_ = sandbox
        entries = view.source_entries("2253+161", lookback_days=7)
        for e in entries:
            assert e.status != "pending"

    def test_newest_first(self, sandbox):
        view, _, _, today, yest, week, old = sandbox
        entries = view.source_entries("0318+164", lookback_days=365)
        dates = [e.date for e in entries]
        assert dates == sorted(dates, reverse=True)

    def test_unknown_source_returns_empty(self, sandbox):
        view, *_ = sandbox
        assert view.source_entries("bogus_source", lookback_days=7) == []

    def test_lookback_filters(self, sandbox):
        view, _, _, today, yest, week, old = sandbox
        recent = view.source_entries("0318+164", lookback_days=14)
        all_ = view.source_entries("0318+164", lookback_days=365)
        assert old in [e.date for e in all_]
        assert old not in [e.date for e in recent]


class TestDayEntries:
    def test_returns_one_entry_per_source_per_date(self, sandbox):
        view, _, _, today, *_ = sandbox
        entries = view.day_entries(today)
        assert "0318+164" in entries
        assert "2253+161" in entries
        assert "0521+166" not in entries

    def test_bad_date_returns_empty(self, sandbox):
        view, *_ = sandbox
        assert view.day_entries("not-a-date") == {}
        assert view.day_entries("9999-99-99") == {}
        assert view.day_entries("") == {}


# ---------------------------------------------------------------------------
# Plots + path-traversal guard
# ---------------------------------------------------------------------------


class TestPlots:
    def test_list_day_plots_includes_sefd_subdir(self, sandbox):
        view, _, _, today, *_ = sandbox
        plots = view.list_day_plots("0318+164", today)
        # Top-level PNGs.
        assert "amp_vs_baseline.png" in plots
        assert "coherence_vs_freq.png" in plots
        # sefd/ subdir keyed with the prefix.
        assert "sefd/sefd_results.png" in plots
        # Non-png filtered out.
        assert "not_a_png.txt" not in plots
        # URLs are relative paths under /sefds/results/.
        for url in plots.values():
            assert url.startswith("/sefds/results/")

    def test_unknown_source_or_date_returns_empty(self, sandbox):
        view, *_ = sandbox
        assert view.list_day_plots("bogus_source", "2026-01-01") == {}
        assert view.list_day_plots("0318+164", "not-a-date") == {}

    def test_resolve_plot_path_serves_real_file(self, sandbox):
        view, _, _, today, *_ = sandbox
        resolved = view.resolve_plot_path(
            f"0318+164/{today}/amp_vs_baseline.png",
        )
        assert resolved is not None
        assert os.path.isfile(resolved)
        # Resolved path is canonically inside the results dir.
        results_real = os.path.realpath(view.results_dir)
        assert resolved.startswith(results_real + os.sep)

    def test_resolve_plot_path_path_traversal_denied(self, sandbox):
        view, *_ = sandbox
        # Even with multiple ".." attempts, the guard must hold.
        assert view.resolve_plot_path("../state.json") is None
        assert view.resolve_plot_path("../../etc/passwd") is None
        assert view.resolve_plot_path("//etc/passwd") is None

    def test_resolve_plot_path_rejects_non_png(self, sandbox):
        view, _, _, today, *_ = sandbox
        assert view.resolve_plot_path(
            f"0318+164/{today}/not_a_png.txt",
        ) is None

    def test_resolve_plot_path_missing_file(self, sandbox):
        view, *_ = sandbox
        assert view.resolve_plot_path("nope/nope/nope.png") is None

    def test_resolve_strips_optional_results_prefix(self, sandbox):
        # The template never emits a "results/" prefix, but a hand-typed
        # URL might.  Be permissive: accept both forms.
        view, _, _, today, *_ = sandbox
        a = view.resolve_plot_path(
            f"0318+164/{today}/amp_vs_baseline.png",
        )
        b = view.resolve_plot_path(
            f"results/0318+164/{today}/amp_vs_baseline.png",
        )
        assert a == b


# ---------------------------------------------------------------------------
# SefdEntry helpers
# ---------------------------------------------------------------------------


class TestSefdEntry:
    def test_updated_age_zero_for_now(self):
        e = SefdEntry(
            key="x", date="2026-01-01", source="0318+164",
            status="complete",
            updated=datetime.now(timezone.utc).isoformat(),
        )
        age = e.updated_age_s()
        assert age is not None and age >= 0
        assert age < 5

    def test_updated_age_none_for_missing(self):
        e = SefdEntry(key="x", date="2026-01-01",
                      source="0318+164", status="pending")
        assert e.updated_age_s() is None

    def test_is_error(self):
        for status in ("light_error", "full_error", "error"):
            e = SefdEntry(key="x", date="2026-01-01",
                          source="0318+164", status=status)
            assert e.is_error
        for status in ("complete", "pending", "light_done"):
            e = SefdEntry(key="x", date="2026-01-01",
                          source="0318+164", status=status)
            assert not e.is_error

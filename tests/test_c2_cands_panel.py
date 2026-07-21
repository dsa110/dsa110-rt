"""Tests for tools/dashboard/dsa_monitor/cands_panel_funcs.py.

The module is part of the deployed Flask app on h23 (port 5778); we
test the filesystem-only helpers — no Flask import required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_annot_db(tmp_path, monkeypatch):
    """Never let list_events()/_fetch_annotated_ids() touch the live
    ~/.dsa_monitor/annotations.db during tests — point it at a throwaway
    path so the DB read is empty + side-effect-free."""
    monkeypatch.setenv("DSA_MONITOR_ANNOT_DB", str(tmp_path / "annot.db"))


@pytest.fixture(scope="module")
def cands_panel():
    """Import the dashboard sibling module without polluting sys.modules."""
    here = Path(__file__).resolve().parent
    panel_dir = here.parent / "tools" / "dashboard" / "dsa_monitor"
    sys.path.insert(0, str(panel_dir))
    try:
        import cands_panel_funcs  # type: ignore
        return cands_panel_funcs
    finally:
        # Leave the entry for the rest of the test session — harmless.
        pass


def _layout_event(root: Path, name: str, *, with_meta: bool = True,
                  n_plots: int = 0, n_cubes: int = 0) -> Path:
    ev = root / name
    (ev / "Level2" / "plots").mkdir(parents=True, exist_ok=True)
    (ev / "Level3").mkdir(parents=True, exist_ok=True)
    (ev / "cubes").mkdir(parents=True, exist_ok=True)
    if with_meta:
        meta = {
            "event_name": name,
            "schema_version": 1,
            "trigger": {"class": "bright_frb", "action": "dump_all_gpus"},
            "c2": {
                "n_events": 7,
                "snr_max": 14.3,
                "dm_median": 350.0,
                "l_median": 1.5e-3,
                "m_median": -2.5e-3,
                "t_peak_mjd": 60781.123456789,
            },
        }
        (ev / "Level3" / f"{name}.json").write_text(json.dumps(meta))
        # Also create the C2/C1 CSVs so the detail page reports them present.
        (ev / "Level2" / f"C2_{name}.csv").write_text("a\n1\n")
        (ev / "Level2" / f"C1_window_{name}.csv").write_text("a\n1\n")
    for i in range(n_plots):
        (ev / "Level2" / "plots" / f"panel_{i}.png").write_text("png")
    for i in range(n_cubes):
        (ev / "cubes" / f"cube_s1_g0_{i}.npz").write_text("npz")
    return ev


def test_archive_browser_missing_root_is_empty(tmp_path, cands_panel) -> None:
    ab = cands_panel.ArchiveBrowser(tmp_path / "nope")
    assert ab.is_available is False
    assert ab.list_events() == []


def test_archive_browser_lists_events_newest_first(
    tmp_path, cands_panel,
) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    # Two valid + one decoy non-event dir.
    a = _layout_event(root, "260521aaaa", n_plots=2, n_cubes=4)
    b = _layout_event(root, "260521bbbb", n_plots=0, n_cubes=0)
    (root / "T2_legacy").mkdir()
    # Force a clear mtime order.
    import os
    now = 1_700_000_000
    os.utime(a, (now - 100, now - 100))
    os.utime(b, (now, now))
    ab = cands_panel.ArchiveBrowser(root)
    events = ab.list_events()
    names = [e.name for e in events]
    assert names == ["260521bbbb", "260521aaaa"]


def test_archive_browser_pulls_l3_fields(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _layout_event(root, "260521abcd", n_plots=3, n_cubes=2)
    ab = cands_panel.ArchiveBrowser(root)
    events = ab.list_events()
    assert len(events) == 1
    e = events[0]
    assert e.name == "260521abcd"
    assert e.trigger_class == "bright_frb"
    assert e.snr_max == pytest.approx(14.3)
    assert e.dm_median == pytest.approx(350.0)
    assert e.n_events == 7
    assert e.n_cubes == 2
    assert e.n_plots == 3


def test_archive_browser_event_detail_resolves(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _layout_event(root, "260521abcd", n_plots=4, n_cubes=8)
    ab = cands_panel.ArchiveBrowser(root)
    det = ab.event_detail("260521abcd")
    assert det is not None
    assert det.name == "260521abcd"
    assert det.has_c1_csv is True
    assert det.has_c2_csv is True
    assert len(det.plots) == 4
    assert len(det.cubes) == 8


def test_archive_browser_event_detail_rejects_bad_name(
    tmp_path, cands_panel,
) -> None:
    ab = cands_panel.ArchiveBrowser(tmp_path)
    assert ab.event_detail("../escape") is None
    assert ab.event_detail("contains/slash") is None
    assert ab.event_detail("260521abcd!") is None  # punctuation rejected
    assert ab.event_detail("nonexistent_event") is None


def test_archive_browser_plot_path_refuses_traversal(
    tmp_path, cands_panel,
) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _layout_event(root, "260521abcd", n_plots=1)
    ab = cands_panel.ArchiveBrowser(root)
    # Legitimate
    p = ab.plot_path("260521abcd", "panel_0.png")
    assert p is not None
    assert p.is_file()
    # Reject traversal
    assert ab.plot_path("260521abcd", "../../etc/passwd") is None
    assert ab.plot_path("260521abcd", "panel_0.txt") is None
    assert ab.plot_path("../etc", "passwd") is None


def test_archive_browser_max_events_caps_list(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    for i in range(10):
        _layout_event(root, f"26052{i:01d}aaaa")
    ab = cands_panel.ArchiveBrowser(root, max_events=3)
    events = ab.list_events()
    assert len(events) == 3


# ---- C3 decision + voltages + time-of-day (2026-07-14 candidates tab) -----


def _write_c3(ev: Path, *, action: str, rules=(), notes: str = "",
              is_injection: bool = False, flag_only: bool = True) -> None:
    (ev / "C3_decision.json").write_text(json.dumps({
        "event_name": ev.name,
        "action": action,
        "rules_fired": list(rules),
        "notes": notes,
        "is_injection": is_injection,
        "flag_only": flag_only,
    }))


def _write_voltages(ev: Path, n: int) -> None:
    vdir = ev / "Level2" / "voltages"
    vdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (vdir / f"{ev.name}_sb{i:02d}_data.out").write_text("v")


def test_summary_carries_c3_keep_and_voltages(tmp_path, cands_panel) -> None:
    ev = _layout_event(tmp_path, "260714aaaa")
    _write_c3(ev, action="KEEP", notes="all clean")
    _write_voltages(ev, 16)
    (s,) = cands_panel.ArchiveBrowser(tmp_path).list_events()
    assert s.c3_action == "KEEP"
    assert s.c3_status == "pass"
    assert s.c3_rules == ()
    assert s.n_voltages == 16
    # t_peak_mjd 60781.123456789 -> fractional day .123456789 ≈ 02:57:47
    assert s.utc_hms == "02:57:46" or s.utc_hms == "02:57:47"


def test_summary_carries_c3_reject_rules(tmp_path, cands_panel) -> None:
    ev = _layout_event(tmp_path, "260714bbbb")
    _write_c3(ev, action="REJECT",
              rules=["R1_image_offset", "R10_cube_unconfirmed"],
              notes="tier-1/R10 high-confidence false positive")
    (s,) = cands_panel.ArchiveBrowser(tmp_path).list_events()
    assert s.c3_status == "fail"
    assert s.c3_rules == ("R1_image_offset", "R10_cube_unconfirmed")
    assert s.n_voltages == 0
    assert s.c3_flag_only is True


def test_summary_pending_without_c3_file(tmp_path, cands_panel) -> None:
    _layout_event(tmp_path, "260714cccc")
    (s,) = cands_panel.ArchiveBrowser(tmp_path).list_events()
    assert s.c3_action is None
    assert s.c3_status == "pending"


def test_summary_time_falls_back_to_mtime(tmp_path, cands_panel) -> None:
    # No Level3 meta -> no t_peak_mjd -> mtime fallback marked '~'.
    ev = _layout_event(tmp_path, "260714dddd", with_meta=False)
    (ev / "Level3" / "260714dddd.json").write_text("{}")
    (s,) = cands_panel.ArchiveBrowser(tmp_path).list_events()
    assert s.utc_hms is not None and s.utc_hms.endswith("~")


def test_event_detail_carries_c3_and_voltages(tmp_path, cands_panel) -> None:
    ev = _layout_event(tmp_path, "260714eeee")
    _write_c3(ev, action="KEEP")
    _write_voltages(ev, 3)
    d = cands_panel.ArchiveBrowser(tmp_path).event_detail("260714eeee")
    assert d.c3_decision["action"] == "KEEP"
    assert d.n_voltages == 3


# ---- annotation-aware listing: annotated events never age out -------------
#
# list_events_detailed() unions the newest max_events dirs with EVERY
# annotated event, so a human-classified/named event stays visible +
# searchable forever. annotated_ids is injected here so the union logic is
# tested without touching the annotations DB.


def _mk_events_by_age(root: Path, n: int) -> list:
    """Create n event dirs, oldest → newest by mtime. Returns names in
    newest-first order (matching list_events output order)."""
    import os
    names = [f"2605{i:02d}zzzz"[:10] for i in range(n)]
    now = 1_700_000_000
    for i, name in enumerate(names):
        ev = _layout_event(root, name)
        # i=0 oldest, i=n-1 newest.
        os.utime(ev, (now + i, now + i))
    return list(reversed(names))  # newest first


def test_annotated_event_older_than_cap_is_still_returned(
    tmp_path, cands_panel,
) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    newest_first = _mk_events_by_age(root, 10)
    old_annotated = newest_first[-1]   # the single oldest dir
    ab = cands_panel.ArchiveBrowser(root, max_events=3)
    # Without the annotation the cap drops it.
    plain = ab.list_events_detailed(annotated_ids=set())
    assert old_annotated not in [e.name for e in plain.events]
    assert plain.truncated is True
    assert plain.n_total == 10
    assert plain.n_newest == 3
    assert plain.n_annotated == 0
    # With the annotation it is merged back — at its true (oldest) position,
    # NOT pinned to the top.
    detailed = ab.list_events_detailed(annotated_ids={old_annotated})
    names = [e.name for e in detailed.events]
    assert old_annotated in names
    assert names[:3] == newest_first[:3]     # newest three unchanged, on top
    assert names[-1] == old_annotated        # annotated one sits at the bottom
    assert detailed.n_total == 10
    assert detailed.n_newest == 3
    assert detailed.n_annotated == 1
    assert detailed.truncated is True


def test_annotated_event_inside_window_not_double_counted(
    tmp_path, cands_panel,
) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    newest_first = _mk_events_by_age(root, 10)
    inside = newest_first[0]  # newest dir, already inside the cap
    ab = cands_panel.ArchiveBrowser(root, max_events=3)
    detailed = ab.list_events_detailed(annotated_ids={inside})
    names = [e.name for e in detailed.events]
    assert names.count(inside) == 1          # no duplicate row
    assert len(detailed.events) == 3
    assert detailed.n_annotated == 0         # it was already in-window


def test_annotation_pointing_at_missing_dir_is_skipped(
    tmp_path, cands_panel,
) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 5)
    ab = cands_panel.ArchiveBrowser(root, max_events=3)
    # An annotation whose candidate dir was deleted must not crash the page.
    detailed = ab.list_events_detailed(annotated_ids={"260101dead"})
    assert "260101dead" not in [e.name for e in detailed.events]
    assert detailed.n_annotated == 0
    assert len(detailed.events) == 3


def test_listing_not_truncated_when_under_cap(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 2)
    ab = cands_panel.ArchiveBrowser(root, max_events=10)
    detailed = ab.list_events_detailed(annotated_ids=set())
    assert detailed.truncated is False
    assert detailed.n_total == 2
    assert detailed.n_newest == 2
    assert detailed.n_annotated == 0


def test_list_events_wrapper_matches_detailed(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 4)
    ab = cands_panel.ArchiveBrowser(root, max_events=2)
    # list_events() reads annotated ids from the DB (empty here) — the row
    # list must equal the detailed listing's events.
    assert [e.name for e in ab.list_events()] == [
        e.name for e in ab.list_events_detailed().events
    ]


# ---- EventIndexCache: server-side cached index -----------------------------
#
# The /bursts page now serves the whole event index from an in-process,
# incrementally-refreshed cache instead of re-enumerating the NFS archive
# per request. These tests exercise the cache directly against a fake
# archive under tmp_path: cold build, incremental pickup, TTL gating and
# the NFS-failure (serve-stale) path.


def test_cache_cold_build_full_index(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    newest_first = _mk_events_by_age(root, 5)
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=60, active_window_s=0,
    )
    snap = cache.snapshot()
    # Whole index, newest-first — NOT a capped window.
    assert snap.n_total == 5
    assert [e.name for e in snap.events] == newest_first
    assert snap.stale is False
    assert snap.last_success_unix is not None


def test_cache_incremental_picks_up_new_event(tmp_path, cands_panel) -> None:
    import os
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 3)
    # active_window_s=0 so ONLY genuinely new/changed dirs get re-read.
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=0, active_window_s=0,
    )
    first = cache.snapshot()
    assert first.n_total == 3
    # A brand-new event dir lands, newer than everything cached.
    now = 1_700_001_000
    ev = _layout_event(root, "260601newv")
    os.utime(ev, (now, now))
    # ttl_s=0 → next access refreshes; the new dir is picked up + sorted in.
    second = cache.snapshot()
    assert second.n_total == 4
    assert second.events[0].name == "260601newv"


def test_cache_ttl_defers_rescan(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 2)
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=9_999, active_window_s=0,
    )
    assert cache.snapshot().n_total == 2
    # A new dir appears, but within the (huge) TTL the cache must NOT
    # re-scan, so it stays at the cached count.
    _layout_event(root, "260601aaab")
    assert cache.snapshot().n_total == 2
    # An explicit refresh bypasses the TTL and picks it up.
    assert cache.snapshot(force_refresh=True).n_total == 3


def test_cache_nfs_failure_serves_stale(tmp_path, cands_panel) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 3)
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=0, active_window_s=0,
    )
    good = cache.snapshot()
    assert good.n_total == 3 and good.stale is False
    # Simulate the NFS mount going away: scan_dirs raises OSError. The
    # cache must keep the last good index and flag it stale — never crash.
    import unittest.mock as mock
    with mock.patch.object(cache._browser, "scan_dirs",
                           side_effect=OSError("stale NFS handle")):
        stale = cache.snapshot()
    assert stale.stale is True
    assert stale.n_total == 3                    # last good index retained
    assert stale.error and "scan failed" in stale.error
    assert stale.last_success_unix == good.last_success_unix


def test_cache_cold_failure_is_empty_not_500(tmp_path, cands_panel) -> None:
    # Root that does not exist at all → is_available False → empty + stale,
    # never an exception (the route degrades gracefully).
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(tmp_path / "nope"),
    )
    snap = cache.snapshot()
    assert snap.events == []
    assert snap.n_total == 0
    assert snap.stale is True


def test_cache_active_window_rereads_young_events(tmp_path, cands_panel) -> None:
    """A young event whose dir mtime does NOT change still gets re-read
    (C3 decision / voltage fragments land late under nested dirs that don't
    bump the event-dir mtime)."""
    import os
    root = tmp_path / "candidates"
    root.mkdir()
    now = 1_700_002_000
    ev = _layout_event(root, "260601cccc")
    os.utime(ev, (now - 10, now - 10))           # "young"
    # Wide active window so this event is always re-summarised.
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=0,
        active_window_s=1e12,
    )
    first = cache.snapshot()
    assert first.events[0].c3_status == "pending"
    # C3 lands its KEEP decision without touching the event-dir mtime.
    _write_c3(ev, action="KEEP")
    os.utime(ev, (now - 10, now - 10))           # pin mtime unchanged
    second = cache.snapshot()
    assert second.events[0].c3_status == "pass"


def test_cache_drops_deleted_event(tmp_path, cands_panel) -> None:
    import shutil
    root = tmp_path / "candidates"
    root.mkdir()
    _mk_events_by_age(root, 3)
    cache = cands_panel.EventIndexCache(
        cands_panel.ArchiveBrowser(root), ttl_s=0, active_window_s=0,
    )
    names = [e.name for e in cache.snapshot().events]
    assert len(names) == 3
    shutil.rmtree(root / names[0])
    after = cache.snapshot()
    assert names[0] not in [e.name for e in after.events]
    assert after.n_total == 2

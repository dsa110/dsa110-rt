"""Tests for tools/dashboard/dsa_monitor/cands_panel_funcs.py.

The module is part of the deployed Flask app on h23 (port 5778); we
test the filesystem-only helpers — no Flask import required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


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

"""Tests for click-to-sort columns on the /bursts events table.

Two layers:

* ``sort_events`` (cands_panel_funcs.py) — the pure-Python sort helper:
  numeric asc/desc, lexical case-insensitive asc/desc, None/missing always
  last regardless of direction, unknown/absent key is a no-op.
* the Flask ``/bursts`` route — ``?sort=`` / ``&dir=`` reorders the WHOLE
  cached index (not just the visible page) and composes with ``?page=``.

Follows the mocking patterns in test_annotations.py / test_positions.py:
mock ``cands_index.snapshot`` to hand the route a fixed, hand-built index
so no filesystem/NFS access is needed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSA_MONITOR_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "tools", "dashboard", "dsa_monitor")
)
DSART_SRC = os.path.join(REPO_ROOT, "src")
for _p in (DSART_SRC, DSA_MONITOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cands_panel_funcs as cpf  # noqa: E402


# ---------------------------------------------------------------------------
# sort_events() unit tests
# ---------------------------------------------------------------------------


def _ev(name, *, snr=None, dm=None, trigger_class=None, mjd=None) -> cpf.EventSummary:
    return cpf.EventSummary(
        name=name, mtime_unix=time.time(), mjd_peak=mjd,
        trigger_class=trigger_class, n_events=None, snr_max=snr,
        dm_median=dm, l_median=None, m_median=None,
        n_cubes=0, n_plots=0,
    )


def test_sort_events_numeric_desc_and_asc():
    events = [_ev("a", snr=5.0), _ev("b", snr=20.0), _ev("c", snr=1.0)]
    desc = cpf.sort_events(events, "snr", "desc")
    assert [e.name for e in desc] == ["b", "a", "c"]
    asc = cpf.sort_events(events, "snr", "asc")
    assert [e.name for e in asc] == ["c", "a", "b"]


def test_sort_events_none_last_both_directions():
    events = [_ev("has_snr", snr=10.0), _ev("no_snr", snr=None),
              _ev("also_has_snr", snr=3.0)]
    desc = cpf.sort_events(events, "snr", "desc")
    assert desc[-1].name == "no_snr"
    assert [e.name for e in desc[:-1]] == ["has_snr", "also_has_snr"]
    asc = cpf.sort_events(events, "snr", "asc")
    assert asc[-1].name == "no_snr"
    assert [e.name for e in asc[:-1]] == ["also_has_snr", "has_snr"]


def test_sort_events_lexical_case_insensitive():
    events = [_ev("e1", trigger_class="Bright_FRB"),
              _ev("e2", trigger_class="alpha"),
              _ev("e3", trigger_class="ZETA"),
              _ev("e4", trigger_class=None)]
    asc = cpf.sort_events(events, "class", "asc")
    assert [e.name for e in asc] == ["e2", "e1", "e3", "e4"]
    desc = cpf.sort_events(events, "class", "desc")
    assert [e.name for e in desc] == ["e3", "e1", "e2", "e4"]


def test_sort_events_event_name_lexical():
    events = [_ev("260714cccc"), _ev("260714aaaa"), _ev("260714bbbb")]
    asc = cpf.sort_events(events, "event", "asc")
    assert [e.name for e in asc] == ["260714aaaa", "260714bbbb", "260714cccc"]
    desc = cpf.sort_events(events, "event", "desc")
    assert [e.name for e in desc] == ["260714cccc", "260714bbbb", "260714aaaa"]


def test_sort_events_unknown_or_absent_key_is_noop():
    events = [_ev("b", snr=1.0), _ev("a", snr=2.0)]
    assert cpf.sort_events(events, None) == events
    assert cpf.sort_events(events, "") == events
    assert cpf.sort_events(events, "not_a_real_column") == events
    # Returns a new list object (copy), not the same list, even when a no-op.
    assert cpf.sort_events(events, None) is not events


def test_sort_events_time_uses_mjd_peak():
    events = [_ev("newer", mjd=60800.0), _ev("older", mjd=60700.0),
              _ev("no_mjd", mjd=None)]
    desc = cpf.sort_events(events, "time", "desc")
    assert [e.name for e in desc] == ["newer", "older", "no_mjd"]


# ---------------------------------------------------------------------------
# Flask /bursts route: ?sort=/&dir= composes with pagination
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_annot_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DSA_MONITOR_ANNOT_DB", str(tmp_path / "annot.db"))


@pytest.fixture(scope="module")
def app_module():
    try:
        with mock.patch("rfi_store.RFIPoller.start", return_value=None):
            import app  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"app import needs live resources: {exc!r}")
    app.app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app_module):
    return app_module.app.test_client()


def _summary(name: str, *, snr: float) -> cpf.EventSummary:
    return cpf.EventSummary(
        name=name, mtime_unix=time.time(), mjd_peak=60000.0,
        trigger_class="bright_frb", n_events=1, snr_max=snr,
        dm_median=100.0, l_median=0.001, m_median=-0.001,
        n_cubes=0, n_plots=0, c3_action="KEEP",
    )


def _snapshot(events):
    return cpf.CacheSnapshot(
        events=events, n_total=len(events), last_success_unix=time.time(),
        stale=False, error=None,
    )


def test_bursts_sort_reorders_whole_index(client, app_module):
    # Newest-first (default cache order) does NOT match SNR order, so a
    # naive "sort the visible page only" implementation would fail this.
    events = [_summary(f"26071{i}aaaa", snr=snr)
              for i, snr in enumerate([3.0, 50.0, 1.0, 20.0])]
    with mock.patch.object(app_module.cands_index, "snapshot",
                           return_value=_snapshot(events)):
        html = client.get("/bursts?sort=snr&dir=desc").get_data(as_text=True)
    names_in_order = [n for n in
                       ["260710aaaa", "260711aaaa", "260712aaaa", "260713aaaa"]
                       if n in html]
    # Extract row order by finding each event name's position in the HTML.
    positions = {e.name: html.index(f"/bursts/{e.name}") for e in events}
    ranked = sorted(positions, key=positions.get)
    assert ranked == ["260711aaaa", "260713aaaa", "260710aaaa", "260712aaaa"]


def test_bursts_sort_composes_with_pagination(client, app_module, monkeypatch):
    # Force a tiny page size so a 5-event index needs multiple pages, and
    # verify ?sort= reorders the FULL index before paging, not just the
    # events on the requested page.
    monkeypatch.setattr(app_module, "DEFAULT_PAGE_SIZE", 2)
    events = [_summary(f"26072{i}aaaa", snr=snr)
              for i, snr in enumerate([10.0, 40.0, 20.0, 5.0, 30.0])]
    with mock.patch.object(app_module.cands_index, "snapshot",
                           return_value=_snapshot(events)):
        page1 = client.get("/bursts?sort=snr&dir=desc&page=1").get_data(as_text=True)
        page2 = client.get("/bursts?sort=snr&dir=desc&page=2").get_data(as_text=True)
        page3 = client.get("/bursts?sort=snr&dir=desc&page=3").get_data(as_text=True)
    by_snr_desc = [e.name for e in
                   sorted(events, key=lambda e: e.snr_max, reverse=True)]
    assert by_snr_desc[0] in page1 and by_snr_desc[1] in page1
    assert by_snr_desc[2] in page2 and by_snr_desc[3] in page2
    assert by_snr_desc[4] in page3
    # Page 2/3 must NOT contain the top-ranked (page-1) events.
    assert by_snr_desc[0] not in page2
    assert by_snr_desc[0] not in page3


def test_bursts_default_no_sort_params_unchanged(client, app_module):
    """No ?sort=/&dir= present -> byte-identical to pre-sort behaviour:
    newest-first (cache order), no sort/dir echoed as active state."""
    events = [_summary(f"26073{i}aaaa", snr=snr)
              for i, snr in enumerate([10.0, 40.0, 20.0])]
    with mock.patch.object(app_module.cands_index, "snapshot",
                           return_value=_snapshot(events)):
        html_plain = client.get("/bursts").get_data(as_text=True)
        html_explicit_default = client.get(
            "/bursts?sort=bogus_key").get_data(as_text=True)
    positions = {e.name: html_plain.index(f"/bursts/{e.name}")
                 for e in events}
    ranked = sorted(positions, key=positions.get)
    # Cache already hands back newest-first (as constructed above);
    # unsorted request must preserve that order verbatim.
    assert ranked == [e.name for e in events]
    assert html_plain == html_explicit_default
    assert " sort-active" not in html_plain

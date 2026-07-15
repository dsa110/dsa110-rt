"""Refined-localization tests: store, parsers, adaptive formatting, API, UI.

Covers the operator-entered refined-position feature end to end:

* ``annotations.event_positions`` store — set / clear / get / bulk /
  history semantics (last-write-wins, append-only audit, pipeline
  snapshot, validation, lazy schema upgrade of a pre-feature DB).
* ``event_astrometry.parse_ra_str`` / ``parse_dec_str`` operator input.
* Adaptive-precision sexagesimal formatting (psrcat-style
  parentheticals across sigma decades, carry rounding, pole guard).
* Flask API ``/api/position/<event>`` (GET / POST / clear) and the two
  page renders: the ``/bursts`` table (column order + superscript-R
  marker) and the event page (entry card).

Flask tests skip cleanly if ``app`` can't import outside the live
observatory environment (same guard as test_annotations.py).
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

import annotations as ann  # noqa: E402
import cands_panel_funcs as cpf  # noqa: E402
import event_astrometry as ea  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    d = str(tmp_path / "annot.db")
    ann.add_user("vishnu", d)
    return d


def _set(db, event="260715twmx", ra=234.09627, dec=17.55895,
         ra_err=108.0, dec_err=108.0, method="M8 oversampled UV",
         user="vishnu", snap=None):
    return ann.set_position(event, ra, dec, ra_err, dec_err, method, user,
                            pipe_snapshot=snap, db_path=db)


# ---------------------------------------------------------------------------
# Store: set / get / clear / bulk / history
# ---------------------------------------------------------------------------


def test_set_get_roundtrip(db):
    row = _set(db, snap={"ra_deg": 234.38, "dec_deg": 17.54,
                         "source": "level3"})
    assert row["active"] is True
    assert row["ra_deg"] == pytest.approx(234.09627)
    assert row["pipe_ra_deg"] == pytest.approx(234.38)
    assert row["pipe_source"] == "level3"
    got = ann.get_position("260715twmx", db)
    assert got["id"] == row["id"]
    assert got["method"] == "M8 oversampled UV"
    assert got["username"] == "vishnu"
    assert got["created_utc"].endswith("Z") or "T" in got["created_utc"]


def test_set_requires_known_user(db):
    with pytest.raises(ann.UnknownUserError):
        _set(db, user="ghost")


def test_set_last_write_wins(db):
    first = _set(db, ra=234.0)
    second = _set(db, ra=234.1, method="M9 outrigger + NVSS tie")
    cur = ann.get_position("260715twmx", db)
    assert cur["id"] == second["id"]
    assert cur["ra_deg"] == pytest.approx(234.1)
    hist = ann.get_position_history("260715twmx", db)
    assert [h["id"] for h in hist] == [first["id"], second["id"]]
    assert hist[0]["active"] is False       # deactivated, not deleted
    assert hist[1]["active"] is True


def test_clear_audited(db):
    _set(db)
    assert ann.clear_position("260715twmx", "vishnu", db) is True
    assert ann.get_position("260715twmx", db) is None
    hist = ann.get_position_history("260715twmx", db)
    assert len(hist) == 2                   # set row + clear audit row
    assert hist[-1]["ra_deg"] is None
    assert hist[-1]["username"] == "vishnu"
    assert hist[-1]["active"] is False
    # Clearing again: no active row, but the audit row is still written.
    assert ann.clear_position("260715twmx", "vishnu", db) is False
    assert len(ann.get_position_history("260715twmx", db)) == 3


def test_clear_requires_known_user(db):
    _set(db)
    with pytest.raises(ann.UnknownUserError):
        ann.clear_position("260715twmx", "ghost", db)
    assert ann.get_position("260715twmx", db) is not None


def test_get_positions_bulk(db):
    ann.add_user("vikram", db)
    _set(db, event="260715twmx")
    _set(db, event="260709vmue", ra=293.94927, dec=16.27777,
         ra_err=72.0, dec_err=72.0, method="pulsar catalog match",
         user="vikram")
    ann.clear_position("260709vmue", "vikram", db)   # only twmx survives
    bulk = ann.get_positions_bulk(db)
    assert set(bulk) == {"260715twmx"}
    assert bulk["260715twmx"]["ra_deg"] == pytest.approx(234.09627)


def test_get_position_none_for_unknown_event(db):
    assert ann.get_position("260101aaaa", db) is None
    assert ann.get_position_history("260101aaaa", db) == []


def test_lazy_schema_upgrade_of_existing_db(tmp_path):
    """A DB created before the feature gains event_positions on first use."""
    import sqlite3

    d = str(tmp_path / "old.db")
    ann.add_user("vishnu", d)
    with sqlite3.connect(d) as conn:
        conn.execute("DROP TABLE event_positions")
    _set(d)                                  # transparently re-creates
    assert ann.get_position("260715twmx", d)["ra_deg"] is not None


# ---------------------------------------------------------------------------
# Store: validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw,msg", [
    (dict(ra=-0.1), "ra out of range"),
    (dict(ra=360.0), "ra out of range"),
    (dict(dec=-90.5), "dec out of range"),
    (dict(dec=91.0), "dec out of range"),
    (dict(ra_err=0.0), "ra_err_arcsec out of range"),
    (dict(ra_err=-3.0), "ra_err_arcsec out of range"),
    (dict(dec_err=3600.0), "dec_err_arcsec out of range"),
    (dict(ra=float("nan")), "not finite"),
    (dict(dec_err=float("inf")), "not finite"),
    (dict(ra="abc"), "not a number"),
    (dict(method="  "), "method is empty"),
])
def test_set_position_rejects_bad_input(db, kw, msg):
    with pytest.raises(ValueError, match=msg):
        _set(db, **kw)
    assert ann.get_position("260715twmx", db) is None   # nothing persisted


# ---------------------------------------------------------------------------
# Operator coordinate parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("txt,deg", [
    ("234.0900", 234.09),
    (" 0.0 ", 0.0),
    ("359.9999", 359.9999),
    ("15:36:21.87", (15 + 36 / 60 + 21.87 / 3600) * 15.0),
    ("15h36m21.87s", (15 + 36 / 60 + 21.87 / 3600) * 15.0),
    ("00:00:00", 0.0),
    ("23:59:59.9", (23 + 59 / 60 + 59.9 / 3600) * 15.0),
])
def test_parse_ra_ok(txt, deg):
    assert ea.parse_ra_str(txt) == pytest.approx(deg, abs=1e-9)


@pytest.mark.parametrize("txt", [
    "", "  ", "360.0", "-1.0", "24:00:00", "12:60:00", "12:00:60",
    "twelve", "12:34", "12 34 56",
])
def test_parse_ra_rejects(txt):
    with pytest.raises(ValueError):
        ea.parse_ra_str(txt)


@pytest.mark.parametrize("txt,deg", [
    ("+17.5598", 17.5598),
    ("-5.2", -5.2),
    ("17.5598", 17.5598),
    ("+17:33:35.3", 17 + 33 / 60 + 35.3 / 3600),
    ("-05:12:00", -(5 + 12 / 60)),
    ("17d33m35.3s", 17 + 33 / 60 + 35.3 / 3600),
    ("+90.0", 90.0),
    ("-90:00:00", -90.0),
])
def test_parse_dec_ok(txt, deg):
    assert ea.parse_dec_str(txt) == pytest.approx(deg, abs=1e-9)


@pytest.mark.parametrize("txt", [
    "", "  ", "90.1", "-91", "17:60:00", "17:00:60", "91:00:00",
    "north", "17 33 35",
])
def test_parse_dec_rejects(txt):
    with pytest.raises(ValueError):
        ea.parse_dec_str(txt)


def test_parse_format_roundtrip_m8():
    """The M8 refined position round-trips through parse -> format."""
    ra = ea.parse_ra_str("234.09627")
    dec = ea.parse_dec_str("+17.55895")
    hms, dms = ea.sexagesimal_refined(ra, dec, 108.0, 108.0)  # 1.8 arcmin
    # sigma_RA = 108" / (15 cos dec) = 7.55 s of time -> 0.1 s digit, (76);
    # sigma_Dec = 108" -> whole-arcminute digit, (2).
    assert hms == "15:36:23.1(76)"
    assert dms == "+17:34(2)"


# ---------------------------------------------------------------------------
# Adaptive sexagesimal formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma,expect", [
    (0.1, "+17:33:35.33(10)"),      # 0.01" last digit
    (1.0, "+17:33:35.3(10)"),       # 0.1"
    (10.0, "+17:33:35(10)"),        # 1"
    (60.0, "+17:33:35(60)"),        # 1"
    (300.0, "+17:34(5)"),           # 1' last digit (seconds dropped)
    (5940.0, "+17:34(99)"),         # 99' — coarsest representable
])
def test_dec_adaptive_precision_ladder(sigma, expect):
    assert ea.format_dec_dms_adaptive(17.559814, sigma) == expect


def test_dec_adaptive_overflow_capped():
    assert ea.format_dec_dms_adaptive(17.5598, 6000.0) == "+17:34(>99)"


def test_dec_adaptive_tiny_sigma_clamps_to_one():
    out = ea.format_dec_dms_adaptive(17.559814, 1e-6)
    assert out.endswith("(1)")


def test_dec_adaptive_negative_and_none():
    assert ea.format_dec_dms_adaptive(-5.25, 10.0) == "-05:15:00(10)"
    assert ea.format_dec_dms_adaptive(None, 10.0) is None
    assert ea.format_dec_dms_adaptive(float("nan"), 10.0) is None


def test_dec_adaptive_carry_rounding():
    # 17°59'59.97" at 0.1" precision must carry to 18:00:00.0.
    dec = (17 * 3600 + 59 * 60 + 59.97) / 3600.0
    assert ea.format_dec_dms_adaptive(dec, 1.0) == "+18:00:00.0(10)"


def test_ra_adaptive_cos_dec_conversion():
    # sigma 15" on-sky at dec 0 -> exactly 1.0 s of time -> "(10)" at
    # the 0.1 s level.
    out = ea.format_ra_hms_adaptive(234.089979, 15.0, 0.0)
    assert out == "15:36:21.6(10)"


def test_ra_adaptive_scales_with_dec():
    # Same on-sky sigma is more seconds of time at high dec (1/cos).
    lo = ea.format_ra_hms_adaptive(234.0, 15.0, 0.0)
    hi = ea.format_ra_hms_adaptive(234.0, 15.0, 60.0)   # 2 s of time
    assert lo.endswith("(10)") and hi.endswith("(20)")


def test_ra_adaptive_pole_guard():
    out = ea.format_ra_hms_adaptive(234.0, 15.0, 89.0)
    assert out.endswith("(>99)")


def test_ra_adaptive_wraps_and_none():
    assert ea.format_ra_hms_adaptive(360.0 + 15.0, 15.0, 0.0).startswith("01:")
    assert ea.format_ra_hms_adaptive(None, 15.0, 0.0) is None


def test_ra_adaptive_carry_to_24h():
    # 23:59:59.999 shown at the 0.01 s digit rounds up and wraps to 0 h.
    ra = (23 + 59 / 60 + 59.999 / 3600) * 15.0
    out = ea.format_ra_hms_adaptive(ra, 1.5, 0.0)   # sigma 0.1 s -> 2 dp
    assert out == "00:00:00.00(10)"


def test_sexagesimal_refined_none_safety():
    assert ea.sexagesimal_refined(None, 17.0, 1.0, 1.0) == (None, None)
    assert ea.sexagesimal_refined(234.0, None, 1.0, 1.0) == (None, None)
    # Missing errors fall back to the fixed-precision formatters.
    hms, dms = ea.sexagesimal_refined(234.089979, 17.559814, None, None)
    assert hms == ea.format_ra_hms(234.089979)
    assert dms == ea.format_dec_dms(17.559814)


# ---------------------------------------------------------------------------
# Flask API + page renders (skip if app can't import outside prod)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    dbfile = tmp_path_factory.mktemp("annot_pos") / "annot.db"
    os.environ["DSA_MONITOR_ANNOT_DB"] = str(dbfile)
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


def _detail(name: str) -> cpf.EventDetail:
    return cpf.EventDetail(
        name=name, archive_dir=Path("/tmp/nonexistent") / name,
        metadata={}, plots=(), cubes=(), has_c2_csv=False,
        has_c1_csv=False, ra_deg=234.38, dec_deg=17.54,
        radec_source="level3", ra_hms="15:37:31.2(42)",
        dec_dms="+17:32:24(60)",
    )


def test_api_position_lifecycle(client, app_module):
    ev = "260715tttt"
    r = client.get(f"/api/position/{ev}")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] and j["refined"] is None and j["history"] == []

    client.post("/annotations/user", json={"name": "vishnu"})

    # Unknown user -> 400.
    r = client.post(f"/api/position/{ev}", json={
        "ra": "234.09627", "dec": "+17.55895", "ra_err_arcsec": "108",
        "dec_err_arcsec": "108", "method": "M8", "user": "ghost"})
    assert r.status_code == 400

    # Decimal-degree entry.
    with mock.patch.object(app_module.cands_browser, "event_detail",
                           return_value=_detail(ev)):
        r = client.post(f"/api/position/{ev}", json={
            "ra": "234.09627", "dec": "+17.55895", "ra_err_arcsec": "108",
            "dec_err_arcsec": "108", "method": "M8 oversampled UV",
            "user": "vishnu"})
    assert r.status_code == 200, r.get_data(as_text=True)
    ref = r.get_json()["refined"]
    assert ref["ra_hms"] == "15:36:23.1(76)"
    assert ref["pipe_ra_deg"] == pytest.approx(234.38)   # snapshot taken
    assert ref["pipe_source"] == "level3"
    assert "M8 oversampled UV" in ref["tooltip"]

    # Sexagesimal entry overrides (last-write-wins).
    r = client.post(f"/api/position/{ev}", json={
        "ra": "15:36:23.1", "dec": "+17:33:32", "ra_err_arcsec": "17",
        "dec_err_arcsec": "17", "method": "M9 outrigger + NVSS tie",
        "user": "vishnu"})
    assert r.status_code == 200
    ref = r.get_json()["refined"]
    assert ref["ra_deg"] == pytest.approx(234.09625, abs=1e-4)
    assert ref["method"] == "M9 outrigger + NVSS tie"

    # GET: current + both audit rows.
    j = client.get(f"/api/position/{ev}").get_json()
    assert j["refined"]["method"] == "M9 outrigger + NVSS tie"
    assert len(j["history"]) == 2

    # Clear (POST alias) -> refined gone, audit row appended.
    r = client.post(f"/api/position/{ev}/clear", json={"user": "vishnu"})
    assert r.status_code == 200 and r.get_json()["cleared"] is True
    j = client.get(f"/api/position/{ev}").get_json()
    assert j["refined"] is None and len(j["history"]) == 3

    # DELETE alias also works.
    r = client.delete(f"/api/position/{ev}", json={"user": "vishnu"})
    assert r.status_code == 200 and r.get_json()["cleared"] is False


@pytest.mark.parametrize("payload,frag", [
    ({"ra": "bogus", "dec": "17", "ra_err_arcsec": "1",
      "dec_err_arcsec": "1", "method": "m", "user": "vishnu"},
     "cannot parse RA"),
    ({"ra": "234", "dec": "95", "ra_err_arcsec": "1",
      "dec_err_arcsec": "1", "method": "m", "user": "vishnu"},
     "out of range"),
    ({"ra": "234", "dec": "17", "ra_err_arcsec": "0",
      "dec_err_arcsec": "1", "method": "m", "user": "vishnu"},
     "out of range"),
    ({"ra": "234", "dec": "17", "ra_err_arcsec": "1",
      "dec_err_arcsec": "1", "method": "  ", "user": "vishnu"},
     "method is empty"),
])
def test_api_position_post_rejects(client, payload, frag):
    client.post("/annotations/user", json={"name": "vishnu"})
    r = client.post("/api/position/260715uuuu", json=payload)
    assert r.status_code == 400
    assert frag in r.get_json()["error"]


def test_api_position_bad_event_name(client):
    assert client.get("/api/position/..%2Fetc").status_code in (400, 404)
    r = client.post("/api/position/has spaces", json={"user": "vishnu"})
    assert r.status_code in (400, 404)


def _summary(name: str) -> cpf.EventSummary:
    rd = ea.RaDec(234.089979, 17.559814, "filterbank")
    hms, dms = ea.sexagesimal_for(rd)
    return cpf.EventSummary(
        name=name, mtime_unix=time.time(), mjd_peak=60000.5,
        trigger_class="bright_frb", n_events=1, snr_max=18.4,
        dm_median=1702.6, l_median=0.005, m_median=-0.017,
        n_cubes=0, n_plots=0, c3_action="KEEP",
        ra_deg=rd.ra_deg, dec_deg=rd.dec_deg, radec_source="filterbank",
        ra_hms=hms, dec_dms=dms,
    )


def test_bursts_page_column_order_and_marker(client, app_module):
    """RA/Dec are columns 2-3 right after event; refined rows carry the
    superscript-R marker and the refined tooltip."""
    ev = "260715wwww"
    client.post("/annotations/user", json={"name": "vishnu"})
    with mock.patch.object(app_module.cands_browser, "event_detail",
                           return_value=_detail(ev)):
        r = client.post(f"/api/position/{ev}", json={
            "ra": "234.09627", "dec": "+17.55895", "ra_err_arcsec": "108",
            "dec_err_arcsec": "108", "method": "M8 oversampled UV",
            "user": "vishnu"})
        assert r.status_code == 200
    events = [_summary(ev), _summary("260101kkkk")]
    with mock.patch.object(app_module.cands_browser, "list_events",
                           return_value=events):
        html = client.get("/bursts").get_data(as_text=True)

    # Header order: event, RA, Dec, UTC time, ... (RA/Dec moved up).
    head = html[html.index("<thead>"):html.index("</thead>")]
    idx = [head.index("event"), head.index("RA (J2000)"),
           head.index("Dec (J2000)"), head.index("UTC time"),
           head.index("n_events")]
    assert idx == sorted(idx)

    # The refined event renders the refined value + R marker; the other
    # event renders the pipeline value with no marker in its row.
    assert "15:36:23.1(76)" in html                 # refined RA shown
    assert 'class="refined-mark"' in html
    assert "M8 oversampled UV" in html              # refined tooltip
    assert "15:36:21.6(42)" in html                 # pipeline RA (other row)
    ann.clear_position(ev, "vishnu")                # leave no residue


def test_event_page_renders_position_card(client, app_module):
    ev = "260715xxxx"
    client.post("/annotations/user", json={"name": "vishnu"})
    with mock.patch.object(app_module.cands_browser, "event_detail",
                           return_value=_detail(ev)):
        r = client.post(f"/api/position/{ev}", json={
            "ra": "234.09627", "dec": "+17.55895", "ra_err_arcsec": "108",
            "dec_err_arcsec": "108", "method": "M8 oversampled UV",
            "user": "vishnu"})
        assert r.status_code == 200
        html = client.get(f"/bursts/{ev}").get_data(as_text=True)
    assert 'id="pos-refined"' in html
    assert 'id="pos-set"' in html
    assert "positions.js" in html
    # Server-embedded state for positions.js.
    assert "window.POS" in html
    assert "15:36:23.1(76)" in html
    ann.clear_position(ev, "vishnu")

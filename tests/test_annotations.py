"""Unit tests for the dashboard human-annotation storage layer.

Pure SQLite against a ``tmp_path`` DB — no Flask, no observatory env.
Covers: user add / case-insensitive uniqueness, per-user last-wins,
clear semantics, event-level source last-wins, custom-tag
normalisation + rejection, current-vs-history queries, the
"unclassified" set, and the count-by-source-and-date (B1913+16) query.

A couple of Flask test-client smoke tests for the endpoints are
appended, guarded so they skip cleanly if ``app`` can't import outside
the live observatory environment.
"""

from __future__ import annotations

import os
import sys

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


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "annot.db")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def test_add_user_and_list(db):
    ann.add_user("vishnu", db)
    ann.add_user("vikram", db)
    assert ann.list_users(db) == ["vishnu", "vikram"]


def test_add_user_strips_and_rejects_empty(db):
    ann.add_user("  vishnu  ", db)
    assert ann.list_users(db) == ["vishnu"]
    with pytest.raises(ValueError):
        ann.add_user("   ", db)


def test_add_user_case_insensitive_unique(db):
    ann.add_user("Vishnu", db)
    ann.add_user("vishnu", db)  # idempotent; first casing wins
    assert ann.list_users(db) == ["Vishnu"]


# ---------------------------------------------------------------------------
# Classifications: last-wins, clear
# ---------------------------------------------------------------------------


def test_classify_requires_known_user(db):
    with pytest.raises(ann.UnknownUserError):
        ann.classify("EV1", "nobody", "FRB", db)


def test_classify_last_wins_per_user(db):
    ann.add_user("vishnu", db)
    ann.add_user("vikram", db)
    ann.classify("EV1", "vishnu", "RFI", db)
    ann.classify("EV1", "vishnu", "FRB", db)  # overrides RFI
    blk = ann.classify("EV1", "vikram", "RFI", db)
    users = {c["user"]: c["label"] for c in blk["classifications"]}
    assert users == {"vishnu": "FRB", "vikram": "RFI"}
    assert blk["labels"] == {"FRB": 1, "RFI": 1}


def test_classify_lowercase_builtin_normalised(db):
    ann.add_user("vishnu", db)
    blk = ann.classify("EV1", "vishnu", "frb", db)
    assert blk["classifications"][0]["label"] == "FRB"


def test_classify_clear_removes_current(db):
    ann.add_user("vishnu", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    blk = ann.classify("EV1", "vishnu", None, db)  # clear
    assert blk["classifications"] == []
    assert blk["labels"] == {}
    # ...but the audit trail retains both the set and the clear.
    hist = ann.event_history("EV1", db)
    labels = [h["label"] for h in hist if h["kind"] == "classification"]
    assert labels == ["FRB", None]


def test_check_offline_voltages_builtin(db):
    """CHECK_OFFLINE_VOLTAGES is a first-class built-in label: present
    in the vocabulary, classifiable, and reserved against custom tags."""
    assert "CHECK_OFFLINE_VOLTAGES" in ann.BUILTIN_LABELS
    ann.add_user("vishnu", db)
    blk = ann.classify("EV1", "vishnu", "CHECK_OFFLINE_VOLTAGES", db)
    assert blk["classifications"][0]["label"] == "CHECK_OFFLINE_VOLTAGES"
    assert blk["labels"] == {"CHECK_OFFLINE_VOLTAGES": 1}
    # Collides with the built-in -> custom-tag creation rejected
    # (including via normalisation from a spaced lowercase form).
    with pytest.raises(ValueError):
        ann.create_tag("check offline voltages", "vishnu", db)


def test_classify_unknown_label_rejected(db):
    ann.add_user("vishnu", db)
    with pytest.raises(ValueError):
        ann.classify("EV1", "vishnu", "MADEUP", db)


# ---------------------------------------------------------------------------
# Custom tags
# ---------------------------------------------------------------------------


def test_create_tag_normalises(db):
    ann.add_user("vishnu", db)
    t = ann.create_tag("  side lobe!! ", "vishnu", db)
    assert t == "SIDE_LOBE"
    assert ann.list_custom_tags(db) == ["SIDE_LOBE"]


def test_create_tag_keeps_plus_minus(db):
    ann.add_user("vishnu", db)
    assert ann.create_tag("B1913+16-ish", "vishnu", db) == "B1913+16-ISH"


def test_create_tag_rejects_builtin_and_empty(db):
    ann.add_user("vishnu", db)
    with pytest.raises(ValueError):
        ann.create_tag("frb", "vishnu", db)
    with pytest.raises(ValueError):
        ann.create_tag("   ", "vishnu", db)


def test_create_tag_truncates(db):
    ann.add_user("vishnu", db)
    t = ann.create_tag("A" * 40, "vishnu", db)
    assert len(t) == ann.MAX_TAG_LEN


def test_classify_with_custom_tag_after_create(db):
    ann.add_user("vishnu", db)
    ann.create_tag("scintillator", "vishnu", db)
    blk = ann.classify("EV1", "vishnu", "scintillator", db)
    assert blk["classifications"][0]["label"] == "SCINTILLATOR"


# ---------------------------------------------------------------------------
# Source names (event-level, last-wins, records who)
# ---------------------------------------------------------------------------


def test_source_last_wins_event_level(db):
    ann.add_user("vishnu", db)
    ann.add_user("vikram", db)
    ann.set_source("EV1", "vishnu", "B1913", db)
    blk = ann.set_source("EV1", "vikram", "B1913+16", db)  # overrides
    assert blk["source_name"]["source_name"] == "B1913+16"
    assert blk["source_name"]["user"] == "vikram"  # records who set it


def test_source_clear(db):
    ann.add_user("vishnu", db)
    ann.set_source("EV1", "vishnu", "B1913+16", db)
    blk = ann.set_source("EV1", "vishnu", "", db)  # clear
    assert blk["source_name"] is None


def test_source_not_uppercased(db):
    ann.add_user("vishnu", db)
    blk = ann.set_source("EV1", "vishnu", "  B1913+16  ", db)
    assert blk["source_name"]["source_name"] == "B1913+16"


# ---------------------------------------------------------------------------
# Source-name purge (typo cleanup)
# ---------------------------------------------------------------------------


def test_purge_moves_rows_and_drops_vocab(db):
    ann.add_user("vishnu", db)
    ann.set_source("EV1", "vishnu", "B1913+I6", db)   # typo
    ann.set_source("EV2", "vishnu", "b1913+i6", db)   # same typo, other case
    n = ann.purge_source_name("B1913+I6", "vishnu", db)
    assert n == 2
    # Gone from the vocabulary...
    assert ann.vocab(db)["source_names"] == []
    # ...and from both events' current source.
    assert ann.event_annotations("EV1", db)["source_name"] is None
    assert ann.event_annotations("EV2", db)["source_name"] is None
    # Rows preserved in the graveyard with the purger stamped.
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT event, source_name, purged_by FROM source_names_purged "
            "ORDER BY orig_id"
        ).fetchall()
    finally:
        conn.close()
    assert [(r[0], r[2]) for r in rows] == [("EV1", "vishnu"), ("EV2", "vishnu")]


def test_purge_current_falls_back_to_latest_surviving_row(db):
    """Purge removes the matching rows entirely, so an event's current
    source becomes the latest SURVIVING row — an older different name
    resurfaces."""
    ann.add_user("vishnu", db)
    ann.set_source("EV1", "vishnu", "3C48", db)       # older, different
    ann.set_source("EV1", "vishnu", "3C48-typo", db)  # latest (typo)
    assert ann.event_annotations("EV1", db)["source_name"]["source_name"] \
        == "3C48-typo"
    ann.purge_source_name("3C48-typo", "vishnu", db)
    cur = ann.event_annotations("EV1", db)["source_name"]
    assert cur is not None and cur["source_name"] == "3C48"


def test_purge_rejects_unknown_user_and_empty(db):
    ann.add_user("vishnu", db)
    ann.set_source("EV1", "vishnu", "3C48", db)
    with pytest.raises(ann.UnknownUserError):
        ann.purge_source_name("3C48", "ghost", db)
    with pytest.raises(ValueError):
        ann.purge_source_name("   ", "vishnu", db)
    # Nothing was purged by the failed attempts.
    assert ann.vocab(db)["source_names"] == ["3C48"]


def test_purge_nonexistent_name_is_noop(db):
    ann.add_user("vishnu", db)
    assert ann.purge_source_name("NEVER_USED", "vishnu", db) == 0


# ---------------------------------------------------------------------------
# Bulk current + unclassified filter
# ---------------------------------------------------------------------------


def test_all_current_counts_agreement(db):
    ann.add_user("vishnu", db)
    ann.add_user("vikram", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    ann.classify("EV1", "vikram", "FRB", db)
    ann.classify("EV2", "vishnu", "RFI", db)
    ann.set_source("EV1", "vishnu", "B1913+16", db)
    cur = ann.all_current(db)
    assert cur["EV1"]["labels"] == {"FRB": 2}
    assert cur["EV1"]["source_name"] == "B1913+16"
    assert cur["EV2"]["labels"] == {"RFI": 1}


def test_unclassified_set(db):
    ann.add_user("vishnu", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    ann.classify("EV2", "vishnu", "RFI", db)
    ann.classify("EV2", "vishnu", None, db)  # cleared → unclassified again
    ann.set_source("EV3", "vishnu", "B1913+16", db)  # source only, no label
    classified = ann.classified_events(db)
    assert classified == {"EV1"}
    # EV2 (cleared) and EV3 (source only) are NOT classified.


def test_annotated_events_union_includes_cleared_and_sources(db):
    """annotated_events() = every event a human ever touched (union of
    classifications, source_names, event_positions) — including ones later
    cleared, so the /bursts list keeps them forever."""
    ann.add_user("vishnu", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    ann.classify("EV2", "vishnu", "RFI", db)
    ann.classify("EV2", "vishnu", None, db)          # cleared, still touched
    ann.set_source("EV3", "vishnu", "B1913+16", db)  # source only
    ann.set_position("EV4", 10.0, 20.0, 5.0, 5.0, "vlbi", "vishnu",
                     db_path=db)                      # refined position only
    assert ann.annotated_events(db) == {"EV1", "EV2", "EV3", "EV4"}
    # classified_events() is the strict subset with a CURRENT label.
    assert ann.classified_events(db) == {"EV1"}


def test_annotated_events_empty_db(db):
    assert ann.annotated_events(db) == set()


# ---------------------------------------------------------------------------
# Query surface: current vs history, B1913+16 count-by-date
# ---------------------------------------------------------------------------


def test_query_current_by_label(db):
    ann.add_user("vishnu", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    ann.classify("EV2", "vishnu", "RFI", db)
    res = ann.query_annotations(db, label="FRB")
    assert [r["event"] for r in res] == ["EV1"]


def test_query_history_full_trail(db):
    ann.add_user("vishnu", db)
    ann.classify("EV1", "vishnu", "FRB", db)
    ann.classify("EV1", "vishnu", "RFI", db)
    cur = ann.query_annotations(db, event="EV1")
    assert cur[0]["classifications"][0]["label"] == "RFI"  # latest only
    hist = ann.query_annotations(db, event="EV1", history=True)
    labels = [h["label"] for h in hist]
    assert labels == ["FRB", "RFI"]  # full trail, oldest first


def test_count_by_source_and_date(db):
    """The B1913+16 agent use case: count detections since a date."""
    ann.add_user("vishnu", db)
    # Two events currently named B1913+16, one named something else.
    ann.set_source("EV_A", "vishnu", "B1913+16", db)
    ann.set_source("EV_B", "vishnu", "B1913+16", db)
    ann.set_source("EV_C", "vishnu", "3C48", db)
    res = ann.query_annotations(db, source="B1913+16")
    assert len(res) == 2
    assert {r["event"] for r in res} == {"EV_A", "EV_B"}
    # Case-insensitive match, and URL-decoded '+' round-trips.
    assert len(ann.query_annotations(db, source="b1913+16")) == 2
    # A future 'since' filters by the source-set timestamp.
    assert len(ann.query_annotations(db, source="B1913+16",
                                     since="2999-01-01")) == 0
    assert len(ann.query_annotations(db, source="B1913+16",
                                     since="2000-01-01")) == 2


def test_vocab(db):
    ann.add_user("vishnu", db)
    ann.create_tag("scint", "vishnu", db)
    ann.set_source("EV1", "vishnu", "B1913+16", db)
    v = ann.vocab(db)
    assert v["builtin_labels"] == list(ann.BUILTIN_LABELS)
    assert "SCINT" in v["custom_tags"]
    assert "SCINT" in v["labels"]
    assert v["users"] == ["vishnu"]
    assert v["source_names"] == ["B1913+16"]


def test_wal_mode_enabled(db):
    import sqlite3
    ann.add_user("vishnu", db)  # creates the DB
    conn = sqlite3.connect(db)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


# ---------------------------------------------------------------------------
# Zombie partition (bursts tab rule) — pure functions, no Flask
# ---------------------------------------------------------------------------

import time as _time

import cands_panel_funcs as cpf


#: A "now" safely inside the C3 era for the zombie tests.
_NOW = cpf.C3_ERA_START_UNIX + 30 * 86400.0


def _ev(name, *, c3_action=None, mtime_unix=0.0, mjd_peak=None,
        has_l3=False):
    return cpf.EventSummary(
        name=name, mtime_unix=mtime_unix, mjd_peak=mjd_peak,
        trigger_class=None, n_events=None, snr_max=None, dm_median=None,
        l_median=None, m_median=None, n_cubes=0, n_plots=0,
        c3_action=c3_action, has_l3=has_l3,
    )


def test_zombie_name_pattern_gate():
    """Only real event names (YYMMDD + 4 lowercase letters) can be
    zombies; calibrator scans / test triggers / misc dirs keep today's
    behaviour no matter how old and pending they are."""
    now = _NOW
    old = now - 2 * 3600.0                      # 2 h old, pending
    assert cpf.is_zombie(_ev("260714nmeh", mtime_unix=old), now)
    for bad in ("dumpnow_12345678", "3C286_a", "DUMP1", "releases",
                "dtest2"):
        assert not cpf.is_zombie(_ev(bad, mtime_unix=old), now), bad


def test_zombie_age_and_status_rules():
    now = _NOW
    fresh = now - 0.5 * 3600.0
    old = now - 2 * 3600.0
    # Fresh pending -> not a zombie (stays in pass).
    assert not cpf.is_zombie(_ev("260714aaaa", mtime_unix=fresh), now)
    # Old pending -> zombie.
    assert cpf.is_zombie(_ev("260714aaaa", mtime_unix=old), now)
    # Judged events are never zombies, however old.
    assert not cpf.is_zombie(
        _ev("260714aaaa", c3_action="KEEP", mtime_unix=old), now)
    assert not cpf.is_zombie(
        _ev("260714aaaa", c3_action="REJECT", mtime_unix=old), now)
    # mjd_peak wins over mtime when present: fresh t_peak, stale mtime.
    mjd_fresh = (now - 0.25 * 3600.0) / 86400.0 + 40587.0
    assert not cpf.is_zombie(
        _ev("260714aaaa", mtime_unix=old, mjd_peak=mjd_fresh), now)


def test_zombie_level3_gate():
    """Real-named pre-C3-era events whose legacy Level3 JSON is on disk
    are NOT zombies (their metadata landed; C3 simply never existed to
    judge them). A genuine zombie has no Level3 at all."""
    now = _NOW
    old = now - 2 * 3600.0
    assert cpf.is_zombie(_ev("240122aaag", mtime_unix=old, has_l3=False), now)
    assert not cpf.is_zombie(
        _ev("240119aacg", mtime_unix=old, has_l3=True), now)


def test_zombie_c3_era_gate():
    """Dirs untouched since before the C3 veto went live are legacy —
    'pending' is not a stuck state for them, however old and however
    real their names look."""
    now = _NOW
    pre_era = cpf.C3_ERA_START_UNIX - 86400.0     # touched before C3
    in_era = cpf.C3_ERA_START_UNIX + 86400.0      # touched after C3
    assert not cpf.is_zombie(
        _ev("230913aaao", mtime_unix=pre_era), now)
    assert cpf.is_zombie(_ev("260711irzt", mtime_unix=in_era), now)


def test_partition_events_c3_buckets():
    now = _NOW
    old = now - 2 * 3600.0
    fresh = now - 60.0
    events = [
        _ev("260714aaaa", c3_action="KEEP", mtime_unix=old),      # pass
        _ev("260714bbbb", c3_action="REJECT", mtime_unix=old),    # fail
        _ev("260714cccc", mtime_unix=fresh),                      # pass (fresh pending)
        _ev("260714dddd", mtime_unix=old),                        # zombie
        _ev("dumpnow_12345678", mtime_unix=old),                  # pass (name gate)
    ]
    ev_pass, ev_fail, ev_zomb = cpf.partition_events_c3(events, now_unix=now)
    assert [e.name for e in ev_pass] == ["260714aaaa", "260714cccc",
                                         "dumpnow_12345678"]
    assert [e.name for e in ev_fail] == ["260714bbbb"]
    assert [e.name for e in ev_zomb] == ["260714dddd"]


# ---------------------------------------------------------------------------
# Flask endpoint smoke tests (skip if app can't import outside prod)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from unittest import mock

    dbfile = tmp_path_factory.mktemp("annot_app") / "annot.db"
    os.environ["DSA_MONITOR_ANNOT_DB"] = str(dbfile)
    try:
        with mock.patch("rfi_store.RFIPoller.start", return_value=None):
            import app  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"app import needs live resources: {exc!r}")
    app.app.config.update(TESTING=True)
    return app.app.test_client()


def test_endpoint_roundtrip(client):
    # Add a user.
    r = client.post("/annotations/user", json={"name": "vishnu"})
    assert r.status_code == 200, r.get_data(as_text=True)
    # Classify.
    r = client.post(
        "/annotations/classify",
        json={"event": "EVX", "user": "vishnu", "label": "FRB"},
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["labels"] == {"FRB": 1}
    # Unknown user → 400.
    r = client.post(
        "/annotations/classify",
        json={"event": "EVX", "user": "ghost", "label": "FRB"},
    )
    assert r.status_code == 400
    # Source set + read via API.
    r = client.post(
        "/annotations/source",
        json={"event": "EVX", "user": "vishnu", "source_name": "B1913+16"},
    )
    assert r.status_code == 200
    r = client.get("/api/annotations?source=B1913%2B16")
    assert r.status_code == 200
    events = [x["event"] for x in r.get_json()["annotations"]]
    assert "EVX" in events
    # Vocab.
    r = client.get("/api/annotations/vocab")
    assert r.status_code == 200
    assert "FRB" in r.get_json()["labels"]


def test_endpoint_purge_roundtrip(client):
    r = client.post("/annotations/user", json={"name": "purger"})
    assert r.status_code == 200
    r = client.post(
        "/annotations/source",
        json={"event": "EVP", "user": "purger", "source_name": "TYP0"},
    )
    assert r.status_code == 200
    r = client.post(
        "/annotations/source/purge", json={"name": "TYP0", "user": "purger"},
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["n_purged"] == 1
    assert "TYP0" not in j["vocab"]["source_names"]
    # Unknown user -> 400.
    r = client.post(
        "/annotations/source/purge", json={"name": "x", "user": "ghost"},
    )
    assert r.status_code == 400
    # Empty name -> 400.
    r = client.post(
        "/annotations/source/purge", json={"name": "  ", "user": "purger"},
    )
    assert r.status_code == 400


def test_bursts_page_zombie_tab_and_source_filter(client):
    """The list page renders the Zombies tab + source search and the
    ?source= filter round-trips (mocked archive)."""
    from unittest import mock
    import app as app_mod

    now = _time.time()
    events = [
        _ev("260714zzzz", c3_action="KEEP", mtime_unix=now - 60),
        _ev("260713qqqq", mtime_unix=now - 7200),   # zombie
    ]
    # The /bursts route now reads the whole index from the in-process
    # EventIndexCache (incremental TTL refresh); mock the snapshot rather
    # than the underlying per-request archive scan.
    snap = cpf.CacheSnapshot(
        events=events, n_total=len(events),
        last_success_unix=now, stale=False, error=None,
    )
    with mock.patch.object(app_mod.cands_index, "snapshot",
                           return_value=snap):
        r = client.get("/bursts")
        html = r.get_data(as_text=True)
        assert r.status_code == 200
        assert "Zombies" in html
        assert "260713qqqq" in html
        assert 'id="src-search"' in html
        r = client.get("/bursts?source=B1913%2B16")
        assert r.status_code == 200
        assert "src-chip" in r.get_data(as_text=True)


def test_bursts_page_warming_shows_rebuild_banner_not_empty_state(client):
    """Post-restart, cold-cache case: EventIndexCache has never completed
    a scan (warming=True, n_total=0). The page must show the reassuring
    rebuild banner and auto-refresh meta tag, and must NOT show the
    misleading "of 0 archived" / "All-time totals — C3 pass: 0" line that
    looks like total data loss."""
    from unittest import mock
    import app as app_mod

    snap = cpf.CacheSnapshot(
        events=[], n_total=0, last_success_unix=None, stale=True,
        error=None, warming=True, scan_progress=7,
    )
    with mock.patch.object(app_mod.cands_index, "snapshot",
                           return_value=snap):
        r = client.get("/bursts")
        html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "rebuilding" in html.lower()
    assert "nothing is lost" in html.lower()
    assert "7 event dir" in html                    # scan_progress hint
    assert '<meta http-equiv="refresh" content="30">' in html
    assert "of 0 archived" not in html
    assert "All-time totals" not in html
    # The scary "stale" banner must not also render alongside it.
    assert "is <strong>stale</strong>" not in html


def test_bursts_page_warm_renders_normally_no_banner(client):
    """A normal, already-warm snapshot renders the usual totals line and
    carries no rebuild banner and no auto-refresh meta tag."""
    from unittest import mock
    import app as app_mod

    now = _time.time()
    events = [_ev("260714warm", c3_action="KEEP", mtime_unix=now - 60)]
    snap = cpf.CacheSnapshot(
        events=events, n_total=len(events), last_success_unix=now,
        stale=False, error=None, warming=False, scan_progress=None,
    )
    with mock.patch.object(app_mod.cands_index, "snapshot",
                           return_value=snap):
        r = client.get("/bursts")
        html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "rebuilding" not in html.lower()
    assert '<meta http-equiv="refresh"' not in html
    assert "All-time totals" in html
    assert "of <strong>1</strong> archived" in html

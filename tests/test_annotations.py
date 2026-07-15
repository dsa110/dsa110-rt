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

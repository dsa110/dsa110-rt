"""dsart.services.slack_annotation_relay — website -> Slack classification
relay.

No network: ``requests`` is monkeypatched. Uses real (tmp) SQLite files
for the annotations DB and the post-map DB, since the mtime-gating
behaviour under test depends on real file stat signatures.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.services.slack_annotation_relay import (  # noqa: E402
    LABEL_COLORS,
    RelayConfig,
    SlackAnnotationRelay,
)


class _FakeResp:
    def __init__(self, json_doc: Dict[str, Any], status: int = 200) -> None:
        self._json = json_doc
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._json


@pytest.fixture(autouse=True)
def _token_env(monkeypatch):
    monkeypatch.delenv("SLACK_TOKEN_DSA110", raising=False)


def _make_annotations_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE classifications ("
            "id INTEGER PRIMARY KEY, event TEXT, user TEXT, label TEXT, "
            "ts_utc TEXT)")
        conn.commit()
    finally:
        conn.close()


def _insert_row(
    path: Path, event: str, user: str, label: Any, ts_utc: str = "2026-07-23T00:00:00Z",
) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO classifications (event, user, label, ts_utc) "
            "VALUES (?, ?, ?, ?)", (event, user, label, ts_utc))
        conn.commit()
    finally:
        conn.close()


def _make_post_map_db(path: Path, rows: Dict[str, tuple]) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE posts (event TEXT PRIMARY KEY, channel TEXT, "
            "ts TEXT, posted_utc TEXT)")
        for event, (channel, ts) in rows.items():
            conn.execute(
                "INSERT INTO posts (event, channel, ts, posted_utc) "
                "VALUES (?, ?, ?, ?)", (event, channel, ts, "2026-07-23"))
        conn.commit()
    finally:
        conn.close()


def _cfg(tmp_path: Path, **kw) -> RelayConfig:
    kw.setdefault("enabled", True)
    kw.setdefault("channel", "C01NUV2M0HM")
    kw.setdefault("annotations_db", str(tmp_path / "annotations.db"))
    kw.setdefault("post_map_db", str(tmp_path / "posts.db"))
    kw.setdefault("state_path", str(tmp_path / "state.json"))
    return RelayConfig(**kw)


def _setup(tmp_path: Path, *, post_map: Dict[str, tuple], **cfg_kw):
    ann_db = tmp_path / "annotations.db"
    posts_db = tmp_path / "posts.db"
    _make_annotations_db(ann_db)
    _make_post_map_db(posts_db, post_map)
    cfg = _cfg(tmp_path, annotations_db=str(ann_db), post_map_db=str(posts_db),
              **cfg_kw)
    return cfg, ann_db, posts_db


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


def test_config_from_dict_defaults() -> None:
    cfg = RelayConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.broadcast_labels == ("FRB",)
    assert cfg.poll_s == 15.0
    assert cfg.fallback_sweep_s == 600.0
    assert cfg.token_env == "SLACK_TOKEN_DSA110"


def test_config_from_yaml_slack_relay_block(tmp_path: Path) -> None:
    cfg_path = tmp_path / "search_rt.yaml"
    cfg_path.write_text(
        "slack_relay:\n"
        "  enabled: true\n"
        "  channel: C01NUV2M0HM\n"
        "  token_file: /home/ubuntu/.config/slack_api_dsa110\n"
        "  poll_s: 15.0\n"
        "  fallback_sweep_s: 600.0\n"
        "  broadcast_labels: [FRB]\n"
    )
    cfg = RelayConfig.from_yaml(cfg_path)
    assert cfg.enabled is True
    assert cfg.channel == "C01NUV2M0HM"
    assert cfg.broadcast_labels == ("FRB",)
    assert cfg.poll_s == 15.0
    assert cfg.fallback_sweep_s == 600.0


# ---------------------------------------------------------------------------
# happy path: new tag, changed tag, cleared tag
# ---------------------------------------------------------------------------


def test_new_tag_posts_with_color_broadcast_and_empty_top_text(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101abcd": ("C01NUV2M0HM", "111.222")})
    _insert_row(ann_db, "260101abcd", "alice", "FRB")

    posted: List[Dict[str, Any]] = []

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        posted.append(json)
        return _FakeResp({"ok": True, "ts": "999.1"})

    monkeypatch.setattr("requests.post", _fake_post)

    relay = SlackAnnotationRelay(cfg)
    summary = relay.poll_once()
    assert summary["queried"] is True
    assert summary["relayed"] == 1
    assert len(posted) == 1

    p = posted[0]
    assert p["channel"] == "C01NUV2M0HM"
    assert p["thread_ts"] == "111.222"
    assert p["reply_broadcast"] is True    # FRB is in default broadcast set
    assert p["text"] == ""                 # top-level text MUST be empty
    att = p["attachments"][0]
    assert att["color"] == LABEL_COLORS["FRB"]
    assert "alice" in att["fallback"]
    assert "classified this as" in att["fallback"]
    assert "FRB" in att["fallback"]
    assert "alice" in att["text"]
    assert "classified this as" in att["text"]
    assert "`FRB`" in att["text"]
    assert att["mrkdwn_in"] == ["text"]


def test_rfi_tag_is_thread_only_not_broadcast(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101rfi1": ("C01NUV2M0HM", "222.333")})
    _insert_row(ann_db, "260101rfi1", "bob", "RFI")

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once()
    assert posted[0]["reply_broadcast"] is False
    assert posted[0]["attachments"][0]["color"] == LABEL_COLORS["RFI"]


def test_changed_tag_phrasing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101chng": ("C01NUV2M0HM", "333.444")})
    _insert_row(ann_db, "260101chng", "carol", "RFI")
    _insert_row(ann_db, "260101chng", "carol", "PULSAR")

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    summary = relay.poll_once()
    assert summary["relayed"] == 2
    assert "classified this as" in posted[0]["attachments"][0]["text"]
    assert "changed their classification to" in posted[1]["attachments"][0]["text"]
    assert "`PULSAR`" in posted[1]["attachments"][0]["text"]
    assert posted[1]["attachments"][0]["color"] == LABEL_COLORS["PULSAR"]


def test_cleared_tag_gray_bar_no_chip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101clr1": ("C01NUV2M0HM", "444.555")})
    _insert_row(ann_db, "260101clr1", "dave", "NOISE")
    _insert_row(ann_db, "260101clr1", "dave", None)

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once()
    cleared = posted[1]
    att = cleared["attachments"][0]
    assert "cleared their classification" in att["text"]
    assert "dave" in att["text"]
    assert "`" not in att["text"]     # no label chip
    assert att["color"] == "#7f8c8d"
    assert cleared["reply_broadcast"] is False


def test_unmapped_event_is_skipped(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(tmp_path, post_map={})  # no mapping at all
    _insert_row(ann_db, "260101nomap", "erin", "FRB")

    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))

    relay = SlackAnnotationRelay(cfg)
    summary = relay.poll_once()
    assert summary["relayed"] == 0
    assert summary["unmapped"] == 1
    assert calls == []


def test_broadcast_labels_frb_only_default_config(monkeypatch, tmp_path) -> None:
    """Default config (broadcast_labels=["FRB"]): only an FRB verdict
    gets reply_broadcast; RFI/NOISE/etc. stay thread-only."""
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101bc01": ("C01NUV2M0HM", "1.1")})
    _insert_row(ann_db, "260101bc01", "fay", "NOISE")

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once()
    assert posted[0]["reply_broadcast"] is False

    _insert_row(ann_db, "260101bc01", "fay", "FRB")
    relay.poll_once()
    assert posted[1]["reply_broadcast"] is True


def test_broadcast_labels_wildcard_broadcasts_everything(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101bcst": ("C01NUV2M0HM", "1.1")},
        broadcast_labels=("*",))
    _insert_row(ann_db, "260101bcst", "gia", "NOISE")

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once()
    assert posted[0]["reply_broadcast"] is True


# ---------------------------------------------------------------------------
# state persistence across poll cycles (no double-post)
# ---------------------------------------------------------------------------


def test_state_persistence_prevents_double_post_across_two_cycles(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={"260101dbl1": ("C01NUV2M0HM", "1.1")})
    _insert_row(ann_db, "260101dbl1", "hank", "FRB")

    posted: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "requests.post",
        lambda url, headers=None, json=None, timeout=None, **kw:
            posted.append(json) or _FakeResp({"ok": True, "ts": "1.1"}))

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once()
    assert len(posted) == 1

    # Fresh relay instance loading the same persisted state file (as
    # would happen across a process restart) -- unchanged DB mtime, but
    # even a forced re-query must not re-relay the already-seen row.
    relay2 = SlackAnnotationRelay(cfg)
    os.utime(ann_db, None)   # touch mtime to force a real requery
    summary = relay2.poll_once()
    assert summary["queried"] is True
    assert summary["relayed"] == 0
    assert len(posted) == 1   # no new post


# ---------------------------------------------------------------------------
# mtime-gated polling
# ---------------------------------------------------------------------------


def test_unchanged_mtime_skips_sqlite_connect(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp({"ok": True, "ts": "1.1"}))
    cfg, ann_db, _posts_db = _setup(tmp_path, post_map={})

    connect_calls: List[str] = []
    real_connect = sqlite3.connect

    def _counting_connect(path, *a, **k):
        connect_calls.append(str(path))
        return real_connect(path, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", _counting_connect)

    relay = SlackAnnotationRelay(cfg)
    now = 1000.0
    summary1 = relay.poll_once(now=now)
    # first cycle: no prior signature -> always "changed" -> queries once
    assert summary1["queried"] is True
    n_after_first = len(connect_calls)
    assert n_after_first > 0

    # second cycle, same clock/no file change, well within fallback
    # window -> must NOT touch sqlite at all.
    summary2 = relay.poll_once(now=now + 1.0)
    assert summary2["queried"] is False
    assert len(connect_calls) == n_after_first


def test_changed_mtime_triggers_query(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp({"ok": True, "ts": "1.1"}))
    cfg, ann_db, _posts_db = _setup(tmp_path, post_map={})
    relay = SlackAnnotationRelay(cfg)

    now = 2000.0
    relay.poll_once(now=now)

    connect_calls: List[str] = []
    real_connect = sqlite3.connect

    def _counting_connect(path, *a, **k):
        connect_calls.append(str(path))
        return real_connect(path, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", _counting_connect)

    time.sleep(0.01)
    _insert_row(ann_db, "260101mtim", "ivy", "FRB")   # changes db mtime/size
    summary = relay.poll_once(now=now + 1.0)
    assert summary["queried"] is True
    assert len(connect_calls) > 0


def test_fallback_sweep_fires_after_time_passage(monkeypatch, tmp_path) -> None:
    """Even with a completely unchanged file, a full query must run once
    fallback_sweep_s has elapsed."""
    monkeypatch.setenv("SLACK_TOKEN_DSA110", "xoxb-fake")
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp({"ok": True, "ts": "1.1"}))
    cfg, ann_db, _posts_db = _setup(
        tmp_path, post_map={}, fallback_sweep_s=100.0)

    connect_calls: List[str] = []
    real_connect = sqlite3.connect

    def _counting_connect(path, *a, **k):
        connect_calls.append(str(path))
        return real_connect(path, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", _counting_connect)

    relay = SlackAnnotationRelay(cfg)
    relay.poll_once(now=0.0)
    n0 = len(connect_calls)
    assert n0 > 0

    relay.poll_once(now=10.0)   # well within the sweep window, unchanged file
    assert len(connect_calls) == n0

    relay.poll_once(now=150.0)  # past fallback_sweep_s -> forced requery
    assert len(connect_calls) > n0

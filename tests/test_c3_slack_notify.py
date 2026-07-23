"""C3 KEEP -> Slack notification (dsart.services.slack_notify).

Two-stage flow (v3.8+): ``post_keep_card`` fires immediately at KEEP.
THE CARD IMAGE IS THE MESSAGE — no text-first ``chat.postMessage``. The
rendered card is uploaded via the external-upload flow with the
dashboard link as the upload's ``initial_comment``; the resulting
message ``ts`` is discovered by polling ``files.info`` and persisted to
a sqlite post-map db. If the render or upload fails, a plain
``chat.postMessage`` (header text + link) is sent instead, so the event
is never silently unposted. ``post_keep_followup`` fires later, once
the filterbank stage completes, and uploads BOTH bbproc variants
(unflagged + SK RFI-flagged) into the card's thread.

No network: all ``requests``/``sqlite3`` calls are exercised against
temp files or monkeypatched. Verifies the best-effort contract
(disabled/missing-token/network-error never raise and never call out)
and the message/upload content on the happy path.
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

from dsart.services.slack_notify import (   # noqa: E402
    SlackNotifier,
    SlackNotifyConfig,
)


C2ROW = {
    "snr_max": 21.4,
    "dm_median": 168.8,
    "l_median": 0.0011,
    "m_median": -0.0022,
    "width_median": 4.0,
    "t_peak_mjd": 61236.5,
}
KEEP_REPORT = {"n_fragments_present": 14, "n_fragments_total": 16,
               "n_manifests": 14}
FB_REPORT_OK = {"ok": True, "outputs": ["evt.fil", "evt.png",
                                        "evt_rfi.fil", "evt_rfi.png"]}
FB_REPORT_FAILED = {"ok": False, "error": "beamform failed"}
FB_REPORT_SKIPPED = {"ok": True, "skipped": "no voltage fragments"}
FB_REPORT_DISABLED = {"skipped": "disabled"}


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
    # ensure a clean slate; individual tests set/unset as needed
    monkeypatch.delenv("SLACK_TOKEN_DSA", raising=False)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # the files.info poll loop sleeps between attempts — never actually
    # wait in tests.
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)


def _cfg(tmp_path: Path, **kw) -> SlackNotifyConfig:
    kw.setdefault("enabled", True)
    kw.setdefault("channel", "C01NUV2M0HM")
    kw.setdefault("post_map_db", str(tmp_path / "posts.db"))
    return SlackNotifyConfig(**kw)


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


def test_config_from_dict_defaults() -> None:
    cfg = SlackNotifyConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.channel == ""
    assert cfg.token_env == "SLACK_TOKEN_DSA"
    assert cfg.token_file == ""
    assert cfg.timeout_s == 10.0
    assert cfg.upload_plot is True
    assert cfg.card is True
    assert cfg.dashboard_base_url == "http://localhost:5778"
    assert cfg.post_map_db == "~/.dsa_monitor/slack_candidate_posts.db"


def test_config_from_dict_matches_yaml_block() -> None:
    # mirrors configs/dsart_search_rt.yaml c3.slack: block
    d = {
        "enabled": True,
        "channel": "C01NUV2M0HM",
        "token_file": "/home/ubuntu/.config/slack_api_dsa110",
        "token_env": "SLACK_TOKEN_DSA",
        "timeout_s": 10.0,
        "upload_plot": True,
        "dashboard_base_url": "http://localhost:5778",
        "post_map_db": "~/.dsa_monitor/slack_candidate_posts.db",
    }
    cfg = SlackNotifyConfig.from_dict(d)
    assert cfg.enabled is True
    assert cfg.channel == "C01NUV2M0HM"
    assert cfg.token_file == "/home/ubuntu/.config/slack_api_dsa110"
    assert cfg.token_env == "SLACK_TOKEN_DSA"
    assert cfg.upload_plot is True
    assert cfg.dashboard_base_url == "http://localhost:5778"
    assert cfg.post_map_db == "~/.dsa_monitor/slack_candidate_posts.db"


def test_c3_config_parses_slack_block(tmp_path: Path) -> None:
    from dsart.services import c3 as c3mod

    cfg_path = tmp_path / "search_rt.yaml"
    cfg_path.write_text(
        "c3:\n"
        "  archive_root: /tmp/cands\n"
        "  slack:\n"
        "    enabled: true\n"
        "    channel: C01NUV2M0HM\n"
        "    token_file: /home/ubuntu/.config/slack_api_dsa110\n"
        "    token_env: SLACK_TOKEN_DSA\n"
        "    timeout_s: 5.0\n"
        "    upload_plot: false\n"
        "    dashboard_base_url: http://localhost:5778\n"
        "    post_map_db: ~/.dsa_monitor/slack_candidate_posts.db\n"
    )
    cfg = c3mod.C3Config.from_yaml(cfg_path)
    assert cfg.slack.enabled is True
    assert cfg.slack.channel == "C01NUV2M0HM"
    assert cfg.slack.token_file == "/home/ubuntu/.config/slack_api_dsa110"
    assert cfg.slack.upload_plot is False
    assert cfg.slack.dashboard_base_url == "http://localhost:5778"
    assert cfg.slack.post_map_db == "~/.dsa_monitor/slack_candidate_posts.db"


# ---------------------------------------------------------------------------
# post_keep_card behaviour
# ---------------------------------------------------------------------------


def test_card_disabled_makes_no_http_calls(monkeypatch, tmp_path) -> None:
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: calls.append("get") or _FakeResp({}))
    notifier = SlackNotifier(_cfg(tmp_path, enabled=False))
    status = notifier.post_keep_card("evt1", C2ROW)
    assert status["ok"] is False
    assert calls == []


def test_card_missing_token_no_raise_no_http(monkeypatch, tmp_path) -> None:
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    monkeypatch.delenv("SLACK_TOKEN_DSA", raising=False)
    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card("evt1", C2ROW)
    assert status["ok"] is False
    assert calls == []


def _fake_upload_and_share(
    calls: List[str], ts: str = "111.222", channel: str = "C01NUV2M0HM",
):
    """A matched set of requests.get/post fakes for the full
    render-upload-poll happy path: getUploadURLExternal -> upload bytes
    -> completeUploadExternal -> files.info (share ts)."""
    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, files=None, timeout=None,
                   **kw):
        if files is not None:
            calls.append("uploadBytes")
            return _FakeResp({}, status=200)
        if url.endswith("files.completeUploadExternal"):
            calls.append("completeUploadExternal")
            posted["complete_json"] = json
            return _FakeResp({"ok": True})
        raise AssertionError(f"unexpected POST {url}")

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        if url.endswith("files.getUploadURLExternal"):
            calls.append("getUploadURLExternal")
            return _FakeResp({"ok": True,
                              "upload_url": "https://files.slack.com/upload/xyz",
                              "file_id": "F123"})
        if url.endswith("files.info"):
            calls.append("files.info")
            return _FakeResp({
                "ok": True,
                "file": {"shares": {"public": {channel: [{"ts": ts}]}}},
            })
        raise AssertionError(f"unexpected GET {url}")

    return _fake_post, _fake_get, posted


def test_card_is_the_message_no_text_first(monkeypatch, tmp_path) -> None:
    """The card image IS the message: no chat.postMessage before/instead
    of the upload on the happy path."""
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake-token-not-real")
    name = "evt_card"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _fake_render_card(ev_dir_arg, name_arg, c2row_arg, out_path,
                          keep_report_arg, mode="cubes"):
        Path(out_path).write_bytes(b"PNG")
        return {"ok": True, "path": str(out_path), "error": None, "panels": []}

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _fake_render_card)

    calls: List[str] = []
    _fake_post, _fake_get, posted = _fake_upload_and_share(calls)
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)

    assert status["ok"] is True
    assert status["ts"] == "111.222"
    assert "chat.postMessage" not in calls
    assert calls == ["getUploadURLExternal", "uploadBytes",
                      "completeUploadExternal", "files.info"]

    complete_json = posted["complete_json"]
    assert complete_json["channel_id"] == "C01NUV2M0HM"
    assert complete_json["files"][0]["title"] == name
    comment = complete_json["initial_comment"]
    # metadata line: name, sigma, DM, UTC, dashboard link — no emoji, no
    # TEST marker.
    assert name in comment
    assert "21.4" in comment
    assert "168.8" in comment
    assert "Open in dashboard" in comment
    assert "http://localhost:5778/bursts/evt_card" in comment
    assert not any(ord(ch) > 0x2100 for ch in comment)
    assert "TEST" not in comment


def test_card_persists_post_map(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_persist"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _fake_render_card(*a, **k):
        out_path = a[3]
        Path(out_path).write_bytes(b"PNG")
        return {"ok": True, "path": str(out_path), "error": None, "panels": []}

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _fake_render_card)

    calls: List[str] = []
    _fake_post, _fake_get, _posted = _fake_upload_and_share(
        calls, ts="777.888")
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    db_path = tmp_path / "posts.db"
    notifier = SlackNotifier(_cfg(tmp_path, post_map_db=str(db_path)))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)
    assert status["ok"] is True

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT event, channel, ts FROM posts WHERE event=?",
            (name,)).fetchone()
    finally:
        conn.close()
    assert row == (name, "C01NUV2M0HM", "777.888")


def test_card_render_failure_falls_back_to_text_message(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_card_fail"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _broken_render_card(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _broken_render_card)

    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        posted.update(json)
        return _FakeResp({"ok": True, "ts": "222.333"})

    monkeypatch.setattr("requests.post", _fake_post)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)
    assert status["ok"] is True
    assert status["ts"] == "222.333"
    assert name in posted["text"]
    assert "Open in dashboard" in posted["text"]


def test_card_upload_failure_falls_back_to_text_message(
    monkeypatch, tmp_path,
) -> None:
    """Render succeeds but getUploadURLExternal fails -> fallback text."""
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_upload_fail"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _fake_render_card(ev_dir_arg, name_arg, c2row_arg, out_path,
                          keep_report_arg, mode="cubes"):
        Path(out_path).write_bytes(b"PNG")
        return {"ok": True, "path": str(out_path), "error": None, "panels": []}

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _fake_render_card)

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        return _FakeResp({"ok": False, "error": "invalid_auth"})

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        return _FakeResp({"ok": True, "ts": "999.111"})

    monkeypatch.setattr("requests.get", _fake_get)
    monkeypatch.setattr("requests.post", _fake_post)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)
    assert status["ok"] is True
    assert status["ts"] == "999.111"


def test_card_no_ev_dir_falls_back_to_text_message(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        posted.update(json)
        return _FakeResp({"ok": True, "ts": "1234.5678"})

    monkeypatch.setattr("requests.post", _fake_post)
    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card("evt1", C2ROW, ev_dir=None)
    assert status["ok"] is True
    assert status["ts"] == "1234.5678"
    text = posted["text"]
    assert "evt1" in text
    assert "21.4" in text
    assert "168.8" in text
    # the token must never be logged/leaked in the payload
    assert "xoxb-fake" not in str(posted)


def test_card_metadata_line_na_fallback_for_missing_values(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        posted.update(json)
        return _FakeResp({"ok": True, "ts": "1.2"})

    monkeypatch.setattr("requests.post", _fake_post)
    notifier = SlackNotifier(_cfg(tmp_path, card=False))
    status = notifier.post_keep_card("evt_na", {}, ev_dir=None)
    assert status["ok"] is True
    text = posted["text"]
    assert "evt_na" in text
    assert "n/a" in text
    assert "Open in dashboard" in text


def test_card_disabled_card_flag_skips_render_goes_straight_to_text(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_no_card"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _should_not_be_called(*a, **k):
        raise AssertionError("render_card should not be called when card=False")

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _should_not_be_called)

    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        posted.update(json)
        return _FakeResp({"ok": True, "ts": "444.555"})

    monkeypatch.setattr("requests.post", _fake_post)
    notifier = SlackNotifier(_cfg(tmp_path, card=False))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)
    assert status["ok"] is True
    assert status["ts"] == "444.555"


def test_card_connection_error_is_caught_not_raised(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")

    def _raise(*a, **k):
        import requests as _r
        raise _r.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr("requests.post", _raise)
    monkeypatch.setattr("requests.get", _raise)
    notifier = SlackNotifier(_cfg(tmp_path, card=False))
    status = notifier.post_keep_card("evt1", C2ROW)
    assert status["ok"] is False
    assert status.get("error")


def test_card_api_error_response_is_not_ok(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResp({"ok": False, "error": "channel_not_found"}))
    notifier = SlackNotifier(_cfg(tmp_path, card=False, channel="Cbad"))
    status = notifier.post_keep_card("evt1", C2ROW)
    assert status["ok"] is False
    assert "channel_not_found" in status["error"]


def test_card_no_share_ts_found_still_ok_but_ts_none(monkeypatch, tmp_path) -> None:
    """files.info never reports a share within the poll window: card
    still posts (ts=None), the followup will just go standalone."""
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_no_share"
    ev_dir = tmp_path / name
    ev_dir.mkdir()

    def _fake_render_card(*a, **k):
        out_path = a[3]
        Path(out_path).write_bytes(b"PNG")
        return {"ok": True, "path": str(out_path), "error": None, "panels": []}

    monkeypatch.setattr(
        "dsart.services.candidate_card.render_card", _fake_render_card)

    def _fake_post(url, headers=None, json=None, files=None, timeout=None,
                   **kw):
        if files is not None:
            return _FakeResp({}, status=200)
        if url.endswith("files.completeUploadExternal"):
            return _FakeResp({"ok": True})
        raise AssertionError(f"unexpected POST {url}")

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        if url.endswith("files.getUploadURLExternal"):
            return _FakeResp({"ok": True,
                              "upload_url": "https://files.slack.com/upload/xyz",
                              "file_id": "F123"})
        if url.endswith("files.info"):
            return _FakeResp({"ok": True, "file": {"shares": {}}})
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    # Fast-forward the poll deadline so this test doesn't burn 15 real
    # seconds spinning on the (mocked-no-op) time.sleep.
    clock = iter([0.0, 100.0, 100.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock, 100.0))

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_card(name, C2ROW, ev_dir=ev_dir)
    assert status["ok"] is True
    assert status["ts"] is None


# ---------------------------------------------------------------------------
# post_keep_followup behaviour
# ---------------------------------------------------------------------------


def _mk_event_with_variants(
    tmp_path: Path, name: str, *, unflagged: bool = True, rfi: bool = True,
) -> Path:
    ev = tmp_path / name
    fb_dir = ev / "filterbank"
    fb_dir.mkdir(parents=True)
    if unflagged:
        (fb_dir / f"{name}.png").write_bytes(b"PNG")
    if rfi:
        (fb_dir / f"{name}_rfi.png").write_bytes(b"PNG")
    return ev


def test_followup_disabled_makes_no_http_calls(monkeypatch, tmp_path) -> None:
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    notifier = SlackNotifier(_cfg(tmp_path, enabled=False))
    status = notifier.post_keep_followup(
        "evt1", tmp_path / "evt1", FB_REPORT_OK, KEEP_REPORT,
        thread_ts="111.222")
    assert status["ok"] is False
    assert calls == []


def test_followup_uploads_both_variants_into_thread(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_followup"
    ev_dir = _mk_event_with_variants(tmp_path, name)
    fb_report = {"ok": True, "outputs": [f"{name}.png", f"{name}_rfi.png"]}

    completes: List[Dict[str, Any]] = []
    calls: List[str] = []

    def _fake_post(url, headers=None, json=None, files=None, timeout=None,
                   **kw):
        if files is not None:
            calls.append("uploadBytes")
            return _FakeResp({}, status=200)
        if url.endswith("files.completeUploadExternal"):
            calls.append("completeUploadExternal")
            completes.append(json)
            assert json["channel_id"] == "C01NUV2M0HM"
            assert json.get("thread_ts") == "111.222"
            return _FakeResp({"ok": True})
        raise AssertionError(f"unexpected POST {url}")

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        calls.append("getUploadURLExternal")
        return _FakeResp({"ok": True,
                          "upload_url": "https://files.slack.com/upload/xyz",
                          "file_id": "F123"})

    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        name, ev_dir, fb_report, KEEP_REPORT, thread_ts="111.222")
    assert status["ok"] is True
    assert sorted(status["uploaded"]) == ["rfi_flagged", "unflagged"]
    assert calls.count("getUploadURLExternal") == 2
    assert calls.count("completeUploadExternal") == 2

    titles = {c["files"][0]["title"] for c in completes}
    comments = {c["initial_comment"] for c in completes}
    assert titles == {f"{name} bbproc - unflagged",
                       f"{name} bbproc - SK RFI-flagged"}
    assert comments == {"bbproc - unflagged", "bbproc - SK RFI-flagged"}


def test_followup_missing_variant_skipped_silently(monkeypatch, tmp_path) -> None:
    """Only the unflagged PNG exists; the RFI-flagged one is missing --
    it's just skipped, not an error."""
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_one_variant"
    ev_dir = _mk_event_with_variants(tmp_path, name, unflagged=True, rfi=False)
    fb_report = {"ok": True, "outputs": [f"{name}.png"]}

    calls: List[str] = []

    def _fake_post(url, headers=None, json=None, files=None, timeout=None,
                   **kw):
        if files is not None:
            calls.append("uploadBytes")
            return _FakeResp({}, status=200)
        if url.endswith("files.completeUploadExternal"):
            calls.append("completeUploadExternal")
            return _FakeResp({"ok": True})
        raise AssertionError(f"unexpected POST {url}")

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        calls.append("getUploadURLExternal")
        return _FakeResp({"ok": True,
                          "upload_url": "https://files.slack.com/upload/xyz",
                          "file_id": "F123"})

    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        name, ev_dir, fb_report, KEEP_REPORT, thread_ts="111.222")
    assert status["ok"] is True
    assert status["uploaded"] == ["unflagged"]
    assert status["failed"] == []
    assert calls.count("getUploadURLExternal") == 1


def test_followup_posts_standalone_when_card_thread_ts_is_none(
    monkeypatch, tmp_path,
) -> None:
    """If the card post failed, thread_ts is None — the followup upload
    must still land, just as a fresh top-level message rather than a
    threaded reply (no `thread_ts` in the completeUploadExternal call)."""
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_standalone"
    ev_dir = _mk_event_with_variants(tmp_path, name)
    fb_report = {"ok": True, "outputs": [f"{name}.png", f"{name}_rfi.png"]}

    seen_payloads: List[Dict[str, Any]] = []

    def _fake_post(url, headers=None, json=None, files=None, timeout=None,
                   **kw):
        if files is not None:
            return _FakeResp({}, status=200)
        if url.endswith("files.completeUploadExternal"):
            seen_payloads.append(json)
            return _FakeResp({"ok": True})
        raise AssertionError(f"unexpected POST {url}")

    def _fake_get(url, headers=None, params=None, timeout=None, **kw):
        return _FakeResp({"ok": True,
                          "upload_url": "https://files.slack.com/upload/xyz",
                          "file_id": "F123"})

    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("requests.get", _fake_get)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        name, ev_dir, fb_report, KEEP_REPORT, thread_ts=None)
    assert status["ok"] is True
    assert sorted(status["uploaded"]) == ["rfi_flagged", "unflagged"]
    assert all("thread_ts" not in p for p in seen_payloads)


def test_followup_disabled_filterbank_skips_silently(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: calls.append("get") or _FakeResp({}))

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        "evt1", tmp_path / "evt1", FB_REPORT_DISABLED, KEEP_REPORT,
        thread_ts="111.222")
    assert status["ok"] is True
    assert status.get("skipped") == "disabled"
    assert calls == []


def test_followup_upload_plot_false_skips_silently(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: calls.append("get") or _FakeResp({}))
    name = "evt_no_upload"
    ev_dir = _mk_event_with_variants(tmp_path, name)

    notifier = SlackNotifier(_cfg(tmp_path, upload_plot=False))
    status = notifier.post_keep_followup(
        name, ev_dir, FB_REPORT_OK, KEEP_REPORT, thread_ts="111.222")
    assert status["ok"] is True
    assert status.get("skipped") == "upload disabled"
    assert calls == []


def test_followup_both_missing_and_ok_skips_silently(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    ev_dir = tmp_path / "evt_no_plot"
    ev_dir.mkdir()
    # ok run, but no PNGs at all (e.g. .fil only)
    fb_report_ok_no_png = {"ok": True, "outputs": ["evt_no_plot.fil"]}

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        "evt_no_plot", ev_dir, fb_report_ok_no_png, KEEP_REPORT,
        thread_ts="111.222")
    assert status["ok"] is True
    assert status.get("skipped") == "no plot"
    assert calls == []


def test_followup_both_missing_and_failed_posts_brief_thread_note(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    ev_dir = tmp_path / "evt_fb_failed"
    ev_dir.mkdir()
    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        assert url.endswith("chat.postMessage")
        posted.update(json)
        return _FakeResp({"ok": True, "ts": "333.444"})

    monkeypatch.setattr("requests.post", _fake_post)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        "evt_fb_failed", ev_dir, FB_REPORT_FAILED, KEEP_REPORT,
        thread_ts="111.222")
    assert status["ok"] is True
    assert posted["thread_ts"] == "111.222"
    assert "failed" in posted["text"]
    assert "14/16" in posted["text"]


def test_followup_connection_error_is_caught_not_raised(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    name = "evt_conn_err"
    ev_dir = _mk_event_with_variants(tmp_path, name)
    fb_report = {"ok": True, "outputs": [f"{name}.png", f"{name}_rfi.png"]}

    def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("requests.get", _raise)

    notifier = SlackNotifier(_cfg(tmp_path))
    status = notifier.post_keep_followup(
        name, ev_dir, fb_report, KEEP_REPORT, thread_ts="111.222")
    # both uploads failed but must not raise; ok reflects the failure
    assert status["ok"] is False


# ---------------------------------------------------------------------------
# C3Service._do_keep integration: card-then-followup ordering + the
# injection guard (injections always KEEP per cube_veto.decide, so
# _do_keep IS reached for them — C3 must still never post to Slack).
# ---------------------------------------------------------------------------


class _KeepDecision:
    action = "KEEP"
    keep = True
    rules_fired: tuple = ()
    notes = "stub"


class _Metrics:
    def __init__(self) -> None:
        self.ok = True


class FakeMonStore:
    def get_dict(self, key):
        return None

    def put_dict(self, key, value):
        pass


def _seed_keep_event(archive_root: Path, name: str) -> Path:
    ev_dir = archive_root / name
    (ev_dir / "Level3").mkdir(parents=True, exist_ok=True)
    (ev_dir / "Level3" / f"{name}.json").write_text(
        '{"c2": {"snr_max": 12.3, "dm_median": 100.0, '
        '"t_peak_mjd": 61236.5}}')
    return ev_dir


def _make_c3_service(tmp_path: Path, *, is_inj: bool):
    from dsart.services import c3 as c3mod

    cfg = c3mod.C3Config(
        archive_root=tmp_path / "candidates",
        rejected_root=tmp_path / "candidates_rejected",
        state_path=tmp_path / "state.json",
        fired_injection_log=None,
        corr_nodes={},              # no fragments to collect -> no broadcaster
        flag_only=True,
        slack=_cfg(tmp_path, card=False),  # skip render machinery in this test
    )
    svc = c3mod.C3Service(cfg, mon_store=FakeMonStore())
    return svc, cfg


def test_do_keep_posts_card_before_followup_and_threads_ts(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("SLACK_TOKEN_DSA", "xoxb-fake")
    svc, cfg = _make_c3_service(tmp_path, is_inj=False)

    from dsart.services import c3 as c3mod
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: False)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: _Metrics())
    monkeypatch.setattr(c3mod, "decide", lambda *a, **k: _KeepDecision())

    name = "260101keep"
    _seed_keep_event(cfg.archive_root, name)

    order: List[str] = []
    posted: Dict[str, Any] = {}

    def _fake_post(url, headers=None, json=None, timeout=None, **kw):
        if url.endswith("chat.postMessage"):
            order.append("card" if "card" not in posted else "followup")
            posted.setdefault("card", json)
            return _FakeResp({"ok": True, "ts": "555.666"})
        raise AssertionError(f"unexpected POST {url}")

    monkeypatch.setattr("requests.post", _fake_post)

    rec = svc.process_event(name)
    assert rec["action"] == "KEEP"
    assert rec["slack"]["card"]["ok"] is True
    assert rec["slack"]["card"]["ts"] == "555.666"
    # filterbank is disabled by default -> followup has nothing to post,
    # skips silently, but must still report ok (never a failure).
    assert rec["slack"]["followup"]["ok"] is True
    assert rec["slack"]["followup"].get("skipped") == "disabled"
    assert order == ["card"]
    assert svc._counters["slack_ok"] == 1
    assert svc._counters["slack_failed"] == 0
    assert svc._counters["slack_followup_ok"] == 1
    assert svc._counters["slack_followup_failed"] == 0


def test_do_keep_injection_makes_zero_http_calls(monkeypatch, tmp_path) -> None:
    svc, cfg = _make_c3_service(tmp_path, is_inj=True)
    from dsart.services import c3 as c3mod
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: True)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: _Metrics())
    monkeypatch.setattr(c3mod, "decide", lambda *a, **k: _KeepDecision())

    name = "260101inj"
    _seed_keep_event(cfg.archive_root, name)

    calls: List[str] = []
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: calls.append("post") or _FakeResp({}))
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: calls.append("get") or _FakeResp({}))

    rec = svc.process_event(name)
    assert rec["action"] == "KEEP"
    assert rec["is_injection"] is True
    assert calls == []
    assert rec["slack"]["card"]["error"] == "injection"
    assert rec["slack"]["followup"]["error"] == "injection"
    # Counters only move when a post is actually attempted — injections
    # never attempt, so neither ok nor failed should increment.
    assert svc._counters["slack_ok"] == 0
    assert svc._counters["slack_failed"] == 0
    assert svc._counters["slack_followup_ok"] == 0
    assert svc._counters["slack_followup_failed"] == 0

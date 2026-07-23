"""Website -> Slack annotation relay (best-effort, standalone process).

The dsa_monitor dashboard lets operators tag a burst event with a
classification (RFI / NOISE / PULSAR / FRB / INJECTION, or clear a
previous tag) — those live in a small SQLite "annotations" DB the
dashboard writes to. This module polls that DB for new/changed/cleared
tags and relays each one as a threaded reply on the event's existing
Slack card message (the one :mod:`dsart.services.slack_notify` posted
at C3 KEEP time), so ops following the Slack channel sees classification
activity without needing the dashboard open.

Runs as its own systemd unit (``systemd/dsart_slack_relay.service``),
entirely decoupled from C3/the KEEP path — a relay failure (missing
token, DB unreadable, Slack API error) never affects C3's veto decision
or archive; every failure mode here is caught, logged, and simply
skipped for that cycle.

Polling is **mtime-gated**: each cycle only ``os.stat()``s the
annotations DB (and its ``-wal`` sibling, if present under WAL mode) and
opens/queries SQLite only when one of those changed since the last
check — plus one unconditional full query every ``fallback_sweep_s``
seconds as a safety net against a missed mtime update (e.g. a write that
lands between two stats with an identical size, or a filesystem with
coarse mtime resolution). The "last seen" stat signature is tracked in
memory only (not persisted) — a process restart just re-queries once
via the fallback-sweep path, which is harmless (``last_seen_id`` still
prevents any row from being relayed twice).

Run standalone: ``python -m dsart.services.slack_annotation_relay
--config configs/dsart_search_rt.yaml`` (reads the top-level
``slack_relay:`` block).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

LOG = logging.getLogger("dsart.services.slack_annotation_relay")

_SLACK_API = "https://slack.com/api"

#: Slack attachment bar color per classification label; anything not
#: listed here (including an unrecognised/typo'd label) falls back to
#: the same neutral gray used for "cleared".
LABEL_COLORS: Dict[str, str] = {
    "RFI": "#c0392b",
    "NOISE": "#7f8c8d",
    "PULSAR": "#2980b9",
    "FRB": "#27ae60",
    "INJECTION": "#e67e22",
}
_DEFAULT_COLOR = "#7f8c8d"


@dataclass(frozen=True)
class RelayConfig:
    enabled: bool = False
    channel: str = ""
    token_file: str = ""
    token_env: str = "SLACK_TOKEN_DSA110"
    annotations_db: str = "~/.dsa_monitor/annotations.db"
    post_map_db: str = "~/.dsa_monitor/slack_candidate_posts.db"
    state_path: str = "~/.dsa_monitor/slack_relay_state.json"
    poll_s: float = 15.0
    timeout_s: float = 10.0
    #: Labels that get `reply_broadcast: true` (surfaced in the main
    #: channel, not just the thread). ``["*"]`` broadcasts every label;
    #: an explicit list broadcasts only those. Default is FRB-only so
    #: routine RFI/NOISE/etc. classification chatter stays thread-only.
    broadcast_labels: Tuple[str, ...] = ("FRB",)
    #: Safety-net unconditional full query interval (seconds), in case
    #: an mtime-based change is ever missed. See module docstring.
    fallback_sweep_s: float = 600.0

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "RelayConfig":
        d = d or {}
        labels = d.get("broadcast_labels", ["FRB"])
        return cls(
            enabled=bool(d.get("enabled", False)),
            channel=str(d.get("channel", "")),
            token_file=str(d.get("token_file", "")),
            token_env=str(d.get("token_env", "SLACK_TOKEN_DSA110")),
            annotations_db=str(
                d.get("annotations_db", "~/.dsa_monitor/annotations.db")),
            post_map_db=str(
                d.get("post_map_db",
                      "~/.dsa_monitor/slack_candidate_posts.db")),
            state_path=str(
                d.get("state_path", "~/.dsa_monitor/slack_relay_state.json")),
            poll_s=float(d.get("poll_s", 15.0)),
            timeout_s=float(d.get("timeout_s", 10.0)),
            broadcast_labels=tuple(str(x) for x in (labels or [])),
            fallback_sweep_s=float(d.get("fallback_sweep_s", 600.0)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "RelayConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return cls.from_dict(doc.get("slack_relay"))

    def is_broadcast(self, label: Optional[str]) -> bool:
        if "*" in self.broadcast_labels:
            return True
        if label is None:
            return False
        return label in self.broadcast_labels


def _phrase_and_text(
    user: str, label: Optional[str], *, seen_before: bool,
    prev_label: Optional[str],
) -> Tuple[str, str, str]:
    """Returns ``(fallback, text, color)`` for the attachment. See the
    module docstring / brief for the exact phrasing rules."""
    if label is None:
        text = f"*{user}* cleared their classification"
        fallback = f"{user} cleared their classification"
        return fallback, text, _DEFAULT_COLOR

    if seen_before and prev_label is not None and prev_label != label:
        phrase = "changed their classification to"
    else:
        phrase = "classified this as"
    text = f"*{user}* {phrase} `{label}`"
    fallback = f"{user} {phrase} {label}"
    color = LABEL_COLORS.get(label, _DEFAULT_COLOR)
    return fallback, text, color


class SlackAnnotationRelay:
    """Best-effort poller relaying dashboard classification annotations
    into the corresponding event's Slack thread."""

    def __init__(self, config: RelayConfig) -> None:
        self._cfg = config
        self._warned_no_token = False
        self._last_seen_id = 0
        #: {"event::user": label_or_None} — the last label relayed for
        #: that (event, user) pair, used to distinguish "first tag" from
        #: "changed" from "reposted unchanged".
        self._last_label: Dict[str, Optional[str]] = {}
        self._load_state()
        # In-memory only (see module docstring) — a fresh process always
        # starts "changed" so the first poll_once() after startup runs a
        # real query rather than trusting a stale signature.
        self._last_sig: Optional[Tuple[Any, ...]] = None
        self._last_sweep_at: Optional[float] = None

    # ----- state persistence ---------------------------------------------

    def _state_path(self) -> Path:
        return Path(self._cfg.state_path).expanduser()

    def _load_state(self) -> None:
        try:
            p = self._state_path()
            if not p.is_file():
                return
            doc = json.loads(p.read_text())
            self._last_seen_id = int(doc.get("last_seen_id", 0))
            self._last_label = dict(doc.get("last_label", {}))
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.warning("slack_relay: failed to load state (%s): %s",
                        self._cfg.state_path, exc)

    def _save_state(self) -> None:
        try:
            p = self._state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps({
                "last_seen_id": self._last_seen_id,
                "last_label": self._last_label,
            }))
            tmp.replace(p)
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.warning("slack_relay: failed to save state (%s): %s",
                        self._cfg.state_path, exc)

    # ----- token -----------------------------------------------------------

    def _token(self) -> Optional[str]:
        if self._cfg.token_file:
            try:
                tok = Path(self._cfg.token_file).read_text().strip()
            except OSError as exc:
                LOG.warning(
                    "slack_relay: token_file %s unreadable (%s); falling "
                    "back to env var %s",
                    self._cfg.token_file, exc, self._cfg.token_env)
                tok = ""
            if tok:
                return tok
        tok = os.environ.get(self._cfg.token_env)
        if not tok:
            if not self._warned_no_token:
                LOG.warning(
                    "slack_relay: neither token_file (%s) nor env var %s "
                    "yielded a token; relay disabled (best-effort, not "
                    "fatal)", self._cfg.token_file or "(unset)",
                    self._cfg.token_env)
                self._warned_no_token = True
            return None
        return tok

    # ----- mtime-gated polling ---------------------------------------------

    def _stat_signature(self) -> Tuple[Any, ...]:
        """(path, mtime_ns, size) for the annotations DB and its ``-wal``
        sibling (present under SQLite WAL mode) — ``(path, None, None)``
        for whichever doesn't exist. Comparable/hashable so it can be
        diffed cheaply between cycles without opening the DB."""
        db_path = Path(self._cfg.annotations_db).expanduser()
        sigs: List[Tuple[str, Optional[int], Optional[int]]] = []
        for p in (db_path, db_path.with_name(db_path.name + "-wal")):
            try:
                st = p.stat()
                sigs.append((str(p), st.st_mtime_ns, st.st_size))
            except OSError:
                sigs.append((str(p), None, None))
        return tuple(sigs)

    def poll_once(self, now: Optional[float] = None) -> Dict[str, Any]:
        """One relay cycle: cheaply checks whether the annotations DB
        changed (or a fallback sweep is due) and, only then, actually
        queries it. Always returns a summary dict, never raises."""
        try:
            return self._poll_once(now)
        except Exception as exc:  # noqa: BLE001 — must never break the loop
            LOG.exception("slack_relay: unexpected failure in poll cycle")
            return {"queried": False, "relayed": 0,
                     "error": f"{type(exc).__name__}: {exc}"}

    def _poll_once(self, now: Optional[float]) -> Dict[str, Any]:
        if not self._cfg.enabled:
            return {"queried": False, "relayed": 0, "skipped": "disabled"}

        clock_now = time.monotonic() if now is None else now
        sig = self._stat_signature()
        due_sweep = (
            self._last_sweep_at is None
            or (clock_now - self._last_sweep_at) >= self._cfg.fallback_sweep_s
        )
        changed = sig != self._last_sig
        self._last_sig = sig

        if not changed and not due_sweep:
            return {"queried": False, "relayed": 0}

        if due_sweep:
            self._last_sweep_at = clock_now

        return self._query_and_relay()

    # ----- query + relay ----------------------------------------------------

    def _query_and_relay(self) -> Dict[str, Any]:
        token = self._token()
        if not token:
            return {"queried": True, "relayed": 0, "error": "no token"}

        rows = self._fetch_new_rows()
        relayed = 0
        unmapped = 0
        max_id = self._last_seen_id
        for row_id, event, user, label, _ts_utc in rows:
            max_id = max(max_id, row_id)
            mapping = self._lookup_post_map(event)
            if mapping is None:
                LOG.info(
                    "slack_relay: event %s has no post-map entry — "
                    "skipping relay for %s's %r tag", event, user, label)
                unmapped += 1
                continue
            channel, thread_ts = mapping
            ok = self._relay_one(token, channel, thread_ts, event, user, label)
            if ok:
                relayed += 1

        self._last_seen_id = max_id
        self._save_state()
        return {"queried": True, "relayed": relayed, "unmapped": unmapped,
                "n_rows": len(rows)}

    def _fetch_new_rows(
        self,
    ) -> List[Tuple[int, str, str, Optional[str], str]]:
        db_path = Path(self._cfg.annotations_db).expanduser()
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.execute(
                    "SELECT id, event, user, label, ts_utc FROM "
                    "classifications WHERE id > ? ORDER BY id",
                    (self._last_seen_id,),
                )
                return list(cur.fetchall())
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.warning("slack_relay: failed to query annotations db (%s): %s",
                        self._cfg.annotations_db, exc)
            return []

    def _lookup_post_map(self, event: str) -> Optional[Tuple[str, str]]:
        db_path = Path(self._cfg.post_map_db).expanduser()
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.execute(
                    "SELECT channel, ts FROM posts WHERE event = ?", (event,))
                row = cur.fetchone()
                return (row[0], row[1]) if row else None
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.warning("slack_relay: failed to query post map (%s): %s",
                        self._cfg.post_map_db, exc)
            return None

    def _relay_one(
        self, token: str, channel: str, thread_ts: str, event: str,
        user: str, label: Optional[str],
    ) -> bool:
        key = f"{event}::{user}"
        seen_before = key in self._last_label
        prev_label = self._last_label.get(key)

        fallback, text, color = _phrase_and_text(
            user, label, seen_before=seen_before, prev_label=prev_label)
        broadcast = self._cfg.is_broadcast(label)

        payload = {
            "channel": channel,
            "thread_ts": thread_ts,
            "reply_broadcast": broadcast,
            "text": "",
            "attachments": [{
                "color": color,
                "fallback": fallback,
                "text": text,
                "mrkdwn_in": ["text"],
            }],
        }
        resp = self._api_post("chat.postMessage", token, payload)
        self._last_label[key] = label
        if resp is None or not resp.get("ok"):
            err = (resp or {}).get("error", "postMessage failed") if resp \
                else "postMessage failed"
            LOG.warning("slack_relay %s/%s: chat.postMessage failed: %s",
                        event, user, err)
            return False
        return True

    def _api_post(
        self, method: str, token: str, payload: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            import requests
            r = requests.post(
                f"{_SLACK_API}/{method}",
                headers={"Authorization": f"Bearer {token}"},
                json=dict(payload),
                timeout=self._cfg.timeout_s,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("slack_relay: %s request failed: %s", method, exc)
            return None

    # ----- run loop ---------------------------------------------------------

    def run_forever(self, stop_event: Optional[Any] = None) -> None:
        """Poll in a loop at ``poll_s`` cadence until ``stop_event`` is
        set (or a SIGTERM/SIGINT is received when run via ``main()``).
        Every cycle is independently best-effort — an exception in one
        poll never aborts the loop."""
        import threading
        stop_event = stop_event or threading.Event()
        LOG.info("slack_relay: starting poll loop (poll_s=%.1f)",
                 self._cfg.poll_s)
        while not stop_event.is_set():
            summary = self.poll_once()
            if summary.get("queried"):
                LOG.info("slack_relay: cycle summary: %s", summary)
            stop_event.wait(self._cfg.poll_s)
        LOG.info("slack_relay: stopped")


def main(argv: Optional[List[str]] = None) -> int:
    import threading

    parser = argparse.ArgumentParser(
        description="Website -> Slack annotation relay")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = RelayConfig.from_yaml(args.config)
    relay = SlackAnnotationRelay(cfg)

    stop_event = threading.Event()

    def _handle_sigterm(signum: int, frame: Any) -> None:
        LOG.info("slack_relay: received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    relay.run_forever(stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

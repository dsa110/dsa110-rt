#!/usr/bin/env python3
"""Human-annotation (classification tagging) storage for the burst dashboard.

No-login human classification of burst events: anyone on the page can
tag an event, and per (event, user) the latest click wins. This module
is the **storage layer only** — pure functions over a SQLite database,
with *no* Flask imports, so it is trivially unit-testable against a
throwaway ``tmp_path`` DB.

Durability
----------
The DB lives **outside** the repo and **outside**
``/dataz/dsa110/candidates`` (archive dirs get pruned; annotations must
outlive them). Default path ``~/.dsa_monitor/annotations.db``,
overridable via ``$DSA_MONITOR_ANNOT_DB``. The directory + schema are
created lazily on first use. WAL mode + a short busy-timeout keeps
concurrent web workers safe; every public call opens its own
short-lived connection.

Schema (append-only audit trail; "current" = latest row)
--------------------------------------------------------
* ``users(name PK, added_utc)`` — people who clicked "add user".
* ``classifications(id PK, event, user, label NULL, ts_utc)`` —
  append-only. ``label=NULL`` records "cleared my classification".
  The current classification for (event, user) is the latest row.
* ``source_names(id PK, event, source_name NULL, user, ts_utc)`` —
  append-only. The source name is **event-level**: the one canonical
  current value for an event is the latest row (NULL = cleared), but
  every edit records who set it.
* ``custom_tags(tag PK, created_by, created_utc)`` — shared vocabulary;
  once created a tag is a valid label for every event, for everyone.

All timestamps are UTC ISO-8601.

Labels
------
Built-ins ``FRB, RFI, NOISE, PULSAR, INJECTION,
CHECK_OFFLINE_VOLTAGES`` (uppercase) plus every
custom tag. Custom tags/labels normalise to: strip → uppercase →
spaces-to-underscores → keep only ``[A-Z0-9_+-]`` → truncate to 24
chars. Empties and collisions with a built-in are rejected.

Classify does **not** auto-create tags: a custom label must have been
created via :func:`create_tag` first (the UI's "+ tag" affordance does
exactly that before applying). This keeps the shared vocabulary
deliberate rather than letting typos accrete as labels.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILTIN_LABELS: tuple[str, ...] = (
    "FRB", "RFI", "NOISE", "PULSAR", "INJECTION",
    # Parking state for ambiguous events: "needs offline voltage
    # analysis before a verdict" — users reclassify later.
    "CHECK_OFFLINE_VOLTAGES",
)
_BUILTIN_SET = frozenset(BUILTIN_LABELS)

MAX_TAG_LEN = 24
_TAG_STRIP_RE = re.compile(r"[^A-Z0-9_+\-]")

DEFAULT_DB_PATH = "~/.dsa_monitor/annotations.db"


class UnknownUserError(ValueError):
    """Raised when a classify/tag/source op names a user that has not
    been added. The Flask layer maps this to HTTP 400."""


# ---------------------------------------------------------------------------
# Normalisation helpers (pure)
# ---------------------------------------------------------------------------


def normalize_user(name: Optional[str]) -> str:
    """Strip a user name; reject empty. Case is preserved (uniqueness is
    enforced case-insensitively by :func:`add_user`)."""
    n = (name or "").strip()
    if not n:
        raise ValueError("user name is empty")
    return n


def normalize_tag(tag: Optional[str]) -> str:
    """Normalise a custom tag / label to the canonical vocabulary form.

    strip → uppercase → spaces→underscores → keep only ``[A-Z0-9_+-]``
    → truncate to :data:`MAX_TAG_LEN`. Raises ``ValueError`` if the
    result is empty.
    """
    t = (tag or "").strip().upper().replace(" ", "_")
    t = _TAG_STRIP_RE.sub("", t)
    t = t[:MAX_TAG_LEN]
    if not t:
        raise ValueError("tag is empty after normalisation")
    return t


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Path resolution + connection management
# ---------------------------------------------------------------------------


def resolve_db_path(db_path: Optional[str] = None) -> str:
    """Resolve the DB path: explicit arg → ``$DSA_MONITOR_ANNOT_DB`` →
    :data:`DEFAULT_DB_PATH`. ``~`` is expanded."""
    p = db_path or os.environ.get("DSA_MONITOR_ANNOT_DB") or DEFAULT_DB_PATH
    return os.path.abspath(os.path.expanduser(p))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name       TEXT PRIMARY KEY,
    added_utc  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS classifications (
    id      INTEGER PRIMARY KEY,
    event   TEXT NOT NULL,
    user    TEXT NOT NULL,
    label   TEXT,            -- NULL = "cleared my classification"
    ts_utc  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_names (
    id           INTEGER PRIMARY KEY,
    event        TEXT NOT NULL,
    source_name  TEXT,       -- NULL = "cleared"
    user         TEXT NOT NULL,
    ts_utc       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS custom_tags (
    tag          TEXT PRIMARY KEY,
    created_by   TEXT NOT NULL,
    created_utc  TEXT NOT NULL
);
-- Graveyard for purged source names (typo cleanup). Rows keep their
-- original columns (original source_names id in orig_id) plus who
-- purged them and when, so a purge is auditable and reversible by
-- hand. CREATE TABLE IF NOT EXISTS means existing live DBs upgrade
-- transparently on first use.
CREATE TABLE IF NOT EXISTS source_names_purged (
    id           INTEGER PRIMARY KEY,
    orig_id      INTEGER NOT NULL,
    event        TEXT NOT NULL,
    source_name  TEXT,
    user         TEXT NOT NULL,
    ts_utc       TEXT NOT NULL,
    purged_by    TEXT NOT NULL,
    purged_utc   TEXT NOT NULL
);
-- Operator-entered refined localizations (offline/outrigger follow-up).
-- Append-only audit + current-value semantics like classifications:
-- last-write-wins, the single current row per event has active=1; a
-- 'clear' appends an audit row with NULL ra/dec and leaves no active
-- row. Each row also snapshots the pipeline position as of entry time
-- (pipe_*), so the DB backup is self-contained with both positions.
-- CREATE TABLE IF NOT EXISTS upgrades existing live DBs transparently
-- on first use (same precedent as source_names_purged).
CREATE TABLE IF NOT EXISTS event_positions (
    id              INTEGER PRIMARY KEY,
    event           TEXT NOT NULL,
    ra_deg          REAL,      -- NULL = 'cleared' audit row
    dec_deg         REAL,
    ra_err_arcsec   REAL,
    dec_err_arcsec  REAL,
    method          TEXT,
    username        TEXT NOT NULL,
    created_utc     TEXT NOT NULL,
    active          INTEGER NOT NULL DEFAULT 0,
    pipe_ra_deg     REAL,
    pipe_dec_deg    REAL,
    pipe_source     TEXT
);
CREATE INDEX IF NOT EXISTS ix_class_event      ON classifications(event);
CREATE INDEX IF NOT EXISTS ix_class_event_user ON classifications(event, user);
CREATE INDEX IF NOT EXISTS ix_source_event     ON source_names(event);
CREATE INDEX IF NOT EXISTS ix_source_name      ON source_names(source_name);
CREATE INDEX IF NOT EXISTS ix_pos_event        ON event_positions(event);
CREATE INDEX IF NOT EXISTS ix_pos_active       ON event_positions(active);
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


@contextlib.contextmanager
def _conn(db_path: Optional[str] = None):
    """Short-lived connection context manager. Creates the parent dir +
    schema lazily, enables WAL, commits on clean exit."""
    path = resolve_db_path(db_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _init_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _event_ok(event: Optional[str]) -> str:
    e = (event or "").strip()
    if not e:
        raise ValueError("event is empty")
    return e


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _canonical_user(conn: sqlite3.Connection, name: str) -> Optional[str]:
    row = conn.execute(
        "SELECT name FROM users WHERE lower(name) = lower(?)", (name,)
    ).fetchone()
    return row["name"] if row else None


def add_user(name: Optional[str], db_path: Optional[str] = None) -> str:
    """Add a user (idempotent, unique case-insensitively). Returns the
    canonical stored name (the first-seen casing wins)."""
    n = normalize_user(name)
    with _conn(db_path) as conn:
        existing = _canonical_user(conn, n)
        if existing is not None:
            return existing
        conn.execute(
            "INSERT INTO users(name, added_utc) VALUES (?, ?)", (n, _now_iso())
        )
        return n


def list_users(db_path: Optional[str] = None) -> List[str]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM users ORDER BY added_utc, name"
        ).fetchall()
    return [r["name"] for r in rows]


# ---------------------------------------------------------------------------
# Custom tags
# ---------------------------------------------------------------------------


def create_tag(
    tag: Optional[str], user: Optional[str], db_path: Optional[str] = None
) -> str:
    """Create a shared custom tag. Returns the normalised tag.

    Rejects empties and collisions with a built-in label. Idempotent for
    an already-existing custom tag. The creating user must exist.
    """
    t = normalize_tag(tag)
    if t in _BUILTIN_SET:
        raise ValueError(f"{t!r} is already a built-in label")
    u = normalize_user(user)
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        conn.execute(
            "INSERT OR IGNORE INTO custom_tags(tag, created_by, created_utc) "
            "VALUES (?, ?, ?)",
            (t, canon, _now_iso()),
        )
    return t


def list_custom_tags(db_path: Optional[str] = None) -> List[str]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT tag FROM custom_tags ORDER BY created_utc, tag"
        ).fetchall()
    return [r["tag"] for r in rows]


def _valid_labels(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT tag FROM custom_tags").fetchall()
    return set(_BUILTIN_SET) | {r["tag"] for r in rows}


# ---------------------------------------------------------------------------
# Classifications
# ---------------------------------------------------------------------------


def classify(
    event: Optional[str],
    user: Optional[str],
    label: Optional[str],
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a classification for (event, user). ``label=None`` clears.

    A non-null label must be a built-in or a pre-created custom tag
    (unknown labels are rejected — see the module docstring). Returns
    the updated current-annotations block for the event.
    """
    e = _event_ok(event)
    u = normalize_user(user)
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        norm_label: Optional[str]
        if label is None:
            norm_label = None
        else:
            norm_label = normalize_tag(label)
            if norm_label not in _valid_labels(conn):
                raise ValueError(
                    f"unknown label {norm_label!r}; create it as a custom "
                    f"tag first"
                )
        conn.execute(
            "INSERT INTO classifications(event, user, label, ts_utc) "
            "VALUES (?, ?, ?, ?)",
            (e, canon, norm_label, _now_iso()),
        )
        return _event_current(conn, e)


# ---------------------------------------------------------------------------
# Source names (event-level)
# ---------------------------------------------------------------------------


def set_source(
    event: Optional[str],
    user: Optional[str],
    source_name: Optional[str],
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Set or clear the event-level source name. Empty / None clears.

    Source names are stored as typed (stripped) — they are proper names
    (e.g. ``B1913+16``) and are *not* upper-cased. Returns the updated
    current-annotations block for the event.
    """
    e = _event_ok(event)
    u = normalize_user(user)
    src = (source_name or "").strip() or None
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        conn.execute(
            "INSERT INTO source_names(event, source_name, user, ts_utc) "
            "VALUES (?, ?, ?, ?)",
            (e, src, canon, _now_iso()),
        )
        return _event_current(conn, e)


def purge_source_name(
    name: Optional[str],
    user: Optional[str],
    db_path: Optional[str] = None,
) -> int:
    """Purge a source name from the vocabulary (typo cleanup).

    Moves EVERY ``source_names`` row whose name matches
    (case-insensitively) into ``source_names_purged`` (stamped with who
    purged and when), then deletes them. The name vanishes from the
    type-ahead vocabulary and from every event's current source; each
    affected event's current source falls back to its latest
    *surviving* row (an older different name, or nothing).

    Returns the number of rows purged. Unknown user or empty name
    raises (``UnknownUserError`` / ``ValueError``).
    """
    n = (name or "").strip()
    if not n:
        raise ValueError("source name is empty")
    u = normalize_user(user)
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        now = _now_iso()
        cur = conn.execute(
            "INSERT INTO source_names_purged"
            " (orig_id, event, source_name, user, ts_utc,"
            "  purged_by, purged_utc)"
            " SELECT id, event, source_name, user, ts_utc, ?, ?"
            " FROM source_names WHERE lower(source_name) = lower(?)",
            (canon, now, n),
        )
        n_purged = cur.rowcount
        conn.execute(
            "DELETE FROM source_names WHERE lower(source_name) = lower(?)",
            (n,),
        )
        return n_purged


# ---------------------------------------------------------------------------
# Current-state readers
# ---------------------------------------------------------------------------

_CURRENT_CLASS_SQL = """
SELECT c.event AS event, c.user AS user, c.label AS label, c.ts_utc AS ts_utc
FROM classifications c
JOIN (
    SELECT event, user, MAX(id) AS mid
    FROM classifications
    GROUP BY event, user
) m ON c.id = m.mid
"""

_CURRENT_SOURCE_SQL = """
SELECT s.event AS event, s.source_name AS source_name,
       s.user AS user, s.ts_utc AS ts_utc
FROM source_names s
JOIN (
    SELECT event, MAX(id) AS mid
    FROM source_names
    GROUP BY event
) m ON s.id = m.mid
"""


def _current_classifications(
    conn: sqlite3.Connection, event: Optional[str] = None
) -> Dict[str, List[dict]]:
    """event → list of current {user, label, ts_utc} (label may be None
    for a user who cleared)."""
    sql = _CURRENT_CLASS_SQL
    params: tuple = ()
    if event is not None:
        sql += " WHERE c.event = ?"
        params = (event,)
    out: Dict[str, List[dict]] = {}
    for r in conn.execute(sql, params).fetchall():
        out.setdefault(r["event"], []).append(
            {"user": r["user"], "label": r["label"], "ts_utc": r["ts_utc"]}
        )
    return out


def _current_sources(
    conn: sqlite3.Connection, event: Optional[str] = None
) -> Dict[str, dict]:
    """event → current {source_name, user, ts_utc} (source_name may be
    None for a cleared event)."""
    sql = _CURRENT_SOURCE_SQL
    params: tuple = ()
    if event is not None:
        sql += " WHERE s.event = ?"
        params = (event,)
    out: Dict[str, dict] = {}
    for r in conn.execute(sql, params).fetchall():
        out[r["event"]] = {
            "source_name": r["source_name"],
            "user": r["user"],
            "ts_utc": r["ts_utc"],
        }
    return out


def _event_current(conn: sqlite3.Connection, event: str) -> Dict[str, Any]:
    """The current-annotations block for one event (the POST return
    shape). Excludes users whose current label is NULL (cleared)."""
    classes = _current_classifications(conn, event).get(event, [])
    active = [c for c in classes if c["label"] is not None]
    active.sort(key=lambda c: c["user"].lower())
    labels: Dict[str, int] = {}
    for c in active:
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    src_row = _current_sources(conn, event).get(event)
    source_block = None
    if src_row and src_row["source_name"] is not None:
        source_block = src_row
    return {
        "event": event,
        "classifications": active,
        "labels": labels,
        "source_name": source_block,
    }


def event_annotations(
    event: str, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Public: current-annotations block for one event."""
    e = _event_ok(event)
    with _conn(db_path) as conn:
        return _event_current(conn, e)


def event_history(event: str, db_path: Optional[str] = None) -> List[dict]:
    """Full audit trail for one event (classifications + source edits),
    oldest first."""
    e = _event_ok(event)
    with _conn(db_path) as conn:
        rows = []
        for r in conn.execute(
            "SELECT user, label, ts_utc FROM classifications "
            "WHERE event = ? ORDER BY id",
            (e,),
        ).fetchall():
            rows.append(
                {
                    "kind": "classification",
                    "user": r["user"],
                    "label": r["label"],
                    "ts_utc": r["ts_utc"],
                }
            )
        for r in conn.execute(
            "SELECT user, source_name, ts_utc FROM source_names "
            "WHERE event = ? ORDER BY id",
            (e,),
        ).fetchall():
            rows.append(
                {
                    "kind": "source_name",
                    "user": r["user"],
                    "source_name": r["source_name"],
                    "ts_utc": r["ts_utc"],
                }
            )
    rows.sort(key=lambda x: x["ts_utc"])
    return rows


def all_current(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Bulk current-annotations for **every** event, in one/two queries
    (no per-event round-trip). Powers the list page badges + filter.

    Returns ``{event: {"source_name": str|None, "labels": {label: count},
    "users": {user: label}}}``. Only events with at least one current
    (non-null) classification and/or a current source appear.
    """
    with _conn(db_path) as conn:
        classes = _current_classifications(conn)
        sources = _current_sources(conn)
    out: Dict[str, Dict[str, Any]] = {}
    for event, clist in classes.items():
        labels: Dict[str, int] = {}
        users: Dict[str, str] = {}
        for c in clist:
            if c["label"] is None:
                continue
            labels[c["label"]] = labels.get(c["label"], 0) + 1
            users[c["user"]] = c["label"]
        if labels:
            out.setdefault(event, {})["labels"] = labels
            out[event]["users"] = users
    for event, srow in sources.items():
        if srow["source_name"] is not None:
            out.setdefault(event, {})["source_name"] = srow["source_name"]
    # Normalise shape.
    for event, blk in out.items():
        blk.setdefault("labels", {})
        blk.setdefault("users", {})
        blk.setdefault("source_name", None)
    return out


def classified_events(db_path: Optional[str] = None) -> set[str]:
    """Events that have at least one current (non-null) classification."""
    cur = all_current(db_path)
    return {e for e, blk in cur.items() if blk.get("labels")}


# ---------------------------------------------------------------------------
# Vocabulary (autocomplete surface)
# ---------------------------------------------------------------------------


def vocab(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Labels, custom tags, users, and all distinct source names ever
    used (current + historical) — for the UI autocomplete + label row."""
    with _conn(db_path) as conn:
        custom = [r["tag"] for r in conn.execute(
            "SELECT tag FROM custom_tags ORDER BY created_utc, tag"
        ).fetchall()]
        users = [r["name"] for r in conn.execute(
            "SELECT name FROM users ORDER BY added_utc, name"
        ).fetchall()]
        sources = [r["source_name"] for r in conn.execute(
            "SELECT DISTINCT source_name FROM source_names "
            "WHERE source_name IS NOT NULL ORDER BY source_name COLLATE NOCASE"
        ).fetchall()]
    return {
        "builtin_labels": list(BUILTIN_LABELS),
        "custom_tags": custom,
        "labels": list(BUILTIN_LABELS) + custom,
        "users": users,
        "source_names": sources,
    }


# ---------------------------------------------------------------------------
# Refined localizations (event_positions)
# ---------------------------------------------------------------------------

#: Refined-position error bounds (arcsec): finite, > 0, < 1 degree.
MAX_POS_ERR_ARCSEC = 3600.0


def validate_position(
    ra_deg: float,
    dec_deg: float,
    ra_err_arcsec: float,
    dec_err_arcsec: float,
    method: Optional[str],
    username: Optional[str],
) -> None:
    """Server-side validation shared by the store + Flask layer.

    Raises ``ValueError`` with a human-readable message on the first
    violated constraint. (Numeric-ness is asserted here too, so the
    store never persists NaN/inf.)"""
    import math

    for label, v in (("ra", ra_deg), ("dec", dec_deg),
                     ("ra_err_arcsec", ra_err_arcsec),
                     ("dec_err_arcsec", dec_err_arcsec)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{label} is not a number")
        if not math.isfinite(f):
            raise ValueError(f"{label} is not finite")
    if not (0.0 <= float(ra_deg) < 360.0):
        raise ValueError("ra out of range: need 0 <= ra < 360 deg")
    if not (-90.0 <= float(dec_deg) <= 90.0):
        raise ValueError("dec out of range: need -90 <= dec <= +90 deg")
    for label, v in (("ra_err_arcsec", ra_err_arcsec),
                     ("dec_err_arcsec", dec_err_arcsec)):
        if not (0.0 < float(v) < MAX_POS_ERR_ARCSEC):
            raise ValueError(
                f"{label} out of range: need 0 < err < "
                f"{MAX_POS_ERR_ARCSEC:g} arcsec"
            )
    if not (method or "").strip():
        raise ValueError("method is empty")
    normalize_user(username)  # raises ValueError on empty


def _position_row(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "event": r["event"],
        "ra_deg": r["ra_deg"],
        "dec_deg": r["dec_deg"],
        "ra_err_arcsec": r["ra_err_arcsec"],
        "dec_err_arcsec": r["dec_err_arcsec"],
        "method": r["method"],
        "username": r["username"],
        "created_utc": r["created_utc"],
        "active": bool(r["active"]),
        "pipe_ra_deg": r["pipe_ra_deg"],
        "pipe_dec_deg": r["pipe_dec_deg"],
        "pipe_source": r["pipe_source"],
    }


def set_position(
    event: Optional[str],
    ra_deg: float,
    dec_deg: float,
    ra_err_arcsec: float,
    dec_err_arcsec: float,
    method: Optional[str],
    username: Optional[str],
    pipe_snapshot: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a refined localization (last-write-wins).

    Deactivates any prior active row for the event and inserts the new
    one with ``active=1``. ``pipe_snapshot`` (``{"ra_deg", "dec_deg",
    "source"}``, any of which may be None) is the contemporaneous
    pipeline position, stored alongside so the backup is
    self-contained. Returns the stored current row."""
    e = _event_ok(event)
    validate_position(ra_deg, dec_deg, ra_err_arcsec, dec_err_arcsec,
                      method, username)
    u = normalize_user(username)
    snap = pipe_snapshot or {}
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        conn.execute(
            "UPDATE event_positions SET active = 0 "
            "WHERE event = ? AND active = 1", (e,),
        )
        cur = conn.execute(
            "INSERT INTO event_positions"
            " (event, ra_deg, dec_deg, ra_err_arcsec, dec_err_arcsec,"
            "  method, username, created_utc, active,"
            "  pipe_ra_deg, pipe_dec_deg, pipe_source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            (
                e, float(ra_deg), float(dec_deg),
                float(ra_err_arcsec), float(dec_err_arcsec),
                str(method).strip(), canon, _now_iso(),
                snap.get("ra_deg"), snap.get("dec_deg"),
                snap.get("source"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM event_positions WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
        return _position_row(row)


def clear_position(
    event: Optional[str],
    username: Optional[str],
    db_path: Optional[str] = None,
) -> bool:
    """Clear the current refined position (audited).

    Deactivates the active row and appends an audit row with NULL
    coordinates (``active=0``) recording who cleared and when — the
    history must show the clear. Returns True if there was an active
    position to clear (the audit row is written either way)."""
    e = _event_ok(event)
    u = normalize_user(username)
    with _conn(db_path) as conn:
        canon = _canonical_user(conn, u)
        if canon is None:
            raise UnknownUserError(f"unknown user: {u!r}")
        cur = conn.execute(
            "UPDATE event_positions SET active = 0 "
            "WHERE event = ? AND active = 1", (e,),
        )
        had_active = cur.rowcount > 0
        conn.execute(
            "INSERT INTO event_positions"
            " (event, ra_deg, dec_deg, ra_err_arcsec, dec_err_arcsec,"
            "  method, username, created_utc, active,"
            "  pipe_ra_deg, pipe_dec_deg, pipe_source)"
            " VALUES (?, NULL, NULL, NULL, NULL, NULL, ?, ?, 0,"
            "         NULL, NULL, NULL)",
            (e, canon, _now_iso()),
        )
        return had_active


def get_position(
    event: str, db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The current (active) refined position for an event, or None."""
    e = _event_ok(event)
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM event_positions "
            "WHERE event = ? AND active = 1 ORDER BY id DESC LIMIT 1",
            (e,),
        ).fetchone()
    return _position_row(row) if row else None


def get_positions_bulk(
    db_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """{event: current refined-position row} for every event that has
    one — a single query, for the /bursts table page."""
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM event_positions WHERE active = 1"
        ).fetchall()
    return {r["event"]: _position_row(r) for r in rows}


def get_position_history(
    event: str, db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Full audit trail for an event's refined positions, oldest first
    (set rows carry coordinates; clear rows have NULL ra/dec)."""
    e = _event_ok(event)
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM event_positions WHERE event = ? ORDER BY id",
            (e,),
        ).fetchall()
    return [_position_row(r) for r in rows]


# ---------------------------------------------------------------------------
# Query surface (the future agent-query API)
# ---------------------------------------------------------------------------


def _in_range(ts: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    if ts is None:
        return False
    if since is not None and ts < since:
        return False
    if until is not None and ts > until:
        return False
    return True


def query_annotations(
    db_path: Optional[str] = None,
    *,
    event: Optional[str] = None,
    user: Optional[str] = None,
    label: Optional[str] = None,
    source: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    history: bool = False,
) -> List[dict]:
    """Read-only query surface for agents.

    ``history=False`` (default) → one record per matching event carrying
    its **current** annotations. ``history=True`` → the full flat audit
    trail (classification + source rows) matching the filters, oldest
    first.

    Filters (all optional, ANDed):
      * ``event``  — exact event name.
      * ``user``   — restrict to that user's classifications (current
                     mode: an event is kept only if that user currently
                     labels it).
      * ``label``  — normalised; current mode keeps events where some
                     current classification equals it.
      * ``source`` — current/row source name equals it (case-insensitive).
      * ``since`` / ``until`` — inclusive ISO-8601 bounds on ``ts_utc``.

    The B1913+16 use case ("count detections of B1913+16 since
    2026-06-15") is ``query_annotations(source="B1913+16",
    since="2026-06-15")`` → the returned list length is the count.
    """
    norm_label = normalize_tag(label) if label else None
    with _conn(db_path) as conn:
        if history:
            return _query_history(
                conn, event=event, user=user, label=norm_label,
                source=source, since=since, until=until,
            )
        return _query_current(
            conn, event=event, user=user, label=norm_label,
            source=source, since=since, until=until,
        )


def _query_current(
    conn, *, event, user, label, source, since, until
) -> List[dict]:
    classes = _current_classifications(conn, event)
    sources = _current_sources(conn, event)
    events = set(classes) | set(sources)
    if event is not None:
        events &= {event}
    src_lc = source.strip().lower() if source else None
    out: List[dict] = []
    for e in sorted(events):
        # Current classifications (drop cleared + apply user/time filters).
        clist = [
            c for c in classes.get(e, [])
            if c["label"] is not None and _in_range(c["ts_utc"], since, until)
        ]
        if user is not None:
            clist = [c for c in clist if c["user"].lower() == user.lower()]
        # Current source (apply time filter).
        srow = sources.get(e)
        cur_source = None
        if srow and srow["source_name"] is not None and _in_range(
            srow["ts_utc"], since, until
        ):
            cur_source = srow

        # Explicit filters.
        if src_lc is not None:
            if cur_source is None or cur_source["source_name"].lower() != src_lc:
                continue
        if label is not None and not any(c["label"] == label for c in clist):
            continue
        if user is not None and source is None and label is None and not clist:
            continue
        # Drop empty events unless the caller pinned this event explicitly.
        if event is None and not clist and cur_source is None:
            continue
        clist.sort(key=lambda c: c["user"].lower())
        out.append({
            "event": e,
            "classifications": clist,
            "source_name": cur_source,
        })
    return out


def _query_history(
    conn, *, event, user, label, source, since, until
) -> List[dict]:
    rows: List[dict] = []
    src_lc = source.strip().lower() if source else None
    # Classification rows.
    if source is None:  # a source filter can't match classification rows
        sql = "SELECT event, user, label, ts_utc FROM classifications"
        clauses, params = [], []
        if event is not None:
            clauses.append("event = ?"); params.append(event)
        if user is not None:
            clauses.append("lower(user) = lower(?)"); params.append(user)
        if label is not None:
            clauses.append("label = ?"); params.append(label)
        if since is not None:
            clauses.append("ts_utc >= ?"); params.append(since)
        if until is not None:
            clauses.append("ts_utc <= ?"); params.append(until)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        for r in conn.execute(sql, params).fetchall():
            rows.append({
                "kind": "classification", "event": r["event"],
                "user": r["user"], "label": r["label"], "ts_utc": r["ts_utc"],
            })
    # Source rows (skip when a label filter is set — labels are class-only).
    if label is None:
        sql = "SELECT event, source_name, user, ts_utc FROM source_names"
        clauses, params = [], []
        if event is not None:
            clauses.append("event = ?"); params.append(event)
        if user is not None:
            clauses.append("lower(user) = lower(?)"); params.append(user)
        if src_lc is not None:
            clauses.append("lower(source_name) = ?"); params.append(src_lc)
        if since is not None:
            clauses.append("ts_utc >= ?"); params.append(since)
        if until is not None:
            clauses.append("ts_utc <= ?"); params.append(until)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        for r in conn.execute(sql, params).fetchall():
            rows.append({
                "kind": "source_name", "event": r["event"],
                "user": r["user"], "source_name": r["source_name"],
                "ts_utc": r["ts_utc"],
            })
    rows.sort(key=lambda x: x["ts_utc"])
    return rows

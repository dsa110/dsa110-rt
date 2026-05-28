"""dsa_monitor Phase 6c: dump-gate state reader + writer.

Front-end for ``/cmd/c2/dumps_enabled`` — the runtime kill-switch that
the C2 coincidencer polls (via :class:`dsart.services.coincidencer.DumpsGate`)
to decide whether to broadcast UDP dump triggers or silently
classify-and-audit them as ``WOULD-DUMP`` log rows.

Operator workflow
-----------------

1. Open the Control tab. The "Dumps" panel shows a status pill
   ("Dumps: ENABLED" / "Dumps: SUPPRESSED") read via
   :func:`get_dumps_state`.
2. To flip the gate, type ``suppress`` or ``enable`` into the
   confirmation box (matching the direction of the requested change),
   enter a free-text reason (mandatory, max 240 chars), and submit.
3. The Flask route :py:func:`tools.dashboard.dsa_monitor.app.control_dumps_enabled`
   calls :func:`set_dumps_state`, which:

   * Writes ``/cmd/c2/dumps_enabled`` =
     ``{"enabled": bool, "ts": float, "actor": str, "reason": str}``
     (DsaStore ``put_dict``).
   * Writes an audit row at
     ``/mon/audit/control/dumps_toggle/<unix_ms>`` so the
     dump-toggle history is independently queryable from any etcd
     reader.

4. The C2 coincidencer's :class:`DumpsGate` cache picks the new value
   up within :data:`DumpsGate.DEFAULT_DUMPS_CACHE_TTL_S` (≈ 200 ms).

Semantics
---------

The etcd payload wraps the boolean to capture *who* flipped the
switch and *why* — a stale ``enabled=False`` set during a known-bad-
data window must still be auditable months later. The wrapper shape
is:

    {
      "enabled": bool,         # required
      "ts":      float,        # unix seconds (server clock)
      "actor":   str,          # human or remote_addr ("anon" fallback)
      "reason":  str,          # 1..240 chars, free text
    }

A missing key is treated as ``enabled=True`` (fail-OPEN; we never want
a cold etcd to silently suppress dumps). :func:`get_dumps_state`
surfaces the missing-key case by setting ``"default": True`` in the
returned dict so the UI can render "Dumps: ENABLED (default)" rather
than pretending a real operator chose to enable.

This module is stdlib-only at import time (the
``dsautils.dsa_store.DsaStore`` import sits on the
:class:`control_store.ControlStore` constructor's lazy path so this
module's import never fails because etcd is briefly down).
"""

from __future__ import annotations

import datetime as _dt
import logging
import socket
import time
from typing import Any, Iterable, Optional

LOG = logging.getLogger("dsa_monitor.dumps_gate")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical etcd key the coincidencer polls. Mirrored from
#: :data:`dsart.services.coincidencer.DUMPS_ENABLED_KEY`; pinned here
#: so the dashboard env (which historically does not always have
#: ``dsart`` on PYTHONPATH) can build the URL without that import.
#: Drift between the two copies is caught by the unit tests in
#: ``tests/test_dumps_gate_dashboard.py``.
DUMPS_ENABLED_KEY: str = "/cmd/c2/dumps_enabled"

#: Prefix for the per-flip audit rows. One key per toggle, keyed by
#: ``<unix_ms>`` so chronological ordering falls out of an etcd
#: ``get_prefix`` (lexicographic sort == numeric sort once you pad to
#: 13 digits; unix_ms stays 13 digits until 2286).
DUMPS_AUDIT_PREFIX: str = "/mon/audit/control/dumps_toggle/"

#: Maximum length of the operator-supplied reason. Wider than a
#: typical journal line but still small enough that the audit panel
#: can wrap without truncation.
MAX_REASON_LEN: int = 240


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_ts(unix_seconds: float) -> str:
    """ISO-8601 UTC timestamp with microseconds + trailing ``Z``.

    Matches :func:`control_store._iso_ts_utc` exactly so the unified
    audit panel (which sorts by ``iso_ts``) renders dumps_toggle rows
    in the right slot alongside restart_all / inject rows.
    """
    return _dt.datetime.fromtimestamp(
        float(unix_seconds), _dt.timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _audit_key(unix_ms: int) -> str:
    """Build the ``/mon/audit/control/dumps_toggle/<unix_ms>`` key."""
    return f"{DUMPS_AUDIT_PREFIX}{int(unix_ms)}"


def _normalise_reason(reason: Any) -> str:
    """Trim whitespace + enforce length cap. Raises ``ValueError``
    when empty or over :data:`MAX_REASON_LEN` chars.
    """
    if reason is None:
        raise ValueError("reason is required (non-empty string)")
    txt = str(reason).strip()
    if not txt:
        raise ValueError("reason is required (non-empty string)")
    if len(txt) > MAX_REASON_LEN:
        raise ValueError(
            f"reason too long: {len(txt)} chars (max {MAX_REASON_LEN})"
        )
    return txt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dumps_state(store: Any) -> dict[str, Any]:
    """Return the current dump-gate state.

    Reads ``/cmd/c2/dumps_enabled`` via ``store.get_dict``. The store
    must expose the same surface as
    :class:`tools.dashboard.dsa_monitor.control_store.ControlStore`
    (i.e. ``get_dict(key) -> dict | None``).

    Shape on a present-and-well-formed key::

        {
          "enabled": True | False,
          "ts": float | None,
          "actor": str | None,
          "reason": str | None,
          "default": False,
        }

    Shape on a missing or malformed key::

        {
          "enabled": True,
          "ts": None,
          "actor": None,
          "reason": None,
          "default": True,
        }

    Any etcd error bubbles up to the caller (the Flask route wraps it
    in a 500 + audit row). We deliberately do NOT fall back to a
    default state on transport errors — the operator must see the
    failure rather than have it silently treated as "dumps enabled".
    """
    raw = store.get_dict(DUMPS_ENABLED_KEY)
    if not isinstance(raw, dict) or "enabled" not in raw:
        return {
            "enabled": True,
            "ts": None,
            "actor": None,
            "reason": None,
            "default": True,
        }
    return {
        "enabled": bool(raw["enabled"]),
        "ts": raw.get("ts"),
        "actor": raw.get("actor"),
        "reason": raw.get("reason"),
        "default": False,
    }


def set_dumps_state(
    store: Any,
    *,
    enabled: bool,
    reason: Any,
    actor: Optional[str] = None,
    host: Optional[str] = None,
    now_unix: Optional[float] = None,
) -> dict[str, Any]:
    """Write the new dump-gate state + a per-flip audit row.

    Parameters
    ----------
    store:
        DsaStore-like object exposing ``put_dict(key, dict)``.
    enabled:
        New value for ``/cmd/c2/dumps_enabled.enabled``.
    reason:
        Free-text justification (1..:data:`MAX_REASON_LEN` chars).
        Empty / whitespace-only / over-length values raise
        :class:`ValueError` BEFORE any etcd write.
    actor:
        Operator label; falls back to ``"anon"``.
    host:
        Local hostname recorded into the audit row; defaults to
        :func:`socket.gethostname`.
    now_unix:
        Injected wall-clock for tests (so the audit key is
        deterministic). ``None`` → :func:`time.time`.

    Returns
    -------
    dict
        Same shape as :func:`get_dumps_state` plus an ``audit_key``
        pointer so the caller can echo it back to the operator.

    Raises
    ------
    ValueError
        On empty / too-long ``reason``. Etcd transport errors
        propagate unchanged (the Flask route wraps them into a 500 +
        audit-log entry).
    """
    reason_txt = _normalise_reason(reason)
    actor_txt = str(actor or "anon")
    host_txt = str(host or socket.gethostname())
    now = float(now_unix) if now_unix is not None else time.time()
    enabled_bool = bool(enabled)

    payload = {
        "enabled": enabled_bool,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
    }
    # 1. Write the cmd key — the C2 coincidencer picks this up on
    #    its next DumpsGate refresh (≤ 200 ms cache TTL).
    store.put_dict(DUMPS_ENABLED_KEY, payload)

    # 2. Write the per-flip audit row. Failures here are logged but
    #    NEVER raised — losing an audit row on a transient etcd
    #    hiccup must not block the operator from flipping the gate.
    unix_ms = int(now * 1000.0)
    audit_key = _audit_key(unix_ms)
    audit_row = {
        "iso_ts": _iso_ts(now),
        "ts_unix_ms": unix_ms,
        "user": actor_txt,
        "host": host_txt,
        "namespace": "c2.dumps_toggle",
        "cn_target": "h23",
        "cmd": "dumps_toggle",
        "val": {"enabled": enabled_bool},
        "ok": True,
        "note": reason_txt,
    }
    try:
        store.put_dict(audit_key, audit_row)
    except Exception as exc:  # noqa: BLE001
        LOG.error(
            "dumps_toggle audit write to %s failed: %s",
            audit_key, exc,
        )

    return {
        "enabled": enabled_bool,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
        "default": False,
        "audit_key": audit_key,
    }


def list_recent_toggles(
    store: Any,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent dumps_toggle audit rows.

    Uses the raw ``etcd3`` client surface
    (``store.get_etcd().get_prefix(...)``) the same way
    :func:`control_store.list_recent_audit` does. Best-effort: any
    etcd transport failure / JSON parse failure is swallowed + logged
    so the audit panel never breaks the dashboard.

    The returned rows are sorted by ``iso_ts`` descending (newest
    first) so the UI can render them straight into a table without
    re-sorting.
    """
    import json

    # control_store.ControlStore exposes a lazy ``_ensure`` /
    # ``_store`` pair; duck-type that path so tests can inject a
    # plain object with a ``.get_etcd()`` method.
    es = store
    if hasattr(store, "_ensure") and hasattr(store, "_store"):
        try:
            store._ensure()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("list_recent_toggles: _ensure failed: %s", exc)
            return []
        es = store._store
    try:
        client = es.get_etcd()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("list_recent_toggles: get_etcd() failed: %s", exc)
        return []
    try:
        pairs = list(client.get_prefix(DUMPS_AUDIT_PREFIX))
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "list_recent_toggles: get_prefix(%s) failed: %s",
            DUMPS_AUDIT_PREFIX, exc,
        )
        return []
    rows: list[dict[str, Any]] = []
    for value, _meta in pairs:
        try:
            if isinstance(value, (bytes, bytearray)):
                payload = json.loads(value.decode("utf-8"))
            elif isinstance(value, str):
                payload = json.loads(value)
            elif isinstance(value, dict):
                payload = value
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            LOG.warning("list_recent_toggles: bad row JSON: %s", exc)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda r: r.get("iso_ts", ""), reverse=True)
    return rows[: int(limit)]


__all__ = [
    "DUMPS_ENABLED_KEY",
    "DUMPS_AUDIT_PREFIX",
    "MAX_REASON_LEN",
    "get_dumps_state",
    "set_dumps_state",
    "list_recent_toggles",
]

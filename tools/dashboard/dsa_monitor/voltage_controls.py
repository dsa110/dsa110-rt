"""dsa_monitor: runtime toggles for the M8 voltage-dump pipeline.

Two operator switches, both modelled on
:mod:`tools.dashboard.dsa_monitor.dumps_gate` (per-flip audit row + a
wrapped boolean payload that records *who* flipped it and *why*):

1. ``/cmd/c2/voltages_enabled`` — the C2 voltage-broadcast kill-switch.
   The coincidencer polls it via
   :class:`dsart.services.coincidencer.DumpsGate` (constructed with the
   voltages key + ``default_enabled=False``) and only fans out the
   ``DUMP_VOLTAGE`` UDP triggers to the 16 corr nodes when ``enabled``
   is True. Unlike the cube ``dumps_enabled`` gate this is **fail-
   CLOSED**: a missing key means voltages are DISABLED, so a cold etcd
   (or a fresh deploy that never set the key) never silently fills the
   corr-node NVMe staging.

2. ``/cmd/c3/flag_only`` — the C3 keep/delete mode. C3
   (:mod:`dsart.services.c3`) reads it per event: ``flag_only=True``
   (the safe default) collects voltages + logs every veto decision but
   performs NO destructive REJECT cleanup; ``flag_only=False`` enables
   the conservative delete path (delete staged voltages + cubes, move
   metadata/plots to ``candidates_rejected/``). A missing key means C3
   falls back to its configured default (also ``True``), so this too is
   fail-safe.

Both helpers write an independent audit row per flip under
``/mon/audit/control/<...>_toggle/<unix_ms>`` so the history is
queryable from any etcd reader, exactly like ``dumps_gate``.

The keys here are pinned copies of
:data:`dsart.services.coincidencer.VOLTAGES_ENABLED_KEY` and
:data:`dsart.services.c3.C3_FLAG_ONLY_KEY`; the dashboard env does not
always have ``dsart`` on PYTHONPATH. Drift is caught by
``tests/test_voltage_controls.py``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import socket
import time
from typing import Any, Optional

LOG = logging.getLogger("dsa_monitor.voltage_controls")


# ---------------------------------------------------------------------------
# Constants — pinned copies of the dsart-side keys.
# ---------------------------------------------------------------------------

#: C2 voltage-broadcast kill-switch. Mirrors
#: :data:`dsart.services.coincidencer.VOLTAGES_ENABLED_KEY`.
VOLTAGES_ENABLED_KEY: str = "/cmd/c2/voltages_enabled"

#: C3 keep/delete mode. Mirrors
#: :data:`dsart.services.c3.C3_FLAG_ONLY_KEY`.
C3_FLAG_ONLY_KEY: str = "/cmd/c3/flag_only"

#: Per-flip audit prefixes (one key per toggle, keyed by ``<unix_ms>``).
VOLTAGES_AUDIT_PREFIX: str = "/mon/audit/control/voltages_toggle/"
C3_MODE_AUDIT_PREFIX: str = "/mon/audit/control/c3_mode_toggle/"

#: Maximum length of the operator-supplied reason (matches dumps_gate).
MAX_REASON_LEN: int = 240


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_ts(unix_seconds: float) -> str:
    """ISO-8601 UTC timestamp with microseconds + trailing ``Z``."""
    return _dt.datetime.fromtimestamp(
        float(unix_seconds), _dt.timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalise_reason(reason: Any) -> str:
    """Trim + length-cap; raise ``ValueError`` when empty/over-length."""
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


def _get_bool_gate(
    store: Any,
    *,
    key: str,
    field: str,
    default_value: bool,
) -> dict[str, Any]:
    """Read a wrapped boolean gate at ``key``.

    Returns ``{field, "ts", "actor", "reason", "default"}`` where
    ``default`` is True iff the key is missing/malformed (in which case
    ``field`` is ``default_value``). Any etcd transport error bubbles up
    to the caller — we never silently fabricate a default on a transport
    failure (the operator must see the read fail rather than have it read
    as the safe value).
    """
    raw = store.get_dict(key)
    if not isinstance(raw, dict) or field not in raw:
        return {
            field: bool(default_value),
            "ts": None,
            "actor": None,
            "reason": None,
            "default": True,
        }
    return {
        field: bool(raw[field]),
        "ts": raw.get("ts"),
        "actor": raw.get("actor"),
        "reason": raw.get("reason"),
        "default": False,
    }


def _set_bool_gate(
    store: Any,
    *,
    key: str,
    audit_prefix: str,
    audit_namespace: str,
    audit_cmd: str,
    field: str,
    value: bool,
    reason: Any,
    actor: Optional[str],
    host: Optional[str],
    now_unix: Optional[float],
) -> dict[str, Any]:
    """Write a wrapped boolean gate + a per-flip audit row.

    Validates ``reason`` BEFORE any etcd write (raises ``ValueError``).
    Etcd transport errors on the cmd-key write propagate; audit-row
    write failures are logged but never raised.
    """
    reason_txt = _normalise_reason(reason)
    actor_txt = str(actor or "anon")
    host_txt = str(host or socket.gethostname())
    now = float(now_unix) if now_unix is not None else time.time()
    value_bool = bool(value)

    store.put_dict(key, {
        field: value_bool,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
    })

    unix_ms = int(now * 1000.0)
    audit_key = f"{audit_prefix}{unix_ms}"
    audit_row = {
        "iso_ts": _iso_ts(now),
        "ts_unix_ms": unix_ms,
        "user": actor_txt,
        "host": host_txt,
        "namespace": audit_namespace,
        "cn_target": "h23",
        "cmd": audit_cmd,
        "val": {field: value_bool},
        "ok": True,
        "note": reason_txt,
    }
    try:
        store.put_dict(audit_key, audit_row)
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s audit write to %s failed: %s", audit_cmd, audit_key, exc)

    return {
        field: value_bool,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
        "default": False,
        "audit_key": audit_key,
    }


def _list_recent(
    store: Any,
    *,
    prefix: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent audit rows under ``prefix``.

    Best-effort: any etcd / JSON failure is swallowed + logged so the
    audit panel never breaks the dashboard. Mirrors
    :func:`dumps_gate.list_recent_toggles`.
    """
    es = store
    if hasattr(store, "_ensure") and hasattr(store, "_store"):
        try:
            store._ensure()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("list_recent(%s): _ensure failed: %s", prefix, exc)
            return []
        es = store._store
    try:
        client = es.get_etcd()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("list_recent(%s): get_etcd() failed: %s", prefix, exc)
        return []
    try:
        pairs = list(client.get_prefix(prefix))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("list_recent(%s): get_prefix failed: %s", prefix, exc)
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
            LOG.warning("list_recent(%s): bad row JSON: %s", prefix, exc)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda r: r.get("iso_ts", ""), reverse=True)
    return rows[: int(limit)]


# ---------------------------------------------------------------------------
# Public API — voltages_enabled (C2)
# ---------------------------------------------------------------------------


def get_voltages_state(store: Any) -> dict[str, Any]:
    """Current ``/cmd/c2/voltages_enabled`` state (fail-CLOSED default).

    Missing/malformed key ⇒ ``{"enabled": False, ..., "default": True}``.
    """
    return _get_bool_gate(
        store, key=VOLTAGES_ENABLED_KEY, field="enabled", default_value=False,
    )


def set_voltages_state(
    store: Any,
    *,
    enabled: bool,
    reason: Any,
    actor: Optional[str] = None,
    host: Optional[str] = None,
    now_unix: Optional[float] = None,
) -> dict[str, Any]:
    """Flip ``/cmd/c2/voltages_enabled`` + write an audit row."""
    return _set_bool_gate(
        store,
        key=VOLTAGES_ENABLED_KEY,
        audit_prefix=VOLTAGES_AUDIT_PREFIX,
        audit_namespace="c2.voltages_toggle",
        audit_cmd="voltages_toggle",
        field="enabled",
        value=enabled,
        reason=reason,
        actor=actor,
        host=host,
        now_unix=now_unix,
    )


def list_recent_voltage_toggles(
    store: Any, *, limit: int = 5,
) -> list[dict[str, Any]]:
    """Most recent voltages_toggle audit rows, newest first."""
    return _list_recent(store, prefix=VOLTAGES_AUDIT_PREFIX, limit=limit)


# ---------------------------------------------------------------------------
# Public API — C3 keep/delete mode (flag_only)
# ---------------------------------------------------------------------------


def get_c3_mode_state(store: Any) -> dict[str, Any]:
    """Current ``/cmd/c3/flag_only`` state (safe default: keep-only).

    Missing/malformed key ⇒ ``{"flag_only": True, ..., "default": True}``
    (C3 itself then falls back to its configured default, also True).
    """
    return _get_bool_gate(
        store, key=C3_FLAG_ONLY_KEY, field="flag_only", default_value=True,
    )


def set_c3_mode_state(
    store: Any,
    *,
    flag_only: bool,
    reason: Any,
    actor: Optional[str] = None,
    host: Optional[str] = None,
    now_unix: Optional[float] = None,
) -> dict[str, Any]:
    """Flip ``/cmd/c3/flag_only`` + write an audit row.

    ``flag_only=True`` = keep-only (no destructive deletes);
    ``flag_only=False`` = enable the conservative REJECT delete path.
    """
    return _set_bool_gate(
        store,
        key=C3_FLAG_ONLY_KEY,
        audit_prefix=C3_MODE_AUDIT_PREFIX,
        audit_namespace="c3.mode_toggle",
        audit_cmd="c3_mode_toggle",
        field="flag_only",
        value=flag_only,
        reason=reason,
        actor=actor,
        host=host,
        now_unix=now_unix,
    )


def list_recent_c3_toggles(
    store: Any, *, limit: int = 5,
) -> list[dict[str, Any]]:
    """Most recent c3_mode_toggle audit rows, newest first."""
    return _list_recent(store, prefix=C3_MODE_AUDIT_PREFIX, limit=limit)


__all__ = [
    "VOLTAGES_ENABLED_KEY",
    "C3_FLAG_ONLY_KEY",
    "VOLTAGES_AUDIT_PREFIX",
    "C3_MODE_AUDIT_PREFIX",
    "MAX_REASON_LEN",
    "get_voltages_state",
    "set_voltages_state",
    "list_recent_voltage_toggles",
    "get_c3_mode_state",
    "set_c3_mode_state",
    "list_recent_c3_toggles",
]

"""Human authority over the dsa110-operator agent (operator-integration).

The external `dsa110-operator` agent (a separate repo, run on a laptop)
reads a single etcd key, ``/cmd/operator/control``, as a human override it
*cannot* countermand: the agent can read it but its write surface is
confined to ``/operator/`` and ``/cmd/ant/``, so it can never enable
itself, re-point the executor, or extend its own time limit. Only this
dashboard (and ops scripts) write the key.

Fields written here:

* ``agents_enabled`` (bool) — master lockout. ``false`` makes every agent
  control attempt fail closed (the agent's reads / Q&A keep working).
* ``executor_email`` (str | "") — optionally pin the single-executor right
  to one Google identity; empty ⇒ the agent self-arbitrates via its lease.
* ``max_obs_seconds`` (int | 0) — hard cap on how long one observation may
  run. ``dsart_rt`` enforces this with a watchdog that auto-``utc_stop``s
  after the cap, independent of the agent. ``0`` ⇒ no cap.

This module is read-only-safe to import and does no etcd work until called.
"""
from __future__ import annotations

import socket
import time
from typing import Any, Optional

from control_store import ControlStore, audit_log

OPERATOR_CONTROL_KEY = "/cmd/operator/control"
OPERATOR_LEASE_KEY = "/operator/executor/holder"

# Sanity ceiling so a fat-fingered field can't set a multi-week cap.
MAX_OBS_CAP_S = 24 * 3600


def get_operator_control(store: ControlStore) -> dict[str, Any]:
    """Current authority (with safe defaults if the key is absent)."""
    try:
        d = store.get_dict(OPERATOR_CONTROL_KEY)
    except Exception:                                          # noqa: BLE001
        d = None
    if not isinstance(d, dict):
        return {"agents_enabled": True, "executor_email": "",
                "max_obs_seconds": 0, "by": "", "ts": 0.0, "present": False}
    return {
        "agents_enabled": bool(d.get("agents_enabled", True)),
        "executor_email": str(d.get("executor_email") or ""),
        "max_obs_seconds": int(d.get("max_obs_seconds") or 0),
        "by": str(d.get("by", "")),
        "ts": float(d.get("ts", 0.0) or 0.0),
        "present": True,
    }


def get_operator_lease(store: ControlStore) -> Optional[dict[str, Any]]:
    """Who currently holds the operator's single-executor lease (or None)."""
    try:
        d = store.get_dict(OPERATOR_LEASE_KEY)
    except Exception:                                          # noqa: BLE001
        return None
    return d if isinstance(d, dict) else None


def set_operator_control(
    store: ControlStore, *,
    agents_enabled: bool,
    executor_email: str = "",
    max_obs_seconds: int = 0,
    actor: str = "unknown",
) -> dict[str, Any]:
    """Validate + write the authority key, and audit the change."""
    email = (executor_email or "").strip()
    if email and ("@" not in email or len(email) > 254):
        raise ValueError(f"executor_email {email!r} is not a valid address")
    try:
        cap = int(max_obs_seconds or 0)
    except (TypeError, ValueError):
        raise ValueError("max_obs_seconds must be an integer number of seconds")
    if cap < 0:
        raise ValueError("max_obs_seconds must be >= 0 (0 disables the cap)")
    if cap > MAX_OBS_CAP_S:
        raise ValueError(f"max_obs_seconds {cap} exceeds the {MAX_OBS_CAP_S}s ceiling")

    payload = {
        "agents_enabled": bool(agents_enabled),
        "executor_email": email,
        "max_obs_seconds": cap,
        "by": str(actor),
        "ts": time.time(),
        "host": socket.gethostname(),
    }
    store.put_dict(OPERATOR_CONTROL_KEY, payload)
    try:
        audit_log(store, namespace="operator.control", cn_target="agent",
                  cmd="set_operator_control",
                  val={"agents_enabled": payload["agents_enabled"],
                       "executor_email": email, "max_obs_seconds": cap},
                  ok=True, user=actor)
    except Exception:                                          # noqa: BLE001
        pass
    payload["present"] = True
    return payload


__all__ = [
    "OPERATOR_CONTROL_KEY", "OPERATOR_LEASE_KEY", "MAX_OBS_CAP_S",
    "get_operator_control", "get_operator_lease", "set_operator_control",
]

"""dsa_monitor spectral-line (SPL) panel: state reader + writer.

Front-end for ``/cnf/spectral_line`` — the per-sub-band config the
``dsart_rt`` orchestrator reads at ``start`` to decide, for each corr
node's chgroup, whether to spawn the SPL second fringe-stopper
(``meridian_fringestop_spl``, which writes a finer-channelisation
``*_sb<NN>_spl.hdf5`` product) or the no-op ``bada_null_drain`` as the
constant second ``bada`` reader. See
``configs/dsart_pipeline_rt.yaml`` (the two ``spl_gate``'d routines) and
``dsart.services.dsart_rt._load_spl_cfg`` for the consumer side.

Operator workflow
-----------------

1. Open the Control tab → "Spectral-line mode (SPL)" panel. Each of the
   16 chgroups (sub-bands) gets a row: an enable checkbox, an
   integration-time (seconds) field, and an nfreq_int (channels-to-
   average) field. The current state is GET-prefilled from
   ``/cnf/spectral_line``.
2. Edit the rows, type the confirm word, submit. The Flask route
   :py:func:`...app.control_spectral_line` calls
   :func:`set_spectral_line_state`, which validates every row, writes
   ``/cnf/spectral_line`` and an audit row.
3. On the next ``restart_all`` + ``start`` the orchestrators re-read
   ``/cnf/spectral_line`` and gate the per-node second-reader routine
   accordingly. (Editing the panel does NOT hot-swap a running fleet;
   it takes effect on the next start.)

Schema (``/cnf/spectral_line``)
-------------------------------

::

    {
      "version":  1,
      "ts":       float,          # unix seconds
      "actor":    str,
      "reason":   str,
      "subbands": {               # keyed by chgroup index "0".."15"
        "<chgroup>": {
          "enabled":       bool,
          "integration_s": float, # SPL integration time (s)
          "nfreq_int":     int,   # channels to average (divides 384)
        },
        ...
      }
    }

A missing key / missing sub-band entry ⇒ SPL disabled for that
sub-band (fail-SAFE to the production single-fringe-stop topology).

stdlib-only at import time (the ``DsaStore`` import lives behind the
:class:`control_store.ControlStore` lazy path).
"""

from __future__ import annotations

import datetime as _dt
import logging
import socket
import time
from typing import Any, Iterable, Optional

LOG = logging.getLogger("dsa_monitor.spectral_line_gate")


# ---------------------------------------------------------------------------
# Constants (kept in lock-step with dsart.services.dsart_rt; drift is
# caught by tests/test_spectral_line_gate_dashboard.py).
# ---------------------------------------------------------------------------

#: etcd key the orchestrator reads at ``start``.
SPECTRAL_LINE_KEY: str = "/cnf/spectral_line"

#: Per-toggle audit-row prefix (one key per save, ``<unix_ms>``).
SPECTRAL_LINE_AUDIT_PREFIX: str = "/mon/audit/control/spectral_line/"

#: Schema version written into the payload.
SCHEMA_VERSION: int = 1

#: Number of corr-node sub-bands (chgroups 0..15).
N_CHGROUPS: int = 16

#: Per-sub-band channel count (nchan_spw on the corr nodes). nfreq_int
#: must divide this exactly (dsamfs io.py asserts ``nchan % nfreq_int``).
NCHAN_SPW: int = 384

#: Slow-vis block cadence (s). nint = round(integration_s / tsamp).
TSAMP_S: float = 0.134217728

#: Legacy production nint=96 → the default SPL integration time so an
#: operator who only changes channelisation re-uses the production
#: fringe-table cache (no slow regen).
DEFAULT_NINT: int = 96
DEFAULT_INTEGRATION_S: float = round(DEFAULT_NINT * TSAMP_S, 6)
DEFAULT_NFREQ_INT: int = 1

#: Guard rails on the integration time. Lower = one slow-vis block.
#: Upper kept well under the bada ring depth (300 blocks ≈ 40 s) so a
#: single output integration can never span more than ~2/3 of the ring.
MIN_INTEGRATION_S: float = TSAMP_S
MAX_INTEGRATION_S: float = 30.0

#: Maximum operator-supplied reason length.
MAX_REASON_LEN: int = 240


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_ts(unix_seconds: float) -> str:
    return _dt.datetime.fromtimestamp(
        float(unix_seconds), _dt.timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _audit_key(unix_ms: int) -> str:
    return f"{SPECTRAL_LINE_AUDIT_PREFIX}{int(unix_ms)}"


def nint_from_integration_s(integration_s: float) -> int:
    """Slow-vis frame count for an integration time. ``>=1``."""
    return max(1, int(round(float(integration_s) / TSAMP_S)))


def divisors_of_nchan() -> list[int]:
    """Valid ``nfreq_int`` choices (divisors of :data:`NCHAN_SPW`)."""
    return [d for d in range(1, NCHAN_SPW + 1) if NCHAN_SPW % d == 0]


def _normalise_reason(reason: Any) -> str:
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


def _default_subband() -> dict[str, Any]:
    return {
        "enabled": False,
        "integration_s": DEFAULT_INTEGRATION_S,
        "nfreq_int": DEFAULT_NFREQ_INT,
    }


def validate_subband(chgroup: Any, entry: Any) -> dict[str, Any]:
    """Validate + normalise one sub-band row. Raises ``ValueError``.

    Returns the cleaned ``{"enabled", "integration_s", "nfreq_int"}``
    dict. An entry is only fully validated (integration_s / nfreq_int
    range + divisor checks) when ``enabled`` is true; a disabled row
    still has its numeric fields coerced + clamped so a later enable
    starts from a sane value.
    """
    try:
        cg = int(chgroup)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"chgroup {chgroup!r} is not an int") from exc
    if not (0 <= cg < N_CHGROUPS):
        raise ValueError(f"chgroup {cg} out of range 0..{N_CHGROUPS - 1}")
    if not isinstance(entry, dict):
        raise ValueError(f"chgroup {cg}: entry must be a dict, got {type(entry)}")

    enabled = bool(entry.get("enabled", False))

    try:
        integration_s = float(entry.get("integration_s", DEFAULT_INTEGRATION_S))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"chgroup {cg}: integration_s {entry.get('integration_s')!r} "
            f"is not a number"
        ) from exc
    try:
        nfreq_int = int(entry.get("nfreq_int", DEFAULT_NFREQ_INT))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"chgroup {cg}: nfreq_int {entry.get('nfreq_int')!r} is not an int"
        ) from exc

    if enabled:
        if not (MIN_INTEGRATION_S - 1e-9 <= integration_s <= MAX_INTEGRATION_S + 1e-9):
            raise ValueError(
                f"chgroup {cg}: integration_s {integration_s:g} out of range "
                f"[{MIN_INTEGRATION_S:g}, {MAX_INTEGRATION_S:g}] s"
            )
        if nfreq_int < 1 or NCHAN_SPW % nfreq_int != 0:
            raise ValueError(
                f"chgroup {cg}: nfreq_int {nfreq_int} must be a positive divisor "
                f"of {NCHAN_SPW} (valid: {divisors_of_nchan()})"
            )
    return {
        "enabled": enabled,
        "integration_s": round(float(integration_s), 6),
        "nfreq_int": int(nfreq_int),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_spectral_line_state(store: Any) -> dict[str, Any]:
    """Return the current spectral-line config, fully populated.

    Always returns all 16 chgroups (missing entries filled with the
    disabled default) so the panel can render a complete table. Adds
    ``"default": True`` when the etcd key is entirely missing.
    """
    raw = store.get_dict(SPECTRAL_LINE_KEY)
    subbands: dict[str, dict[str, Any]] = {}
    is_default = not (isinstance(raw, dict) and raw.get("subbands"))
    raw_sub = (raw.get("subbands") if isinstance(raw, dict) else None) or {}
    for cg in range(N_CHGROUPS):
        entry = raw_sub.get(str(cg))
        if isinstance(entry, dict):
            merged = _default_subband()
            merged.update({
                "enabled": bool(entry.get("enabled", False)),
                "integration_s": _coerce_float(
                    entry.get("integration_s"), DEFAULT_INTEGRATION_S),
                "nfreq_int": _coerce_int(entry.get("nfreq_int"), DEFAULT_NFREQ_INT),
            })
            merged["nint"] = nint_from_integration_s(merged["integration_s"])
            subbands[str(cg)] = merged
        else:
            d = _default_subband()
            d["nint"] = nint_from_integration_s(d["integration_s"])
            subbands[str(cg)] = d
    n_enabled = sum(1 for v in subbands.values() if v["enabled"])
    return {
        "version": int(raw.get("version", SCHEMA_VERSION)) if isinstance(raw, dict) else SCHEMA_VERSION,
        "subbands": subbands,
        "n_enabled": n_enabled,
        "ts": raw.get("ts") if isinstance(raw, dict) else None,
        "actor": raw.get("actor") if isinstance(raw, dict) else None,
        "reason": raw.get("reason") if isinstance(raw, dict) else None,
        "default": is_default,
        "tsamp_s": TSAMP_S,
        "nchan_spw": NCHAN_SPW,
        "default_integration_s": DEFAULT_INTEGRATION_S,
        "nfreq_int_choices": divisors_of_nchan(),
    }


def _coerce_float(v: Any, default: float) -> float:
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return default


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def set_spectral_line_state(
    store: Any,
    *,
    subbands: dict[Any, Any],
    reason: Any,
    actor: Optional[str] = None,
    host: Optional[str] = None,
    now_unix: Optional[float] = None,
) -> dict[str, Any]:
    """Validate + write ``/cnf/spectral_line`` + a per-save audit row.

    ``subbands`` maps chgroup (int or str) → ``{"enabled",
    "integration_s", "nfreq_int"}``. Every row is validated BEFORE any
    etcd write; a single bad row raises :class:`ValueError` and nothing
    is written. Missing chgroups are filled with the disabled default so
    the persisted doc is always complete.
    """
    reason_txt = _normalise_reason(reason)
    actor_txt = str(actor or "anon")
    host_txt = str(host or socket.gethostname())
    now = float(now_unix) if now_unix is not None else time.time()

    clean: dict[str, dict[str, Any]] = {}
    for cg in range(N_CHGROUPS):
        raw_entry = subbands.get(cg, subbands.get(str(cg)))
        if raw_entry is None:
            clean[str(cg)] = _default_subband()
        else:
            clean[str(cg)] = validate_subband(cg, raw_entry)

    n_enabled = sum(1 for v in clean.values() if v["enabled"])
    payload = {
        "version": SCHEMA_VERSION,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
        "subbands": clean,
    }
    store.put_dict(SPECTRAL_LINE_KEY, payload)

    unix_ms = int(now * 1000.0)
    audit_key = _audit_key(unix_ms)
    enabled_groups = sorted(int(cg) for cg, v in clean.items() if v["enabled"])
    audit_row = {
        "iso_ts": _iso_ts(now),
        "ts_unix_ms": unix_ms,
        "user": actor_txt,
        "host": host_txt,
        "namespace": "control.spectral_line",
        "cn_target": "fleet",
        "cmd": "spectral_line_set",
        "val": {"n_enabled": n_enabled, "enabled_chgroups": enabled_groups},
        "ok": True,
        "note": reason_txt,
    }
    try:
        store.put_dict(audit_key, audit_row)
    except Exception as exc:  # noqa: BLE001
        LOG.error("spectral_line audit write to %s failed: %s", audit_key, exc)

    return {
        "version": SCHEMA_VERSION,
        "ts": now,
        "actor": actor_txt,
        "reason": reason_txt,
        "subbands": clean,
        "n_enabled": n_enabled,
        "enabled_chgroups": enabled_groups,
        "audit_key": audit_key,
    }


def list_recent_changes(store: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    """Most recent spectral_line audit rows, newest first (best-effort)."""
    import json

    es = store
    if hasattr(store, "_ensure") and hasattr(store, "_store"):
        try:
            store._ensure()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("list_recent_changes: _ensure failed: %s", exc)
            return []
        es = store._store
    try:
        client = es.get_etcd()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("list_recent_changes: get_etcd() failed: %s", exc)
        return []
    try:
        pairs = list(client.get_prefix(SPECTRAL_LINE_AUDIT_PREFIX))
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "list_recent_changes: get_prefix(%s) failed: %s",
            SPECTRAL_LINE_AUDIT_PREFIX, exc,
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
            LOG.warning("list_recent_changes: bad row JSON: %s", exc)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda r: r.get("iso_ts", ""), reverse=True)
    return rows[: int(limit)]


__all__ = [
    "SPECTRAL_LINE_KEY",
    "SPECTRAL_LINE_AUDIT_PREFIX",
    "SCHEMA_VERSION",
    "N_CHGROUPS",
    "NCHAN_SPW",
    "TSAMP_S",
    "DEFAULT_NINT",
    "DEFAULT_INTEGRATION_S",
    "DEFAULT_NFREQ_INT",
    "MIN_INTEGRATION_S",
    "MAX_INTEGRATION_S",
    "MAX_REASON_LEN",
    "nint_from_integration_s",
    "divisors_of_nchan",
    "validate_subband",
    "get_spectral_line_state",
    "set_spectral_line_state",
    "list_recent_changes",
]

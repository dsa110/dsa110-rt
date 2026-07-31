"""Antenna and SNAP control for the dashboard's Control tab.

Replaces the antenna-control and SNAP-control surfaces of the retired
``webserverUIservice`` (websrv) on lxd110h20. The scheduling half of that
service is deliberately *not* carried over.

Both surfaces are just etcd writes; there is no RPC anywhere.

Antennas — consumed by ``dsa110-hwmc`` (``hwmc/dsa_labjack.py``). Every
``DsaAntLabJack`` watches **two** keys: its own ``/cmd/ant/<n>`` and the
broadcast ``/cmd/ant/0``. So commanding "all" is a single write to
``/cmd/ant/0``, not a fan-out.

    {"cmd": "move",        "val": <degrees>}
    {"cmd": "noise_a_on",  "val": true|false}
    {"cmd": "noise_b_on",  "val": true|false}
    {"cmd": "halt"}                            # no "val" -- see below

``execute_cmd`` branches on ``len(cmd) > 1``: with two keys it calls the
handler with ``cmd['val']``, with one key it calls with no argument. So
``halt`` must be sent as a *single-key* dict or hwmc raises a TypeError.

⚠ ``_move_ang`` performs **no range check** -- it only coerces to float via
``_validate_num``. This module is the only guard against a typo slewing the
array, hence :data:`EL_MIN` / :data:`EL_MAX`.

SNAPs — consumed by ``SNAP_control/scripts/snap.py`` on lxd110h20, which
watches ``/cmd/snap/<n>`` for n in 1..32. All five verbs exposed here are
**zero-argument** (``dsaX_snap.process`` does ``known_commands[cmd]()``), and
``set_delay`` sources its delays server-side from ``beamformer_weights.yaml``:

    {"cmd": "arm"|"progonly"|"prong"|"set_delay"|"level"}

There is **no** broadcast key for SNAPs, so "all" here really is 32 writes.

``armed_mjd`` is published to ``/mon/snap/<n>/armed_mjd`` as
``{"armed_mjd": <float>}``. **55000.0 is the "never armed" sentinel** set at
construction and re-set by ``prog``/``progonly``; treat it as "not armed"
rather than as a timestamp.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterable

LOG = logging.getLogger("dsa_monitor.antenna_snap_control")

# --- antennas -------------------------------------------------------------

ANT_CMD_PREFIX = "/cmd/ant/"
#: Broadcast key. hwmc has every antenna watch this in addition to its own.
ANT_BROADCAST = 0
#: Highest antenna number hwmc instantiates (see startSnaps/dsa.yaml).
ANT_MAX = 117

#: Commanded-elevation guard. Operator-chosen 2026-07-31: hwmc itself does no
#: range checking, and observed ``ant_cmd_el`` tops out near 124, so 143 bounds
#: the mechanical range without blocking stow.
EL_MIN = 0.0
EL_MAX = 143.0

ANT_VERBS = ("move", "noise_a_on", "noise_b_on", "halt")

# --- SNAPs ----------------------------------------------------------------

SNAP_CMD_PREFIX = "/cmd/snap/"
SNAP_MON_PREFIX = "/mon/snap/"
#: snap.py is launched with -n 1..32 (scripts/startSnaps.sh).
SNAP_MIN = 1
SNAP_MAX = 32
#: Only the verbs the operator asked for. `prog`, `mon` and `test` exist in
#: dsaX_snap but are deliberately not exposed.
SNAP_VERBS = ("arm", "progonly", "prong", "set_delay", "level")

#: dsaX_snap sets this at construction and on prog/progonly; it is not a real
#: timestamp.
ARMED_MJD_SENTINEL = 55000.0

#: Unix epoch as MJD, for converting armed_mjd to an age.
MJD_UNIX_EPOCH = 40587.0


class SelectionError(ValueError):
    """Operator input that we refuse to turn into an etcd write."""


def parse_number_list(raw: Any, *, lo: int, hi: int, label: str) -> list[int]:
    """Parse an operator-typed selection like ``"1, 5, 7-9"``.

    Accepts comma and/or whitespace separation and inclusive ``a-b`` ranges.
    Returns a sorted, de-duplicated list. An **empty** box returns ``[]``,
    which callers interpret as "all" -- the two surfaces then diverge, because
    antennas have a broadcast key and SNAPs do not.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    out: set[int] = set()
    for tok in re.split(r"[,\s]+", text):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                raise SelectionError(f"{label} range {tok!r} is backwards")
            span = range(a, b + 1)
            if len(span) > (hi - lo + 1):
                raise SelectionError(f"{label} range {tok!r} is too large")
            out.update(span)
            continue
        if not re.fullmatch(r"\d+", tok):
            raise SelectionError(f"{label} {tok!r} is not a number or a-b range")
        out.add(int(tok))

    bad = sorted(n for n in out if n < lo or n > hi)
    if bad:
        raise SelectionError(
            f"{label} out of range {lo}-{hi}: {', '.join(map(str, bad[:8]))}"
        )
    return sorted(out)


def parse_antennas(raw: Any) -> list[int]:
    """Operator antenna box → explicit antenna numbers (``[]`` means all)."""
    return parse_number_list(raw, lo=1, hi=ANT_MAX, label="antenna")


def parse_snaps(raw: Any) -> list[int]:
    """Operator SNAP box → explicit SNAP numbers (``[]`` means all)."""
    return parse_number_list(raw, lo=SNAP_MIN, hi=SNAP_MAX, label="SNAP")


def validate_elevation(raw: Any) -> float:
    """Coerce and bounds-check a commanded elevation.

    This is the *only* guard: hwmc's ``_move_ang`` accepts any float.
    """
    if raw is None or str(raw).strip() == "":
        raise SelectionError("elevation is required for a move")
    try:
        val = float(str(raw).strip())
    except ValueError:
        raise SelectionError(f"elevation {raw!r} is not a number") from None
    if val != val or val in (float("inf"), float("-inf")):
        raise SelectionError("elevation must be finite")
    if not (EL_MIN <= val <= EL_MAX):
        raise SelectionError(
            f"elevation {val:g} outside allowed {EL_MIN:g}-{EL_MAX:g} deg"
        )
    return val


def ant_payload(cmd: str, val: Any = None) -> dict[str, Any]:
    """Build the exact dict hwmc's ``execute_cmd`` expects.

    ``halt`` must be single-key: hwmc passes ``cmd['val']`` to the handler
    whenever the dict has more than one key, and ``_halt`` takes no argument.
    """
    if cmd not in ANT_VERBS:
        raise SelectionError(f"unknown antenna command {cmd!r}")
    if cmd == "halt":
        return {"cmd": "halt"}
    return {"cmd": cmd, "val": val}


def send_ant_command(
    store: Any,
    antennas: Iterable[int],
    cmd: str,
    val: Any = None,
) -> list[str]:
    """Write an antenna command; returns the etcd keys written.

    An empty ``antennas`` writes the single broadcast key ``/cmd/ant/0``,
    which every antenna is already watching.
    """
    payload = ant_payload(cmd, val)
    targets = sorted(set(int(a) for a in antennas)) or [ANT_BROADCAST]
    keys: list[str] = []
    for n in targets:
        key = f"{ANT_CMD_PREFIX}{n}"
        store.put_dict(key, payload)
        keys.append(key)
    LOG.info("antenna command %s -> %d key(s): %s", payload, len(keys), keys[:6])
    return keys


def send_snap_command(store: Any, snaps: Iterable[int], cmd: str) -> list[str]:
    """Write a SNAP command; returns the etcd keys written.

    No broadcast key exists for SNAPs, so an empty ``snaps`` fans out to all
    of :data:`SNAP_MIN`..:data:`SNAP_MAX`.
    """
    if cmd not in SNAP_VERBS:
        raise SelectionError(f"unknown SNAP command {cmd!r}")
    targets = sorted(set(int(s) for s in snaps)) or list(
        range(SNAP_MIN, SNAP_MAX + 1)
    )
    payload = {"cmd": cmd}
    keys: list[str] = []
    for n in targets:
        key = f"{SNAP_CMD_PREFIX}{n}"
        store.put_dict(key, payload)
        keys.append(key)
    LOG.info("SNAP command %s -> %d key(s)", payload, len(keys))
    return keys


def _mjd_now() -> float:
    return MJD_UNIX_EPOCH + time.time() / 86400.0


def get_snap_armed(store: Any, *, now_mjd: float | None = None) -> list[dict[str, Any]]:
    """Read ``armed_mjd`` for every SNAP, with age in seconds.

    Returns one row per SNAP in :data:`SNAP_MIN`..:data:`SNAP_MAX`. Rows carry
    ``armed`` = False when the key is missing (snap.py not running, or never
    armed since etcd was rebuilt) or when the value is the 55000.0 sentinel.
    A missing key and a sentinel are distinguished via ``reason`` so the
    operator can tell "no process" from "not armed".
    """
    now = _mjd_now() if now_mjd is None else now_mjd
    rows: list[dict[str, Any]] = []
    for n in range(SNAP_MIN, SNAP_MAX + 1):
        row: dict[str, Any] = {
            "snap": n, "armed": False, "armed_mjd": None,
            "age_s": None, "reason": "",
        }
        try:
            blob = store.get_dict(f"{SNAP_MON_PREFIX}{n}/armed_mjd")
        except Exception as exc:                                   # noqa: BLE001
            row["reason"] = f"etcd error: {exc}"
            rows.append(row)
            continue
        if not isinstance(blob, dict) or "armed_mjd" not in blob:
            row["reason"] = "no key (snap.py not running?)"
            rows.append(row)
            continue
        try:
            mjd = float(blob["armed_mjd"])
        except (TypeError, ValueError):
            row["reason"] = f"unparseable: {blob['armed_mjd']!r}"
            rows.append(row)
            continue
        row["armed_mjd"] = mjd
        if abs(mjd - ARMED_MJD_SENTINEL) < 1e-6:
            row["reason"] = "not armed (sentinel)"
        else:
            row["armed"] = True
            row["age_s"] = max(0.0, (now - mjd) * 86400.0)
        rows.append(row)
    return rows


def format_age(age_s: float | None) -> str:
    """Human-readable elapsed time for the armed_mjd column."""
    if age_s is None:
        return "--"
    s = int(age_s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"

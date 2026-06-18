"""Durable, registry-independent fired-injection log + coincidence test.

Motivation
----------

The live injection registry (``/cnf/inject/active/<inj_id>``, see
:mod:`dsart.coinc.inject_match`) is intentionally short-lived — entries
age out ~60 s after fire (``ttl_s``) and the matcher actively evicts
them. That is exactly right for the C2 hot-path matcher, but it leaves a
gap for **C3**, which runs minutes after an event is archived:

  * C2 stamps the durable ``injection`` marker into the Level3 JSON *only
    when the live matcher tagged a member at fire time*. If the matcher
    missed (registry already expired, an l/m or specnum gate just barely
    rejected the row, the search side bounced, …) the event is archived
    with **no** injection marker, and C3's cube veto would then treat a
    real injection as a candidate FP — and could delete its dump.

This module is the conservative backstop: every fired injection is also
appended to an **append-only JSONL log** that never expires. C3 reads it
and, as a final fallback after the L3 marker + CSV ``inj_id`` checks,
asks "did any archived candidate in this event coincide (DM + sky +
time) with a fired injection?". A hit EXEMPTS the event from the veto.

Because the only consequence of a coincidence hit is *keeping* a dump we
might otherwise have vetoed, the matching here is deliberately **looser**
than the calibration-grade matcher in :mod:`inject_match` — erring toward
"injection" is the safe direction (KEEP), in line with the whole C3
conservativeness mandate.

The log is written by the dashboard's ``publish_active_inject`` (the
single chokepoint that arms an injection) and read by C3. Duplicate
records for one ``inj_id`` are expected (the dashboard re-publishes with
the derived ``apply_at_specnum`` after an auto-arm) and are de-duplicated
on read, keeping the most-recently-fired record per id.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOG = logging.getLogger("dsart.coinc.inject_log")

__all__ = [
    "DEFAULT_FIRED_LOG_PATH",
    "DEFAULT_COINC_DM_TOL_FRAC",
    "DEFAULT_COINC_LM_TOL_RAD",
    "DEFAULT_COINC_WALL_PRE_S",
    "DEFAULT_COINC_WALL_POST_S",
    "FiredInjection",
    "append_fired_injection",
    "load_fired_injections",
    "event_coincident_inj_id",
]

#: Default location of the durable fired-injection log. Lives under the
#: operations tree (same root as the C2/C3 CSVs + C3 state) so it is on
#: shared storage both the dashboard (writer) and C3 (reader) can see.
DEFAULT_FIRED_LOG_PATH: str = (
    "/dataz/dsa110/operations/inject/fired_injections.jsonl"
)

#: Coincidence tolerances — deliberately looser than the inject_match
#: calibration gate (a hit only EXEMPTS from the veto = KEEP, the safe
#: direction). DM within 10 %, sky within ~0.06 rad (~3.4°).
DEFAULT_COINC_DM_TOL_FRAC: float = 0.10
DEFAULT_COINC_LM_TOL_RAD: float = 0.06

#: Wall-clock coincidence window around the injection's ``fired_at_unix``.
#: The event's burst time (C1 row MJD) should fall in
#: ``[fired_at - PRE, fired_at + POST]``. POST is generous (covers the
#: band-bottom dispersion sweep at high DM + corr→search→C2 pipeline +
#: C2 clustering/holdoff latency); PRE tolerates specnum→UTC skew.
DEFAULT_COINC_WALL_PRE_S: float = 10.0
DEFAULT_COINC_WALL_POST_S: float = 120.0

#: Unix epoch in MJD (1970-01-01 00:00:00 UTC).
_MJD_UNIX_EPOCH: float = 40587.0


@dataclass(frozen=True)
class FiredInjection:
    """One durable fired-injection record (a superset of the registry row
    plus the wall-clock fire time, which is all the coincidence test
    needs)."""

    inj_id: str
    dm_pc_cm3: float
    l_rad: float
    m_rad: float
    apply_at_specnum: int
    fired_at_unix: float
    ttl_s: float = 60.0
    width_samples: int = 0
    fluence_jy_ms: float = 0.0
    fired_by: str = "unknown"
    target_snr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "inj_id": self.inj_id,
            "dm_pc_cm3": self.dm_pc_cm3,
            "l_rad": self.l_rad,
            "m_rad": self.m_rad,
            "apply_at_specnum": self.apply_at_specnum,
            "fired_at_unix": self.fired_at_unix,
            "ttl_s": self.ttl_s,
            "width_samples": self.width_samples,
            "fluence_jy_ms": self.fluence_jy_ms,
            "fired_by": self.fired_by,
        }
        if self.target_snr is not None:
            d["target_snr"] = self.target_snr
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FiredInjection":
        inj_id = str(d["inj_id"]).strip()
        if not inj_id:
            raise ValueError("empty inj_id")
        ts_raw = d.get("target_snr")
        target_snr: Optional[float] = None
        if ts_raw is not None:
            try:
                target_snr = float(ts_raw)
            except (TypeError, ValueError):
                target_snr = None
        return cls(
            inj_id=inj_id,
            dm_pc_cm3=float(d["dm_pc_cm3"]),
            l_rad=float(d["l_rad"]),
            m_rad=float(d["m_rad"]),
            apply_at_specnum=int(d.get("apply_at_specnum", 0)),
            fired_at_unix=float(d["fired_at_unix"]),
            ttl_s=float(d.get("ttl_s", 60.0)),
            width_samples=int(d.get("width_samples", 0)),
            fluence_jy_ms=float(d.get("fluence_jy_ms", 0.0)),
            fired_by=str(d.get("fired_by", "unknown")),
            target_snr=target_snr,
        )


def append_fired_injection(
    log_path: os.PathLike | str,
    inj: FiredInjection,
) -> bool:
    """Atomically append one fired-injection record to the JSONL log.

    Best-effort: returns True on success, False (with a logged warning)
    on any error — a failed durable-log write must NEVER block arming an
    injection. The append is a single ``write()`` of one ``\\n``-
    terminated JSON line opened ``"a"`` (POSIX guarantees an O_APPEND
    write of < PIPE_BUF is atomic vs concurrent appenders).
    """
    p = Path(log_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(inj.to_dict(), separators=(",", ":"))
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except OSError as exc:
        LOG.warning("append_fired_injection(%s) failed: %s", p, exc)
        return False


def load_fired_injections(
    log_path: os.PathLike | str,
    *,
    since_unix: Optional[float] = None,
) -> List[FiredInjection]:
    """Read the durable log, de-duplicating to the latest record per id.

    Tolerant of a missing file (returns ``[]``) and of individual
    malformed lines (skipped with a warning) so a partial / truncated
    write never poisons the whole read. When ``since_unix`` is given,
    records fired before it are dropped (bounds the scan for very old
    logs).
    """
    p = Path(log_path)
    if not p.is_file():
        return []
    latest: Dict[str, FiredInjection] = {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = FiredInjection.from_dict(json.loads(raw))
                except (ValueError, KeyError, TypeError) as exc:
                    LOG.warning(
                        "inject_log %s:%d: skipping bad record: %s",
                        p, lineno, exc,
                    )
                    continue
                if since_unix is not None and rec.fired_at_unix < since_unix:
                    continue
                prev = latest.get(rec.inj_id)
                if prev is None or rec.fired_at_unix >= prev.fired_at_unix:
                    latest[rec.inj_id] = rec
    except OSError as exc:
        LOG.warning("load_fired_injections(%s) failed: %s", p, exc)
        return []
    return list(latest.values())


def _mjd_to_unix(mjd: float) -> Optional[float]:
    if not (isinstance(mjd, (int, float)) and math.isfinite(mjd)) or mjd <= 0.0:
        return None
    return (float(mjd) - _MJD_UNIX_EPOCH) * 86400.0


def event_coincident_inj_id(
    fired: Iterable[FiredInjection],
    *,
    mjd: float,
    dm_pc_cc: float,
    l_rad: float,
    m_rad: float,
    dm_tol_frac: float = DEFAULT_COINC_DM_TOL_FRAC,
    lm_tol_rad: float = DEFAULT_COINC_LM_TOL_RAD,
    wall_pre_s: float = DEFAULT_COINC_WALL_PRE_S,
    wall_post_s: float = DEFAULT_COINC_WALL_POST_S,
) -> Optional[str]:
    """Return the ``inj_id`` of a fired injection coincident with one
    archived candidate row, else ``None``.

    Coincidence = DM (fractional) AND sky (l/m) AND wall-clock time all
    within tolerance. Time uses the row's MJD vs the injection's
    ``fired_at_unix``; a row with no usable MJD skips the time gate (DM +
    sky only) since the whole point is a generous KEEP-side backstop.
    """
    row_unix = _mjd_to_unix(mjd)
    best: Optional[str] = None
    best_lm = float("inf")
    for inj in fired:
        if inj.dm_pc_cm3 == 0.0 or not math.isfinite(inj.dm_pc_cm3):
            continue
        try:
            dm_diff = abs(float(dm_pc_cc) - inj.dm_pc_cm3)
        except (TypeError, ValueError):
            continue
        if dm_diff > dm_tol_frac * abs(inj.dm_pc_cm3):
            continue
        try:
            lm_dist = math.hypot(
                float(l_rad) - inj.l_rad, float(m_rad) - inj.m_rad,
            )
        except (TypeError, ValueError):
            continue
        if lm_dist > lm_tol_rad:
            continue
        if row_unix is not None:
            dt = row_unix - inj.fired_at_unix
            if not (-wall_pre_s <= dt <= wall_post_s):
                continue
        if lm_dist < best_lm:
            best_lm = lm_dist
            best = inj.inj_id
    return best

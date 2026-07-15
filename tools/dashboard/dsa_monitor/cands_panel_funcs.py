"""Burst-candidates dashboard panel helpers (h23 dsa_monitor).

The panel reads ``/dataz/dsa110/candidates/<name>/`` (populated by
``dsart_c2.service``) and renders:

  * a recent-events table on ``/bursts``
  * a per-event detail view on ``/bursts/<name>`` that surfaces the
    Level3 JSON metadata + the four PNGs in ``Level2/plots/``.

Module-level constants are tuned for the h23 deployment but can be
overridden via environment variables (``CANDS_ARCHIVE_ROOT``,
``CANDS_MAX_EVENTS``) for dev hosts that don't have ``/dataz``.

Pure-Python (stdlib + Flask), so it stays inside the existing
``dsa_monitor`` deployment surface and doesn't drag any heavy ML
dependencies into the Flask process.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "ArchiveBrowser",
    "EventSummary",
    "EventDetail",
    "DEFAULT_ARCHIVE_ROOT",
    "DEFAULT_MAX_EVENTS",
    "ZOMBIE_PENDING_AGE_S",
    "C3_ERA_START_UNIX",
    "is_zombie",
    "partition_events_c3",
]


_LOG = logging.getLogger("dsa_monitor.cands")


DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get("CANDS_ARCHIVE_ROOT", "/dataz/dsa110/candidates")
)
DEFAULT_MAX_EVENTS = int(os.environ.get("CANDS_MAX_EVENTS", "200"))


_EVENT_NAME_RE = re.compile(r"^\d{6}[a-z]{4}$")
_SAFE_EVENT_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


#: Fragments a complete voltage collection holds (one per corr node).
N_VOLTAGE_FRAGMENTS_TOTAL = 16


@dataclass(frozen=True)
class EventSummary:
    """One row in the events table."""

    name: str
    mtime_unix: float
    mjd_peak: Optional[float]
    trigger_class: Optional[str]
    n_events: Optional[int]
    snr_max: Optional[float]
    dm_median: Optional[float]
    l_median: Optional[float]
    m_median: Optional[float]
    n_cubes: int
    n_plots: int
    # C3 cube-veto decision (from C3_decision.json; None = C3 has not
    # processed the event yet).
    c3_action: Optional[str] = None          # "KEEP" | "REJECT"
    c3_rules: Tuple[str, ...] = ()
    c3_notes: Optional[str] = None
    c3_is_injection: Optional[bool] = None
    c3_flag_only: Optional[bool] = None
    # Voltage fragments actually on h23 under Level2/voltages/.
    n_voltages: int = 0
    # Event time of day (UTC HH:MM:SS), from t_peak_mjd with a
    # dir-mtime fallback (suffixed '~' to mark it approximate).
    utc_hms: Optional[str] = None
    # Whether Level3/<name>.json exists on h23. Used by the zombie
    # partition: a genuine zombie never received its Level3 metadata.
    has_l3: bool = False

    @property
    def c3_status(self) -> str:
        """'pass' (KEEP), 'fail' (REJECT) or 'pending' (no decision)."""
        if self.c3_action == "KEEP":
            return "pass"
        if self.c3_action == "REJECT":
            return "fail"
        return "pending"


#: A "C3 pending" event older than this is a zombie: C1/C2 triggered but
#: cubes/metadata never landed on h23, so C3 will never judge it.
ZOMBIE_PENDING_AGE_S = 3600.0

#: MJD of the unix epoch (1970-01-01).
_MJD_UNIX_EPOCH = 40587.0

#: When the C3 cube veto went live (2026-06-01 UTC). Dirs untouched
#: since before this predate the veto entirely — "C3 pending" is not a
#: meaningful (stuck) state for them, so they can never be zombies.
C3_ERA_START_UNIX = 1780272000.0  # 2026-06-01T00:00:00Z


def _event_age_s(event: "EventSummary", now_unix: float) -> float:
    """Trigger age in seconds. Uses ``t_peak_mjd`` (via ``mjd_peak``)
    when the Level3 metadata provided one; zombies typically have NO
    Level3 on h23, so the realistic source is the event dir mtime
    (``mtime_unix``), which C2 stamps at trigger time."""
    if event.mjd_peak is not None:
        return now_unix - (event.mjd_peak - _MJD_UNIX_EPOCH) * 86400.0
    return now_unix - event.mtime_unix


def is_zombie(event: "EventSummary", now_unix: float) -> bool:
    """True iff C3 has not judged the event and it is old enough that
    it never will (cube/metadata transfer to h23 failed).

    Two gates keep non-events out (2026-07-15 archive audit):

    * real event-name pattern (``YYMMDD`` + 4 lowercase letters,
      :data:`_EVENT_NAME_RE`) — excludes calibrator scans (3C286\\*),
      synthetic/manual test triggers (dtest, dumpnow_\\*, DUMP1-4)
      and misc dirs (releases);
    * no ``Level3/<name>.json`` on disk (``has_l3`` False) — a genuine
      zombie is one whose metadata never landed. This excludes the
      real-named pre-C3-era events whose legacy Level3 JSON is present
      but which the C3 veto (which didn't exist then) never judged;
    * dir touched since :data:`C3_ERA_START_UNIX` — legacy dirs
      untouched since before the C3 veto existed were never going to be
      judged, so "pending" is not a stuck state for them.

    Everything excluded keeps today's behaviour (PASS tab).
    """
    return (
        bool(_EVENT_NAME_RE.match(event.name))
        and event.c3_status == "pending"
        and not event.has_l3
        and event.mtime_unix >= C3_ERA_START_UNIX
        and _event_age_s(event, now_unix) > ZOMBIE_PENDING_AGE_S
    )


def partition_events_c3(
    events: List["EventSummary"],
    now_unix: Optional[float] = None,
) -> Tuple[List["EventSummary"], List["EventSummary"], List["EventSummary"]]:
    """Split events into the three /bursts tabs.

    Returns ``(events_pass, events_fail, events_zombie)``:

      * pass    — C3 KEEP, plus *fresh* pending events (< 1 h old): the
                  operator wants new triggers front and centre while C3
                  is still expected to judge them.
      * fail    — C3 REJECT.
      * zombie  — pending for > :data:`ZOMBIE_PENDING_AGE_S`: the
                  transfer/copy path failed, C3 will never run.
    """
    now = time.time() if now_unix is None else now_unix
    ev_pass: List[EventSummary] = []
    ev_fail: List[EventSummary] = []
    ev_zombie: List[EventSummary] = []
    for e in events:
        if e.c3_status == "fail":
            ev_fail.append(e)
        elif is_zombie(e, now):
            ev_zombie.append(e)
        else:
            ev_pass.append(e)
    return ev_pass, ev_fail, ev_zombie


@dataclass(frozen=True)
class EventDetail:
    """Per-event detail page data."""

    name: str
    archive_dir: Path
    metadata: Mapping[str, Any]
    plots: Tuple[str, ...]  # relative filenames under Level2/plots/
    cubes: Tuple[str, ...]  # filenames under cubes/
    has_c2_csv: bool
    has_c1_csv: bool
    c3_decision: Optional[Mapping[str, Any]] = None
    n_voltages: int = 0
    # bbproc coherent-filterbank products under <cand>/filterbank/
    fil_plots: Tuple[str, ...] = ()   # inspection PNGs
    fil_files: Tuple[str, ...] = ()   # .fil filenames (size-linked only)
    fil_meta: Optional[Mapping[str, Any]] = None  # filterbank.json


class ArchiveBrowser:
    """Filesystem-only browser for ``<root>/<name>/``."""

    def __init__(
        self,
        root: Path = DEFAULT_ARCHIVE_ROOT,
        max_events: int = DEFAULT_MAX_EVENTS,
    ) -> None:
        self._root = Path(root)
        self._max_events = max_events

    @property
    def root(self) -> Path:
        return self._root

    @property
    def is_available(self) -> bool:
        return self._root.is_dir()

    # ----- table view -----------------------------------------------------

    def list_events(self) -> List[EventSummary]:
        """Return up to ``max_events`` summaries, newest first."""
        if not self.is_available:
            return []
        candidates: List[Tuple[float, Path]] = []
        try:
            entries = list(self._root.iterdir())
        except OSError as exc:
            _LOG.warning("list_events: %s readdir failed: %s",
                         self._root, exc)
            return []
        for p in entries:
            if not p.is_dir():
                continue
            if not _looks_like_event_dir(p):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, p))
        candidates.sort(reverse=True)
        out: List[EventSummary] = []
        for mtime, p in candidates[: self._max_events]:
            out.append(self._summarise(p, mtime))
        return out

    def _summarise(self, event_dir: Path, mtime: float) -> EventSummary:
        name = event_dir.name
        meta = self._read_l3(event_dir, name)
        c2 = (meta or {}).get("c2", {}) if isinstance(meta, dict) else {}
        trigger = (meta or {}).get("trigger", {}) if isinstance(meta, dict) else {}
        n_cubes = _count_files(event_dir / "cubes", "cube_s*_g*_*.npz")
        n_plots = _count_files(event_dir / "Level2" / "plots", "*.png")
        c3 = self._read_c3(event_dir) or {}
        mjd_peak = _as_optional_float(c2.get("t_peak_mjd"))
        return EventSummary(
            name=name,
            mtime_unix=mtime,
            mjd_peak=mjd_peak,
            trigger_class=_as_optional_str(trigger.get("class")),
            n_events=_as_optional_int(c2.get("n_events")),
            snr_max=_as_optional_float(c2.get("snr_max")),
            dm_median=_as_optional_float(c2.get("dm_median")),
            l_median=_as_optional_float(c2.get("l_median")),
            m_median=_as_optional_float(c2.get("m_median")),
            n_cubes=n_cubes,
            n_plots=n_plots,
            c3_action=_as_optional_str(c3.get("action")),
            c3_rules=tuple(str(r) for r in (c3.get("rules_fired") or [])),
            c3_notes=_as_optional_str(c3.get("notes")) or None,
            c3_is_injection=(
                bool(c3["is_injection"]) if "is_injection" in c3 else None
            ),
            c3_flag_only=(
                bool(c3["flag_only"]) if "flag_only" in c3 else None
            ),
            n_voltages=_count_files(
                event_dir / "Level2" / "voltages", "*_data.out"
            ),
            utc_hms=_utc_hms(mjd_peak, mtime),
            has_l3=meta is not None,
        )

    # ----- detail view ----------------------------------------------------

    def event_detail(self, name: str) -> Optional[EventDetail]:
        if not _SAFE_EVENT_RE.match(name):
            return None
        event_dir = self._root / name
        if not event_dir.is_dir():
            return None
        meta = self._read_l3(event_dir, name) or {}
        plots_dir = event_dir / "Level2" / "plots"
        plots: List[str] = []
        if plots_dir.is_dir():
            for p in sorted(plots_dir.iterdir()):
                if p.suffix.lower() == ".png":
                    plots.append(p.name)
        cubes_dir = event_dir / "cubes"
        cubes: List[str] = []
        if cubes_dir.is_dir():
            for p in sorted(cubes_dir.iterdir()):
                if p.suffix.lower() == ".npz":
                    cubes.append(p.name)
        fb_dir = event_dir / "filterbank"
        fil_plots: List[str] = []
        fil_files: List[str] = []
        fil_meta = None
        if fb_dir.is_dir():
            for p in sorted(fb_dir.iterdir()):
                if p.suffix.lower() == ".png":
                    fil_plots.append(p.name)
                elif p.suffix.lower() == ".fil":
                    fil_files.append(p.name)
            mpath = fb_dir / "filterbank.json"
            if mpath.is_file():
                try:
                    fil_meta = json.loads(mpath.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    fil_meta = None
        return EventDetail(
            name=name,
            archive_dir=event_dir,
            metadata=meta,
            plots=tuple(plots),
            cubes=tuple(cubes),
            has_c2_csv=(event_dir / "Level2" / f"C2_{name}.csv").is_file(),
            has_c1_csv=(
                event_dir / "Level2" / f"C1_window_{name}.csv"
            ).is_file(),
            c3_decision=self._read_c3(event_dir),
            n_voltages=_count_files(
                event_dir / "Level2" / "voltages", "*_data.out"
            ),
            fil_plots=tuple(fil_plots),
            fil_files=tuple(fil_files),
            fil_meta=fil_meta,
        )

    def fil_plot_path(self, name: str, plot_name: str) -> Optional[Path]:
        """Resolve a filterbank inspection PNG path, refusing traversal."""
        if not _SAFE_EVENT_RE.match(name):
            return None
        if "/" in plot_name or ".." in plot_name:
            return None
        if not plot_name.endswith(".png"):
            return None
        p = self._root / name / "filterbank" / plot_name
        try:
            p_resolved = p.resolve()
            root_resolved = (self._root / name).resolve()
        except OSError:
            return None
        if not str(p_resolved).startswith(str(root_resolved)):
            return None
        return p_resolved if p_resolved.is_file() else None

    def plot_path(self, name: str, plot_name: str) -> Optional[Path]:
        """Resolve a plot PNG path, refusing traversal."""
        if not _SAFE_EVENT_RE.match(name):
            return None
        if "/" in plot_name or ".." in plot_name:
            return None
        if not plot_name.endswith(".png"):
            return None
        p = self._root / name / "Level2" / "plots" / plot_name
        try:
            p_resolved = p.resolve()
            root_resolved = (self._root / name).resolve()
        except OSError:
            return None
        if not str(p_resolved).startswith(str(root_resolved)):
            return None
        return p_resolved if p_resolved.is_file() else None

    # ----- internals ------------------------------------------------------

    @staticmethod
    def _read_l3(event_dir: Path, name: str) -> Optional[dict]:
        path = event_dir / "Level3" / f"{name}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("read_l3 %s failed: %s", path, exc)
            return None

    @staticmethod
    def _read_c3(event_dir: Path) -> Optional[dict]:
        """The C3 cube-veto audit sidecar (written by
        ``dsart.services.c3.C3Service._write_audit``); absent until C3
        processes the event (~minutes after the trigger)."""
        path = event_dir / "C3_decision.json"
        if not path.is_file():
            return None
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("read_c3 %s failed: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _looks_like_event_dir(p: Path) -> bool:
    """Heuristic: a YYMMDDxxxx-shaped name is the canonical event-name
    layout; we also accept dirs that contain a Level3/<name>.json so a
    re-named or replayed event still shows up.
    """
    if _EVENT_NAME_RE.match(p.name):
        return True
    if (p / "Level3" / f"{p.name}.json").is_file():
        return True
    return False


def _count_files(d: Path, glob: str) -> int:
    if not d.is_dir():
        return 0
    try:
        return sum(1 for _ in d.glob(glob))
    except OSError:
        return 0


def _as_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v)


def _utc_hms(mjd: Optional[float], mtime_unix: float) -> Optional[str]:
    """UTC time-of-day for the table. Prefers the C2 peak MJD; falls
    back to the directory mtime with a '~' suffix so the operator can
    tell it's the archive-write time, not the burst time."""
    from datetime import datetime, timezone
    if mjd is not None:
        try:
            dt = datetime.fromtimestamp(
                (float(mjd) - 40587.0) * 86400.0, tz=timezone.utc
            )
            return dt.strftime("%H:%M:%S")
        except (ValueError, OverflowError, OSError):
            pass
    try:
        dt = datetime.fromtimestamp(float(mtime_unix), tz=timezone.utc)
        return dt.strftime("%H:%M:%S") + "~"
    except (ValueError, OverflowError, OSError):
        return None

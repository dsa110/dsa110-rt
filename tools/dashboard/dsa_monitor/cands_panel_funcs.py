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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "ArchiveBrowser",
    "EventSummary",
    "EventDetail",
    "DEFAULT_ARCHIVE_ROOT",
    "DEFAULT_MAX_EVENTS",
]


_LOG = logging.getLogger("dsa_monitor.cands")


DEFAULT_ARCHIVE_ROOT = Path(
    os.environ.get("CANDS_ARCHIVE_ROOT", "/dataz/dsa110/candidates")
)
DEFAULT_MAX_EVENTS = int(os.environ.get("CANDS_MAX_EVENTS", "200"))


_EVENT_NAME_RE = re.compile(r"^\d{6}[a-z]{4}$")
_SAFE_EVENT_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class EventDetail:
    """Per-event detail page data."""

    name: str
    archive_dir: Path
    metadata: Mapping[str, Any]
    plots: Tuple[str, ...]  # relative filenames under Level2/plots/
    cubes: Tuple[str, ...]  # filenames under cubes/
    has_c2_csv: bool
    has_c1_csv: bool


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
        return EventSummary(
            name=name,
            mtime_unix=mtime,
            mjd_peak=_as_optional_float(c2.get("t_peak_mjd")),
            trigger_class=_as_optional_str(trigger.get("class")),
            n_events=_as_optional_int(c2.get("n_events")),
            snr_max=_as_optional_float(c2.get("snr_max")),
            dm_median=_as_optional_float(c2.get("dm_median")),
            l_median=_as_optional_float(c2.get("l_median")),
            m_median=_as_optional_float(c2.get("m_median")),
            n_cubes=n_cubes,
            n_plots=n_plots,
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
        )

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

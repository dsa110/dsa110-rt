"""Read-only view of the SEFD scanner state for native rendering in
``dsa_monitor``.

The SEFD scanner (``sefd_dashboard.service``) runs in a casa38 conda
env and writes two things to disk:

* a JSON state file (``state.json``) keyed by ``<YYYY-MM-DD>_<src>``
  with status / metrics / full_metrics dicts per calibrator MS, and
* a tree of PNG diagnostic plots under ``results/<src>/<date>/``
  (with a deeper ``sefd/`` subdirectory once the full pipeline ran).

``dsa_monitor`` runs in the ``dsart_h23`` conda env and does **not**
import the casa38 stack; instead it consumes the scanner's outputs
read-only.  This module is the seam between the two: pure ``json``
parsing + ``os.listdir`` walks, with bounded results and a tiny
mtime-keyed cache so the dashboard never blocks on disk for more than
a millisecond on a hot path.

The Flask handlers in ``app.py`` (``/sefds``, ``/sefds/source/<name>``,
``/sefds/day/<date>``, ``/sefds/results/<path>``) all go through the
``SefdView`` singleton defined here.  No write paths live in this
module — the scanner remains the sole writer of ``state.json``.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dsa_monitor.sefd_view")

# ---------------------------------------------------------------------------
# Defaults (overridable via env in app.py)
# ---------------------------------------------------------------------------

DEFAULT_STATE_FILE = "/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/state.json"
DEFAULT_RESULTS_DIR = "/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/results"
# Heartbeat file touched by the scanner every poll cycle
# (whether or not anything changed in state.json).  We prefer this
# over state.json's mtime for the liveness pill so a quiet night
# doesn't show as a dead scanner.  Conventionally lives next to
# state.json under the same directory.
DEFAULT_HEARTBEAT_FILENAME = "scanner_heartbeat"

# Hard-coded calibrator catalog (kept in sync with the scanner's
# ``SOURCES`` dict so the dashboard renders the right flux annotation
# even before the scanner has produced any state entries).  If the
# scanner adds a new source, add it here too.
DEFAULT_SOURCES: Dict[str, Dict[str, float]] = {
    "0318+164": {"flux_jy": 7.81},
    "0521+166": {"flux_jy": 8.47},
    "2253+161": {"flux_jy": 10.0},
}

# Status strings the scanner writes.  Kept in one place so the
# template renderers and tests agree.
STATUS_PENDING = "pending"
STATUS_LIGHT_PROCESSING = "light_processing"
STATUS_LIGHT_DONE = "light_done"
STATUS_FULL_PROCESSING = "full_processing"
STATUS_COMPLETE = "complete"
STATUS_LIGHT_ERROR = "light_error"
STATUS_FULL_ERROR = "full_error"
STATUS_ERROR = "error"

ERROR_STATUSES = frozenset(
    [STATUS_LIGHT_ERROR, STATUS_FULL_ERROR, STATUS_ERROR],
)

# When the scanner has run at least light_diagnostics for an entry.
HAS_METRICS_STATUSES = frozenset(
    [STATUS_LIGHT_DONE, STATUS_FULL_PROCESSING, STATUS_COMPLETE],
)


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class SefdEntry:
    """One ``state.json`` entry, normalised for template use."""

    key: str
    date: str
    source: str
    status: str
    updated: Optional[str] = None
    path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    full_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.status in ERROR_STATUSES

    @property
    def has_metrics(self) -> bool:
        return self.status in HAS_METRICS_STATUSES and bool(self.metrics)

    def updated_age_s(self, now_unix: Optional[float] = None) -> Optional[float]:
        if not self.updated:
            return None
        try:
            dt = datetime.fromisoformat(self.updated)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if now_unix is None:
            now_unix = datetime.now(timezone.utc).timestamp()
        return now_unix - dt.timestamp()


@dataclass
class SefdSummary:
    """Top-level summary payload for the SEFD landing page."""

    dates: List[str]
    sources: List[str]
    source_flux: Dict[str, Dict[str, float]]
    grid: Dict[str, Dict[str, Optional[SefdEntry]]]
    lookback_days: int
    state_path: str
    state_mtime_unix: Optional[float]
    scanner_alive: bool
    scanner_age_s: Optional[float]
    currently_processing: Optional[str]


# ---------------------------------------------------------------------------
# SefdView
# ---------------------------------------------------------------------------


# Scanner heartbeat: the scanner rewrites state.json every time an entry
# transitions (light_processing / light_done / full_processing /
# complete / *_error).  If state.json hasn't been touched in this many
# seconds AND the scanner has at least one MS to chew on, we report
# the scanner as stale on the dashboard.  Conservatively long (1 h) so
# we don't false-alarm on legitimate idle nights.
SCANNER_STALE_S = 3600.0

# Don't keep the state file open for more than a few hundred ms; if a
# concurrent scanner write trips us up we'll just re-read on the next
# poll.
_OPEN_TIMEOUT_S = 2.0


class SefdView:
    """Read-only access to the scanner's on-disk outputs.

    Thread-safe (a single internal ``threading.Lock`` guards the
    mtime-keyed state cache).  Construction is cheap; the underlying
    ``state.json`` is re-read only when its mtime changes, so a hot
    page render never re-parses the file.

    Constructor parameters:

    state_file
        Path to the scanner's JSON state file.  Defaults to the
        sefd_dashboard.service location on h23.
    results_dir
        Path to the scanner's per-source plot tree.  Defaults to the
        sibling ``results/`` of the state file's repo.
    sources
        Optional override of the source catalog.  Keys are source
        names, values are ``{"flux_jy": float}`` dicts.  Defaults to
        :data:`DEFAULT_SOURCES`.
    """

    def __init__(
        self,
        state_file: str = DEFAULT_STATE_FILE,
        results_dir: str = DEFAULT_RESULTS_DIR,
        sources: Optional[Dict[str, Dict[str, float]]] = None,
        heartbeat_file: Optional[str] = None,
    ) -> None:
        self.state_file = state_file
        self.results_dir = results_dir
        self.sources: Dict[str, Dict[str, float]] = dict(
            sources if sources is not None else DEFAULT_SOURCES
        )
        if heartbeat_file is None:
            heartbeat_file = os.path.join(
                os.path.dirname(state_file) or ".",
                DEFAULT_HEARTBEAT_FILENAME,
            )
        self.heartbeat_file = heartbeat_file
        self._lock = threading.Lock()
        self._cache_mtime: Optional[float] = None
        self._cache_state: Dict[str, Dict[str, Any]] = {}
        self._cache_error: Optional[str] = None

    # ----- low-level: state.json + filesystem -----------------------------

    def _read_state_unlocked(self) -> Dict[str, Dict[str, Any]]:
        """Read ``state.json``, using the mtime-keyed cache.

        Returns an empty dict (and stores ``self._cache_error``) if the
        file is missing / unreadable / malformed.  Never raises.
        """
        import json

        if not os.path.exists(self.state_file):
            self._cache_error = f"state file missing: {self.state_file}"
            self._cache_state = {}
            self._cache_mtime = None
            return self._cache_state

        try:
            mtime = os.path.getmtime(self.state_file)
        except OSError as exc:
            self._cache_error = f"stat failed: {exc}"
            self._cache_state = {}
            self._cache_mtime = None
            return self._cache_state

        if self._cache_mtime is not None and mtime == self._cache_mtime:
            return self._cache_state

        try:
            with open(self.state_file, "r") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self._cache_error = f"parse failed: {exc}"
            return self._cache_state  # last good state stays cached

        if not isinstance(data, dict):
            self._cache_error = "state.json root is not a dict"
            return self._cache_state

        self._cache_state = data
        self._cache_mtime = mtime
        self._cache_error = None
        return self._cache_state

    def _read_state(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._read_state_unlocked())

    def state_mtime(self) -> Optional[float]:
        """Return the cached mtime of ``state.json`` (epoch seconds)."""
        with self._lock:
            self._read_state_unlocked()
            return self._cache_mtime

    def state_error(self) -> Optional[str]:
        """Return the last load error, or ``None`` if the file is good."""
        with self._lock:
            self._read_state_unlocked()
            return self._cache_error

    # ----- entry parsing --------------------------------------------------

    @staticmethod
    def _entry_from_raw(key: str, raw: Dict[str, Any]) -> SefdEntry:
        """Normalise a single ``state.json`` value into a SefdEntry.

        Tolerates missing fields: an old-format entry without
        ``date`` / ``source`` falls back to parsing the
        ``<date>_<source>`` key, and any unknown status passes through
        verbatim.  Float-cast on metric values is left to the template
        renderers (the source JSON is already JSON-numeric).
        """
        date = str(raw.get("date") or "")
        source = str(raw.get("source") or "")
        if not (date and source) and "_" in key:
            head, _, tail = key.partition("_")
            date = date or head
            source = source or tail
        status = str(raw.get("status") or STATUS_PENDING)
        metrics_raw = raw.get("metrics")
        full_raw = raw.get("full_metrics")
        return SefdEntry(
            key=key,
            date=date,
            source=source,
            status=status,
            updated=raw.get("updated"),
            path=raw.get("path"),
            metrics=dict(metrics_raw) if isinstance(metrics_raw, dict) else {},
            full_metrics=(
                dict(full_raw) if isinstance(full_raw, dict) else {}
            ),
            error=raw.get("error"),
        )

    # ----- summary --------------------------------------------------------

    def summary(self, lookback_days: int = 7) -> SefdSummary:
        """Build the grid + nav payload for the ``/sefds`` landing page.

        ``lookback_days`` is clamped to ``[1, 365]``.  Only state.json
        entries whose ``date`` parses as ISO yyyy-mm-dd are included
        in the grid (malformed dates are dropped silently rather than
        crashing the page); these always remain in the per-source
        listing accessible via :meth:`source_entries`.
        """
        lookback_days = max(1, min(int(lookback_days), 365))
        raw = self._read_state()
        today = datetime.now(timezone.utc).date()
        cutoff_days = lookback_days

        sources = sorted(self.sources.keys())

        kept_dates: Dict[str, bool] = {}
        entries_by_key: Dict[str, SefdEntry] = {}
        for key, val in raw.items():
            if not isinstance(val, dict):
                continue
            entry = self._entry_from_raw(key, val)
            try:
                obs_date = datetime.strptime(entry.date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if (today - obs_date).days > cutoff_days or (
                today - obs_date
            ).days < 0:
                continue
            entries_by_key[entry.key] = entry
            kept_dates[entry.date] = True

        dates = sorted(kept_dates.keys(), reverse=True)

        grid: Dict[str, Dict[str, Optional[SefdEntry]]] = {}
        for d in dates:
            row: Dict[str, Optional[SefdEntry]] = {}
            for s in sources:
                row[s] = entries_by_key.get(f"{d}_{s}")
            grid[d] = row

        state_mtime = self.state_mtime()
        # Prefer the scanner's heartbeat file for liveness so a quiet
        # night (no transitions in state.json) doesn't false-alarm.
        # Fall back to state.json if the heartbeat file isn't there
        # yet (e.g. very first scanner cycle, or pre-upgrade scanner).
        try:
            hb_mtime = (
                os.path.getmtime(self.heartbeat_file)
                if os.path.exists(self.heartbeat_file)
                else None
            )
        except OSError:
            hb_mtime = None
        liveness_mtime = hb_mtime if hb_mtime is not None else state_mtime
        scanner_age = (
            (datetime.now(timezone.utc).timestamp() - liveness_mtime)
            if liveness_mtime is not None
            else None
        )
        scanner_alive = (
            (scanner_age is not None) and (scanner_age < SCANNER_STALE_S)
        )

        # Reverse-derive ``currently_processing`` from state: the
        # scanner sets *_processing on the entry it's working on and
        # clears it as soon as the next status lands.  This keeps us
        # in sync with the scanner without a separate IPC channel.
        currently_processing: Optional[str] = None
        for k, e in entries_by_key.items():
            if e.status in (STATUS_LIGHT_PROCESSING, STATUS_FULL_PROCESSING):
                currently_processing = k
                break

        return SefdSummary(
            dates=dates,
            sources=sources,
            source_flux=dict(self.sources),
            grid=grid,
            lookback_days=lookback_days,
            state_path=self.state_file,
            state_mtime_unix=state_mtime,
            scanner_alive=scanner_alive,
            scanner_age_s=scanner_age,
            currently_processing=currently_processing,
        )

    # ----- per-source / per-day -------------------------------------------

    def source_entries(
        self, source_name: str, lookback_days: int = 7,
    ) -> List[SefdEntry]:
        """All entries for one calibrator, newest-first.

        Returns the full set of non-pending entries (so the per-source
        page can show errors too).  ``lookback_days`` is applied to
        ``entry.date``.
        """
        if source_name not in self.sources:
            return []
        lookback_days = max(1, min(int(lookback_days), 365))
        raw = self._read_state()
        today = datetime.now(timezone.utc).date()
        out: List[SefdEntry] = []
        for key, val in raw.items():
            if not isinstance(val, dict):
                continue
            entry = self._entry_from_raw(key, val)
            if entry.source != source_name:
                continue
            if entry.status == STATUS_PENDING:
                continue
            try:
                obs_date = datetime.strptime(entry.date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if (today - obs_date).days > lookback_days:
                continue
            out.append(entry)
        out.sort(key=lambda e: e.date, reverse=True)
        return out

    def day_entries(self, date: str) -> Dict[str, SefdEntry]:
        """All entries for one observation date, keyed by source.

        Sources without an entry for the date are omitted (the
        renderer should look at ``self.sources`` for the canonical
        list and treat missing ones as ``---``).
        """
        if not _is_iso_date(date):
            return {}
        raw = self._read_state()
        out: Dict[str, SefdEntry] = {}
        for key, val in raw.items():
            if not isinstance(val, dict):
                continue
            entry = self._entry_from_raw(key, val)
            if entry.date != date:
                continue
            if entry.source not in self.sources:
                continue
            out[entry.source] = entry
        return out

    # ----- plots ----------------------------------------------------------

    def list_day_plots(self, source: str, date: str) -> Dict[str, str]:
        """Return ``{filename: url_path}`` for every PNG in the
        ``results/<source>/<date>/`` tree (including the ``sefd/``
        subdir).  ``url_path`` is the path the Flask
        ``/sefds/results/<path>`` route will serve.

        Returns an empty dict if the directory doesn't exist.
        """
        if source not in self.sources or not _is_iso_date(date):
            return {}
        base = os.path.join(self.results_dir, source, date)
        if not os.path.isdir(base):
            return {}
        plots: Dict[str, str] = {}
        for fname in sorted(os.listdir(base)):
            full = os.path.join(base, fname)
            if os.path.isfile(full) and fname.lower().endswith(".png"):
                plots[fname] = self._results_url(source, date, fname)
        sefd_sub = os.path.join(base, "sefd")
        if os.path.isdir(sefd_sub):
            for fname in sorted(os.listdir(sefd_sub)):
                full = os.path.join(sefd_sub, fname)
                if os.path.isfile(full) and fname.lower().endswith(".png"):
                    plots[f"sefd/{fname}"] = self._results_url(
                        source, date, f"sefd/{fname}",
                    )
        return plots

    @staticmethod
    def _results_url(source: str, date: str, fname: str) -> str:
        return f"/sefds/results/{source}/{date}/{fname}"

    def resolve_plot_path(self, rel_path: str) -> Optional[str]:
        """Resolve a ``/sefds/results/...`` URL to an absolute path.

        Returns ``None`` if the path tries to escape ``results_dir``
        (path-traversal guard) or doesn't resolve to an existing PNG
        under the results tree.  This is the function the Flask
        ``send_file`` handler must consult before serving any byte.
        """
        if not rel_path:
            return None
        cleaned = rel_path.lstrip("/")
        if cleaned.startswith("results/"):
            cleaned = cleaned[len("results/"):]
        base = os.path.realpath(self.results_dir)
        # Strip any leading slashes and collapse .. so realpath
        # below catches symlink escapes too.
        joined = os.path.normpath(os.path.join(base, cleaned))
        resolved = os.path.realpath(joined)
        if not (resolved == base or resolved.startswith(base + os.sep)):
            return None
        if not os.path.isfile(resolved):
            return None
        if not resolved.lower().endswith(".png"):
            return None
        return resolved


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_iso_date(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return False
    return True

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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from event_astrometry import (
    EventAstrometry,
    RaDec,
    sexagesimal_for,
)

__all__ = [
    "ArchiveBrowser",
    "EventIndexCache",
    "CacheSnapshot",
    "EventSummary",
    "EventDetail",
    "EventListing",
    "DEFAULT_ARCHIVE_ROOT",
    "DEFAULT_MAX_EVENTS",
    "DEFAULT_CACHE_TTL_S",
    "DEFAULT_ACTIVE_WINDOW_S",
    "DEFAULT_PAGE_SIZE",
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

#: Server-side event-index cache tunables. The archive lives on slow NFS
#: (/dataz); enumerating + reading metadata for every event dir on every
#: /bursts render is what made the page crawl once the age cap was removed.
#: The cache holds the WHOLE index in memory and refreshes it incrementally.
#:
#:  * TTL — max age of the cache before an access triggers a (cheap,
#:    incremental) refresh. One readdir+stat per dir on a warm cache; the
#:    expensive per-event reads run only for new/changed/active dirs.
#:  * ACTIVE window — freshly-triggered events keep mutating after their
#:    dir mtime settles (C3 writes its decision, voltage fragments stage in
#:    under Level2/voltages/ which does NOT bump the event-dir mtime). Any
#:    event younger than this is re-summarised on every refresh so those
#:    late-arriving fields stay current; older events are immutable and stay
#:    cached untouched.
#:  * PAGE size — default rows rendered per /bursts page (newest-first).
DEFAULT_CACHE_TTL_S = float(os.environ.get("CANDS_CACHE_TTL_S", "60"))
DEFAULT_ACTIVE_WINDOW_S = float(
    os.environ.get("CANDS_CACHE_ACTIVE_WINDOW_S", "21600")  # 6 h
)
DEFAULT_PAGE_SIZE = int(os.environ.get("CANDS_PAGE_SIZE", "200"))


#: Sentinel so _summarise can tell "meta not supplied" from "meta is None".
_UNSET = object()


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
    # ICRS J2000 sky position (deg) + provenance of the pointing-dec used
    # to derive it ("level3" | "filterbank" | "legacy" | None). None
    # ra/dec means no
    # pointing-dec source was available; the UI renders "—". The table
    # shows the sexagesimal strings; the degrees stay for the per-cell
    # tooltip and for API/test stability.
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    radec_source: Optional[str] = None
    ra_hms: Optional[str] = None      # "HH:MM:SS.s(NN)"; no (NN) for legacy
    dec_dms: Optional[str] = None     # "+DD:MM:SS(NN)"; no (NN) for legacy

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
    # ICRS J2000 sky position (deg + sexagesimal) + pointing-dec
    # provenance ("level3" | "filterbank" | "legacy" | None).
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    radec_source: Optional[str] = None
    ra_hms: Optional[str] = None      # "HH:MM:SS.s(NN)"; no (NN) for legacy
    dec_dms: Optional[str] = None     # "+DD:MM:SS(NN)"; no (NN) for legacy


@dataclass(frozen=True)
class EventListing:
    """Result of :meth:`ArchiveBrowser.list_events_detailed`.

    ``events`` is the merged newest-first table (the newest ``max_events``
    dirs by mtime, plus every human-annotated event that fell outside that
    window, re-sorted into chronological position). The counts drive the
    /bursts truncation notice:

      * ``n_total`` (M)     — every event dir on disk.
      * ``n_newest`` (N)    — how many of the newest-by-mtime dirs are
                              shown (``min(max_events, n_total)``).
      * ``n_annotated`` (K) — annotated events shown that are OLDER than
                              the newest-N window (added back so they are
                              never dropped by the cap).
      * ``truncated``       — True iff the cap actually bit
                              (``n_total > max_events``).
    """

    events: List[EventSummary]
    n_total: int
    n_newest: int
    n_annotated: int
    truncated: bool


class ArchiveBrowser:
    """Filesystem-only browser for ``<root>/<name>/``."""

    def __init__(
        self,
        root: Path = DEFAULT_ARCHIVE_ROOT,
        max_events: int = DEFAULT_MAX_EVENTS,
    ) -> None:
        self._root = Path(root)
        self._max_events = max_events
        # Per-process astrometry cache + per-render batcher. Persists
        # across list_events() calls so re-renders are free; keyed on the
        # exact inputs so a later filterbank.json recomputes.
        self._astro = EventAstrometry()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def is_available(self) -> bool:
        return self._root.is_dir()

    # ----- table view -----------------------------------------------------

    def list_events(self) -> List[EventSummary]:
        """Return the /bursts table rows, newest first.

        Thin wrapper over :meth:`list_events_detailed`; keeps the old
        signature for callers that only want the row list (the recent-
        events widgets)."""
        return self.list_events_detailed().events

    def scan_dirs(self) -> List[Tuple[float, Path]]:
        """Cheap archive enumeration: one readdir + one stat per event dir.

        Returns ``(mtime, path)`` for every dir that looks like an event
        (unsorted). Raises ``OSError`` if the root readdir itself fails
        (NFS down) — callers decide whether to serve a stale cache or an
        empty listing. Per-dir stat failures are skipped, not fatal.

        This is the only step that must touch NFS on a warm cache; the
        expensive per-event metadata reads live in :meth:`summarise_pairs`
        and are skipped for unchanged dirs by :class:`EventIndexCache`."""
        out: List[Tuple[float, Path]] = []
        for p in self._root.iterdir():          # OSError propagates
            if not p.is_dir():
                continue
            if not _looks_like_event_dir(p):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            out.append((mtime, p))
        return out

    def summarise_pairs(
        self, pairs: List[Tuple[float, Path]]
    ) -> List[EventSummary]:
        """Turn ``(mtime, path)`` pairs into :class:`EventSummary` rows.

        This is the expensive path: per event it reads Level3/C3 JSON and
        globs the cubes/plots/voltages dirs, then resolves every sky
        position in a single batched TETE->ICRS transform (see
        ``event_astrometry``). Order matches ``pairs``."""
        prepped: List[Tuple[Path, float, Optional[dict]]] = [
            (p, mtime, self._read_l3(p, p.name)) for mtime, p in pairs
        ]
        try:
            radecs = self._astro.compute(
                [(p.name, p, meta) for p, _, meta in prepped]
            )
        except Exception:                                      # noqa: BLE001
            _LOG.exception("astrometry batch failed; positions blank")
            radecs = {}
        return [
            self._summarise(p, mtime, meta=meta,
                            radec=radecs.get(p.name))
            for p, mtime, meta in prepped
        ]

    def list_events_detailed(
        self, annotated_ids: Optional[set] = None
    ) -> EventListing:
        """Return the events table + truncation counts.

        The row list is the UNION of (a) the newest ``max_events`` dirs by
        mtime and (b) EVERY event that carries a human annotation, so an
        annotated event is *never* dropped by the cap no matter how old it
        gets. The union is re-sorted newest-first, so old annotated events
        sit at their true chronological position (deep in the list —
        ``?source=`` / ``?tag=`` search finds them; they are NOT pinned to
        the top).

        ``annotated_ids`` is the set of annotated event names; when None it
        is read from the annotations DB (best-effort). It is a parameter so
        the union logic is unit-testable without a DB. Annotated ids with
        no candidate dir on disk are skipped (logged at debug) so a deleted
        archive dir never crashes the page. Cost stays at one readdir + one
        stat per dir (today's budget) plus one small DB query: annotated
        events outside the window reuse the stat already taken in the
        readdir pass."""
        if not self.is_available:
            return EventListing([], 0, 0, 0, False)
        try:
            candidates = self.scan_dirs()
        except OSError as exc:
            _LOG.warning("list_events: %s readdir failed: %s",
                         self._root, exc)
            return EventListing([], 0, 0, 0, False)
        candidates.sort(reverse=True)
        n_total = len(candidates)
        picked = candidates[: self._max_events]
        picked_names = {p.name for _, p in picked}

        # (b) Merge back annotated events that fell outside the newest-N
        # window. Reuse the mtimes already gathered above — no extra stat.
        if annotated_ids is None:
            annotated_ids = self._fetch_annotated_ids()
        extra: List[Tuple[float, Path]] = []
        if annotated_ids:
            by_name = {p.name: (mtime, p) for mtime, p in candidates}
            for eid in annotated_ids:
                if eid in picked_names:
                    continue
                hit = by_name.get(eid)
                if hit is None:
                    _LOG.debug(
                        "annotated event %s has no candidate dir under %s; "
                        "skipping", eid, self._root,
                    )
                    continue
                extra.append(hit)

        merged = picked + extra
        merged.sort(reverse=True)
        events = self.summarise_pairs(merged)
        return EventListing(
            events=events,
            n_total=n_total,
            n_newest=len(picked),
            n_annotated=len(extra),
            truncated=n_total > self._max_events,
        )

    @staticmethod
    def _fetch_annotated_ids() -> set:
        """Event ids with any human annotation, from the annotations DB.

        Best-effort + lazily imported so ``cands_panel_funcs`` stays
        importable (and unit-testable) on hosts without the annotations
        module or its DB — a failure just means no annotated events are
        pinned, i.e. today's cap-only behaviour."""
        try:
            import annotations as _ann  # sibling module, pure sqlite
            return _ann.annotated_events()
        except Exception:                                      # noqa: BLE001
            _LOG.exception(
                "annotated_events() failed; annotated-event pinning skipped")
            return set()

    def _summarise(
        self, event_dir: Path, mtime: float,
        meta: Any = _UNSET, radec: Optional[RaDec] = None,
    ) -> EventSummary:
        name = event_dir.name
        if meta is _UNSET:
            meta = self._read_l3(event_dir, name)
        rd = radec if radec is not None else RaDec(None, None, None)
        rd_hms, rd_dms = sexagesimal_for(rd)
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
            ra_deg=rd.ra_deg,
            dec_deg=rd.dec_deg,
            radec_source=rd.source,
            ra_hms=rd_hms,
            dec_dms=rd_dms,
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
        rd = self._astro.compute_one(name, event_dir, meta)
        rd_hms, rd_dms = sexagesimal_for(rd)
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
            ra_deg=rd.ra_deg,
            dec_deg=rd.dec_deg,
            radec_source=rd.source,
            ra_hms=rd_hms,
            dec_dms=rd_dms,
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
# Server-side cached event index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSnapshot:
    """Immutable view of the cached index handed to a request.

    ``events`` is the FULL index, newest-first (every event dir on disk,
    not a capped window). Rendering/pagination/search all operate on this
    in-memory list — no NFS access on the request path once warm."""

    events: List[EventSummary]
    n_total: int
    #: Wall-clock of the last *successful* refresh (unix seconds), or None
    #: on a cold cache that has never completed a scan.
    last_success_unix: Optional[float]
    #: True when the most recent refresh attempt failed (NFS down / readdir
    #: error) so the caller is serving a stale index. The page shows a
    #: "stale as of <time>" note instead of 500-ing.
    stale: bool
    #: Short reason string for the last failure (for the stale note), or None.
    error: Optional[str]
    #: True when no scan has EVER succeeded since process start — e.g. right
    #: after a dashboard restart, while the first (slow, NFS-bound) build is
    #: still in flight on a background thread. Distinct from ``stale``: a
    #: stale cache had a good index at some point and is failing to refresh
    #: it; a warming cache has never had one, so the empty listing/"0
    #: archived" totals below would otherwise look like data loss. Callers
    #: should show a reassuring "still building" banner instead of the
    #: normal empty-state / stale-state UI.
    #: Defaults to False so existing call sites (tests constructing a
    #: CacheSnapshot directly for a warm/mocked cache) don't need updating.
    warming: bool = False
    #: Best-effort progress hint for the in-flight cold build: number of
    #: event dirs the directory scan has found so far (before the slower
    #: per-event summarise pass). ``None`` if unknown (no build has reached
    #: that point yet) or once the cache has gone warm. Purely cosmetic —
    #: do not rely on it for correctness.
    scan_progress: Optional[int] = None


class EventIndexCache:
    """Thread-safe, incrementally-refreshed in-memory index of the event
    archive, layered over :class:`ArchiveBrowser`.

    Motivation: with the age cap lifted, ``/bursts`` was enumerating and
    reading metadata for the entire ``/dataz`` archive over NFS on every
    render, growing linearly with the archive (~400 dirs now, tens of
    thousands/yr expected). This cache does that work at most once per
    ``ttl_s`` and, on a warm cache, only re-reads dirs that are new,
    changed, or still "active" (young enough to still be mutating).

    Refresh policy (incremental):

      * Cold start / empty cache — full scan + summarise of every dir.
      * Warm access past the TTL — one readdir+stat sweep; re-summarise
        only dirs whose mtime changed or that are younger than the ACTIVE
        window (freshly-triggered events still accreting C3 decisions and
        voltage fragments). Everything else is served from cache.
      * ``refresh(force=True)`` — same sweep, TTL ignored (explicit).

    Failure policy: an NFS failure during a refresh never propagates and
    never clears the cache — the last good index is kept and served with
    :attr:`CacheSnapshot.stale` set. A cold cache that has never succeeded
    degrades to an empty, stale listing (still no 500).

    Concurrency: a single non-blocking refresh lock guarantees at most one
    thread scans NFS at a time; the others serve the current snapshot
    immediately rather than piling up behind the slow scan. A short-held
    data lock guards the index dicts. This suits Flask's threaded server
    (the dashboard serves requests on multiple worker threads)."""

    def __init__(
        self,
        browser: ArchiveBrowser,
        ttl_s: float = DEFAULT_CACHE_TTL_S,
        active_window_s: float = DEFAULT_ACTIVE_WINDOW_S,
    ) -> None:
        self._browser = browser
        self._ttl_s = float(ttl_s)
        self._active_window_s = float(active_window_s)
        self._data_lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._by_name: Dict[str, EventSummary] = {}
        self._dir_mtime: Dict[str, float] = {}
        self._built = False
        self._last_attempt = 0.0
        self._last_success: Optional[float] = None
        self._stale = False
        self._error: Optional[str] = None
        #: Dirs found by the most recent (possibly still in-flight) scan,
        #: for the cold-build progress hint. See CacheSnapshot.scan_progress.
        self._scan_progress: Optional[int] = None

    # ----- public API -----------------------------------------------------

    def snapshot(self, *, force_refresh: bool = False) -> CacheSnapshot:
        """Refresh if due (or forced), then return the full index.

        Cheap to call per request: the refresh is TTL-gated and single-
        flighted, so most calls just copy the in-memory index out."""
        self._maybe_refresh(force=force_refresh)
        with self._data_lock:
            events = sorted(
                self._by_name.values(),
                key=lambda e: e.mtime_unix,
                reverse=True,
            )
            warming = self._last_success is None
            return CacheSnapshot(
                events=events,
                n_total=len(events),
                last_success_unix=self._last_success,
                stale=self._stale,
                error=self._error,
                warming=warming,
                scan_progress=self._scan_progress if warming else None,
            )

    def invalidate(self) -> None:
        """Force the next :meth:`snapshot` to do a full re-scan."""
        with self._data_lock:
            self._built = False
            self._last_attempt = 0.0

    # ----- refresh internals ---------------------------------------------

    def _maybe_refresh(self, force: bool = False) -> None:
        now = time.time()
        with self._data_lock:
            due = (
                force
                or not self._built
                or (now - self._last_attempt) >= self._ttl_s
            )
        if not due:
            return
        # Single-flight: if another thread is already refreshing, don't
        # block this request behind the slow NFS scan — serve what we have.
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._do_refresh(now)
        finally:
            self._refresh_lock.release()

    def _do_refresh(self, now: float) -> None:
        # Take a lock-free copy of the current state to diff against, so the
        # (slow, NFS-bound) scan/summarise below holds no lock.
        with self._data_lock:
            self._last_attempt = now
            prev_summaries = dict(self._by_name)
            prev_mtime = dict(self._dir_mtime)

        if not self._browser.is_available:
            with self._data_lock:
                self._stale = True
                self._error = "archive root not present"
            _LOG.warning("event index: archive root %s not present; "
                         "serving stale (%d cached)",
                         self._browser.root, len(prev_summaries))
            return

        try:
            pairs = self._browser.scan_dirs()
        except OSError as exc:
            with self._data_lock:
                self._stale = True
                self._error = f"archive scan failed: {exc}"
            _LOG.warning("event index: scan of %s failed (%s); serving "
                         "stale (%d cached)",
                         self._browser.root, exc, len(prev_summaries))
            return

        with self._data_lock:
            self._scan_progress = len(pairs)

        current = {p.name: (mtime, p) for mtime, p in pairs}
        # Decide what needs the expensive per-event summarise: anything new,
        # anything whose dir mtime moved, and anything still young enough to
        # be accreting late fields (C3 decision, staged voltages).
        to_summarise: List[Tuple[float, Path]] = []
        for name, (mtime, p) in current.items():
            old = prev_mtime.get(name)
            if (
                old is None
                or old != mtime
                or (now - mtime) < self._active_window_s
            ):
                to_summarise.append((mtime, p))

        try:
            fresh = self._browser.summarise_pairs(to_summarise)
        except Exception as exc:                               # noqa: BLE001
            with self._data_lock:
                self._stale = True
                self._error = f"summarise failed: {exc}"
            _LOG.exception("event index: summarise of %d dirs failed; "
                           "serving stale", len(to_summarise))
            return
        fresh_by_name = {s.name: s for s in fresh}

        # Commit: rebuild the index from the on-disk name set (drops deleted
        # dirs), carrying forward cached summaries for untouched events.
        with self._data_lock:
            by_name: Dict[str, EventSummary] = {}
            dir_mtime: Dict[str, float] = {}
            for name, (mtime, _p) in current.items():
                summ = fresh_by_name.get(name) or prev_summaries.get(name)
                if summ is None:
                    # New dir that somehow wasn't summarised — skip rather
                    # than emit a half-built row.
                    continue
                by_name[name] = summ
                dir_mtime[name] = mtime
            self._by_name = by_name
            self._dir_mtime = dir_mtime
            self._built = True
            self._stale = False
            self._error = None
            self._last_success = now
            self._scan_progress = None
        _LOG.info("event index refreshed: %d events (%d re-read) from %s",
                  len(current), len(to_summarise), self._browser.root)


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

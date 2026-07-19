"""Search-node cube-staging retention sweeper.

Background (2026-07-19 disk-full incident)
------------------------------------------
Every C2 ``dump_all_gpus`` event stages ~1.1 GB of NPZ cubes per GPU
half under ``c1.dump_root``/<event>/ on each search node. The bounded
uploader (:mod:`dsart.coinc.cube_uploader`) rsyncs them to the h23
candidate archive but — until 2026-07-19 — nothing ever deleted the
local copies, so ~2.2 GB/event/node accumulated until all four search
nodes hit 100% root-disk, failing new dumps (0-byte NPZs) and every
other write under ``/home/ubuntu/data``.

Two complementary fixes:

* the uploader's ``purge_pattern`` deletes a half's own NPZs after a
  verified (rc==0) rsync and appends an ``UPLOAD_OK`` marker to the
  per-event ``upload.log`` (the primary, self-healing path);
* THIS sweeper is the last-resort backstop bounding the staging dir by
  age and total size, so upload outages / stranded events (e.g. C2
  ``events_incomplete_discarded`` like 260718hoxr, which never get an
  archive dir) cannot walk the disk to full again.

Sweep policy (per pass, in order):

1. tier 0 — hygiene: top-level ``*.tmp`` and 0-byte ``*.npz`` older
   than ``tmp_age_h`` (failed/interrupted writes) are deleted.
2. tier 1 — age cap: any event dir (or top-level NPZ) whose NEWEST
   member file is older than ``max_age_h`` is deleted.
3. tier 2 — size cap: if the staging total still exceeds
   ``max_total_gb``, event dirs are deleted oldest-first until the
   total drops below ``low_water_gb`` — preferring dirs whose
   ``upload.log`` carries an ``UPLOAD_OK`` marker; un-uploaded dirs
   are only sacrificed if the uploaded ones weren't enough, each with
   a loud warning (that data exists nowhere else).

Both GPU halves run a sweeper over the same ``dump_root``; a
non-blocking ``flock`` on ``<dump_root>/.retention.lock`` makes the
passes mutually exclusive, and every delete tolerates
``FileNotFoundError`` so an operator (or the sibling half between
locks) racing us is harmless.
"""

from __future__ import annotations

import dataclasses
import fcntl
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["CubeRetentionConfig", "CubeRetentionSweeper"]

_LOG = logging.getLogger("dsart.dump.cube_retention")

_LOCK_NAME = ".retention.lock"
_UPLOAD_OK_MARKER = "UPLOAD_OK"


@dataclass(frozen=True)
class CubeRetentionConfig:
    """Knobs for :class:`CubeRetentionSweeper` (yaml: ``c1.retention``)."""

    dump_root: Path
    #: Delete event dirs / stray NPZs whose newest file is older than
    #: this many hours, uploaded or not. Sized so a multi-day h23 /
    #: network outage doesn't silently discard fresh events, while a
    #: forgotten dir can't live forever.
    max_age_h: float = 96.0
    #: Start deleting oldest-first when the staging total exceeds this.
    max_total_gb: float = 150.0
    #: ... and stop once the total is back under this.
    low_water_gb: float = 120.0
    #: Age for tier-0 hygiene deletes of ``*.tmp`` / 0-byte NPZs.
    tmp_age_h: float = 2.0
    #: Seconds between sweep passes.
    sweep_interval_s: float = 600.0


@dataclass
class _Entry:
    path: Path
    is_dir: bool
    newest_mtime: float
    n_bytes: int
    uploaded: bool


def _scan_entry(path: Path) -> Optional[_Entry]:
    """Size/mtime/uploaded summary for one event dir or loose file."""
    try:
        if path.is_dir():
            newest = 0.0
            total = 0
            uploaded = False
            for f in path.iterdir():
                try:
                    st = f.stat()
                except FileNotFoundError:
                    continue
                newest = max(newest, st.st_mtime)
                total += st.st_size
                if f.name == "upload.log":
                    try:
                        uploaded = _UPLOAD_OK_MARKER in f.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        uploaded = False
            if newest == 0.0:
                newest = path.stat().st_mtime
            return _Entry(path, True, newest, total, uploaded)
        st = path.stat()
        return _Entry(path, False, st.st_mtime, st.st_size, False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _LOG.warning("cube_retention: scan failed for %s: %r", path, exc)
        return None


def _delete(entry: _Entry) -> int:
    """Best-effort delete; returns bytes freed (0 on failure)."""
    try:
        if entry.is_dir:
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            entry.path.unlink()
        return entry.n_bytes
    except FileNotFoundError:
        return 0
    except OSError as exc:
        _LOG.warning(
            "cube_retention: delete failed for %s: %r", entry.path, exc
        )
        return 0


def sweep_once(config: CubeRetentionConfig, *,
               now: Optional[float] = None) -> dict:
    """One sweep pass. Returns a stats dict (also used by tests).

    Caller is responsible for locking (see
    :meth:`CubeRetentionSweeper._run`); this function only enumerates
    and deletes.
    """
    t_now = time.time() if now is None else float(now)
    root = Path(config.dump_root)
    stats = {
        "n_tmp_deleted": 0,
        "n_age_deleted": 0,
        "n_size_deleted": 0,
        "n_unuploaded_sacrificed": 0,
        "bytes_freed": 0,
        "bytes_after": 0,
    }
    if not root.is_dir():
        return stats

    # ---- tier 0: hygiene --------------------------------------------------
    tmp_cutoff = t_now - config.tmp_age_h * 3600.0
    try:
        top = list(root.iterdir())
    except OSError as exc:
        _LOG.warning("cube_retention: cannot list %s: %r", root, exc)
        return stats
    for f in top:
        if f.is_dir() or f.name == _LOCK_NAME:
            continue
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        is_tmp = f.name.endswith(".tmp")
        is_zero_npz = f.suffix == ".npz" and st.st_size == 0
        if (is_tmp or is_zero_npz) and st.st_mtime < tmp_cutoff:
            try:
                f.unlink()
                stats["n_tmp_deleted"] += 1
                stats["bytes_freed"] += st.st_size
            except OSError:
                pass

    # ---- enumerate --------------------------------------------------------
    entries = [e for e in (_scan_entry(p) for p in root.iterdir()
                           if p.name != _LOCK_NAME) if e is not None]

    # ---- tier 1: age cap --------------------------------------------------
    age_cutoff = t_now - config.max_age_h * 3600.0
    survivors = []
    for e in entries:
        if e.newest_mtime < age_cutoff:
            freed = _delete(e)
            stats["n_age_deleted"] += 1
            stats["bytes_freed"] += freed
            _LOG.info(
                "cube_retention: age-expired %s (%.1f GB, uploaded=%s, "
                "age %.1f h > max_age_h %.1f)",
                e.path.name, e.n_bytes / 1e9, e.uploaded,
                (t_now - e.newest_mtime) / 3600.0, config.max_age_h,
            )
        else:
            survivors.append(e)

    # ---- tier 2: size cap -------------------------------------------------
    total = sum(e.n_bytes for e in survivors)
    if total > config.max_total_gb * 1e9:
        low_water = config.low_water_gb * 1e9
        # uploaded dirs first (oldest first), then un-uploaded ones.
        dirs = [e for e in survivors if e.is_dir]
        ordered = (
            sorted((e for e in dirs if e.uploaded),
                   key=lambda e: e.newest_mtime)
            + sorted((e for e in dirs if not e.uploaded),
                     key=lambda e: e.newest_mtime)
        )
        for e in ordered:
            if total <= low_water:
                break
            if not e.uploaded:
                _LOG.warning(
                    "cube_retention: size cap sacrificing UN-UPLOADED "
                    "event dir %s (%.1f GB) — this event's cubes exist "
                    "nowhere else!", e.path.name, e.n_bytes / 1e9,
                )
                stats["n_unuploaded_sacrificed"] += 1
            freed = _delete(e)
            stats["n_size_deleted"] += 1
            stats["bytes_freed"] += freed
            total -= e.n_bytes
    stats["bytes_after"] = total
    return stats


class CubeRetentionSweeper:
    """Periodic background sweeper thread (one per search-compute half).

    Start/stop lifecycle mirrors the other dump-side workers. The
    inter-process ``flock`` means the two halves' sweepers never run a
    pass concurrently — whichever wins the lock does the work and the
    other skips its tick.
    """

    def __init__(self, config: CubeRetentionConfig,
                 thread_name: str = "cube-retention") -> None:
        self._config = config
        self._thread_name = str(thread_name)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._n_sweeps = 0
        self._last_stats: dict = {}

    @property
    def n_sweeps(self) -> int:
        return self._n_sweeps

    @property
    def last_stats(self) -> dict:
        return dict(self._last_stats)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=self._thread_name, daemon=True
        )
        self._thread.start()
        _LOG.info(
            "CubeRetentionSweeper started: root=%s max_age_h=%.0f "
            "max_total_gb=%.0f low_water_gb=%.0f interval_s=%.0f",
            self._config.dump_root, self._config.max_age_h,
            self._config.max_total_gb, self._config.low_water_gb,
            self._config.sweep_interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def sweep_now(self) -> dict:
        """One locked pass (also the periodic body). Public for tests
        and for an eager sweep at service start."""
        root = Path(self._config.dump_root)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return {}
        lock_path = root / _LOCK_NAME
        try:
            with open(lock_path, "w") as lock_fh:
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return {}  # sibling half is sweeping — skip
                stats = sweep_once(self._config)
        except OSError as exc:
            _LOG.warning("cube_retention: sweep skipped: %r", exc)
            return {}
        self._n_sweeps += 1
        self._last_stats = stats
        if stats.get("bytes_freed"):
            _LOG.info(
                "cube_retention: freed %.2f GB (tmp=%d age=%d size=%d "
                "unuploaded_sacrificed=%d), %.1f GB staged after sweep",
                stats["bytes_freed"] / 1e9, stats["n_tmp_deleted"],
                stats["n_age_deleted"], stats["n_size_deleted"],
                stats["n_unuploaded_sacrificed"],
                stats["bytes_after"] / 1e9,
            )
        return stats

    def _run(self) -> None:
        # Eager pass at start (catches a backlog accumulated while the
        # service was down), then periodic.
        self.sweep_now()
        while not self._stop_event.wait(self._config.sweep_interval_s):
            self.sweep_now()

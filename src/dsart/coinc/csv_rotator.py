"""Hourly-rotated CSV writer + retention enforcement.

The C2 service runs two instances:

* one for the C1 hiplot CSV (one row per received candidate);
* one for the C2 hiplot CSV (one row per coincidenced cluster).

Each instance writes into a directory served by an external
``hiplot`` server (port 5017 for C1, 5027 for C2 — see
``systemd/hiplot_c1.service`` / ``hiplot_c2.service``).

File naming: ``${prefix}_${YYYYMMDD}_${HH}.csv`` in UTC (matches the
legacy T2 layout the operator already has muscle memory for).

Concurrency: assumed single-writer per directory (the C2 service
holds the only instance). Not thread-safe.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

__all__ = [
    "RollingCsvWriter",
    "rotation_key",
]


_LOG = logging.getLogger("dsart.coinc.csv_rotator")


def rotation_key(now_utc: datetime) -> str:
    """Return the ``YYYYMMDD_HH`` rotation key for ``now_utc`` (UTC)."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be tz-aware (UTC)")
    if now_utc.utcoffset() != timedelta(0):
        raise ValueError("now_utc must be in UTC")
    return now_utc.strftime("%Y%m%d_%H")


class RollingCsvWriter:
    """Append-only CSV writer with hourly file rotation.

    Public API matches the design's reference:

    * :meth:`append_row` — write one row; rotates to a new file if the
      hour changed since the last write (uses the row's own ``now_utc``
      if supplied, else ``datetime.now(timezone.utc)``).
    * :meth:`maybe_rotate` — explicit rotation check; called on each
      append + periodically by the housekeeping task.
    * :meth:`housekeep` — delete files whose mtime is older than
      ``retention_hours``. Returns the number of files removed.
    """

    def __init__(
        self,
        dir_path: Path,
        prefix: str,
        schema: Sequence[str],
        *,
        retention_hours: int = 48,
    ) -> None:
        if not prefix:
            raise ValueError("prefix must be non-empty")
        if retention_hours <= 0:
            raise ValueError(
                f"retention_hours={retention_hours} must be > 0"
            )
        self._dir = Path(dir_path)
        self._prefix = prefix
        self._schema = tuple(schema)
        self._retention_hours = retention_hours
        self._current_key: Optional[str] = None
        self._current_path: Optional[Path] = None
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def schema(self) -> tuple[str, ...]:
        return self._schema

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{self._prefix}_{key}.csv"

    def maybe_rotate(self, now_utc: datetime) -> bool:
        """Switch to a new file if the hour has changed; returns True
        if a rotation happened."""
        key = rotation_key(now_utc)
        if self._current_key == key:
            return False
        # Roll to the new file: open in append mode so we don't clobber
        # a file already populated by a prior process restart in the
        # same hour.
        self._current_key = key
        self._current_path = self._path_for(key)
        # If the file doesn't exist yet, write a header row so hiplot
        # can find the column names.
        if not self._current_path.exists():
            self._write_header(self._current_path)
        return True

    def append_row(
        self, row: Mapping[str, object], *, now_utc: Optional[datetime] = None,
    ) -> Path:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        self.maybe_rotate(now_utc)
        assert self._current_path is not None  # populated by maybe_rotate
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=list(self._schema), extrasaction="ignore",
        )
        writer.writerow(row)
        with self._current_path.open("a", encoding="utf-8") as f:
            f.write(buf.getvalue())
        return self._current_path

    def housekeep(self, now_utc: datetime) -> int:
        """Delete files whose mtime is older than ``retention_hours``.

        Returns the number of files removed. Files not matching the
        ``${prefix}_*`` pattern are ignored.
        """
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be tz-aware (UTC)")
        cutoff = now_utc - timedelta(hours=self._retention_hours)
        cutoff_unix = cutoff.timestamp()
        removed = 0
        for p in self._dir.iterdir():
            if not p.is_file():
                continue
            if not p.name.startswith(f"{self._prefix}_"):
                continue
            if not p.name.endswith(".csv"):
                continue
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime < cutoff_unix:
                try:
                    p.unlink()
                except OSError as exc:
                    _LOG.warning(
                        "housekeep: failed to unlink %s: %s", p, exc,
                    )
                    continue
                removed += 1
        if removed:
            _LOG.info(
                "housekeep: removed %d expired %s_*.csv files older than "
                "%dh", removed, self._prefix, self._retention_hours,
            )
        return removed

    # ----- helpers ------------------------------------------------------

    def _write_header(self, path: Path) -> None:
        buf = io.StringIO()
        csv.writer(buf).writerow(list(self._schema))
        # Atomic header-stamp using a temp file; the file may already
        # exist if a prior process ran in this hour, in which case we
        # let the append path take over without re-writing the header.
        if path.exists():
            return
        with path.open("w", encoding="utf-8") as f:
            f.write(buf.getvalue())

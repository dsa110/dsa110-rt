"""Rolling time-window buffer for in-flight C1 candidates on C2.

The C2 receiver feeds parsed :class:`wire.C1Batch` payloads into the
window; the window keeps a chronologically ordered view of the most
recent ``window_s`` seconds (measured against the *most recent* MJD
seen so far — *not* wall-clock) and emits the candidates that have
aged out on the latest add.

The window is the single source of truth used by
:mod:`dsart.coinc.components` (for connected-components edges) and
:mod:`dsart.coinc.stats` (for cluster characterisation), so we keep
the in-window membership *sorted by MJD ascending* so downstream
modules can scan candidates monotonically.

See ``docs/c1c2/C1C2_DESIGN.md`` §3.3 for the receiver-side flow this
sits inside.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Tuple

from .wire import C1BatchHeader, C1CandidateRow

__all__ = [
    "WindowEntry",
    "TimeWindow",
]


@dataclass(frozen=True, slots=True)
class WindowEntry:
    """One in-window candidate (a C1 row + its provenance + MJD)."""

    mjd: float
    snr: float
    l_rad: float
    m_rad: float
    l_pix: int
    m_pix: int
    dm_pc_cc: float
    dm_idx_global: int
    fine_dm_idx: int
    event_specnum: int
    width_samples: int
    kernel_id: str
    flags: int
    # Provenance: the header carries sample_period_us etc. that the
    # connected-components edge predicate needs.
    search_node_id: int
    gpu_half: int
    cube_id: int
    sample_period_us: float

    @classmethod
    def from_row(
        cls,
        header: C1BatchHeader,
        row: C1CandidateRow,
        mjd: float,
    ) -> "WindowEntry":
        return cls(
            mjd=mjd,
            snr=row.snr,
            l_rad=row.l_rad,
            m_rad=row.m_rad,
            l_pix=row.l_pix,
            m_pix=row.m_pix,
            dm_pc_cc=row.dm_pc_cc,
            dm_idx_global=row.dm_idx_global,
            fine_dm_idx=row.fine_dm_idx,
            event_specnum=row.event_specnum,
            width_samples=row.width_samples,
            kernel_id=row.kernel_id,
            flags=row.flags,
            search_node_id=header.search_node_id,
            gpu_half=header.gpu_half,
            cube_id=header.cube_id,
            sample_period_us=header.sample_period_us,
        )


class TimeWindow:
    """Bounded buffer of :class:`WindowEntry` ordered by MJD.

    Concurrency: this class is *not* thread-safe. The C2 service drives
    one reader task per (s, g) connection, all multiplexed onto a single
    asyncio loop, so the writes serialise naturally.
    """

    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s={window_s} must be > 0")
        self._window_s = float(window_s)
        # Keep MJDs and entries in lock-step so bisect on _mjds gives
        # us the correct slice into _entries.
        self._entries: Deque[WindowEntry] = deque()
        self._mjds: Deque[float] = deque()
        # The latest MJD ever inserted; anchors the age-out cutoff
        # (so a single late batch can't shrink the window).
        self._latest_mjd: float = -1.0
        # Stash the entries removed by the most recent age-out so the
        # service can pass them to the components graph for eviction.
        self._last_aged_out: List[WindowEntry] = []

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def latest_mjd(self) -> float:
        """The most recent MJD seen (anchors the age-out cutoff)."""
        return self._latest_mjd

    def __len__(self) -> int:
        return len(self._entries)

    # ----- mutation -------------------------------------------------------

    def add(
        self,
        header: C1BatchHeader,
        rows: Iterable[C1CandidateRow],
    ) -> List[WindowEntry]:
        """Append a batch of C1 rows; returns the entries just inserted.

        Updates the age-out anchor to ``max(latest_mjd, max(row mjd))``.
        Late-arriving rows older than the cutoff are still inserted (so
        we don't drop them silently) but will not extend the window.

        Call :meth:`aged_out` to retrieve the entries the age-out
        pushed out of the window on this add.
        """
        new_entries: List[WindowEntry] = []
        max_mjd = self._latest_mjd
        for row in rows:
            mjd = header.candidate_mjd(row.event_specnum)
            entry = WindowEntry.from_row(header, row, mjd)
            self._insert_sorted(entry)
            new_entries.append(entry)
            if mjd > max_mjd:
                max_mjd = mjd
        if max_mjd > self._latest_mjd:
            self._latest_mjd = max_mjd
        self._last_aged_out = self._age_out_below(self._cutoff_mjd())
        return new_entries

    def aged_out(self) -> List[WindowEntry]:
        """Return + clear the entries removed by the most recent add."""
        out = self._last_aged_out
        self._last_aged_out = []
        return out

    # ----- read -----------------------------------------------------------

    def snapshot(self) -> List[WindowEntry]:
        """All currently in-window entries, MJD-ascending."""
        return list(self._entries)

    def __iter__(self) -> Iterator[WindowEntry]:
        return iter(self._entries)

    # ----- internals ------------------------------------------------------

    def _cutoff_mjd(self) -> float:
        if self._latest_mjd < 0.0:
            return -1.0
        return self._latest_mjd - (self._window_s / 86400.0)

    def _insert_sorted(self, entry: WindowEntry) -> None:
        # bisect needs a sliceable sequence; deque supports __getitem__
        # but not bisect.insort directly (it tries to assign by index).
        # For window depths typically < 100k entries the convert-to-list
        # cost would dominate, so we maintain a parallel mjds list view
        # for bisect and insert into the deque manually.
        idx = bisect.bisect_right(self._mjds, entry.mjd)
        # Cheap fast-path: typical append at the right end.
        if idx == len(self._mjds):
            self._mjds.append(entry.mjd)
            self._entries.append(entry)
            return
        # Rare out-of-order insertion (very late batch). deque.insert
        # is O(n); these are infrequent so we accept the cost.
        self._mjds.insert(idx, entry.mjd)
        self._entries.insert(idx, entry)

    def _age_out_below(self, cutoff: float) -> List[WindowEntry]:
        if cutoff < 0.0 or not self._mjds:
            return []
        removed: List[WindowEntry] = []
        while self._mjds and self._mjds[0] < cutoff:
            removed.append(self._entries.popleft())
            self._mjds.popleft()
        return removed
